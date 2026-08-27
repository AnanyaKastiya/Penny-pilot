import json
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from ..ai.parser import parse_expense_text, parse_receipt_image, ParsedTransaction
from ..ai.memory import memory_store
from ..database.db import add_transaction, get_user_categories

class AgentState(TypedDict):
    user_id: str
    raw_text: Optional[str]
    image_bytes: Optional[bytes]
    mime_type: Optional[str]
    api_key: Optional[str]
    allowed_categories: Optional[List[str]]
    parsed_transactions: List[Dict[str, Any]]
    needs_clarification: bool
    pending_questions: List[Dict[str, Any]]
    recorded_ids: List[int]
    error: Optional[str]

# ----------------- NODE DEFINITIONS ----------------- #

def parse_input_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Extracts ALL transactions (debits & credits) from text or UPI screenshot."""
    api_key = state.get("api_key")
    user_id = state.get("user_id", "guest")
    allowed_cats = state.get("allowed_categories") or get_user_categories(user_id)
    transactions = []
    
    try:
        if state.get("image_bytes"):
            tx_list = parse_receipt_image(
                image_bytes=state["image_bytes"],
                mime_type=state.get("mime_type", "image/jpeg"),
                api_key=api_key,
                allowed_categories=allowed_cats
            )
            transactions = [t.model_dump() for t in tx_list]
        elif state.get("raw_text"):
            tx_list = parse_expense_text(
                state["raw_text"], 
                api_key=api_key,
                allowed_categories=allowed_cats
            )
            transactions = [t.model_dump() for t in tx_list]
        else:
            return {"error": "No text or image provided to process."}

        return {"parsed_transactions": transactions, "allowed_categories": allowed_cats, "error": None}
    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}"}

def rag_merchant_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Checks ChromaDB vector memory for permanent rules vs variable friends."""
    transactions = state.get("parsed_transactions", [])
    updated_list = []

    for tx in transactions:
        merchant = tx.get("recipient_name")
        match = memory_store.query_merchant(merchant)
        
        if match:
            if match.get("is_permanent_rule"):
                tx["category"] = match["category"]
                tx["notes"] = f"{tx.get('notes') or ''} (Auto-resolved: {match['notes']})".strip()
                tx["confidence"] = 0.98
                tx["needs_clarification"] = False
                tx["clarification_question"] = None
            else:
                tx["category"] = match["category"]
                tx["clarification_question"] = f"Last time you interacted with {merchant} for '{match.get('notes', match['category'])}'. What was this payment for?"
                tx["needs_clarification"] = True
        
        updated_list.append(tx)

    return {"parsed_transactions": updated_list}

def clarification_router_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Compiles ALL questions for EVERY ambiguous transaction simultaneously."""
    transactions = state.get("parsed_transactions", [])
    pending_questions = []

    for idx, tx in enumerate(transactions):
        if tx.get("needs_clarification"):
            ttype = tx.get("transaction_type", "DEBIT")
            action_word = "received from" if ttype == "CREDIT" else "paid to"
            q = tx.get("clarification_question") or f"What was this ₹{tx.get('amount')} {action_word} '{tx.get('recipient_name')}' for on {tx.get('date')}?"
            pending_questions.append({
                "index": idx,
                "recipient_name": tx.get("recipient_name"),
                "amount": tx.get("amount"),
                "date": tx.get("date"),
                "transaction_type": ttype,
                "suggested_category": tx.get("category", "Miscellaneous"),
                "question": q,
                "entity_type": tx.get("entity_type", "merchant")
            })

    needs_clarification = len(pending_questions) > 0
    return {
        "needs_clarification": needs_clarification,
        "pending_questions": pending_questions
    }

def save_transaction_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Persists confirmed unambiguous transactions into SQLite and indexes in ChromaDB."""
    transactions = state.get("parsed_transactions", [])
    user_id = state.get("user_id", "guest")
    recorded_ids = []

    for tx in transactions:
        if tx.get("needs_clarification"):
            continue

        tx_id = add_transaction(
            date=tx.get("date"),
            amount=float(tx.get("amount", 0.0)),
            recipient_name=tx.get("recipient_name", "Unknown"),
            recipient_upi=tx.get("recipient_upi"),
            category=tx.get("category", "Miscellaneous"),
            payment_app=tx.get("payment_app", "Manual"),
            transaction_type=tx.get("transaction_type", "DEBIT"),
            raw_input=state.get("raw_text") or "UPI Screenshot",
            notes=tx.get("notes"),
            confidence_score=float(tx.get("confidence", 1.0)),
            is_clarified=1,
            user_id=user_id
        )
        recorded_ids.append(tx_id)

        memory_store.index_transaction(
            tx_id=tx_id,
            date=tx.get("date"),
            amount=float(tx.get("amount", 0.0)),
            recipient=tx.get("recipient_name", "Unknown"),
            category=tx.get("category", "Miscellaneous"),
            notes=tx.get("notes") or ""
        )

    return {"recorded_ids": recorded_ids}

# ----------------- CONDITIONAL EDGES ----------------- #

def route_after_clarification_check(state: AgentState) -> str:
    """Routes to save_node if all clear, otherwise ends turn to wait for user input."""
    if state.get("needs_clarification"):
        return "wait_for_user"
    return "save_transaction"

# ----------------- GRAPH COMPILATION ----------------- #

def build_pennypilot_graph():
    """Compiles the LangGraph State Machine."""
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("rag_merchant", rag_merchant_node)
    workflow.add_node("clarification_check", clarification_router_node)
    workflow.add_node("save_transaction", save_transaction_node)

    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "rag_merchant")
    workflow.add_edge("rag_merchant", "clarification_check")

    workflow.add_conditional_edges(
        "clarification_check",
        route_after_clarification_check,
        {
            "save_transaction": "save_transaction",
            "wait_for_user": END
        }
    )
    workflow.add_edge("save_transaction", END)

    return workflow.compile()

# Global Compiled Graph
pennypilot_agent = build_pennypilot_graph()

def process_batch_clarification(
    state: AgentState,
    clarified_items: List[Dict[str, Any]]
) -> AgentState:
    """Applies user clarification for ALL items, sets DEBIT/CREDIT properly, and commits to database."""
    transactions = state.get("parsed_transactions", [])
    user_id = state.get("user_id", "guest")
    recorded_ids = []

    updates_by_idx = {item["index"]: item for item in clarified_items}

    for idx, tx in enumerate(transactions):
        if idx in updates_by_idx:
            u = updates_by_idx[idx]
            chosen_cat = u.get("category", tx.get("category", "Miscellaneous"))
            chosen_type = u.get("transaction_type", tx.get("transaction_type", "DEBIT"))
            user_note = u.get("notes", "")
            remember_rule = u.get("remember_rule", False)

            tx["category"] = chosen_cat
            tx["transaction_type"] = chosen_type
            tx["notes"] = f"{tx.get('notes') or ''} {user_note}".strip()
            tx["needs_clarification"] = False
            tx["confidence"] = 1.0

            memory_store.save_merchant_mapping(
                merchant_name=tx.get("recipient_name", ""),
                category=chosen_cat,
                notes=user_note,
                upi_id=tx.get("recipient_upi"),
                is_permanent_rule=remember_rule
            )

        tx_id = add_transaction(
            date=tx.get("date"),
            amount=float(tx.get("amount", 0.0)),
            recipient_name=tx.get("recipient_name", "Unknown"),
            recipient_upi=tx.get("recipient_upi"),
            category=tx.get("category", "Miscellaneous"),
            payment_app=tx.get("payment_app", "Manual"),
            transaction_type=tx.get("transaction_type", "DEBIT"),
            raw_input=state.get("raw_text") or "UPI Screenshot",
            notes=tx.get("notes"),
            confidence_score=float(tx.get("confidence", 1.0)),
            is_clarified=1,
            user_id=user_id
        )
        recorded_ids.append(tx_id)

        memory_store.index_transaction(
            tx_id=tx_id,
            date=tx.get("date"),
            amount=float(tx.get("amount", 0.0)),
            recipient=tx.get("recipient_name", "Unknown"),
            category=tx.get("category", "Miscellaneous"),
            notes=tx.get("notes") or ""
        )

    state["parsed_transactions"] = transactions
    state["needs_clarification"] = False
    state["pending_questions"] = []
    state["recorded_ids"] = recorded_ids
    return state
