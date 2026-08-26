import json
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from ..ai.parser import parse_expense_text, parse_receipt_image, ParsedTransaction
from ..ai.memory import memory_store
from ..database.db import add_transaction

class AgentState(TypedDict):
    user_id: str
    raw_text: Optional[str]
    image_bytes: Optional[bytes]
    mime_type: Optional[str]
    api_key: Optional[str]
    parsed_transactions: List[Dict[str, Any]]
    needs_clarification: bool
    clarification_question: Optional[str]
    clarification_index: Optional[int]
    user_response: Optional[str]
    recorded_ids: List[int]
    error: Optional[str]

# ----------------- NODE DEFINITIONS ----------------- #

def parse_input_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Parses natural language text or UPI image screenshot into transactions."""
    api_key = state.get("api_key")
    transactions = []
    
    try:
        if state.get("image_bytes"):
            tx = parse_receipt_image(
                image_bytes=state["image_bytes"],
                mime_type=state.get("mime_type", "image/jpeg"),
                api_key=api_key
            )
            transactions.append(tx.model_dump())
        elif state.get("raw_text"):
            tx_list = parse_expense_text(state["raw_text"], api_key=api_key)
            transactions = [t.model_dump() for t in tx_list]
        else:
            return {"error": "No text or image provided to process."}

        return {"parsed_transactions": transactions, "error": None}
    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}"}

def rag_merchant_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Queries ChromaDB vector memory to auto-resolve ambiguous merchants."""
    transactions = state.get("parsed_transactions", [])
    updated_list = []

    for tx in transactions:
        merchant = tx.get("recipient_name")
        # If confidence is low or needs clarification, query ChromaDB memory
        if tx.get("needs_clarification") or tx.get("confidence", 1.0) < 0.7:
            match = memory_store.query_merchant(merchant)
            if match:
                # Memory match found via RAG!
                tx["category"] = match["category"]
                tx["notes"] = f"{tx.get('notes') or ''} (Auto-resolved from memory: {match['notes']})".strip()
                tx["confidence"] = 0.95
                tx["needs_clarification"] = False
                tx["clarification_question"] = None
        updated_list.append(tx)

    return {"parsed_transactions": updated_list}

def clarification_router_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Checks if any transaction requires human-in-the-loop clarification."""
    transactions = state.get("parsed_transactions", [])
    
    for idx, tx in enumerate(transactions):
        if tx.get("needs_clarification"):
            q = tx.get("clarification_question") or f"What category is ₹{tx.get('amount')} to '{tx.get('recipient_name')}' for?"
            return {
                "needs_clarification": True,
                "clarification_question": q,
                "clarification_index": idx
            }

    return {"needs_clarification": False, "clarification_question": None, "clarification_index": None}

def save_transaction_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Persists all verified transactions into SQLite and indexes in ChromaDB."""
    transactions = state.get("parsed_transactions", [])
    user_id = state.get("user_id", "guest")
    recorded_ids = []

    for tx in transactions:
        # Don't save incomplete transactions that still need clarification
        if tx.get("needs_clarification"):
            continue

        tx_id = add_transaction(
            date=tx.get("date"),
            amount=float(tx.get("amount", 0.0)),
            recipient_name=tx.get("recipient_name", "Unknown"),
            recipient_upi=tx.get("recipient_upi"),
            category=tx.get("category", "Miscellaneous"),
            payment_app=tx.get("payment_app", "Manual"),
            raw_input=state.get("raw_text") or "UPI Screenshot",
            notes=tx.get("notes"),
            confidence_score=float(tx.get("confidence", 1.0)),
            is_clarified=1,
            user_id=user_id
        )
        recorded_ids.append(tx_id)

        # Index transaction into ChromaDB for semantic search
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

def process_clarification_response(
    state: AgentState,
    chosen_category: str,
    user_notes: str = ""
) -> AgentState:
    """Applies user clarification, teaches ChromaDB memory, and saves the transaction."""
    idx = state.get("clarification_index")
    if idx is None:
        idx = 0
    transactions = state.get("parsed_transactions", [])

    if transactions and 0 <= idx < len(transactions):
        tx = transactions[idx]
        tx["category"] = chosen_category
        tx["notes"] = f"{tx.get('notes') or ''} {user_notes}".strip()
        tx["needs_clarification"] = False
        tx["confidence"] = 1.0
        
        # 🧠 Teach ChromaDB vector memory permanently!
        memory_store.save_merchant_mapping(
            merchant_name=tx.get("recipient_name", ""),
            category=chosen_category,
            notes=user_notes,
            upi_id=tx.get("recipient_upi")
        )

        state["parsed_transactions"] = transactions
        state["needs_clarification"] = False

        # Run save node
        save_res = save_transaction_node(state)
        state["recorded_ids"] = save_res.get("recorded_ids", [])

    return state
