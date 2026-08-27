import json
import os
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from PIL import Image
import io

from ..config import GEMINI_API_KEY, DEFAULT_CATEGORIES, MODEL_CANDIDATES

class ParsedTransaction(BaseModel):
    amount: float = Field(..., description="The transaction amount in INR (Rupees)")
    recipient_name: str = Field("Unknown", description="Recipient merchant name or person who paid/received")
    recipient_upi: Optional[str] = Field(None, description="UPI ID if found (e.g., merchant@okaxis)")
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    time: Optional[str] = Field(None, description="Transaction time in HH:MM format if available")
    category: str = Field("Miscellaneous", description="One of the standard categories")
    payment_app: str = Field("Manual", description="GPay, PhonePe, Paytm, CRED, Bank, or Manual")
    transaction_type: str = Field("DEBIT", description="'DEBIT' if user paid/sent money, 'CREDIT' if user received/got money or refund")
    confidence: float = Field(1.0, description="Confidence score between 0.0 and 1.0")
    notes: Optional[str] = Field(None, description="Additional context or items bought")
    entity_type: str = Field("merchant", description="'merchant', 'peer_friend', 'multi_store', or 'unknown'")
    needs_clarification: bool = Field(False, description="True if merchant is an individual, friend, multi-category store, or ambiguous")
    clarification_question: Optional[str] = Field(None, description="Specific, tailored question to ask the user for this exact transaction")

class MultiTransactionResponse(BaseModel):
    transactions: List[ParsedTransaction]

def get_client(api_key: Optional[str] = None) -> genai.Client:
    """Instantiates Google GenAI client with key from parameter or env."""
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured. Please set it in .env or provide it in the UI.")
    return genai.Client(api_key=key)

def parse_expense_text(text: str, api_key: Optional[str] = None) -> List[ParsedTransaction]:
    """Parses natural language expense text into one or more structured transactions."""
    client = get_client(api_key)
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    categories_str = ", ".join(DEFAULT_CATEGORIES)
    system_instruction = f"""You are PennyPilot, an expert financial entity extractor.
Current Date: {today_str}
Allowed Categories: [{categories_str}]

Rules for Transaction Types:
1. 'DEBIT' = User spent, sent, paid, or bought something.
2. 'CREDIT' = User received money, friend repaid split bill, refund, or cashback (e.g., 'received 300 from Tanya', 'Mahima sent me 100 for dinner').
3. For peer transfers (Mahima, Tanya, Rahul), set entity_type='peer_friend' and needs_clarification=True.
"""

    prompt = f"User Input: \"{text}\""
    
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=MultiTransactionResponse,
                    temperature=0.1
                )
            )
            if response.text:
                parsed = json.loads(response.text)
                return [ParsedTransaction(**t) for t in parsed.get("transactions", [])]
        except Exception as e:
            print(f"Fallback from model {model_name}: {e}")
            continue

    return [
        ParsedTransaction(
            amount=0.0,
            recipient_name="Unknown",
            date=today_str,
            category="Miscellaneous",
            transaction_type="DEBIT",
            notes=text,
            needs_clarification=True,
            clarification_question="Could not automatically parse. Please enter details."
        )
    ]

def parse_receipt_image(image_bytes: bytes, mime_type: str = "image/jpeg", api_key: Optional[str] = None) -> List[ParsedTransaction]:
    """Scans and extracts ALL visible transactions (debits & credits) from a UPI history/passbook screenshot."""
    client = get_client(api_key)
    today_str = datetime.now().strftime("%Y-%m-%d")
    categories_str = ", ".join(DEFAULT_CATEGORIES)

    system_instruction = f"""You are PennyPilot, an elite multimodal financial scanner specialized in Indian UPI screenshots (Google Pay, PhonePe, Paytm, CRED, Bank Statements).
Current Date: {today_str}
Allowed Categories: [{categories_str}]

CRITICAL TRANSACTION TYPE DETECTION RULES:
1. Pay close attention to whether the transaction is an Outgoing Expense (DEBIT) or Incoming Money (CREDIT):
   - 'DEBIT' (Paid to, Sent to, Debited from account, minus sign '-', red color text, 'Paid ₹X').
   - 'CREDIT' (Received from, Credited to account, plus sign '+', green color text, 'Received ₹X', 'Cashback', 'Refund').
   - If a friend (e.g. Mahima, Tanya) sent/paid the user money, mark transaction_type='CREDIT', notes='Received payment / Bill reimbursement', and entity_type='peer_friend'.

2. Scan the ENTIRE screenshot from top to bottom and extract EVERY transaction.

3. INDIVIDUAL CLARIFICATION RULES:
   - For 'CREDIT' payments from friends, ask: "What was this ₹X received from [Name] for? (e.g., Food split reimbursement, Rent share)".
   - For 'DEBIT' payments to friends/unknowns, ask what the payment was for.
   - For obvious stores (Uber, BESCOM, Netflix), set needs_clarification=False and confidence=1.0.
"""

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=["Extract all transactions (debits and received credits):", image_part],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=MultiTransactionResponse,
                    temperature=0.1
                )
            )
            if response.text:
                parsed = json.loads(response.text)
                txs = [ParsedTransaction(**t) for t in parsed.get("transactions", [])]
                if txs:
                    return txs
        except Exception as e:
            print(f"Fallback image model {model_name}: {e}")
            continue

    return [
        ParsedTransaction(
            amount=0.0,
            recipient_name="Receipt Image",
            date=today_str,
            category="Miscellaneous",
            payment_app="UPI Screenshot",
            transaction_type="DEBIT",
            needs_clarification=True,
            clarification_question="Could not automatically extract receipt details. Please enter manually."
        )
    ]
