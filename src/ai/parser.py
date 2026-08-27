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
    recipient_name: str = Field("Unknown", description="Recipient merchant name or person name")
    recipient_upi: Optional[str] = Field(None, description="UPI ID if found (e.g., merchant@okaxis)")
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    time: Optional[str] = Field(None, description="Transaction time in HH:MM format if available")
    category: str = Field("Miscellaneous", description="One of the standard categories")
    payment_app: str = Field("Manual", description="GPay, PhonePe, Paytm, CRED, Bank, or Manual")
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
    """Parses natural language expense text into one or more structured transactions with individual questions."""
    client = get_client(api_key)
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    categories_str = ", ".join(DEFAULT_CATEGORIES)
    system_instruction = f"""You are PennyPilot, an expert financial entity extractor.
Current Date: {today_str}
Allowed Categories: [{categories_str}]

Task: Extract ALL financial transactions from the user's input (single or multi-day).

Rules for Ambiguity & Clarification:
1. For every transaction where the recipient is a friend/person (e.g., 'Mahima', 'Rohan', 'Ankit') or a multi-category store (e.g. 'Blinkit', 'Amazon', 'Zepto'), you MUST set needs_clarification=True.
2. Generate a custom, polite clarification_question for EACH ambiguous transaction (e.g., 'What was this ₹320 payment to Mahima for? (e.g., Food, Movie, Cab)' or 'What did you buy on Blinkit? (Groceries / Food / Skincare)').
3. For clear, unambiguous merchants (e.g., 'Uber' -> Travel & Commute, 'BESCOM' -> Bills & Utilities, 'Netflix' -> Entertainment), set needs_clarification=False and confidence=1.0.
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
            notes=text,
            needs_clarification=True,
            clarification_question="Could not automatically parse. Please enter details."
        )
    ]

def parse_receipt_image(image_bytes: bytes, mime_type: str = "image/jpeg", api_key: Optional[str] = None) -> List[ParsedTransaction]:
    """Scans and extracts ALL visible transactions (1 to 20+) from a UPI history/passbook screenshot."""
    client = get_client(api_key)
    today_str = datetime.now().strftime("%Y-%m-%d")
    categories_str = ", ".join(DEFAULT_CATEGORIES)

    system_instruction = f"""You are PennyPilot, an elite multimodal financial scanner specialized in Indian UPI screenshots (Google Pay, PhonePe, Paytm, CRED, Bank Statements).
Current Date: {today_str}
Allowed Categories: [{categories_str}]

CRITICAL INSTRUCTIONS:
1. Scan the ENTIRE image from top to bottom. If the screenshot contains a list/history/passbook of multiple transactions (e.g. 5, 10, or 20 transactions), EXTRACT EVERY SINGLE TRANSACTION. Do not stop after 1 or 4 transactions.
2. For each transaction, extract:
   - amount: Numeric value (exclude cashback, rewards, or wallet balance).
   - recipient_name: Exact name of the person, shop, or company.
   - recipient_upi: UPI ID or phone number if visible.
   - date: Exact YYYY-MM-DD. If year is missing, infer 2026 based on Current Date.
   - payment_app: GPay, PhonePe, Paytm, CRED, or Bank.

3. INDIVIDUAL CLARIFICATION RULES FOR EVERY TRANSACTION:
   - If the recipient is a friend/peer (e.g., 'Mahima', 'Suresh Kumar', 'Ankit', 'Rahul') -> set entity_type='peer_friend', needs_clarification=True, and generate a specific question: "What was this ₹X payment to [Name] for on [Date]? (e.g., Food, Cab, Movie, Rent)".
   - If the recipient is a multi-category store (e.g., 'Blinkit', 'Amazon', 'Zepto', 'Instamart') -> set entity_type='multi_store', needs_clarification=True, and generate question: "What did you purchase on [Name] for ₹X? (Groceries / Food / Skincare)".
   - If the recipient is an unknown personal name / small shop without clear business context -> set needs_clarification=True and formulate a question.
   - If the recipient is a 100% distinct single-category brand (e.g. 'Uber' -> Travel, 'Netflix' -> Entertainment, 'BESCOM' -> Bills, 'Airtel' -> Bills) -> set needs_clarification=False and confidence=1.0.
"""

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=["Extract all transactions and generate clarification questions for ambiguous items:", image_part],
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
            needs_clarification=True,
            clarification_question="Could not automatically extract receipt details. Please enter manually."
        )
    ]
