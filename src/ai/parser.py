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
    needs_clarification: bool = Field(False, description="True if merchant is an individual or category is ambiguous")
    clarification_question: Optional[str] = Field(None, description="Polite question to ask user if clarification is needed")

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
    system_instruction = f"""You are PennyPilot, an intelligent financial parser.
Current Date: {today_str}
Allowed Categories: [{categories_str}]

Task: Extract all financial transactions from the user's input.
Rules:
1. Extract exact amount, recipient/merchant, and infer the most accurate category.
2. If relative dates are used ('today', 'yesterday', 'on 24th Aug', 'last Sunday'), compute exact YYYY-MM-DD based on Current Date.
3. If an expense is to an individual with no context (e.g., 'sent 500 to Ankit'), mark needs_clarification=True, confidence=0.5, and formulate a clarification_question.
4. If multiple expenses are mentioned across multiple days in one prompt, extract all of them into the transactions list.
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

    # Fallback basic extraction
    return [
        ParsedTransaction(
            amount=0.0,
            recipient_name="Unknown",
            date=today_str,
            category="Miscellaneous",
            notes=text,
            needs_clarification=True,
            clarification_question="Could not automatically parse. Please enter the amount and category."
        )
    ]

def parse_receipt_image(image_bytes: bytes, mime_type: str = "image/jpeg", api_key: Optional[str] = None) -> ParsedTransaction:
    """Extracts transaction details from a UPI screenshot (GPay, PhonePe, Paytm, etc.)."""
    client = get_client(api_key)
    today_str = datetime.now().strftime("%Y-%m-%d")
    categories_str = ", ".join(DEFAULT_CATEGORIES)

    system_instruction = f"""You are PennyPilot, an expert UPI payment receipt analyzer.
Current Date: {today_str}
Allowed Categories: [{categories_str}]

Task: Inspect this payment screenshot (Google Pay, PhonePe, Paytm, CRED, or Bank App) and extract structured details.

Rules:
1. Amount: Find the exact numeric payment amount (e.g., ₹240.00 -> 240.0).
2. Recipient: Identify the Paid-to Name and UPI ID.
3. Payment App: Detect whether this is GPay, PhonePe, Paytm, CRED, BHIM, or Other.
4. Date & Time: Extract exact transaction timestamp if present on receipt.
5. Ambiguity Check:
   - If recipient is a recognizable merchant (Swiggy, Uber, Zomato, Starbucks, D-Mart, Shell, BESCOM, Airtel), assign the right category with confidence=1.0 and needs_clarification=False.
   - If recipient is a generic personal name without context (e.g. 'Suresh Kumar'), set category='Miscellaneous', confidence=0.5, needs_clarification=True, and generate a polite clarification_question.
"""

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=["Please analyze this UPI screenshot and extract transaction data.", image_part],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=ParsedTransaction,
                    temperature=0.1
                )
            )
            if response.text:
                parsed = json.loads(response.text)
                return ParsedTransaction(**parsed)
        except Exception as e:
            print(f"Fallback image model {model_name}: {e}")
            continue

    return ParsedTransaction(
        amount=0.0,
        recipient_name="Receipt Image",
        date=today_str,
        category="Miscellaneous",
        payment_app="UPI Screenshot",
        needs_clarification=True,
        clarification_question="Could not automatically extract receipt details. Please enter manually."
    )
