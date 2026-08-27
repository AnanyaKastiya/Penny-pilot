import json
import os
import io
from datetime import datetime
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from PIL import Image

from ..config import GEMINI_API_KEY, DEFAULT_CATEGORIES, MODEL_CANDIDATES

class ParsedTransaction(BaseModel):
    amount: float = Field(..., description="The transaction amount in INR (Rupees)")
    recipient_name: str = Field("Unknown", description="Recipient merchant name or person who paid/received")
    recipient_upi: Optional[str] = Field(None, description="UPI ID if found (e.g., merchant@okaxis)")
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    time: Optional[str] = Field(None, description="Transaction time in HH:MM format if available")
    category: str = Field("Miscellaneous", description="One of the allowed categories")
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

def optimize_image_for_fast_ocr(image_bytes: bytes, max_dimension: int = 1200, quality: int = 85) -> Tuple[bytes, str]:
    """Resizes and compresses screenshots for ultra-fast network transfer and instantaneous OCR."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        
        w, h = img.size
        if max(w, h) > max_dimension:
            ratio = max_dimension / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        out_buf = io.BytesIO()
        img.save(out_buf, format='JPEG', quality=quality, optimize=True)
        return out_buf.getvalue(), 'image/jpeg'
    except Exception:
        return image_bytes, 'image/jpeg'

def parse_expense_text(
    text: str, 
    api_key: Optional[str] = None, 
    allowed_categories: Optional[List[str]] = None
) -> List[ParsedTransaction]:
    """Parses natural language expense text into structured transactions considering custom categories."""
    client = get_client(api_key)
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    cats = allowed_categories or DEFAULT_CATEGORIES
    categories_str = ", ".join(cats)
    system_instruction = f"""You are PennyPilot, an elite multimodal financial scanner specialized in extracting financial transactions from text and Indian UPI screenshots (Google Pay, PhonePe, Paytm, CRED, Bank Statements).
Current Date: {today_str}
Allowed Categories: [{categories_str}]

EXTRACTION & CLARIFICATION RULES:
1. Extract EVERY single transaction listed. If multiple dates/lines or multiple screenshot rows exist, extract ALL of them.
2. 'DEBIT' = User spent, sent, or paid money.
3. 'CREDIT' = User received money, refund, cashback, or friend repaid bill.
4. When to set needs_clarification=FALSE (DO NOT ASK QUESTIONS):
   - When the item/purpose is clear (e.g. 'tiffin', 'canteen', 'lunch', 'groceries', 'cab', 'auto', 'metro', 'blinkit', 'swiggy', 'zomato', 'treat'). Set category='Food & Dining', 'Groceries', 'Travel & Commute', etc. and needs_clarification=False.
   - For recognized merchants, shops, stores, and enterprises (e.g., 'Gautam Enterprises', 'Chai Point'), categorize directly and set needs_clarification=False.
5. When to set needs_clarification=TRUE:
   - ONLY for peer transfers to individuals/friends where the purpose is completely unspecified (e.g. 'Paid 300 to Mahima' with no items mentioned, or 'Received 200 from Tanya').
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
                    temperature=0.0
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

def parse_receipt_image(
    image_bytes: bytes, 
    mime_type: str = "image/jpeg", 
    api_key: Optional[str] = None,
    allowed_categories: Optional[List[str]] = None
) -> List[ParsedTransaction]:
    """Scans and extracts ALL transactions from a screenshot considering user custom categories with ultra-low latency."""
    client = get_client(api_key)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cats = allowed_categories or DEFAULT_CATEGORIES
    categories_str = ", ".join(cats)

    # 1. Optimize image payload for instantaneous vision tokenization
    opt_bytes, opt_mime = optimize_image_for_fast_ocr(image_bytes)

    system_instruction = f"""You are PennyPilot, an elite multimodal financial scanner specialized in Indian UPI screenshots (Google Pay, PhonePe, Paytm, CRED, Bank Statements).
Current Date: {today_str}
Allowed Categories: [{categories_str}]

EXTRACTION & CLARIFICATION RULES:
1. Scan the ENTIRE screenshot from top to bottom and extract EVERY transaction row/card visible.
2. Pay close attention to whether the transaction is an Outgoing Expense (DEBIT) or Incoming Money (CREDIT):
   - 'DEBIT' (Paid to, Sent to, Debited from account, minus sign '-', red color text, 'Paid ₹X').
   - 'CREDIT' (Received from, Credited to account, plus sign '+', green color text, 'Received ₹X', 'Cashback', 'Refund').
   - If a friend (e.g. Mahima, Tanya) sent/paid the user money, mark transaction_type='CREDIT', notes='Received payment / Bill reimbursement', and entity_type='peer_friend'.
3. Set needs_clarification=TRUE ONLY for peer transfers to friends/individuals where purpose is completely ambiguous. For shops, merchants, enterprises, and recognizable services, set needs_clarification=FALSE.
"""

    image_part = types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime)
    
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=["Extract all transactions (debits and received credits):", image_part],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=MultiTransactionResponse,
                    temperature=0.0
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
