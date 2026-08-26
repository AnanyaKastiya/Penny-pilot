import json
import pandas as pd
from typing import Optional, Dict, Any
from google import genai
from google.genai import types
from ..config import MODEL_CANDIDATES, GEMINI_API_KEY
from ..analytics.engine import (
    get_overall_summary, get_category_breakdown, get_week_on_week_trends,
    get_month_on_month_trends, get_daily_burn_rate, detect_recurring_subscriptions
)
from ..ai.parser import get_client

def ask_financial_advisor(
    question: str,
    df: pd.DataFrame,
    monthly_budget: float = 15000.0,
    api_key: Optional[str] = None
) -> str:
    """Answers financial questions using exact database numbers and Gemini reasoning."""
    client = get_client(api_key)
    
    # Compute deterministic context using Pandas
    summary = get_overall_summary(df)
    cat_df = get_category_breakdown(df)
    burn = get_daily_burn_rate(df, monthly_budget=monthly_budget)
    wow = get_week_on_week_trends(df)
    mom = get_month_on_month_trends(df)
    recurring = detect_recurring_subscriptions(df)

    cat_summary_str = "\n".join([f"- {r['category']}: ₹{r['amount']:,.2f} ({r['percentage']}%) [{r['count']} purchases]" for _, r in cat_df.iterrows()]) if not cat_df.empty else "No expense records."
    
    recurring_str = "\n".join([f"- {r['merchant']}: ₹{r['amount']:,.2f} (Annual: ₹{r['annual_cost']:,.2f})" for r in recurring]) if recurring else "None detected."

    financial_context = f"""
=== REAL USER EXPENSE CONTEXT ===
- Total Historical Spend: ₹{summary['total_spend']:,.2f} ({summary['transaction_count']} transactions)
- Current Month ({burn['current_month']}) Spend: ₹{burn['month_spend']:,.2f}
- Monthly Target Budget: ₹{burn['monthly_budget']:,.2f} (Remaining: ₹{burn['remaining_budget']:,.2f})
- Days Left in Month: {burn['days_remaining']} days
- Current Daily Burn Rate: ₹{burn['daily_burn_so_far']:,.2f}/day
- Recommended Safe Daily Allowance: ₹{burn.get('safe_daily_allowance') or 'N/A'}/day
- Budget Health Status: {burn['status']}
- Week-on-Week Change: {wow['percentage_change']}% ({wow['status']})
- Month-on-Month Change: {mom['percentage_change']}%

=== SPENDING BY CATEGORY ===
{cat_summary_str}

=== FIXED & RECURRING CHARGES (SUBSCRIPTIONS/BILLS) ===
{recurring_str}
"""

    system_instruction = f"""You are PennyPilot AI, an elite, friendly, and practical Personal Financial Advisor & Expense Copilot.
You have direct access to the user's real, verified financial numbers.

Rules:
1. ALWAYS quote exact numbers, categories, and merchants from the provided context.
2. Give structured, highly readable advice with bold text, bullet points, and clear actionable takeaways.
3. Categorize expenses into Needs (Fixed: rent/bills/groceries) vs Wants (Discretionary: dining/entertainment/shopping) when asked.
4. When asked how to cut down or save money, give realistic, non-drastic cutback targets (e.g. 'Reduce Food & Dining by 20% to save ₹X').
5. Keep your tone encouraging, empowering, and crisp.
"""

    prompt = f"""{financial_context}

User Question:
"{question}"
"""

    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Advisor fallback from {model_name}: {e}")
            continue

    return "I couldn't analyze the financial data right now. Please verify your Gemini API key."
