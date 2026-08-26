import pandas as pd
import numpy as np
from datetime import datetime, date
import calendar
from typing import Dict, Any, List, Optional

def get_overall_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes high-level aggregate statistics from transactions."""
    if df.empty:
        return {
            "total_spend": 0.0,
            "transaction_count": 0,
            "avg_transaction": 0.0,
            "top_category": "None",
            "top_merchant": "None"
        }

    total_spend = float(df["amount"].sum())
    tx_count = int(len(df))
    avg_tx = float(df["amount"].mean())
    
    top_cat = df.groupby("category")["amount"].sum().idxmax() if "category" in df else "None"
    top_merchant = df["recipient_name"].value_counts().idxmax() if "recipient_name" in df and not df["recipient_name"].isna().all() else "None"

    return {
        "total_spend": round(total_spend, 2),
        "transaction_count": tx_count,
        "avg_transaction": round(avg_tx, 2),
        "top_category": top_cat,
        "top_merchant": top_merchant
    }

def get_category_breakdown(df: pd.DataFrame, month: Optional[str] = None) -> pd.DataFrame:
    """Returns total spend, transaction count, and percentage share per category."""
    if df.empty:
        return pd.DataFrame(columns=["category", "amount", "count", "percentage"])

    data = df.copy()
    if month and "date" in data.columns:
        data["month_str"] = data["date"].dt.strftime("%Y-%m")
        data = data[data["month_str"] == month]

    if data.empty:
        return pd.DataFrame(columns=["category", "amount", "count", "percentage"])

    breakdown = data.groupby("category").agg(
        amount=("amount", "sum"),
        count=("amount", "count")
    ).reset_index()

    total = breakdown["amount"].sum()
    breakdown["percentage"] = (breakdown["amount"] / total * 100).round(1) if total > 0 else 0.0
    breakdown = breakdown.sort_values(by="amount", ascending=False)
    return breakdown

def get_week_on_week_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Week-on-Week (WoW) expenditure trends, percentages, and category surges."""
    if df.empty or "date" not in df.columns:
        return {"current_week_spend": 0.0, "prev_week_spend": 0.0, "percentage_change": 0.0, "status": "No Data", "category_deltas": []}

    data = df.copy()
    data["year_week"] = data["date"].dt.strftime("%Y-W%W")
    
    weeks = sorted(data["year_week"].unique())
    if len(weeks) == 0:
        return {"current_week_spend": 0.0, "prev_week_spend": 0.0, "percentage_change": 0.0, "status": "No Data", "category_deltas": []}

    current_week = weeks[-1]
    prev_week = weeks[-2] if len(weeks) >= 2 else None

    curr_df = data[data["year_week"] == current_week]
    curr_spend = float(curr_df["amount"].sum())

    prev_spend = 0.0
    pct_change = 0.0
    status = "First Week of Data"

    category_deltas = []

    if prev_week:
        prev_df = data[data["year_week"] == prev_week]
        prev_spend = float(prev_df["amount"].sum())
        
        if prev_spend > 0:
            pct_change = round(((curr_spend - prev_spend) / prev_spend) * 100, 1)
            status = "Increased" if pct_change > 0 else "Decreased" if pct_change < 0 else "Unchanged"

        # Compare category-by-category
        curr_cats = curr_df.groupby("category")["amount"].sum()
        prev_cats = prev_df.groupby("category")["amount"].sum()
        
        all_cats = set(curr_cats.index).union(set(prev_cats.index))
        for cat in all_cats:
            c_val = float(curr_cats.get(cat, 0.0))
            p_val = float(prev_cats.get(cat, 0.0))
            delta = c_val - p_val
            pct = round(((c_val - p_val) / p_val * 100), 1) if p_val > 0 else (100.0 if c_val > 0 else 0.0)
            category_deltas.append({
                "category": cat,
                "current_spend": round(c_val, 2),
                "prev_spend": round(p_val, 2),
                "delta": round(delta, 2),
                "pct_change": pct
            })

    category_deltas = sorted(category_deltas, key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "current_week": current_week,
        "current_week_spend": round(curr_spend, 2),
        "prev_week": prev_week,
        "prev_week_spend": round(prev_spend, 2),
        "percentage_change": pct_change,
        "status": status,
        "category_deltas": category_deltas
    }

def get_month_on_month_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Month-on-Month (MoM) expenditure trends and shifts."""
    if df.empty or "date" not in df.columns:
        return {"current_month_spend": 0.0, "prev_month_spend": 0.0, "percentage_change": 0.0, "status": "No Data"}

    data = df.copy()
    data["year_month"] = data["date"].dt.strftime("%Y-%m")
    
    months = sorted(data["year_month"].unique())
    if len(months) == 0:
        return {"current_month_spend": 0.0, "prev_month_spend": 0.0, "percentage_change": 0.0, "status": "No Data"}

    curr_month = months[-1]
    prev_month = months[-2] if len(months) >= 2 else None

    curr_df = data[data["year_month"] == curr_month]
    curr_spend = float(curr_df["amount"].sum())

    prev_spend = 0.0
    pct_change = 0.0
    status = "First Month of Data"

    if prev_month:
        prev_df = data[data["year_month"] == prev_month]
        prev_spend = float(prev_df["amount"].sum())
        if prev_spend > 0:
            pct_change = round(((curr_spend - prev_spend) / prev_spend) * 100, 1)
            status = "Increased" if pct_change > 0 else "Decreased" if pct_change < 0 else "Unchanged"

    return {
        "current_month": curr_month,
        "current_month_spend": round(curr_spend, 2),
        "prev_month": prev_month,
        "prev_month_spend": round(prev_spend, 2),
        "percentage_change": pct_change,
        "status": status
    }

def get_daily_burn_rate(df: pd.DataFrame, monthly_budget: Optional[float] = None) -> Dict[str, Any]:
    """Calculates daily burn rate and safe daily spend allowance for the remaining days of the month."""
    today = date.today()
    curr_month_str = today.strftime("%Y-%m")
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    day_of_month = today.day
    days_remaining = max(1, days_in_month - day_of_month + 1)

    curr_month_spend = 0.0
    if not df.empty and "date" in df.columns:
        data = df.copy()
        data["month_str"] = data["date"].dt.strftime("%Y-%m")
        curr_month_spend = float(data[data["month_str"] == curr_month_str]["amount"].sum())

    daily_spend_so_far = round(curr_month_spend / max(1, day_of_month), 2)
    
    budget = monthly_budget or 0.0
    remaining_budget = max(0.0, budget - curr_month_spend)
    safe_daily_allowance = round(remaining_budget / days_remaining, 2) if budget > 0 else None

    status = "No Budget Set"
    if budget > 0:
        if curr_month_spend > budget:
            status = "Over Budget"
        elif safe_daily_allowance < (daily_spend_so_far * 0.7):
            status = "Caution: High Burn Rate"
        else:
            status = "On Track"

    return {
        "current_month": curr_month_str,
        "month_spend": round(curr_month_spend, 2),
        "monthly_budget": budget,
        "remaining_budget": round(remaining_budget, 2),
        "days_elapsed": day_of_month,
        "days_remaining": days_remaining,
        "daily_burn_so_far": daily_spend_so_far,
        "safe_daily_allowance": safe_daily_allowance,
        "status": status
    }

def detect_recurring_subscriptions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identifies repeating payments to the same merchant with consistent amounts."""
    if df.empty or len(df) < 2:
        return []

    # Group by merchant & amount
    groups = df.groupby(["recipient_name", "amount"]).filter(lambda x: len(x) >= 2)
    if groups.empty:
        return []

    recurring = []
    for (merchant, amount), sub_df in groups.groupby(["recipient_name", "amount"]):
        if merchant and merchant != "Unknown":
            recurring.append({
                "merchant": merchant,
                "amount": float(amount),
                "frequency_count": len(sub_df),
                "category": sub_df["category"].iloc[0],
                "annual_cost": round(float(amount) * 12, 2)
            })

    return recurring

def simulate_round_up_savings(df: pd.DataFrame, round_to: int = 50) -> Dict[str, Any]:
    """Calculates potential micro-savings by rounding up every transaction."""
    if df.empty:
        return {"total_potential_savings": 0.0, "round_to": round_to, "transaction_count": 0}

    amounts = df["amount"].values
    # (round_to - (amt % round_to)) % round_to
    round_ups = [(round_to - (a % round_to)) % round_to for a in amounts if a > 0]
    total_savings = float(np.sum(round_ups))

    return {
        "total_potential_savings": round(total_savings, 2),
        "round_to": round_to,
        "transaction_count": len(amounts)
    }
