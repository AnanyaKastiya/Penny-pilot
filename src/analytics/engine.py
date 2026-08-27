import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares and validates dataframe columns with DEBIT/CREDIT support."""
    if df.empty:
        return df
    clean = df.copy()
    if 'date' in clean.columns:
        clean['date'] = pd.to_datetime(clean['date'])
    if 'amount' in clean.columns:
        clean['amount'] = pd.to_numeric(clean['amount'], errors='coerce').fillna(0.0)
    if 'transaction_type' not in clean.columns:
        clean['transaction_type'] = 'DEBIT'
    else:
        clean['transaction_type'] = clean['transaction_type'].fillna('DEBIT').str.upper()

    # AUTO-DETECT: If notes, raw_input, or recipient_name indicates received money / reimbursement
    if 'notes' in clean.columns:
        rec_notes = clean['notes'].fillna('').str.lower().str.contains('received|credited|refund|cashback|reimbursement|repaid|settled up|got from|sent me')
        clean.loc[rec_notes, 'transaction_type'] = 'CREDIT'
    if 'raw_input' in clean.columns:
        rec_input = clean['raw_input'].fillna('').str.lower().str.contains('received|credited to|cashback|refund|sent you|received from')
        clean.loc[rec_input, 'transaction_type'] = 'CREDIT'

    return clean

def get_overall_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates overall metrics including Gross Spend, Total Received, and Net Spend."""
    df = _clean_df(df)
    if df.empty:
        return {
            "total_spend": 0.0,
            "gross_spend": 0.0,
            "total_received": 0.0,
            "transaction_count": 0,
            "avg_transaction": 0.0,
            "top_category": "None",
            "top_category_amount": 0.0
        }

    debits = df[df['transaction_type'] == 'DEBIT']
    credits = df[df['transaction_type'] == 'CREDIT']

    gross_spend = float(debits['amount'].sum())
    total_received = float(credits['amount'].sum())
    net_spend = max(0.0, gross_spend - total_received)

    cat_totals = debits.groupby('category')['amount'].sum().sort_values(ascending=False) if not debits.empty else pd.Series()
    top_cat = cat_totals.index[0] if not cat_totals.empty else "None"
    top_amt = float(cat_totals.iloc[0]) if not cat_totals.empty else 0.0

    return {
        "total_spend": round(net_spend, 2),
        "gross_spend": round(gross_spend, 2),
        "total_received": round(total_received, 2),
        "transaction_count": len(df),
        "avg_transaction": round(float(debits['amount'].mean()), 2) if not debits.empty else 0.0,
        "top_category": top_cat,
        "top_category_amount": round(top_amt, 2)
    }

def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Net Category Spending (Debits minus Category Reimbursements)."""
    df = _clean_df(df)
    if df.empty:
        return pd.DataFrame(columns=['category', 'amount', 'percentage', 'count'])

    debits = df[df['transaction_type'] == 'DEBIT']
    credits = df[df['transaction_type'] == 'CREDIT']

    deb_grp = debits.groupby('category').agg(debit_amt=('amount', 'sum'), count=('id', 'count')) if not debits.empty else pd.DataFrame(columns=['debit_amt', 'count'])
    cred_grp = credits.groupby('category')['amount'].sum().rename('credit_amt') if not credits.empty else pd.Series(name='credit_amt')

    merged = deb_grp.join(cred_grp, how='outer').fillna(0.0)
    merged['amount'] = (merged['debit_amt'] - merged['credit_amt']).clip(lower=0.0)
    merged['count'] = merged['count'].astype(int)

    total_net = merged['amount'].sum()
    merged['percentage'] = (merged['amount'] / total_net * 100).round(1) if total_net > 0 else 0.0
    merged = merged.reset_index().rename(columns={'index': 'category'})
    return merged[['category', 'amount', 'percentage', 'count']].sort_values(by='amount', ascending=False)

def get_week_on_week_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates exact Net Week-on-Week percentage shift."""
    df = _clean_df(df)
    if df.empty:
        return {"current_week_spend": 0.0, "prev_week_spend": 0.0, "percentage_change": 0.0, "status": "Stable"}

    today = date.today()
    current_week_start = pd.to_datetime(today - timedelta(days=today.weekday()))
    prev_week_start = current_week_start - timedelta(days=7)
    prev_week_end = current_week_start - timedelta(seconds=1)

    def calc_net(sub_df):
        if sub_df.empty:
            return 0.0
        deb = sub_df[sub_df['transaction_type'] == 'DEBIT']['amount'].sum()
        cred = sub_df[sub_df['transaction_type'] == 'CREDIT']['amount'].sum()
        return max(0.0, float(deb - cred))

    curr_df = df[df['date'] >= current_week_start]
    prev_df = df[(df['date'] >= prev_week_start) & (df['date'] <= prev_week_end)]

    curr_spend = calc_net(curr_df)
    prev_spend = calc_net(prev_df)

    if prev_spend == 0.0:
        pct_change = 0.0 if curr_spend == 0.0 else 100.0
    else:
        pct_change = round(((curr_spend - prev_spend) / prev_spend) * 100, 1)

    status = "Increased" if pct_change > 5 else ("Decreased" if pct_change < -5 else "Stable")

    return {
        "current_week_spend": round(curr_spend, 2),
        "prev_week_spend": round(prev_spend, 2),
        "percentage_change": pct_change,
        "status": status
    }

def get_month_on_month_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Net Month-on-Month percentage shift."""
    df = _clean_df(df)
    if df.empty:
        return {"current_month": date.today().strftime("%Y-%m"), "current_month_spend": 0.0, "prev_month": None, "prev_month_spend": 0.0, "percentage_change": 0.0}

    df['month'] = df['date'].dt.strftime('%Y-%m')
    months = sorted(df['month'].unique())
    curr_m = months[-1]

    def calc_net_m(m_str):
        sub = df[df['month'] == m_str]
        deb = sub[sub['transaction_type'] == 'DEBIT']['amount'].sum()
        cred = sub[sub['transaction_type'] == 'CREDIT']['amount'].sum()
        return max(0.0, float(deb - cred))

    curr_spend = calc_net_m(curr_m)

    if len(months) > 1:
        prev_m = months[-2]
        prev_spend = calc_net_m(prev_m)
        if prev_spend == 0.0:
            pct_change = 0.0 if curr_spend == 0.0 else 100.0
        else:
            pct_change = round(((curr_spend - prev_spend) / prev_spend) * 100, 1)
    else:
        prev_m = None
        prev_spend = 0.0
        pct_change = 0.0

    return {
        "current_month": curr_m,
        "current_month_spend": round(curr_spend, 2),
        "prev_month": prev_m,
        "prev_month_spend": round(prev_spend, 2),
        "percentage_change": pct_change
    }

def get_daily_burn_rate(df: pd.DataFrame, monthly_budget: float = 15000.0) -> Dict[str, Any]:
    """Computes daily burn allowance based on Net Expenditure."""
    df = _clean_df(df)
    today = date.today()
    curr_m = today.strftime("%Y-%m")
    
    # Days calculation
    if today.month == 12:
        next_month = date(today.year + 1, 1, 1)
    else:
        next_month = date(today.year, today.month + 1, 1)
    total_days_in_month = (next_month - date(today.year, today.month, 1)).days
    day_of_month = today.day
    days_remaining = max(1, total_days_in_month - day_of_month)

    # Net spend in current month
    if not df.empty:
        df['month'] = df['date'].dt.strftime('%Y-%m')
        m_df = df[df['month'] == curr_m]
        deb = m_df[m_df['transaction_type'] == 'DEBIT']['amount'].sum()
        cred = m_df[m_df['transaction_type'] == 'CREDIT']['amount'].sum()
        month_spend = max(0.0, float(deb - cred))
    else:
        month_spend = 0.0

    daily_burn_so_far = month_spend / day_of_month if day_of_month > 0 else 0.0
    remaining_budget = max(0.0, monthly_budget - month_spend)
    safe_daily_allowance = remaining_budget / days_remaining if remaining_budget > 0 else 0.0

    status = "Healthy"
    if month_spend > monthly_budget:
        status = "Over Budget"
    elif safe_daily_allowance < daily_burn_so_far * 0.7:
        status = "High Burn Rate"

    return {
        "current_month": curr_m,
        "monthly_budget": monthly_budget,
        "month_spend": round(month_spend, 2),
        "remaining_budget": round(remaining_budget, 2),
        "days_remaining": days_remaining,
        "daily_burn_so_far": round(daily_burn_so_far, 2),
        "safe_daily_allowance": round(safe_daily_allowance, 2) if status != "Over Budget" else 0.0,
        "status": status
    }

def detect_recurring_subscriptions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identifies recurring subscription debits."""
    df = _clean_df(df)
    if df.empty:
        return []
    
    debits = df[df['transaction_type'] == 'DEBIT']
    sub_keywords = ['netflix', 'spotify', 'prime', 'hotstar', 'youtube', 'gym', 'broadband', 'wifi', 'rent', 'icloud']
    found = []
    
    for _, row in debits.iterrows():
        merchant = str(row['recipient_name']).lower()
        if any(k in merchant for k in sub_keywords) or row['category'] in ['Entertainment & Subscriptions', 'Bills & Utilities']:
            found.append({
                "merchant": row['recipient_name'],
                "amount": float(row['amount']),
                "category": row['category'],
                "date": row['date'].strftime('%Y-%m-%d'),
                "annual_cost": round(float(row['amount']) * 12, 2)
            })

    unique = {item['merchant']: item for item in found}
    return list(unique.values())

def simulate_round_up_savings(df: pd.DataFrame, round_to: int = 50) -> Dict[str, Any]:
    """Calculates potential micro-savings on Debits."""
    df = _clean_df(df)
    if df.empty:
        return {"total_potential_savings": 0.0, "transaction_count": 0, "avg_saved_per_tx": 0.0}

    debits = df[df['transaction_type'] == 'DEBIT']
    amounts = debits['amount'].values
    if len(amounts) == 0:
        return {"total_potential_savings": 0.0, "transaction_count": 0, "avg_saved_per_tx": 0.0}

    ceil_amounts = np.ceil(amounts / round_to) * round_to
    diffs = ceil_amounts - amounts
    diffs = np.where(diffs == 0, round_to, diffs)
    total_savings = float(np.sum(diffs))

    return {
        "total_potential_savings": round(total_savings, 2),
        "transaction_count": len(debits),
        "avg_saved_per_tx": round(total_savings / len(debits), 2)
    }
