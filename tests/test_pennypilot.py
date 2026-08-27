import pytest
import uuid
import pandas as pd
from datetime import datetime
from pennypilot.src.database.db import (
    init_db, add_transaction, get_all_transactions, 
    update_transaction_category, delete_transaction, set_monthly_budget, get_monthly_budget
)
from pennypilot.src.database.seeder import seed_guest_data_if_empty
from pennypilot.src.analytics.engine import (
    get_overall_summary, get_category_breakdown, get_week_on_week_trends,
    get_month_on_month_trends, get_daily_burn_rate, simulate_round_up_savings
)
from pennypilot.src.ai.memory import MerchantMemory
from pennypilot.src.agents.workflow import build_pennypilot_graph

def test_database_and_guest_seeder():
    init_db()
    seed_guest_data_if_empty()
    guest_df = get_all_transactions(user_id='guest')
    assert len(guest_df) >= 20

def test_clean_new_user_isolation():
    init_db()
    unique_uid = f"user_{uuid.uuid4().hex[:8]}"
    new_user_df = get_all_transactions(user_id=unique_uid)
    assert len(new_user_df) == 0

    add_transaction(
        date="2026-08-27",
        amount=150.0,
        recipient_name="Local Bakery",
        category="Food & Dining",
        transaction_type="DEBIT",
        user_id=unique_uid
    )
    user_after = get_all_transactions(user_id=unique_uid)
    assert len(user_after) == 1
    assert user_after.iloc[0]["recipient_name"] == "Local Bakery"

def test_net_spend_with_received_credit():
    init_db()
    uid = f"user_{uuid.uuid4().hex[:8]}"
    # User spent 500 on Food
    add_transaction("2026-08-25", 500.0, "Restaurant", "Food & Dining", transaction_type="DEBIT", user_id=uid)
    # Friend Mahima reimbursed 200 for Food
    add_transaction("2026-08-25", 200.0, "Mahima", "Food & Dining", transaction_type="CREDIT", notes="Received payment", user_id=uid)

    df = get_all_transactions(user_id=uid)
    summary = get_overall_summary(df)
    
    assert summary["gross_spend"] == 500.0
    assert summary["total_received"] == 200.0
    assert summary["total_spend"] == 300.0  # Net Spend is 500 - 200 = 300!

    cat_df = get_category_breakdown(df)
    food_row = cat_df[cat_df["category"] == "Food & Dining"]
    assert food_row.iloc[0]["amount"] == 300.0

def test_chroma_permanent_vs_variable_memory(tmp_path):
    mem = MerchantMemory(persist_dir=str(tmp_path / "test_chroma"))
    mem.save_merchant_mapping("Sanjay Kumar Yadav", "Food & Dining", "Hostel Canteen", is_permanent_rule=True)
    match_canteen = mem.query_merchant("Sanjay Kumar Yadav")
    assert match_canteen is not None
    assert match_canteen["category"] == "Food & Dining"
    assert match_canteen["is_permanent_rule"] is True

    mem.save_merchant_mapping("Mahima", "Food & Dining", "Dinner split", is_permanent_rule=False)
    match_friend = mem.query_merchant("Mahima")
    assert match_friend is not None
    assert match_friend["is_permanent_rule"] is False

def test_langgraph_compilation():
    graph = build_pennypilot_graph()
    assert graph is not None
