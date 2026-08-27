import pytest
import uuid
import pandas as pd
from datetime import datetime
from pennypilot.src.database.db import (
    init_db, add_transaction, get_all_transactions, 
    update_transaction_category, delete_transaction, set_monthly_budget, get_monthly_budget,
    get_user_categories, add_user_category, delete_user_category, get_all_existing_users
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

def test_returning_user_data_persistence():
    init_db()
    uid = f"user_{uuid.uuid4().hex[:8]}"
    
    # 1. User logs in first time and adds 2 transactions & target budget
    set_monthly_budget("2026-08", 18000.0, user_id=uid)
    add_transaction("2026-08-25", 350.0, "Cafe", "Food & Dining", user_id=uid)
    add_transaction("2026-08-26", 120.0, "Metro", "Travel & Commute", user_id=uid)
    
    # 2. User logs out and logs in again with the same name
    returning_df = get_all_transactions(user_id=uid)
    assert len(returning_df) == 2
    assert get_monthly_budget("2026-08", user_id=uid) == 18000.0

    # 3. Check that user appears in get_all_existing_users
    users = get_all_existing_users()
    user_ids = [u["user_id"] for u in users]
    assert uid in user_ids

def test_custom_user_categories():
    init_db()
    uid = f"user_{uuid.uuid4().hex[:8]}"
    
    # Add custom category
    assert add_user_category("Treat to Juniors", user_id=uid) is True
    assert add_user_category("Freshers Party Contribution", user_id=uid) is True
    
    cats = get_user_categories(user_id=uid)
    assert "Treat to Juniors" in cats
    assert "Freshers Party Contribution" in cats
    
    # Delete category
    delete_user_category("Freshers Party Contribution", user_id=uid)
    updated_cats = get_user_categories(user_id=uid)
    assert "Freshers Party Contribution" not in updated_cats
    assert "Treat to Juniors" in updated_cats

def test_net_spend_with_received_credit():
    init_db()
    uid = f"user_{uuid.uuid4().hex[:8]}"
    add_transaction("2026-08-25", 500.0, "Restaurant", "Food & Dining", transaction_type="DEBIT", user_id=uid)
    add_transaction("2026-08-25", 200.0, "Mahima", "Food & Dining", transaction_type="CREDIT", notes="Received payment", user_id=uid)

    df = get_all_transactions(user_id=uid)
    summary = get_overall_summary(df)
    
    assert summary["gross_spend"] == 500.0
    assert summary["total_received"] == 200.0
    assert summary["total_spend"] == 300.0

def test_chroma_permanent_vs_variable_memory(tmp_path):
    mem = MerchantMemory(persist_dir=str(tmp_path / "test_chroma"))
    mem.save_merchant_mapping("Sanjay Kumar Yadav", "Food & Dining", "Hostel Canteen", is_permanent_rule=True)
    match_canteen = mem.query_merchant("Sanjay Kumar Yadav")
    assert match_canteen is not None
    assert match_canteen["category"] == "Food & Dining"
    assert match_canteen["is_permanent_rule"] is True

def test_langgraph_compilation():
    graph = build_pennypilot_graph()
    assert graph is not None
