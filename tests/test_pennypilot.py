import pytest
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
    
    # Verify Guest has rich multi-month transactions
    guest_df = get_all_transactions(user_id='guest')
    assert len(guest_df) >= 20
    
    # Verify Month-on-Month reduction in spend for guest
    mom = get_month_on_month_trends(guest_df)
    assert mom.get("percentage_change", 0.0) < 0 # Spend is down in Month 2!

def test_clean_new_user_isolation():
    init_db()
    # A new unique user starts completely clean (0 transactions)
    new_user_df = get_all_transactions(user_id='test_ananya_fresh')
    assert len(new_user_df) == 0

    # Adding a transaction to new user does not pollute guest
    add_transaction(
        date="2026-08-27",
        amount=150.0,
        recipient_name="Local Bakery",
        category="Food & Dining",
        user_id='test_ananya_fresh'
    )
    user_after = get_all_transactions(user_id='test_ananya_fresh')
    assert len(user_after) == 1
    assert user_after.iloc[0]["recipient_name"] == "Local Bakery"

def test_chroma_merchant_memory(tmp_path):
    mem = MerchantMemory(persist_dir=str(tmp_path / "test_chroma"))
    mem.save_merchant_mapping("Kishan Lal", "Groceries", "Local vegetable store", "kishan@upi")

    match = mem.query_merchant("Kishan Lal")
    assert match is not None
    assert match["category"] == "Groceries"

def test_langgraph_compilation():
    graph = build_pennypilot_graph()
    assert graph is not None
