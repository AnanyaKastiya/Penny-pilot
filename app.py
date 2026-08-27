import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_CATEGORIES, GEMINI_API_KEY, save_api_key_to_env
from src.database.db import (
    init_db, add_transaction, get_all_transactions, 
    delete_transaction, set_monthly_budget, get_monthly_budget
)
from src.database.seeder import seed_guest_data_if_empty
from src.analytics.engine import (
    get_overall_summary, get_category_breakdown, get_week_on_week_trends,
    get_month_on_month_trends, get_daily_burn_rate, detect_recurring_subscriptions,
    simulate_round_up_savings
)
from src.ai.memory import memory_store
from src.ai.advisor import ask_financial_advisor
from src.agents.workflow import pennypilot_agent, process_batch_clarification

# ----------------- PAGE CONFIG ----------------- #
st.set_page_config(
    page_title="PennyPilot | AI Financial Copilot",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize DB & Seed Demo Data for Guest
init_db()
seed_guest_data_if_empty()

# ----------------- THEME & CSS ----------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: inline-block;
    }
    
    .welcome-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.1);
    }

    .clarify-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .pill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .pill-green { background: #dcfce7; color: #15803d; }
    .pill-blue { background: #dbeafe; color: #1e40af; }
    .pill-purple { background: #f3e8ff; color: #7e22ce; }
    .pill-orange { background: #ffedd5; color: #c2410c; }
    
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE ----------------- #
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "current_user_name" not in st.session_state:
    st.session_state.current_user_name = None
if "active_agent_state" not in st.session_state:
    st.session_state.active_agent_state = None
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = False
if "advisor_chat_history" not in st.session_state:
    st.session_state.advisor_chat_history = []

today = date.today()
curr_month_str = today.strftime("%Y-%m")

# ========================================================================= #
#                    WELCOME & PROFILE SELECTION SCREEN                     #
# ========================================================================= #
if not st.session_state.current_user:
    st.markdown("""
        <div style="text-align: center; padding: 40px 0 20px 0;">
            <span style="font-size: 3rem;">💸</span>
            <h1 class="app-title" style="font-size: 2.8rem;">Welcome to PennyPilot</h1>
            <p style="font-size: 1.1rem; color: #94a3b8; max-width: 600px; margin: 10px auto 30px auto;">
                Your Autonomous AI Financial Copilot. Choose how you would like to proceed:
            </p>
        </div>
    """, unsafe_allow_html=True)

    w_col1, w_col2 = st.columns([1, 1], gap="large")

    with w_col1:
        st.markdown("""
            <div class="welcome-card">
                <span style="font-size: 2.5rem;">🌟</span>
                <h3 style="margin: 10px 0 6px 0;">Explore as Guest</h3>
                <span class="pill-badge pill-purple">Recruiter & Reviewer Demo</span>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 12px;">
                    Pre-loaded with <b>2 full months of realistic financial data</b> showing a <b>38% reduction in spending</b> in Month 2. Instant review of all charts, AI memory, and advisor features.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Launch Guest Recruiter Demo 🚀", type="primary", use_container_width=True):
            st.session_state.current_user = "guest"
            st.session_state.current_user_name = "Guest (Recruiter Demo)"
            st.rerun()

    with w_col2:
        st.markdown("""
            <div class="welcome-card">
                <span style="font-size: 2.5rem;">👤</span>
                <h3 style="margin: 10px 0 6px 0;">Start Fresh</h3>
                <span class="pill-badge pill-green">Clean Personal Ledger</span>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 12px;">
                    Start with a <b>completely blank ledger (0 transactions)</b>. Enter your real daily expenses, upload your own receipts, and build your personal financial health tracker from scratch.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        with st.form("new_user_form"):
            user_name_input = st.text_input("What is your name?", placeholder="e.g. Ananya")
            user_budget_input = st.number_input("Your Monthly Target Budget (₹)", min_value=1000.0, value=15000.0, step=1000.0)
            submitted = st.form_submit_button("Create My Fresh Ledger 🎯", type="secondary", use_container_width=True)
            
            if submitted:
                clean_name = user_name_input.strip() or "User"
                user_id = clean_name.lower().replace(" ", "_")
                set_monthly_budget(curr_month_str, user_budget_input, user_id=user_id)
                st.session_state.current_user = user_id
                st.session_state.current_user_name = clean_name
                st.rerun()

    st.stop()

# ========================================================================= #
#                         AUTHENTICATED DASHBOARD                           #
# ========================================================================= #
active_user_id = st.session_state.current_user
active_user_name = st.session_state.current_user_name
df_all = get_all_transactions(user_id=active_user_id)
current_budget = get_monthly_budget(curr_month_str, user_id=active_user_id) or 15000.0

# ----------------- TOP HEADER ----------------- #
h_col1, h_col2 = st.columns([3, 1.2])

with h_col1:
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 8px;">
            <span style="font-size: 2.2rem;">💸</span>
            <div>
                <h1 class="app-title">PennyPilot</h1>
                <p style="margin: 0; font-size: 0.95rem; color: #94a3b8; font-weight: 500;">
                    Autonomous Financial Copilot & Agentic Expense Tracker
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with h_col2:
    st.markdown(f"""
        <div style="text-align: right; padding-top: 6px;">
            <span class="pill-badge pill-purple">👤 {active_user_name}</span>
            <span class="pill-badge pill-green">● Online</span><br>
            <span style="font-size: 0.8rem; color: #94a3b8;">📅 {today.strftime('%d %B %Y')}</span>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Switch User / Logout", use_container_width=True):
        st.session_state.current_user = None
        st.session_state.current_user_name = None
        st.session_state.advisor_chat_history = []
        st.session_state.active_agent_state = None
        st.session_state.pending_clarification = False
        st.rerun()

st.markdown("<hr style='margin: 12px 0 20px 0; border-color: rgba(226, 232, 240, 0.15);'>", unsafe_allow_html=True)

# ----------------- STEP-BY-STEP WORKFLOW STEPPER ----------------- #
selected_step = st.radio(
    "Workflow Navigation",
    [
        "Step 1: 📝 Add & Scan Expenses",
        "Step 2: 📊 Spending Breakdown & Trends",
        "Step 3: 🤖 AI Financial Copilot & Q&A"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

# Quick Executive Summary Ribbon
summary = get_overall_summary(df_all)
burn = get_daily_burn_rate(df_all, monthly_budget=current_budget)
wow = get_week_on_week_trends(df_all)
mom = get_month_on_month_trends(df_all)

st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px;">
        <span class="pill-badge pill-blue">Net Month Spend: <b>₹{burn['month_spend']:,.2f}</b></span>
        <span class="pill-badge pill-green">Money Received / Reimbursements: <b>+₹{summary['total_received']:,.2f}</b></span>
        <span class="pill-badge pill-purple">Target Budget: <b>₹{current_budget:,.2f}</b></span>
        <span class="pill-badge pill-orange">Safe Daily Allowance: <b>₹{burn['safe_daily_allowance'] or 0:,.2f}/day</b></span>
    </div>
""", unsafe_allow_html=True)

# ========================================================================= #
#                    STEP 1: ADD & SCAN EXPENSES                            #
# ========================================================================= #
if selected_step == "Step 1: 📝 Add & Scan Expenses":
    st.subheader("Step 1: Log Your Expenses (Single, Multi-Day, or Full Statement Screenshot)")
    st.caption("Choose how you want to enter your expenses. PennyPilot automatically scans lists and asks for clarifications.")

    entry_mode = st.segmented_control(
        "Entry Type",
        options=["💬 AI Smart Prompt (Multi-Day Text)", "📸 Upload Full UPI Screenshot", "📅 Manual & Batch Date Entry"],
        default="💬 AI Smart Prompt (Multi-Day Text)"
    )

    # ---------- OPTION A: NATURAL LANGUAGE PROMPT ---------- #
    if entry_mode == "💬 AI Smart Prompt (Multi-Day Text)":
        st.markdown("##### 💬 Type naturally (handles single or multiple days in one prompt)")
        user_prompt = st.text_area(
            "Expense prompt",
            placeholder="e.g. 'Yesterday sent 320 to Mahima for dinner, paid 150 to Sanjay Kumar Yadav at canteen, and spent 280 on Blinkit.'",
            height=100,
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("Process with AI 🚀", type="primary", use_container_width=True):
                if not GEMINI_API_KEY:
                    st.error("Please configure your GEMINI_API_KEY in .env.")
                elif not user_prompt.strip():
                    st.warning("Please type an expense description first.")
                else:
                    with st.spinner("AI extracting all transactions and checking RAG memory..."):
                        initial_state = {
                            "user_id": active_user_id,
                            "raw_text": user_prompt,
                            "image_bytes": None,
                            "mime_type": None,
                            "api_key": GEMINI_API_KEY,
                            "parsed_transactions": [],
                            "needs_clarification": False,
                            "pending_questions": [],
                            "recorded_ids": [],
                            "error": None
                        }
                        res = pennypilot_agent.invoke(initial_state)
                        if res.get("error"):
                            st.error(res["error"])
                        elif res.get("needs_clarification"):
                            st.session_state.active_agent_state = res
                            st.session_state.pending_clarification = True
                            st.rerun()
                        else:
                            st.success(f"✅ Successfully recorded {len(res.get('recorded_ids', []))} transaction(s)!")
                            st.rerun()

    # ---------- OPTION B: FULL UPI RECEIPT SCREENSHOT ---------- #
    elif entry_mode == "📸 Upload Full UPI Screenshot":
        st.markdown("##### 📸 Upload Google Pay / PhonePe / Paytm / Passbook Screenshot (scans 1 to 20+ transactions)")
        up_file = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
        
        if up_file:
            st.image(up_file, caption="Payment Screenshot Preview", width=260)
            if st.button("Extract All Transactions & Check Memory 🔍", type="primary"):
                with st.spinner("Multimodal Vision scanning full screenshot & formulating questions..."):
                    img_bytes = up_file.getvalue()
                    mime = up_file.type
                    initial_state = {
                        "user_id": active_user_id,
                        "raw_text": None,
                        "image_bytes": img_bytes,
                        "mime_type": mime,
                        "api_key": GEMINI_API_KEY,
                        "parsed_transactions": [],
                        "needs_clarification": False,
                        "pending_questions": [],
                        "recorded_ids": [],
                        "error": None
                    }
                    res = pennypilot_agent.invoke(initial_state)
                    if res.get("error"):
                        st.error(res["error"])
                    elif res.get("needs_clarification"):
                        st.session_state.active_agent_state = res
                        st.session_state.pending_clarification = True
                        st.rerun()
                    else:
                        st.success(f"✅ Extracted and recorded {len(res.get('recorded_ids', []))} transaction(s) from screenshot!")
                        st.rerun()

    # ---------- OPTION C: MANUAL DATE-WISE & BATCH ENTRY ---------- #
    elif entry_mode == "📅 Manual & Batch Date Entry":
        st.markdown("##### 📅 Enter expenses for specific dates manually")
        
        m_tab1, m_tab2 = st.tabs(["Single Transaction", "Batch Multi-Day Entry"])
        
        with m_tab1:
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                manual_date = st.date_input("Transaction Date", value=today)
                manual_amt = st.number_input("Amount (₹)", min_value=1.0, value=250.0, step=50.0)
            with col_m2:
                manual_recipient = st.text_input("Recipient / Merchant", placeholder="e.g. Chai Point / Auto / Mahima")
                manual_cat = st.selectbox("Category", DEFAULT_CATEGORIES, index=0)
            with col_m3:
                manual_app = st.selectbox("Payment Mode", ["GPay", "PhonePe", "Paytm", "Card", "Cash", "Auto-Debit", "Manual"])
                manual_notes = st.text_input("Notes (Optional)", placeholder="e.g. Dinner treat")
                
            if st.button("Save Manual Entry 💾", type="primary"):
                if not manual_recipient.strip():
                    st.warning("Please enter a merchant or recipient name.")
                else:
                    tx_id = add_transaction(
                        date=manual_date.strftime("%Y-%m-%d"),
                        amount=float(manual_amt),
                        recipient_name=manual_recipient.strip(),
                        category=manual_cat,
                        payment_app=manual_app,
                        notes=manual_notes.strip(),
                        user_id=active_user_id
                    )
                    memory_store.save_merchant_mapping(manual_recipient.strip(), manual_cat, manual_notes.strip(), is_permanent_rule=False)
                    memory_store.index_transaction(tx_id, manual_date.strftime("%Y-%m-%d"), manual_amt, manual_recipient, manual_cat, manual_notes)
                    st.success(f"✅ Transaction of ₹{manual_amt} saved for {manual_date.strftime('%d %b %Y')}!")
                    st.rerun()

        with m_tab2:
            st.caption("Quickly add multiple days' expenses at once:")
            sample_batch = [
                {"date": today - timedelta(days=2), "merchant": "Uber", "amount": 180.0, "category": "Travel & Commute", "notes": "Office commute"},
                {"date": today - timedelta(days=1), "merchant": "Mahima", "amount": 280.0, "category": "Food & Dining", "notes": "Lunch split"},
                {"date": today, "merchant": "Blinkit", "amount": 340.0, "category": "Groceries", "notes": "Milk & bread"}
            ]
            batch_df = pd.DataFrame(sample_batch)
            edited_df = st.data_editor(batch_df, num_rows="dynamic", use_container_width=True)
            
            if st.button("Save All Batch Transactions 📥", type="primary"):
                saved_count = 0
                for _, row in edited_df.iterrows():
                    d_str = row["date"].strftime("%Y-%m-%d") if isinstance(row["date"], (date, datetime)) else str(row["date"])
                    amt = float(row.get("amount", 0.0))
                    rec = str(row.get("merchant", "Unknown")).strip()
                    cat = str(row.get("category", "Miscellaneous"))
                    note = str(row.get("notes", ""))
                    if amt > 0:
                        t_id = add_transaction(d_str, amt, rec, cat, notes=note, user_id=active_user_id)
                        memory_store.save_merchant_mapping(rec, cat, note, is_permanent_rule=False)
                        memory_store.index_transaction(t_id, d_str, amt, rec, cat, note)
                        saved_count += 1
                st.success(f"✅ Successfully saved {saved_count} transactions!")
                st.rerun()

    # ========================================================================= #
    #      MULTI-TRANSACTION CLARIFICATION & SMART MEMORY REVIEW HUB           #
    # ========================================================================= #
    if st.session_state.pending_clarification and st.session_state.active_agent_state:
        state = st.session_state.active_agent_state
        pending_qs = state.get("pending_questions", [])
        total_extracted = len(state.get("parsed_transactions", []))

        st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid #6366f1; border-radius: 12px; padding: 16px; margin-bottom: 20px;">
                <h3 style="margin: 0 0 6px 0; color: #818cf8;">🤔 PennyPilot Needs Your Input ({len(pending_qs)} of {total_extracted} Transactions)</h3>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">
                    PennyPilot extracted all transactions from your input. Please answer the specific questions below so each item is categorized accurately.
                </p>
            </div>
        """, unsafe_allow_html=True)

        user_clarifications = []

        with st.form("batch_clarification_form"):
            for q_item in pending_qs:
                idx = q_item["index"]
                rec_name = q_item["recipient_name"]
                amt = q_item["amount"]
                dt = q_item["date"]
                sug_cat = q_item["suggested_category"]
                q_text = q_item["question"]
                e_type = q_item.get("entity_type", "merchant")

                ttype = q_item.get("transaction_type", "DEBIT")
                is_credit = (ttype == "CREDIT")
                badge_html = f'<span class="pill-badge pill-green">💰 Money Received (+₹{amt:,.2f})</span>' if is_credit else f'<span class="pill-badge pill-red">💸 Money Spent (-₹{amt:,.2f})</span>'

                st.markdown(f"""
                    <div class="clarify-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">{rec_name}</span>
                            <div>{badge_html} <span class="pill-badge pill-blue">📅 {dt}</span></div>
                        </div>
                        <p style="margin: 0 0 10px 0; color: #cbd5e1; font-weight: 500;">❓ <b>AI Question:</b> {q_text}</p>
                    </div>
                """, unsafe_allow_html=True)

                col_ans1, col_ans2, col_ans3, col_ans4 = st.columns([1.2, 1.4, 1.4, 1.2])

                with col_ans1:
                    chosen_type = st.selectbox(
                        f"Type for '{rec_name}'",
                        ["DEBIT (Spent)", "CREDIT (Received / Refund)"],
                        index=1 if is_credit else 0,
                        key=f"type_{idx}"
                    )

                with col_ans2:
                    default_idx = DEFAULT_CATEGORIES.index(sug_cat) if sug_cat in DEFAULT_CATEGORIES else 0
                    chosen_cat = st.selectbox(f"Category", DEFAULT_CATEGORIES, index=default_idx, key=f"cat_{idx}")
                
                with col_ans3:
                    note_val = st.text_input(f"Context / Notes", placeholder="e.g. Dinner split reimbursement", key=f"note_{idx}")

                with col_ans4:
                    default_remember = True if e_type == "merchant" else False
                    remember_rule = st.checkbox(
                        "📌 Always remember?",
                        value=default_remember,
                        key=f"rem_{idx}",
                        help="Check for shops/canteens (e.g. Sanjay Yadav Canteen). UNCHECK for friends (e.g. Mahima/Tanya)."
                    )

                final_type = "CREDIT" if "CREDIT" in chosen_type else "DEBIT"
                user_clarifications.append({
                    "index": idx,
                    "transaction_type": final_type,
                    "category": chosen_cat,
                    "notes": note_val,
                    "remember_rule": remember_rule
                })
                st.markdown("<hr style='margin: 10px 0; border-color: rgba(226, 232, 240, 0.1);'>", unsafe_allow_html=True)

            submit_all = st.form_submit_button(f"💾 Save All {len(user_clarifications)} Verified Transactions 🚀", type="primary", use_container_width=True)

            if submit_all:
                updated_state = process_batch_clarification(
                    state=state,
                    clarified_items=user_clarifications
                )
                st.session_state.pending_clarification = False
                st.session_state.active_agent_state = None
                st.success(f"✅ Successfully saved and learned rules for {len(updated_state.get('recorded_ids', []))} transactions!")
                st.rerun()

# ========================================================================= #
#                    STEP 2: SPENDING BREAKDOWN & TRENDS                    #
# ========================================================================= #
elif selected_step == "Step 2: 📊 Spending Breakdown & Trends":
    st.subheader("Step 2: Visual Spending Allocation & Mathematical Trends")
    st.caption("100% deterministic calculations using Pandas & NumPy (Zero LLM math hallucinations).")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Net Total Spent", f"₹{summary['total_spend']:,.2f}", help="Gross Spent minus Money Received / Reimbursements")
    with m2:
        st.metric("Total Money Received", f"+₹{summary['total_received']:,.2f}", delta="Reimbursements / Income", delta_color="normal")
    with m3:
        st.metric(f"Net Spend in {today.strftime('%B')}", f"₹{burn['month_spend']:,.2f}", delta=f"{mom['percentage_change']}% MoM" if mom.get('prev_month') else None)
    with m4:
        st.metric("Top Spending Category", summary["top_category"])

    st.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        st.markdown("##### 🍩 Category Breakdown")
        cat_df = get_category_breakdown(df_all)
        if not cat_df.empty:
            fig_pie = px.pie(
                cat_df,
                values="amount",
                names="category",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No transaction data available yet in this profile. Log expenses in Step 1!")

    with col_g2:
        st.markdown("##### 📈 Month-on-Month & Week-on-Week Shift")
        if mom.get("prev_month"):
            mom_chart_df = pd.DataFrame({
                "Period": [f"Prev Month ({mom['prev_month']})", f"This Month ({mom['current_month']})"],
                "Amount (₹)": [mom["prev_month_spend"], mom["current_month_spend"]]
            })
            fig_bar = px.bar(
                mom_chart_df,
                x="Period",
                y="Amount (₹)",
                color="Period",
                text="Amount (₹)",
                color_discrete_sequence=["#f43f5e", "#10b981"]
            )
            fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            m_diff = mom["percentage_change"]
            if m_diff < 0:
                st.success(f"🎉 Fantastic financial discipline! Spending is **DOWN by {abs(m_diff)}%** compared to last month!")
            elif m_diff > 0:
                st.warning(f"⚠️ Spending is **UP by {m_diff}%** compared to last month.")
        else:
            st.info("Log transactions across multiple months or view Guest Demo to see Month-on-Month comparison.")

    # Transaction Ledger
    st.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown("##### 📋 Complete Transaction Ledger")
    
    search_filter = st.text_input("🔍 Search ledger by merchant or category", placeholder="e.g. 'Swiggy', 'Uber', 'Food'")
    display_df = df_all.copy()
    if search_filter.strip() and not display_df.empty:
        display_df = display_df[
            display_df["recipient_name"].str.contains(search_filter, case=False, na=False) |
            display_df["category"].str.contains(search_filter, case=False, na=False) |
            display_df["notes"].str.contains(search_filter, case=False, na=False)
        ]
    
    if not display_df.empty:
        table_show = display_df[["id", "date", "transaction_type", "recipient_name", "amount", "category", "payment_app", "notes"]].copy()
        table_show["date"] = pd.to_datetime(table_show["date"]).dt.strftime("%Y-%m-%d")
        
        # Format Amount with +/- based on transaction_type
        def fmt_amt(row):
            amt = row["amount"]
            if str(row["transaction_type"]).upper() == "CREDIT":
                return f"+₹{amt:,.2f} (Received)"
            return f"-₹{amt:,.2f} (Spent)"

        table_show["amount"] = table_show.apply(fmt_amt, axis=1)
        table_show = table_show.rename(columns={
            "transaction_type": "Type",
            "recipient_name": "Recipient / Sender",
            "payment_app": "Payment Mode"
        })
        st.dataframe(table_show, use_container_width=True, hide_index=True)
    else:
        st.info("No records in this ledger yet.")

# ========================================================================= #
#                    STEP 3: AI FINANCIAL COPILOT & Q&A                     #
# ========================================================================= #
elif selected_step == "Step 3: 🤖 AI Financial Copilot & Q&A":
    st.subheader("Step 3: Interactive Financial Copilot & Q&A")
    st.caption("Ask questions about your spending, discover where to cut down, and explore budget strategies.")

    # 1-Click Question Pills
    st.markdown("##### 💡 Instant Quick Questions:")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    
    clicked_prompt = None
    with q_col1:
        if st.button("🍔 Where am I spending the most?", use_container_width=True):
            clicked_prompt = "Where am I spending the most money, and what percentage of my total budget does it take?"
    with q_col2:
        if st.button("✂️ What can I cut down?", use_container_width=True):
            clicked_prompt = "Looking at my non-essential and discretionary spending, what specific categories and habits should I cut down to save at least 20% next month?"
    with q_col3:
        if st.button("🏠 Fixed vs Discretionary Expenses?", use_container_width=True):
            clicked_prompt = "Break down all my transactions into Fixed Necessary Expenses (Needs: rent, bills, basic groceries) versus Discretionary Expenses (Wants: dining out, shopping, entertainment)."
    with q_col4:
        if st.button("🎯 Next Month Budget Plan?", use_container_width=True):
            clicked_prompt = "Based on my last month's spending patterns, suggest a realistic category-by-category budget allocation for next month."

    # Freeform chat input
    user_q = st.chat_input("Ask PennyPilot anything about your finances (e.g. 'How much did I spend on food this week?')...")
    active_query = clicked_prompt or user_q

    if active_query:
        st.session_state.advisor_chat_history.append({"role": "user", "content": active_query})
        with st.spinner("PennyPilot analyzing your numbers & generating financial advice..."):
            advice = ask_financial_advisor(active_query, df_all, monthly_budget=current_budget, api_key=GEMINI_API_KEY)
            st.session_state.advisor_chat_history.append({"role": "assistant", "content": advice})

    # Render Chat History
    if st.session_state.advisor_chat_history:
        for msg in reversed(st.session_state.advisor_chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.info("👆 Click any of the quick question pills above or type a custom question in the chat bar below!")

    st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)

    # Health Utilities: Burn Rate & Micro-Savings
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        st.markdown("##### 🔥 Safe Daily Spend Allowance")
        st.write(f"**Monthly Target Budget:** ₹{burn['monthly_budget']:,.2f}")
        st.write(f"**Spent so far ({today.strftime('%B')}):** ₹{burn['month_spend']:,.2f}")
        st.write(f"**Days Left in Month:** {burn['days_remaining']} days")
        if burn['safe_daily_allowance'] is not None:
            st.metric("Recommended Daily Spend Limit", f"₹{burn['safe_daily_allowance']:,.2f}/day")
            st.caption(f"Status: **{burn['status']}**")

    with col_u2:
        st.markdown("##### 🪙 Micro-Savings 'Round-Up' Simulator")
        st.caption("See how much you would passively save by rounding up every payment:")
        r_thresh = st.select_slider("Round-Up Threshold", options=[10, 20, 50, 100], value=50)
        micro_sav = simulate_round_up_savings(df_all, round_to=r_thresh)
        st.metric(f"Passive Savings (Round to ₹{r_thresh})", f"₹{micro_sav['total_potential_savings']:,.2f}")
        st.caption(f"Calculated across your {micro_sav['transaction_count']} transactions.")
