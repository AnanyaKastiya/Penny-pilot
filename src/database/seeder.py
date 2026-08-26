import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta
from ..config import DB_PATH
from ..ai.memory import memory_store

def seed_guest_data_if_empty():
    """Seeds 2 full months of realistic data for 'guest' showing a positive decrease in spend in Month 2."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE user_id = 'guest'")
    count = cursor.fetchone()[0]
    
    if count == 0:
        today = date.today()
        # Month 1: 30 to 60 days ago (High Spend: ~₹24,850)
        # Month 2: 0 to 29 days ago (Disciplined Spend: ~₹15,449) -> ~38% decrease!
        
        sample_transactions = [
            # ----------------- MONTH 1 (PREVIOUS MONTH: HIGH SPEND ~₹24,850) ----------------- #
            ((today - timedelta(days=58)).strftime("%Y-%m-%d"), 649.0, "Netflix", "netflix@citi", "Entertainment & Subscriptions", "Auto-Debit", "Monthly Netflix subscription", "Monthly Netflix subscription", 1.0, 1),
            ((today - timedelta(days=57)).strftime("%Y-%m-%d"), 1450.0, "BESCOM Electricity", "bescom@karnataka", "Bills & Utilities", "GPay", "Summer AC high power bill", "Home electricity", 1.0, 1),
            ((today - timedelta(days=55)).strftime("%Y-%m-%d"), 3200.0, "D-Mart Supermarket", "dmart@pos", "Groceries", "Card", "Groceries & household items", "Monthly bulk groceries", 1.0, 1),
            ((today - timedelta(days=53)).strftime("%Y-%m-%d"), 850.0, "Swiggy", "swiggy@icici", "Food & Dining", "GPay", "Late night pizza & dessert", "Late night food", 1.0, 1),
            ((today - timedelta(days=50)).strftime("%Y-%m-%d"), 420.0, "Uber", "uber@axis", "Travel & Commute", "PhonePe", "Surge price cab ride", "Cab", 1.0, 1),
            ((today - timedelta(days=48)).strftime("%Y-%m-%d"), 3800.0, "Zara India", "zara@hdfc", "Shopping & E-Commerce", "Card", "Impulse clothes shopping", "Clothes & shoes", 1.0, 1),
            ((today - timedelta(days=46)).strftime("%Y-%m-%d"), 950.0, "Zomato", "zomato@icici", "Food & Dining", "GPay", "Weekend restaurant dining", "Weekend dinner", 1.0, 1),
            ((today - timedelta(days=44)).strftime("%Y-%m-%d"), 600.0, "Uber", "uber@axis", "Travel & Commute", "PhonePe", "Peak hour cab", "Cab ride", 1.0, 1),
            ((today - timedelta(days=42)).strftime("%Y-%m-%d"), 1400.0, "PVR Cinemas", "pvr@icici", "Entertainment & Subscriptions", "GPay", "Movie tickets & popcorn", "Weekend movie", 1.0, 1),
            ((today - timedelta(days=40)).strftime("%Y-%m-%d"), 780.0, "Swiggy", "swiggy@icici", "Food & Dining", "GPay", "Office lunch delivery", "Lunch order", 1.0, 1),
            ((today - timedelta(days=38)).strftime("%Y-%m-%d"), 2200.0, "Myntra", "myntra@icici", "Shopping & E-Commerce", "GPay", "Sneakers purchase", "Shoes", 1.0, 1),
            ((today - timedelta(days=36)).strftime("%Y-%m-%d"), 1200.0, "Chai Point & Cafe", "chaipoint@hdfc", "Food & Dining", "Paytm", "Frequent cafe meetups", "Cafe snacks", 1.0, 1),
            ((today - timedelta(days=34)).strftime("%Y-%m-%d"), 950.0, "Zomato", "zomato@icici", "Food & Dining", "GPay", "Late dinner with friends", "Food delivery", 1.0, 1),
            ((today - timedelta(days=32)).strftime("%Y-%m-%d"), 800.0, "Uber", "uber@axis", "Travel & Commute", "PhonePe", "Airport drop cab", "Airport cab", 1.0, 1),
            ((today - timedelta(days=31)).strftime("%Y-%m-%d"), 4600.0, "Room Rent / Maintenance", "landlord@sbi", "Bills & Utilities", "GPay", "Monthly room maintenance", "Maintenance", 1.0, 1),

            # ----------------- MONTH 2 (CURRENT MONTH: OPTIMIZED SPEND ~₹15,449) ----------------- #
            ((today - timedelta(days=26)).strftime("%Y-%m-%d"), 649.0, "Netflix", "netflix@citi", "Entertainment & Subscriptions", "Auto-Debit", "Monthly subscription", "Netflix", 1.0, 1),
            ((today - timedelta(days=24)).strftime("%Y-%m-%d"), 1100.0, "BESCOM Electricity", "bescom@karnataka", "Bills & Utilities", "GPay", "Power bill (optimized usage)", "Home electricity", 1.0, 1),
            ((today - timedelta(days=22)).strftime("%Y-%m-%d"), 799.0, "Airtel Broadband", "airtel@icici", "Bills & Utilities", "PhonePe", "Monthly wifi internet", "Wifi bill", 1.0, 1),
            ((today - timedelta(days=20)).strftime("%Y-%m-%d"), 2800.0, "D-Mart Supermarket", "dmart@pos", "Groceries", "Card", "Monthly groceries & cooking ingredients", "Groceries", 1.0, 1),
            ((today - timedelta(days=17)).strftime("%Y-%m-%d"), 350.0, "Swiggy", "swiggy@icici", "Food & Dining", "GPay", "Occasional dinner order", "Dinner", 1.0, 1),
            ((today - timedelta(days=15)).strftime("%Y-%m-%d"), 220.0, "Uber", "uber@axis", "Travel & Commute", "PhonePe", "Urgent commute ride", "Cab", 1.0, 1),
            ((today - timedelta(days=12)).strftime("%Y-%m-%d"), 1100.0, "Amazon India", "amazon@apl", "Shopping & E-Commerce", "GPay", "Essential study books & cables", "Study items", 1.0, 1),
            ((today - timedelta(days=10)).strftime("%Y-%m-%d"), 320.0, "Chai Point", "chaipoint@hdfc", "Food & Dining", "Paytm", "Team tea & biscuits", "Snacks", 1.0, 1),
            ((today - timedelta(days=8)).strftime("%Y-%m-%d"), 350.0, "Metro Card Recharge", "dmrc@sbi", "Travel & Commute", "GPay", "Smart card top-up (Switched from cabs!)", "Metro commute", 1.0, 1),
            ((today - timedelta(days=6)).strftime("%Y-%m-%d"), 380.0, "Ramesh Kumar", "ramesh@okaxis", "Food & Dining", "PhonePe", "College canteen lunch", "Canteen lunch", 1.0, 1),
            ((today - timedelta(days=4)).strftime("%Y-%m-%d"), 480.0, "Zomato", "zomato@icici", "Food & Dining", "GPay", "Weekend lunch treat", "Weekend treat", 1.0, 1),
            ((today - timedelta(days=2)).strftime("%Y-%m-%d"), 120.0, "Auto Rickshaw Fare", "auto@upi", "Travel & Commute", "GPay", "Auto to station", "Commute", 1.0, 1),
            ((today - timedelta(days=1)).strftime("%Y-%m-%d"), 4600.0, "Room Rent / Maintenance", "landlord@sbi", "Bills & Utilities", "GPay", "Monthly room maintenance", "Maintenance", 1.0, 1),
            (today.strftime("%Y-%m-%d"), 70.0, "Chai Point", "chaipoint@hdfc", "Food & Dining", "GPay", "Morning tea", "Tea", 1.0, 1),
        ]
        
        for dt, amt, rec, upi, cat, app, raw, note, conf, is_c in sample_transactions:
            cursor.execute('''
                INSERT INTO transactions (
                    user_id, date, amount, recipient_name, recipient_upi, category,
                    payment_app, raw_input, notes, confidence_score, is_clarified, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', ('guest', dt, amt, rec, upi, cat, app, raw, note, conf, is_c, datetime.now().isoformat()))
            tx_id = cursor.lastrowid
            memory_store.index_transaction(tx_id, dt, amt, rec, cat, note)
            memory_store.save_merchant_mapping(rec, cat, note, upi)

        # Set monthly budget for guest
        prev_m = (today - timedelta(days=35)).strftime("%Y-%m")
        curr_m = today.strftime("%Y-%m")
        
        cursor.execute('''
            INSERT INTO budgets (user_id, month, target_amount, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, month) DO NOTHING
        ''', ('guest', prev_m, 20000.0, datetime.now().isoformat()))

        cursor.execute('''
            INSERT INTO budgets (user_id, month, target_amount, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, month) DO NOTHING
        ''', ('guest', curr_m, 16000.0, datetime.now().isoformat()))

        conn.commit()
    conn.close()
