import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from ..config import DB_PATH, DEFAULT_CATEGORIES

def get_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite tables if they do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Transactions Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'guest',
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                recipient_name TEXT,
                recipient_upi TEXT,
                category TEXT NOT NULL,
                payment_app TEXT DEFAULT 'Manual',
                transaction_type TEXT DEFAULT 'DEBIT',
                raw_input TEXT,
                notes TEXT,
                confidence_score REAL DEFAULT 1.0,
                is_clarified INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        ''')

        # Check and migrate columns
        cursor.execute("PRAGMA table_info(transactions)")
        cols = [c[1] for c in cursor.fetchall()]
        if 'user_id' not in cols:
            cursor.execute("ALTER TABLE transactions ADD COLUMN user_id TEXT DEFAULT 'guest'")
        if 'transaction_type' not in cols:
            cursor.execute("ALTER TABLE transactions ADD COLUMN transaction_type TEXT DEFAULT 'DEBIT'")

        # Monthly Budgets Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'guest',
                month TEXT NOT NULL,
                target_amount REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, month)
            )
        ''')

        # User Custom Categories Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'guest',
                category_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, category_name)
            )
        ''')

        # Merchant Category Overrides
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS merchant_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'guest',
                merchant_key TEXT NOT NULL,
                category TEXT NOT NULL,
                notes TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, merchant_key)
            )
        ''')

        # Auto-heal: Fix any existing transactions where notes or raw_input indicate received/credit
        cursor.execute('''
            UPDATE transactions
            SET transaction_type = 'CREDIT'
            WHERE (
                LOWER(notes) LIKE '%received%' OR
                LOWER(notes) LIKE '%credited%' OR
                LOWER(notes) LIKE '%refund%' OR
                LOWER(notes) LIKE '%cashback%' OR
                LOWER(notes) LIKE '%reimbursement%' OR
                LOWER(raw_input) LIKE '%received%' OR
                LOWER(raw_input) LIKE '%sent you%' OR
                LOWER(raw_input) LIKE '%refund%'
            ) AND transaction_type != 'CREDIT'
        ''')

        conn.commit()

def add_transaction(
    date: str,
    amount: float,
    recipient_name: str,
    category: str,
    recipient_upi: Optional[str] = None,
    payment_app: str = 'Manual',
    transaction_type: str = 'DEBIT',
    raw_input: Optional[str] = None,
    notes: Optional[str] = None,
    confidence_score: float = 1.0,
    is_clarified: int = 1,
    user_id: str = 'guest'
) -> int:
    """Inserts a new transaction for a specific user and returns its ID."""
    created_at = datetime.now().isoformat()
    
    # Check all fields for incoming credit / reimbursement indications
    all_text = f"{transaction_type} {notes or ''} {raw_input or ''}".lower()
    if any(k in all_text for k in ['credit', 'received', 'credited', 'refund', 'cashback', 'reimbursement', 'repaid', 'got from']):
        clean_type = "CREDIT"
    else:
        clean_type = "DEBIT"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (
                user_id, date, amount, recipient_name, recipient_upi, category,
                payment_app, transaction_type, raw_input, notes, confidence_score, is_clarified, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, date, amount, recipient_name, recipient_upi, category,
            payment_app, clean_type, raw_input, notes, confidence_score, is_clarified, created_at
        ))
        conn.commit()
        return cursor.lastrowid

def get_all_transactions(user_id: str = 'guest') -> pd.DataFrame:
    """Returns all transactions for a specific user as a DataFrame with verified DEBIT/CREDIT types."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            'SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC, id DESC', 
            conn, 
            params=(user_id,)
        )
        if not df.empty:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            if 'transaction_type' not in df.columns:
                df['transaction_type'] = 'DEBIT'
            # AUTO-HEAL: If notes or raw_input indicates received money, set type to CREDIT
            if 'notes' in df.columns:
                rec_notes = df['notes'].fillna('').str.lower().str.contains('received|credited|refund|cashback|reimbursement|repaid|settled up|got from|sent me')
                df.loc[rec_notes, 'transaction_type'] = 'CREDIT'
            if 'raw_input' in df.columns:
                rec_input = df['raw_input'].fillna('').str.lower().str.contains('received|credited to|cashback|refund|sent you|received from')
                df.loc[rec_input, 'transaction_type'] = 'CREDIT'
        return df

def get_user_categories(user_id: str = 'guest') -> List[str]:
    """Returns standard default categories plus custom categories created by this user."""
    categories = list(DEFAULT_CATEGORIES)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT category_name FROM user_categories WHERE user_id = ? ORDER BY id ASC', (user_id,))
        rows = cursor.fetchall()
        for r in rows:
            c_name = r['category_name'].strip()
            if c_name and c_name not in categories:
                categories.append(c_name)
    return categories

def add_user_category(category_name: str, user_id: str = 'guest') -> bool:
    """Adds a new custom category for a user."""
    clean_name = category_name.strip()
    if not clean_name:
        return False
    created_at = datetime.now().isoformat()
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_categories (user_id, category_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, category_name) DO NOTHING
            ''', (user_id, clean_name, created_at))
            conn.commit()
            return True
    except Exception:
        return False

def delete_user_category(category_name: str, user_id: str = 'guest') -> bool:
    """Deletes a custom category for a user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_categories WHERE user_id = ? AND category_name = ?', (user_id, category_name))
        conn.commit()
        return True

def get_all_existing_users() -> List[Dict[str, Any]]:
    """Returns a list of all distinct users with transaction counts and latest activity."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                user_id,
                COUNT(id) as tx_count,
                MAX(date) as last_tx_date
            FROM transactions
            WHERE user_id != 'guest'
            GROUP BY user_id
            ORDER BY last_tx_date DESC
        ''')
        rows = cursor.fetchall()
        users = []
        for r in rows:
            uid = r['user_id']
            # Reconstruct pretty name
            display_name = uid.replace("_", " ").title()
            users.append({
                "user_id": uid,
                "display_name": display_name,
                "tx_count": r['tx_count'],
                "last_tx_date": r['last_tx_date']
            })
        return users

def update_transaction_category(transaction_id: int, new_category: str, notes: Optional[str] = None, user_id: str = 'guest'):
    """Updates the category and notes of an existing transaction."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE transactions 
            SET category = ?, notes = COALESCE(?, notes), is_clarified = 1
            WHERE id = ? AND user_id = ?
        ''', (new_category, notes, transaction_id, user_id))
        conn.commit()

def update_transaction_type(transaction_id: int, transaction_type: str, user_id: str = 'guest'):
    """Updates the transaction type (DEBIT or CREDIT) of an existing transaction."""
    clean_type = "CREDIT" if "credit" in transaction_type.lower() or "received" in transaction_type.lower() else "DEBIT"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE transactions 
            SET transaction_type = ? 
            WHERE id = ? AND user_id = ?
        ''', (clean_type, transaction_id, user_id))
        conn.commit()

def delete_transaction(transaction_id: int, user_id: str = 'guest'):
    """Deletes a transaction by ID for a specific user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM transactions WHERE id = ? AND user_id = ?', (transaction_id, user_id))
        conn.commit()

def set_monthly_budget(month: str, target_amount: float, user_id: str = 'guest'):
    """Sets or updates a monthly budget for month format YYYY-MM."""
    created_at = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO budgets (user_id, month, target_amount, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, month) DO UPDATE SET
                target_amount = excluded.target_amount,
                created_at = excluded.created_at
        ''', (user_id, month, target_amount, created_at))
        conn.commit()

def get_monthly_budget(month: str, user_id: str = 'guest') -> Optional[float]:
    """Fetches the monthly target budget amount for month YYYY-MM."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT target_amount FROM budgets WHERE user_id = ? AND month = ?', (user_id, month))
        row = cursor.fetchone()
        return row['target_amount'] if row else None
