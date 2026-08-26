import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from ..config import DB_PATH

def get_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite tables if they do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Transactions Table with user_id
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
                raw_input TEXT,
                notes TEXT,
                confidence_score REAL DEFAULT 1.0,
                is_clarified INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        ''')

        # Check if user_id column exists (for migrations)
        cursor.execute("PRAGMA table_info(transactions)")
        cols = [c[1] for c in cursor.fetchall()]
        if 'user_id' not in cols:
            cursor.execute("ALTER TABLE transactions ADD COLUMN user_id TEXT DEFAULT 'guest'")

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

        conn.commit()

def add_transaction(
    date: str,
    amount: float,
    recipient_name: str,
    category: str,
    recipient_upi: Optional[str] = None,
    payment_app: str = 'Manual',
    raw_input: Optional[str] = None,
    notes: Optional[str] = None,
    confidence_score: float = 1.0,
    is_clarified: int = 1,
    user_id: str = 'guest'
) -> int:
    """Inserts a new transaction for a specific user and returns its ID."""
    created_at = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (
                user_id, date, amount, recipient_name, recipient_upi, category,
                payment_app, raw_input, notes, confidence_score, is_clarified, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, date, amount, recipient_name, recipient_upi, category,
            payment_app, raw_input, notes, confidence_score, is_clarified, created_at
        ))
        conn.commit()
        return cursor.lastrowid

def get_all_transactions(user_id: str = 'guest') -> pd.DataFrame:
    """Returns all transactions for a specific user as a DataFrame."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            'SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC, id DESC', 
            conn, 
            params=(user_id,)
        )
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df

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
