import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'
load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

def save_api_key_to_env(api_key: str):
    """Saves the API key permanently to the .env file."""
    if not api_key:
        return
    with open(ENV_PATH, 'w', encoding='utf-8') as f:
        f.write(f"GEMINI_API_KEY={api_key.strip()}\n")
    os.environ['GEMINI_API_KEY'] = api_key.strip()

# Database Paths
DB_PATH = BASE_DIR / 'pennypilot.db'
CHROMA_PERSIST_DIR = str(BASE_DIR / 'chroma_data')

# Default Expense Categories
DEFAULT_CATEGORIES = [
    'Food & Dining',
    'Groceries',
    'Travel & Commute',
    'Shopping & E-Commerce',
    'Bills & Utilities',
    'Entertainment & Subscriptions',
    'Health & Personal Care',
    'Education & Books',
    'Transfers & Investments',
    'Miscellaneous'
]

# Resilient Model Fallback Sequence
MODEL_CANDIDATES = ['gemini-3.7-flash', 'gemini-3.6-flash', 'gemini-3.5-flash-lite']
PRIMARY_MODEL = 'gemini-3.7-flash'
FALLBACK_MODEL = 'gemini-3.5-flash-lite'
