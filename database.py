import sqlite3
import hashlib
from datetime import datetime
from typing import Optional
import pandas as pd

DB_PATH = "expense_tracker.db"

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Sports & Fitness",
    "Travel & Transport",
    "Shopping",
    "Entertainment",
    "Bills & EMI",
    "Healthcare",
    "Utilities",
    "Personal Care",
    "Indulgence",
    "Other",
]


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            merchant TEXT,
            date TEXT NOT NULL,
            description TEXT,
            category TEXT DEFAULT 'Other',
            source TEXT DEFAULT 'sms',
            raw_text TEXT,
            notes TEXT DEFAULT '',
            is_pending INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Migrate existing DB: add is_pending column if missing
    try:
        c.execute("ALTER TABLE transactions ADD COLUMN is_pending INTEGER DEFAULT 0")
    except Exception:
        pass  # column already exists

    c.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Generate a default webhook token if not set
    import secrets as _secrets
    c.execute("INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)",
              ("webhook_token", _secrets.token_urlsafe(24)))

    c.execute("""
        CREATE TABLE IF NOT EXISTS transaction_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER NOT NULL,
            item_name TEXT,
            quantity TEXT,
            price REAL,
            category TEXT DEFAULT 'Other',
            FOREIGN KEY (transaction_id) REFERENCES transactions(id)
        )
    """)

    default_users = [
        (1, "ankush", _hash("ank123"), "Ankush"),
        (2, "akash",  _hash("aka123"), "Akash"),
        (3, "arpita", _hash("arp123"), "Arpita"),
        (4, "sai",    _hash("sai123"), "Sai"),
    ]
    for uid, username, pwd_hash, display_name in default_users:
        c.execute(
            "INSERT OR IGNORE INTO users (id, username, password_hash, display_name) VALUES (?,?,?,?)",
            (uid, username, pwd_hash, display_name),
        )

    conn.commit()
    conn.close()


def verify_user(username: str, password: str):
    """Returns (user_id, display_name) or None."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, display_name FROM users WHERE username=? AND password_hash=?",
        (username, _hash(password)),
    )
    result = c.fetchone()
    conn.close()
    return result


def get_all_users() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, display_name FROM users ORDER BY id", conn)
    conn.close()
    return df


def get_config(key: str) -> Optional[str]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM config WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def set_config(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?,?)", (key, value))
    conn.commit()
    conn.close()


def save_transaction(
    user_id, amount, merchant, date, description, category, source, raw_text,
    notes="", is_pending=False
) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO transactions
           (user_id, amount, merchant, date, description, category, source, raw_text, notes, is_pending)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (user_id, amount, merchant, date, description, category, source, raw_text, notes, int(is_pending)),
    )
    transaction_id = c.lastrowid
    conn.commit()
    conn.close()
    return transaction_id


def save_transaction_items(transaction_id: int, items: list):
    if not items:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for item in items:
        c.execute(
            "INSERT INTO transaction_items (transaction_id, item_name, quantity, price, category) VALUES (?,?,?,?,?)",
            (
                transaction_id,
                item.get("name", ""),
                item.get("quantity", ""),
                item.get("price", 0.0),
                item.get("category", "Other"),
            ),
        )
    conn.commit()
    conn.close()


def update_transaction_category(transaction_id: int, category: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE transactions SET category=? WHERE id=?", (category, transaction_id)
    )
    conn.commit()
    conn.close()


def confirm_pending_transaction(transaction_id: int, category: str, notes: str = ""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE transactions SET is_pending=0, category=?, notes=? WHERE id=?",
        (category, notes, transaction_id),
    )
    conn.commit()
    conn.close()


def get_pending_transactions(user_id: int = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT t.id, u.display_name as user, t.amount, t.merchant,
               t.date, t.category, t.source, t.raw_text, t.created_at
        FROM transactions t
        JOIN users u ON t.user_id = u.id
        WHERE t.is_pending = 1
    """
    params = []
    if user_id:
        query += " AND t.user_id = ?"
        params.append(user_id)
    query += " ORDER BY t.created_at DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def delete_transaction(transaction_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM transaction_items WHERE transaction_id=?", (transaction_id,))
    conn.execute("DELETE FROM transactions WHERE id=?", (transaction_id,))
    conn.commit()
    conn.close()


def get_transactions(
    user_ids=None, start_date=None, end_date=None, category=None
) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT t.id, u.id as user_id, u.display_name as user, t.amount,
               t.merchant, t.date, t.description, t.category, t.source, t.notes
        FROM transactions t
        JOIN users u ON t.user_id = u.id
        WHERE t.is_pending = 0
    """
    params = []

    if user_ids:
        placeholders = ",".join("?" * len(user_ids))
        query += f" AND t.user_id IN ({placeholders})"
        params.extend(user_ids)

    if start_date:
        query += " AND t.date >= ?"
        params.append(str(start_date))

    if end_date:
        query += " AND t.date <= ?"
        params.append(str(end_date))

    if category and category != "All":
        query += " AND t.category = ?"
        params.append(category)

    query += " ORDER BY t.date DESC, t.id DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_transaction_items(transaction_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM transaction_items WHERE transaction_id=?",
        conn,
        params=(transaction_id,),
    )
    conn.close()
    return df


def get_spending_by_category(user_ids=None, start_date=None, end_date=None) -> pd.DataFrame:
    df = get_transactions(user_ids, start_date, end_date)
    if df.empty:
        return pd.DataFrame(columns=["category", "amount"])
    return df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)


def get_daily_spending(user_ids=None, start_date=None, end_date=None) -> pd.DataFrame:
    df = get_transactions(user_ids, start_date, end_date)
    if df.empty:
        return pd.DataFrame(columns=["date", "amount"])
    return df.groupby("date")["amount"].sum().reset_index().sort_values("date")


def get_spending_summary_text(user_ids=None, start_date=None, end_date=None) -> str:
    df = get_transactions(user_ids, start_date, end_date)
    if df.empty:
        return "No transactions found for this period."

    total = df["amount"].sum()
    by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    by_user = df.groupby("user")["amount"].sum().sort_values(ascending=False)
    by_day = df.groupby("date")["amount"].sum()

    lines = [
        f"Total spending: INR {total:,.2f}",
        f"Number of transactions: {len(df)}",
        f"Date range: {df['date'].min()} to {df['date'].max()}",
        "",
        "Spending by category:",
    ]
    for cat, amt in by_category.items():
        pct = (amt / total) * 100
        lines.append(f"  {cat}: INR {amt:,.2f} ({pct:.1f}%)")

    lines.append("")
    lines.append("Spending by user:")
    for user, amt in by_user.items():
        lines.append(f"  {user}: INR {amt:,.2f}")

    avg_daily = by_day.mean()
    lines.append(f"\nAverage daily spending: INR {avg_daily:,.2f}")

    top_merchants = df.groupby("merchant")["amount"].sum().sort_values(ascending=False).head(5)
    lines.append("\nTop 5 merchants:")
    for merchant, amt in top_merchants.items():
        if merchant:
            lines.append(f"  {merchant}: INR {amt:,.2f}")

    return "\n".join(lines)
