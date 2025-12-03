import os
import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alerts.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    src_ip TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT
);
"""


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_attack(src_ip: str, action: str, details: str):
    """Persist detection results for later dashboard use."""
    timestamp = datetime.utcnow().isoformat()
    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO alerts (timestamp, src_ip, action, details) VALUES (?, ?, ?, ?)",
                (timestamp, src_ip, action, details),
            )
            conn.commit()
    except sqlite3.Error as exc:
        print(f"⚠️ DB yazma hatası: {exc}")


def fetch_logs():
    """Return alert history as a DataFrame ordered by latest timestamp."""
    columns = ["id", "timestamp", "src_ip", "action", "details"]

    try:
        with _get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT id, timestamp, src_ip, action, details FROM alerts "
                "ORDER BY datetime(timestamp) DESC",
                conn,
            )
            return df
    except sqlite3.Error as exc:
        print(f"⚠️ DB okuma hatası: {exc}")
        return pd.DataFrame(columns=columns)
