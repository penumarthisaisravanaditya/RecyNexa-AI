
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "recynexa_history.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL DEFAULT
                    (datetime('now', 'localtime')),
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                contamination TEXT,
                recommendation TEXT
            )
        """)


def save_scan(
    filename,
    predicted_class,
    confidence,
    status,
    contamination=None,
    recommendation=None
):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO scans (
                filename,
                predicted_class,
                confidence,
                status,
                contamination,
                recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            filename,
            predicted_class,
            float(confidence),
            status,
            contamination,
            recommendation
        ))


def get_all_scans():
    with get_connection() as connection:
        cursor = connection.execute("""
            SELECT id, scanned_at, filename, predicted_class,
                   confidence, status, contamination, recommendation
            FROM scans
            ORDER BY id DESC
        """)
        return cursor.fetchall()


def clear_scan_history():
    with get_connection() as connection:
        connection.execute("DELETE FROM scans")


initialize_database()
