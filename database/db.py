import sqlite3

DB_PATH = "app.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        semantic_path TEXT,
        steering_path TEXT
    )
    """)

    conn.commit()
    conn.close()