import sqlite3
from datetime import datetime

DB_PATH = "voice_diary.db"


def init_db():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS diary_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        date TEXT,
        transcription TEXT,
        emotion TEXT,
        summary TEXT,
        topics TEXT,
        full_chat TEXT
    )
    """)

    # Check if full_chat column exists (for migrations)
    c.execute("PRAGMA table_info(diary_entries)")
    columns = [info[1] for info in c.fetchall()]
    if "full_chat" not in columns:
        c.execute("ALTER TABLE diary_entries ADD COLUMN full_chat TEXT")

    conn.commit()
    conn.close()


def save_entry(session_id, transcription, emotion, summary, topics, full_chat=None):

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    topics_str = ",".join(topics) if topics else ""

    c.execute("""
    INSERT INTO diary_entries
    (session_id, date, transcription, emotion, summary, topics, full_chat)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        datetime.now().isoformat(),
        transcription,
        emotion,
        summary,
        topics_str,
        full_chat
    ))

    conn.commit()
    conn.close()


def get_entries():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT date, emotion FROM diary_entries ORDER BY date DESC")

    rows = c.fetchall()

    conn.close()

    return rows


def get_history():
    """Fetch all history entries for the history panel."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    SELECT id, session_id, date, transcription, emotion, summary, topics, full_chat 
    FROM diary_entries 
    ORDER BY date DESC
    """)

    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "session_id": row[1],
            "date": row[2],
            "transcription": row[3],
            "emotion": row[4],
            "summary": row[5],
            "topics": row[6].split(",") if row[6] else [],
            "full_chat": row[7]
        })

    return history