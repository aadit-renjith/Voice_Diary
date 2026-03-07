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
        topics TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_entry(session_id, transcription, emotion, summary, topics):

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    topics_str = ",".join(topics) if topics else ""

    c.execute("""
    INSERT INTO diary_entries
    (session_id, date, transcription, emotion, summary, topics)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        datetime.now().isoformat(),
        transcription,
        emotion,
        summary,
        topics_str
    ))

    conn.commit()
    conn.close()


def get_entries():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT date, emotion FROM diary_entries")

    rows = c.fetchall()

    conn.close()

    return rows