import sqlite3
import json
import re
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


def _merge_text(existing, new_text):
    existing = (existing or "").strip()
    new_text = (new_text or "").strip()

    if not new_text:
        return existing
    if not existing:
        return new_text
    if new_text in existing:
        return existing

    return f"{existing}\n\n{new_text}"


def _merge_topics(existing_topics, new_topics):
    merged = []
    seen = set()

    for topic in (existing_topics or []) + (new_topics or []):
        cleaned = (topic or "").strip()
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        merged.append(cleaned)

    return merged


def _clean_chat_text(text):
    cleaned = (text or "").strip()
    return re.sub(r"^\[.*?\]\s*", "", cleaned)


def _extract_chat_highlights(full_chat):
    if not full_chat:
        return []

    try:
        chat_data = json.loads(full_chat) if isinstance(full_chat, str) else full_chat
    except Exception:
        return []

    highlights = []
    for msg in chat_data or []:
        if msg.get("role") != "user":
            continue

        parts = msg.get("parts", [])
        text = _clean_chat_text(parts[0].get("text", "")) if parts else ""
        if text:
            highlights.append(text)

    return highlights


def _shorten(text, limit=160):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_entry_summary(transcription, full_chat, topics, emotion, explicit_summary=None):
    topics = [topic for topic in (topics or []) if topic]
    highlights = _extract_chat_highlights(full_chat)
    transcript_text = _shorten(transcription, 180)
    mood_text = f"Overall mood: {emotion}." if emotion else None

    if explicit_summary:
        base = explicit_summary.strip()
        if topics:
            topic_text = f"Main topics: {', '.join(topics[:4])}."
            return f"{base} {topic_text}" if topic_text not in base else base
        return base

    if highlights:
        latest = _shorten(highlights[-1], 120)
        parts = [f"Today's chat captured {len(highlights)} update{'s' if len(highlights) != 1 else ''}."]
        parts.append(f"Latest note: {latest}")
        if topics:
            parts.append(f"Main topics: {', '.join(topics[:4])}.")
        if mood_text:
            parts.append(mood_text)
        return " ".join(parts)

    if transcript_text:
        parts = [f"Today's diary note: {transcript_text}"]
        if topics:
            parts.append(f"Main topics: {', '.join(topics[:4])}.")
        if mood_text:
            parts.append(mood_text)
        return " ".join(parts)

    if topics or mood_text:
        parts = []
        if topics:
            parts.append(f"Main topics: {', '.join(topics[:4])}.")
        if mood_text:
            parts.append(mood_text)
        return " ".join(parts)

    return None


def upsert_daily_entry(session_id, emotion=None, summary=None, topics=None, full_chat=None, transcription_append=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now()
    today_key = now.date().isoformat()

    c.execute("""
    SELECT id, session_id, transcription, emotion, summary, topics, full_chat
    FROM diary_entries
    WHERE substr(date, 1, 10) = ?
    ORDER BY date DESC
    LIMIT 1
    """, (today_key,))

    row = c.fetchone()

    if row:
        entry_id = row[0]
        existing_session_id = row[1]
        existing_transcription = row[2]
        existing_emotion = row[3]
        existing_summary = row[4]
        existing_topics = row[5].split(",") if row[5] else []
        existing_full_chat = row[6]

        merged_transcription = _merge_text(existing_transcription, transcription_append)
        merged_topics = _merge_topics(existing_topics, topics)
        merged_full_chat = full_chat if full_chat is not None else existing_full_chat
        merged_summary = _build_entry_summary(
            merged_transcription,
            merged_full_chat,
            merged_topics,
            emotion or existing_emotion,
            explicit_summary=summary,
        )

        c.execute("""
        UPDATE diary_entries
        SET session_id = ?, date = ?, transcription = ?, emotion = ?, summary = ?, topics = ?, full_chat = ?
        WHERE id = ?
        """, (
            session_id or existing_session_id,
            now.isoformat(),
            merged_transcription,
            emotion or existing_emotion,
            merged_summary,
            ",".join(merged_topics),
            merged_full_chat,
            entry_id,
        ))
    else:
        merged_topics = _merge_topics([], topics)
        initial_transcription = (transcription_append or "").strip()
        initial_summary = _build_entry_summary(
            initial_transcription,
            full_chat,
            merged_topics,
            emotion,
            explicit_summary=summary,
        )
        c.execute("""
        INSERT INTO diary_entries
        (session_id, date, transcription, emotion, summary, topics, full_chat)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            now.isoformat(),
            initial_transcription,
            emotion,
            initial_summary,
            ",".join(merged_topics),
            full_chat,
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
