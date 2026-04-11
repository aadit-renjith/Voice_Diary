from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import os
import traceback
import numpy as np
import pickle
import sqlite3
import json
from collections import Counter
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# ML / Audio — Whisper is loaded lazily at startup

# Local modules
from data_processor import extract_multi_features
from train_model import AttentionLayer, WarmupSchedule
from openrouter_chat_engine import ChatEngine
from database import init_db, get_entries, get_history, upsert_daily_entry
from text_emotion import get_text_emotion
from fusion import fuse

# Configuration
MODEL_PATH = "../models/ser_cnn_lstm.keras"
LABEL_ENCODER_PATH = "../models/label_encoder.pkl"
SCALER_PATH = "../models/scaler.pkl"
TEMP_DIR = "temp_uploads"

# Globals
model = None
scaler = None
label_encoder = None
whisper_model = None
chat_engine = ChatEngine()


class ChatRequest(BaseModel):
    message: str
    emotion: Optional[str] = None
    session_id: str


class StartChatRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, label_encoder, whisper_model

    init_db()

    # Load SER model
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'AttentionLayer': AttentionLayer,
                    'WarmupSchedule': WarmupSchedule
                }
            )
            print("CNN-LSTM model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    # Load scaler
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading scaler: {e}")

    # Load label encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        try:
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            print(f"Label encoder loaded.")
        except Exception as e:
            print(f"Error loading label encoder: {e}")

    # Load Whisper (lazy import — only needed if installed)
    try:
        import whisper as _whisper
        whisper_model = _whisper.load_model("base")
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"Whisper not available (text emotion will be skipped): {e}")
        whisper_model = None

    yield
    # Cleanup code can go here if needed

# Initialize App
app = FastAPI(title="SER Voice Diary API", version="2.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# AUDIO EMOTION PREDICTION
# ---------------------------------------------------------

def predict_from_file(file_path):
    features = extract_multi_features(file_path)

    if features is None:
        return None, None

    n_time, n_freq = features.shape

    if scaler is not None:
        features = scaler.transform(
            features.reshape(-1, n_freq)
        ).reshape(n_time, n_freq)

    features = np.expand_dims(features, axis=0)

    probs = model.predict(features, verbose=0)
    pred_idx = np.argmax(probs, axis=1)[0]

    if label_encoder is not None:
        emotion = label_encoder.inverse_transform([pred_idx])[0]
    else:
        emotion = str(pred_idx)

    return emotion, probs[0]


@app.get("/")
def read_root():
    return {"message": "Welcome to SER Voice Diary API"}


# ---------------------------------------------------------
# VOICE → TEXT + MULTIMODAL EMOTION
# ---------------------------------------------------------

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), source: str = Form("standalone")):

    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_DIR, f"chat_{file.filename}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcription = ""
    audio_emotion = None   # (label, confidence) or None
    text_emotion = None    # (label, confidence) or None
    final_emotion = None   # (label, confidence) or None
    is_chat_source = source == "chat"

    try:
        # 1) Audio Emotion (CNN-BiLSTM model) ─────────────────────────
        try:
            if model and not is_chat_source:
                pred_emotion, probs = predict_from_file(temp_path)
                if pred_emotion is not None and probs is not None:
                    audio_emotion = (pred_emotion, float(np.max(probs)))
                print(f"[audio]  {audio_emotion}")
        except Exception as e:
            print(f"[audio]  FAILED: {e}")
            traceback.print_exc()

        # 2) Whisper Speech-to-Text ───────────────────────────────────
        try:
            if whisper_model:
                result = whisper_model.transcribe(temp_path)
                transcription = result.get("text", "").strip()
                print(f"[whisper] \"{transcription}\"")
        except Exception as e:
            print(f"[whisper] FAILED: {e}")
            traceback.print_exc()

        # 3) Text Emotion (DistilRoBERTa) ────────────────────────────
        try:
            if transcription and not is_chat_source:
                text_emotion = get_text_emotion(transcription)
                print(f"[text]   {text_emotion}")
        except Exception as e:
            print(f"[text]   FAILED: {e}")
            traceback.print_exc()

        # 4) Fusion ───────────────────────────────────────────────────
        final_emotion = None if is_chat_source else fuse(audio_emotion, text_emotion)
        print(f"[fusion] {final_emotion}")

        # 5) Save standalone recording to DB ─────────────────────────
        if source != "chat" and (transcription or final_emotion):
            try:
                upsert_daily_entry(
                    session_id=f"daily-{datetime.now().date().isoformat()}",
                    transcription_append=transcription or "",
                    emotion=final_emotion[0] if final_emotion else None,
                    summary=None,
                    topics=[],
                    full_chat=None,
                )
                print("[db]     updated daily diary entry from standalone recording")
            except Exception as e:
                print(f"[db]     FAILED to save: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"[transcribe] Unexpected error: {e}")
        traceback.print_exc()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "transcription": transcription,
        "audio_emotion": list(audio_emotion) if audio_emotion else None,
        "text_emotion": list(text_emotion) if text_emotion else None,
        "final_emotion": list(final_emotion) if final_emotion else None,
    }


# ---------------------------------------------------------
# CHAT START
# ---------------------------------------------------------

@app.post("/chat/start")
async def start_chat(req: StartChatRequest):
    chat_engine.reset_session(req.session_id)
    return chat_engine.get_opening_message(req.session_id)


# ---------------------------------------------------------
# CHAT MESSAGE
# ---------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):

    result = chat_engine.chat(
        req.session_id,
        req.message,
        req.emotion  # send final_emotion from frontend ideally
    )

    history = chat_engine.get_serializable_history(req.session_id)
    full_chat_json = json.dumps(history) if history else None
    upsert_daily_entry(
        session_id=f"daily-{datetime.now().date().isoformat()}",
        transcription_append=None,
        emotion=req.emotion,
        summary=result["summary"],
        topics=result["detected_topics"],
        full_chat=full_chat_json,
    )

    return result


# ---------------------------------------------------------
# HISTORY
# ---------------------------------------------------------

@app.get("/history")
def fetch_history():
    return get_history()


@app.delete("/history/{entry_id}")
def delete_history_entry(entry_id: int):
    conn = sqlite3.connect("voice_diary.db")
    c = conn.cursor()
    c.execute("DELETE FROM diary_entries WHERE id = ?", (entry_id,))
    conn.commit()
    deleted = c.rowcount
    conn.close()

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"message": "History entry deleted", "id": entry_id}


@app.delete("/history")
def clear_all_history():
    conn = sqlite3.connect("voice_diary.db")
    c = conn.cursor()
    c.execute("DELETE FROM diary_entries")
    conn.commit()
    conn.close()
    return {"message": "All history cleared"}


# ---------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------

@app.get("/analytics/emotions")
def emotion_distribution():
    entries = get_entries()
    emotions = [e[1] for e in entries]
    return dict(Counter(emotions))


@app.get("/analytics/weekly")
def weekly_report():
    conn = sqlite3.connect("voice_diary.db")
    c = conn.cursor()

    week_ago = datetime.now() - timedelta(days=7)

    c.execute("""
    SELECT emotion FROM diary_entries
    WHERE date >= ?
    """, (week_ago.isoformat(),))

    rows = c.fetchall()
    conn.close()

    emotions = [r[0] for r in rows]
    counts = dict(Counter(emotions))

    dominant = max(counts, key=counts.get) if counts else None

    return {
        "total_entries": len(rows),
        "emotion_distribution": counts,
        "dominant_emotion": dominant
    }


@app.get("/analytics/monthly")
def monthly_report():
    conn = sqlite3.connect("voice_diary.db")
    c = conn.cursor()

    month_ago = datetime.now() - timedelta(days=30)

    c.execute("""
    SELECT emotion FROM diary_entries
    WHERE date >= ?
    """, (month_ago.isoformat(),))

    rows = c.fetchall()
    conn.close()

    emotions = [r[0] for r in rows]
    counts = dict(Counter(emotions))

    return {
        "total_entries": len(rows),
        "emotion_distribution": counts
    }


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
