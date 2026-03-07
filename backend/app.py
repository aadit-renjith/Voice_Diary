from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import os
import speech_recognition as sr
import numpy as np
import pickle
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

from data_processor import extract_features
from chat_engine import ChatEngine
from database import init_db, save_entry, get_entries


MODEL_PATH = "../models/pro_ser_model.pkl"
TEMP_DIR = "temp_uploads"


app = FastAPI(title="SER Voice Diary API", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
chat_engine = ChatEngine()


class ChatRequest(BaseModel):
    message: str
    emotion: Optional[str] = None
    session_id: str


class StartChatRequest(BaseModel):
    session_id: str


@app.on_event("startup")
def startup():

    global model

    init_db()

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    else:
        print("Model file not found. Ensure training is complete.")


@app.get("/")
def read_root():
    return {"message": "Welcome to SER Voice Diary API"}


# ---------------------------------------------------------
# EMOTION PREDICTION (VOICE RECORDER)
# ---------------------------------------------------------

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    print(f"Received file: {file.filename}")

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    temp_path = os.path.join(TEMP_DIR, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:

        features = extract_features(temp_path)

        if features is None:
            raise HTTPException(status_code=400, detail="Could not process audio file")

        features = features.reshape(1, -1)

        prediction = model.predict(features)

        emotion = prediction[0]

        print("Predicted emotion:", emotion)

        # IMPORTANT: SAVE ENTRY FOR REPORTS
        save_entry(
            session_id="voice_recording",
            transcription="voice emotion capture",
            emotion=emotion,
            summary=None,
            topics=[]
        )

        print("Entry saved to database")

        os.remove(temp_path)

        return {"emotion": emotion}

    except Exception as e:

        print("Prediction error:", e)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# TRANSCRIBE AUDIO
# ---------------------------------------------------------

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    temp_path = os.path.join(TEMP_DIR, f"chat_{file.filename}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcription = ""
    emotion = None

    try:

        recognizer = sr.Recognizer()

        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

        if model:

            features = extract_features(temp_path)

            if features is not None:
                features = features.reshape(1, -1)
                prediction = model.predict(features)
                emotion = prediction[0]

    except Exception as e:
        print("Transcription error:", e)

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "transcription": transcription,
        "emotion": emotion
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
        req.emotion
    )

    if result["is_complete"]:

        save_entry(
            session_id=req.session_id,
            transcription=req.message,
            emotion=req.emotion,
            summary=result["summary"],
            topics=result["detected_topics"]
        )

    return result


# ---------------------------------------------------------
# ANALYTICS: EMOTION DISTRIBUTION
# ---------------------------------------------------------

@app.get("/analytics/emotions")
def emotion_distribution():

    entries = get_entries()

    emotions = [e[1] for e in entries]

    counts = Counter(emotions)

    return dict(counts)


# ---------------------------------------------------------
# WEEKLY REPORT
# ---------------------------------------------------------

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

    counts = Counter(emotions)

    dominant = None

    if counts:
        dominant = max(counts, key=counts.get)

    return {
        "total_entries": len(rows),
        "emotion_distribution": counts,
        "dominant_emotion": dominant
    }


# ---------------------------------------------------------
# MONTHLY REPORT
# ---------------------------------------------------------

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

    counts = Counter(emotions)

    return {
        "total_entries": len(rows),
        "emotion_distribution": counts
    }


if __name__ == "__main__":
    uvicorn.run("app:app", port=8000, reload=True)