<<<<<<< HEAD
from fastapi import FastAPI, File, UploadFile, HTTPException
=======
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import os
import speech_recognition as sr
import numpy as np
import pickle
<<<<<<< HEAD
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
=======
import librosa
from data_processor import extract_mel_spectrogram
from chat_engine import ChatEngine

# Configuration
MODEL_PATH = "../models/ser_cnn_lstm.keras"
LABEL_ENCODER_PATH = "../models/label_encoder.pkl"
SCALER_PATH = "../models/scaler.pkl"
TEMP_DIR = "temp_uploads"

# Initialize App
app = FastAPI(title="SER Voice Diary API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD

model = None
=======
# Load Model, Scaler & Label Encoder
model = None
scaler = None
label_encoder = None
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
chat_engine = ChatEngine()


class ChatRequest(BaseModel):
    message: str
    emotion: Optional[str] = None
    session_id: str


class StartChatRequest(BaseModel):
    session_id: str


@app.on_event("startup")
<<<<<<< HEAD
def startup():

    global model

    init_db()

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    else:
        print("Model file not found. Ensure training is complete.")

=======
def load_model():
    global model, scaler, label_encoder

    # Load Keras model
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
            print("CNN-LSTM model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file not found. Ensure training is complete.")

    # Load scaler
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    else:
        print("Scaler file not found. Features won't be scaled.")

    # Load label encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        try:
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                label_encoder = pickle.load(f)
            print(f"Label encoder loaded. Classes: {list(label_encoder.classes_)}")
        except Exception as e:
            print(f"Error loading label encoder: {e}")
    else:
        print("Label encoder not found.")


def predict_from_file(file_path):
    """Extract features from audio file and predict emotion using CNN-LSTM."""
    features = extract_mel_spectrogram(file_path)

    if features is None:
        return None, None

    # Scale features (reshape → scale → reshape back)
    n_time, n_freq = features.shape
    if scaler is not None:
        features = scaler.transform(features.reshape(-1, n_freq)).reshape(n_time, n_freq)

    # Add batch dimension: (1, time, freq)
    features = np.expand_dims(features, axis=0)

    # Predict
    probs = model.predict(features, verbose=0)
    pred_idx = np.argmax(probs, axis=1)[0]

    if label_encoder is not None:
        emotion = label_encoder.inverse_transform([pred_idx])[0]
    else:
        emotion = str(pred_idx)

    return emotion, probs[0]

>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440

@app.get("/")
def read_root():
    return {"message": "Welcome to SER Voice Diary API"}


<<<<<<< HEAD
# ---------------------------------------------------------
# EMOTION PREDICTION (VOICE RECORDER)
# ---------------------------------------------------------

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):

=======
@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    print(f"Received file: {file.filename}")

<<<<<<< HEAD
=======
    # Save uploaded file temporarily
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    temp_path = os.path.join(TEMP_DIR, file.filename)
<<<<<<< HEAD

=======
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
<<<<<<< HEAD

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

=======
        print(f"Processing file at: {temp_path}")

        emotion, probs = predict_from_file(temp_path)

        if emotion is None:
            print("Feature extraction failed (returned None)")
            raise HTTPException(status_code=400, detail="Could not process audio file")

        print(f"Predicted emotion: {emotion}")
        print(f"Probabilities: {probs}")

        # Cleanup
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
        os.remove(temp_path)

        return {"emotion": emotion}

    except Exception as e:
<<<<<<< HEAD

        print("Prediction error:", e)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# TRANSCRIBE AUDIO
# ---------------------------------------------------------

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

=======
        print(f"Error during prediction: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text and also predict emotion."""
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    temp_path = os.path.join(TEMP_DIR, f"chat_{file.filename}")
<<<<<<< HEAD

=======
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcription = ""
    emotion = None

    try:
<<<<<<< HEAD

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
=======
        # 1) Speech-to-text using Google Speech Recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
        print(f"Transcription: {transcription}")

        # 2) Emotion prediction (if model is loaded)
        if model:
            try:
                pred_emotion, _ = predict_from_file(temp_path)
                if pred_emotion is not None:
                    emotion = pred_emotion
                    print(f"Chat emotion: {emotion}")
            except Exception as e:
                print(f"Emotion prediction in chat failed (non-critical): {e}")

    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        transcription = ""
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        transcription = ""
    except Exception as e:
        print(f"Transcription error: {e}")
        transcription = ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"transcription": transcription, "emotion": emotion}


@app.post("/chat/start")
async def start_chat(req: StartChatRequest):
    """Start a new chat session and get the AI's opening message."""
    chat_engine.reset_session(req.session_id)
    result = chat_engine.get_opening_message(req.session_id)
    return result


@app.post("/chat")
async def chat(req: ChatRequest):
    """Send a message in an ongoing chat session."""
    result = chat_engine.chat(req.session_id, req.message, req.emotion)
    return result


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
