from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import os
import speech_recognition as sr
import numpy as np
import pickle
import librosa
from data_processor import extract_features
from chat_engine import ChatEngine

# Configuration
MODEL_PATH = "../models/pro_ser_model.pkl"
TEMP_DIR = "temp_uploads"

# Initialize App
app = FastAPI(title="SER Voice Diary API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = None
chat_engine = ChatEngine()


class ChatRequest(BaseModel):
    message: str
    emotion: Optional[str] = None
    session_id: str


class StartChatRequest(BaseModel):
    session_id: str


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file not found. Ensure training is complete.")


@app.get("/")
def read_root():
    return {"message": "Welcome to SER Voice Diary API"}


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    print(f"Received file: {file.filename}")

    # Save uploaded file temporarily
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    temp_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(f"Processing file at: {temp_path}")
        # Extract features
        features = extract_features(temp_path)

        if features is None:
            print("Feature extraction failed (returned None)")
            raise HTTPException(status_code=400, detail="Could not process audio file")

        print(f"Features extracted. Shape: {features.shape}")

        # Reshape for prediction (1, n_features)
        features = features.reshape(1, -1)

        # Predict
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)

        emotion = prediction[0]
        print(f"Predicted emotion: {emotion}")
        print(f"Probabilities: {probabilities}")

        # Cleanup
        os.remove(temp_path)

        return {"emotion": emotion}

    except Exception as e:
        print(f"Error during prediction: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text and also predict emotion."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    temp_path = os.path.join(TEMP_DIR, f"chat_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcription = ""
    emotion = None

    try:
        # 1) Speech-to-text using Google Speech Recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
        print(f"Transcription: {transcription}")

        # 2) Emotion prediction (if model is loaded)
        if model:
            try:
                features = extract_features(temp_path)
                if features is not None:
                    features = features.reshape(1, -1)
                    prediction = model.predict(features)
                    emotion = prediction[0]
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
