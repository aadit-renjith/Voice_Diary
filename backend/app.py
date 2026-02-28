from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import numpy as np
import pickle
import librosa
from data_processor import extract_features

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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
