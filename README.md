# 🎤 Voice Diary — Multimodal Emotion-Aware Diary App

A voice-first diary application that detects your emotional state using a **dual-modality pipeline**:
- **Voice analysis** → CNN-Bidirectional LSTM with Attention (trained on RAVDESS)
- **Speech-to-text** → OpenAI Whisper → **Text emotion analysis** → DistilRoBERTa
- **Late fusion** of both signals for a final emotion label

Both a standalone recording mode and an interactive chatbot mode (powered by Google Gemini) are supported, and all sessions are saved to a local SQLite database.

---

## 📁 Project Structure

```
S8_HONOURS_MINIPROJECT/
├── backend/                # FastAPI backend
│   ├── app.py              # Main API server
│   ├── train_model.py      # CNN-BiLSTM model training
│   ├── data_processor.py   # Audio feature extraction
│   ├── chat_engine.py      # Gemini-powered diary chatbot
│   ├── text_emotion.py     # DistilRoBERTa text emotion classifier
│   ├── fusion.py           # Late fusion of audio + text emotions
│   ├── database.py         # SQLite helpers
│   ├── evaluate_model.py   # Standalone model evaluator
│   ├── .env.example        # Copy to apikey.env and fill in your key
│   └── apikey.env          # ← NOT committed (in .gitignore)
├── frontend/               # React + Vite frontend
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── AudioRecorder.jsx     # Standalone voice → emotion
│           ├── ChatWidget.jsx        # Interactive chatbot with voice
│           ├── TranscriptPanel.jsx   # Speech analysis debug panel
│           ├── EmotionChart.jsx
│           ├── HistoryPanel.jsx
│           └── ReportsPanel.jsx
├── models/                 # Saved model files (not committed)
├── dataset/                # RAVDESS dataset (not committed)
└── requirements.txt
```

---

## 🚀 Setup

### 1. Prerequisites

- Python 3.10 or 3.11
- Node.js 18+
- [FFmpeg](https://ffmpeg.org/) installed and on your system PATH (required by Whisper)

### 2. Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Copy the env template and add your Gemini API key
copy backend\.env.example backend\apikey.env
# Edit apikey.env and set:  GEMINI_API_KEY=your_key_here

# Start the backend (from project root)
cd backend
python app.py
# Server runs on http://localhost:8001
```

### 3. Frontend

```bash
# Install Node dependencies
cd frontend
npm install

# Start the dev server
npm run dev
# App runs on http://localhost:5173
```

### 4. Train the model (optional — pre-trained model required)

Download the [RAVDESS dataset](https://zenodo.org/record/1188976) into the `dataset/` folder, then:

```bash
cd backend
python train_model.py
```

This produces `models/ser_cnn_lstm.keras`, `models/label_encoder.pkl`, and `models/scaler.pkl`.

---

## 🎯 How It Works

### Emotion Pipeline (both modes)
```
Microphone audio
    ↓
┌─────────────────────────────────────────┐
│  1. CNN-BiLSTM + Attention              │ → audio_emotion (label, confidence)
│     Mel + MFCC+Δ+ΔΔ + Chroma + Contrast│
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. Whisper (speech → text)             │ → transcription
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. DistilRoBERTa (text → emotion)      │ → text_emotion (label, confidence)
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. Late Fusion (40% audio + 60% text)  │ → final_emotion
└─────────────────────────────────────────┘
    ↓
   Saved to SQLite DB
```

### Mode 1 — Standalone Recording
Click the microphone on the home page. Your emotion is detected and saved immediately. The **Speech Analysis** panel below shows the transcription and emotion breakdown.

### Mode 2 — Voice Chatbot
Click "Talk to your Diary" (bottom-right). The AI asks follow-up questions about your day. Each full conversation is saved with an AI-generated summary once the session is complete.

---

## 🔧 Configuration

| File | Purpose |
|---|---|
| `backend/apikey.env` | Set `GEMINI_API_KEY` |
| `frontend/src/App.jsx` | Comment out `<TranscriptPanel .../>` and its import to hide the debug panel |
| `backend/fusion.py` | Adjust `AUDIO_WEIGHT` / `TEXT_WEIGHT` to tune fusion |

---

## 📊 Emotions Recognised

`neutral` · `happy` · `sad` · `angry` · `fearful` · `surprised`

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Audio model | TensorFlow / Keras (CNN-BiLSTM + Attention) |
| Speech-to-text | OpenAI Whisper (`base` model) |
| Text emotion | HuggingFace Transformers — DistilRoBERTa |
| Backend API | FastAPI + Uvicorn |
| Chatbot AI | Google Gemini 2.0 Flash |
| Frontend | React + Vite |
| Database | SQLite |

---

## 📝 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/transcribe` | Upload WAV → returns transcription + emotions, saves to DB |
| `POST` | `/chat/start` | Start a new diary chat session |
| `POST` | `/chat` | Send a message + emotion context, get AI reply |
| `GET` | `/history` | All diary entries |
| `DELETE` | `/history` | Clear all entries |
| `GET` | `/analytics/emotions` | Emotion distribution counts |
| `GET` | `/analytics/weekly` | Last 7 days emotion summary |
| `GET` | `/analytics/monthly` | Last 30 days emotion summary |
