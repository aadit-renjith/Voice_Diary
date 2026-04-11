import React, { useState, useEffect } from 'react';
import AudioRecorder from './components/AudioRecorder';
import EmotionChart from './components/EmotionChart';
import ChatWidget from './components/ChatWidget';
import ReportsPanel from './components/ReportsPanel';
import HistoryPanel from './components/HistoryPanel';
import TranscriptPanel from './components/TranscriptPanel';   // ← toggle: comment this out to hide
import './App.css';
import axios from 'axios';

const API = "http://localhost:8001";

function App() {
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [transcription, setTranscription] = useState("");
  const [history, setHistory] = useState([]);
  const [predictionCount, setPredictionCount] = useState(0);
  const [lastPrediction, setLastPrediction] = useState(null); // full prediction data

  useEffect(() => {
    fetchInitialHistory();
  }, []);

  const fetchInitialHistory = async () => {
    const res = await axios.get(`${API}/history`);
    const chartData = res.data.map(e => ({
      emotion: e.emotion,
      timestamp: e.date
    })).reverse();
    setHistory(chartData);
  };

  const handlePrediction = (data) => {
    const finalEmotion = data?.finalEmotion || null;
    const emotionLabel = finalEmotion?.[0] || null;

    setCurrentEmotion(emotionLabel);
    setTranscription(data?.transcription || "");
    setLastPrediction(data);     // store full data for TranscriptPanel

    if (emotionLabel) {
      setHistory(prev => [...prev, {
        emotion: emotionLabel,
        timestamp: new Date().toISOString()
      }]);
      setPredictionCount(prev => prev + 1);
    }
  };

  const getEmoji = (e) => ({
    happy: '🤩', sad: '😢', angry: '😠', fearful: '😰',
    neutral: '😐', surprised: '😲', calm: '😌'
  }[e] || '💭');

  const getLabel = (e) => ({
    happy: 'Joyful', sad: 'Melancholy', angry: 'Agitated',
    fearful: 'Anxious', neutral: 'Calm', surprised: 'Surprised'
  }[e] || e);

  return (
    <div className="app">
      <div className="body">

        <div className="col-left">
          <h1>How are you feeling?</h1>

          <div className="card">
            <AudioRecorder onPrediction={handlePrediction} />
          </div>

          <div className="card">
            <p>Detected Emotion</p>

            {currentEmotion ? (
              <>
                <div>
                  {getLabel(currentEmotion)} {getEmoji(currentEmotion)}
                </div>
              </>
            ) : (
              <p>Waiting...</p>
            )}
          </div>

          {/* ── Transcript Panel ────────────────────────────────────
              To disable the transcription panel, comment out the
              <TranscriptPanel ... /> block below (or remove the import above).
          */}
          {lastPrediction && (
            <div className="card">
              <TranscriptPanel
                transcription={lastPrediction.transcription}
                audioEmotion={lastPrediction.audioEmotion}
                textEmotion={lastPrediction.textEmotion}
                finalEmotion={lastPrediction.finalEmotion}
              />
            </div>
          )}
          {/* ── End Transcript Panel ─────────────────────────────── */}

        </div>

        <div className="col-right">
          <EmotionChart data={history} />
          <ReportsPanel refreshKey={predictionCount} />
        </div>
      </div>

      <ChatWidget
        currentEmotion={currentEmotion}
        onChatComplete={() => fetchInitialHistory()}
      />
    </div>
  );
}

export default App;