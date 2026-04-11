import React, { useState, useEffect } from 'react';
import AudioRecorder from './components/AudioRecorder';
import EmotionChart from './components/EmotionChart';
import ChatWidget from './components/ChatWidget';
import ReportsPanel from './components/ReportsPanel';
import HistoryPanel from './components/HistoryPanel';
import TranscriptPanel from './components/TranscriptPanel';
import './App.css';
import axios from 'axios';
import { BookOpen, Mic } from 'lucide-react';

const API = "http://localhost:8000";

function App() {
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [transcription, setTranscription] = useState("");
  const [history, setHistory] = useState([]);
  const [predictionCount, setPredictionCount] = useState(0);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [page, setPage] = useState('home'); // 'home' | 'history'

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
    setLastPrediction(data);

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
      {/* ── Page Nav ─────────────────────────────────── */}
      <div className="page-nav">
        <button
          id="nav-home-btn"
          className={`page-nav-btn ${page === 'home' ? 'pnb-active' : ''}`}
          onClick={() => setPage('home')}
        >
          <Mic size={16} />
          <span>Record</span>
        </button>
        <button
          id="nav-history-btn"
          className={`page-nav-btn ${page === 'history' ? 'pnb-active' : ''}`}
          onClick={() => setPage('history')}
        >
          <BookOpen size={16} />
          <span>History</span>
        </button>
      </div>

      {/* ── Home Page ─────────────────────────────────── */}
      {page === 'home' && (
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
          </div>

          <div className="col-right">
            <EmotionChart data={history} />
            <ReportsPanel refreshKey={predictionCount} />
          </div>
        </div>
      )}

      {/* ── History Page ──────────────────────────────── */}
      {page === 'history' && (
        <div className="history-view">
          <HistoryPanel />
        </div>
      )}

      <ChatWidget
        currentEmotion={currentEmotion}
        onChatComplete={() => fetchInitialHistory()}
      />
    </div>
  );
}

export default App;