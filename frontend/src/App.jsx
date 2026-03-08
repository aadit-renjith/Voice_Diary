import React, { useState, useEffect } from 'react';
import AudioRecorder from './components/AudioRecorder';
import EmotionChart from './components/EmotionChart';
import ChatWidget from './components/ChatWidget';
import ReportsPanel from './components/ReportsPanel';
import HistoryPanel from './components/HistoryPanel';
import './App.css';
import { Mic, Activity, History } from 'lucide-react';
import axios from 'axios';

const API = "http://127.0.0.1:8000";

function App() {
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [history, setHistory] = useState([]);
  const [predictionCount, setPredictionCount] = useState(0);
  const [view, setView] = useState('home'); // 'home' or 'history'

  useEffect(() => {
    fetchInitialHistory();
  }, []);

  const fetchInitialHistory = async () => {
    try {
      const res = await axios.get(`${API}/history`);
      // Only keep the simple date/emotion for the chart
      const chartData = res.data.map(e => ({
        emotion: e.emotion,
        timestamp: e.date
      })).reverse(); // Oldest first for the chart
      setHistory(chartData);
    } catch (err) {
      console.error("Failed to fetch initial history:", err);
    }
  };

  const handlePrediction = (emotion) => {
    setCurrentEmotion(emotion);
    const entry = { emotion, timestamp: new Date().toISOString() };
    setHistory([...history, entry]);
    setPredictionCount(prev => prev + 1);  // trigger reports refresh
  };

  const getEmoji = (e) => ({ happy: '🤩', sad: '😢', angry: '😠', fearful: '😰', neutral: '😐', surprised: '😲', calm: '😌', disgust: '🤢' }[e] || '💭');
  const getLabel = (e) => ({ happy: 'Joyful', sad: 'Melancholy', angry: 'Agitated', fearful: 'Anxious', neutral: 'Calm', surprised: 'Amazed', calm: 'Serene', disgust: 'Averse' }[e] || e);
  const getMessage = (e) => ({ happy: 'Keep shining bright!', sad: 'It\'s okay, better days ahead.', angry: 'Take a deep breath.', neutral: 'Staying balanced.', surprised: 'What a moment!', fearful: 'You are safe here.' }[e] || '');

  return (
    <div className="app">
      {/* Nav */}
      <nav className="nav">
        <div className="nav-left" onClick={() => setView('home')} style={{ cursor: 'pointer' }}>
          <Mic size={20} className="nav-icon" />
          <span className="nav-title">Voice Diary</span>
        </div>
        <div className="nav-center">
          <button className={`nav-link ${view === 'home' ? 'active' : ''}`} onClick={() => setView('home')}>Journal</button>
          <button className={`nav-link ${view === 'history' ? 'active' : ''}`} onClick={() => setView('history')}>History</button>
        </div>
        <div className="nav-right-actions">
          <span className="status-dot"></span>
          <span className="status-text">System Active</span>
        </div>
      </nav>

      {/* Main Content Areas */}
      {view === 'home' ? (
        <div className="body">
          {/* LEFT COLUMN */}
          <div className="col-left">
            <h1 className="hero-title">How are you feeling?</h1>
            <p className="hero-sub">Record your voice to analyze your emotional state.</p>

            {/* Recorder Card */}
            <div className="card recorder-card">
              <h2 className="card-title">Voice Emotion Recorder</h2>
              <AudioRecorder onPrediction={handlePrediction} />
            </div>

            {/* Detected Emotion Card */}
            <div className="card emotion-card">
              <p className="emotion-label">Detected Emotion</p>
              {currentEmotion ? (
                <>
                  <div className="emotion-pill">
                    <span className="emotion-text">{getLabel(currentEmotion)}</span>
                    <span className="emotion-emoji">{getEmoji(currentEmotion)}</span>
                  </div>
                  <p className="emotion-msg">{getMessage(currentEmotion)}</p>
                </>
              ) : (
                <div className="emotion-pill placeholder">
                  <span className="emotion-text">Waiting...</span>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT COLUMN */}
          <div className="col-right">
            <div className="right-header">
              <Activity size={20} className="right-header-icon" />
              <span className="right-header-text">Emotional Trends</span>
            </div>

            {/* Chart Card */}
            <div className="card chart-card">
              <h2 className="chart-title">Mood Trend</h2>
              {history.length > 0 ? (
                <EmotionChart data={history} />
              ) : (
                <p className="chart-empty">Record audio to see your mood trend.</p>
              )}
            </div>

            {/* Reports */}
            <ReportsPanel refreshKey={predictionCount} />
          </div>
        </div>
      ) : (
        <div className="history-view">
          <HistoryPanel />
        </div>
      )}

      {/* Chat Widget (fixed position overlay) */}
      <ChatWidget
        currentEmotion={currentEmotion}
        onChatComplete={() => {
          fetchInitialHistory();
          setPredictionCount(prev => prev + 1);
        }}
      />
    </div>
  );
}

export default App;
