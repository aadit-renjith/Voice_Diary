import React, { useState, useEffect } from 'react';
import AudioRecorder from './components/AudioRecorder';
import EmotionChart from './components/EmotionChart';
import ChatWidget from './components/ChatWidget';
import './App.css';
import { Mic, Activity } from 'lucide-react';

function App() {
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const saved = localStorage.getItem('emotionHistory');
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  const handlePrediction = (emotion) => {
    setCurrentEmotion(emotion);
    const entry = { emotion, timestamp: new Date().toISOString() };
    const updated = [...history, entry];
    setHistory(updated);
    localStorage.setItem('emotionHistory', JSON.stringify(updated));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('emotionHistory');
    setCurrentEmotion(null);
  };

  const getEmoji = (e) => ({ happy: '🤩', sad: '😢', angry: '😠', fearful: '😰', neutral: '😐', surprised: '😲' }[e] || '💭');
  const getLabel = (e) => ({ happy: 'Joyful', sad: 'Melancholy', angry: 'Agitated', fearful: 'Anxious', neutral: 'Calm', surprised: 'Amazed' }[e] || e);
  const getMessage = (e) => ({ happy: 'Keep shining bright!', sad: 'It\'s okay, better days ahead.', angry: 'Take a deep breath.', neutral: 'Staying balanced.', surprised: 'What a moment!', fearful: 'You are safe here.' }[e] || '');

  return (
    <div className="app">
      {/* Nav */}
      <nav className="nav">
        <div className="nav-left">
          <Mic size={20} className="nav-icon" />
          <span className="nav-title">Voice Diary</span>
        </div>
        <button className="nav-btn" onClick={clearHistory}>Reset History</button>
      </nav>

      {/* Body */}
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

          {/* Detected Emotion Card (Left) */}
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

          {/* Small Detected Emotion Card (Right) */}
          <div className="card emotion-card-sm">
            <p className="emotion-label">Detected Emotion</p>
            {currentEmotion ? (
              <>
                <div className="emotion-pill">
                  <span className="emotion-text">{getLabel(currentEmotion)}</span>
                  <span className="emotion-emoji">{getEmoji(currentEmotion)}</span>
                </div>
                <p className="emotion-msg">{getMessage(currentEmotion)} {getEmoji(currentEmotion)}</p>
              </>
            ) : (
              <div className="emotion-pill placeholder">
                <span className="emotion-text">Waiting...</span>
              </div>
            )}
          </div>

          {/* Dark Chart Card */}
          <div className="card chart-card">
            <h2 className="chart-title">Emotional Trends</h2>
            {history.length > 0 ? (
              <EmotionChart data={history} />
            ) : (
              <p className="chart-empty">Record audio to see your mood trend.</p>
            )}
          </div>
        </div>
      </div>

      {/* Chat Widget (fixed position overlay) */}
      <ChatWidget currentEmotion={currentEmotion} />
    </div>
  );
}

export default App;
