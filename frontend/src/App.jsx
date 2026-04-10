import React, { useState, useEffect } from 'react';
import AudioRecorder from './components/AudioRecorder';
import EmotionChart from './components/EmotionChart';
import ChatWidget from './components/ChatWidget';
import ReportsPanel from './components/ReportsPanel';
import HistoryPanel from './components/HistoryPanel';
import './App.css';
import { Mic, Activity } from 'lucide-react';
import axios from 'axios';

const API = "http://localhost:8001";

function App() {
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [transcription, setTranscription] = useState("");
  const [history, setHistory] = useState([]);
  const [predictionCount, setPredictionCount] = useState(0);
  const [view, setView] = useState('home');

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

  // 🔥 FIXED
  const handlePrediction = (data) => {
    const finalEmotion = data?.finalEmotion?.[0] || null;

    setCurrentEmotion(finalEmotion);
    setTranscription(data?.transcription || "");

    if (finalEmotion) {
      setHistory(prev => [...prev, {
        emotion: finalEmotion,
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
    fearful: 'Anxious', neutral: 'Calm'
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

                
                {transcription && (
                  <div style={{ marginTop: 10 }}>
                    <p>You said:</p>
                    <p>"{transcription}"</p>
                  </div>
                )}
              </>
            ) : (
              <p>Waiting...</p>
            )}
          </div>
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