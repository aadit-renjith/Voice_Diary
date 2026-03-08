import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './HistoryPanel.css';
import { Calendar, MessageSquare, ChevronDown, ChevronUp, Trash2, Clock } from 'lucide-react';

const API = "http://127.0.0.1:8000";

const emojiMap = {
    happy: '🤩',
    sad: '😢',
    angry: '😠',
    fearful: '😰',
    neutral: '😐',
    surprised: '😲',
    calm: '😌',
    disgust: '🤢'
};

const HistoryPanel = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [expandedId, setExpandedId] = useState(null);

    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        setLoading(true);
        try {
            const res = await axios.get(`${API}/history`);
            setHistory(res.data);
        } catch (err) {
            console.error("Failed to fetch history:", err);
        } finally {
            setLoading(false);
        }
    };

    const clearHistory = async () => {
        if (!window.confirm("Are you sure you want to clear all history? This cannot be undone.")) return;
        try {
            await axios.delete(`${API}/history`);
            setHistory([]);
        } catch (err) {
            console.error("Failed to clear history:", err);
        }
    };

    const toggleExpand = (id) => {
        setExpandedId(expandedId === id ? null : id);
    };

    const formatDate = (dateStr) => {
        const d = new Date(dateStr);
        return d.toLocaleString(undefined, {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    if (loading) return <div className="history-status">Loading history...</div>;

    return (
        <div className="history-panel">
            <div className="history-header">
                <h2>Your Progress</h2>
                {history.length > 0 && (
                    <button className="clear-btn" onClick={clearHistory}>
                        <Trash2 size={16} /> Clear All
                    </button>
                )}
            </div>

            {history.length === 0 ? (
                <div className="history-empty">
                    <Calendar size={48} className="empty-icon" />
                    <p>No entries yet. Start recording or chatting to see your history!</p>
                </div>
            ) : (
                <div className="history-list">
                    {history.map((entry) => (
                        <div key={entry.id} className={`history-item ${expandedId === entry.id ? 'expanded' : ''}`}>
                            <div className="history-item-main" onClick={() => toggleExpand(entry.id)}>
                                <div className="entry-icon">
                                    {emojiMap[entry.emotion] || '💭'}
                                </div>
                                <div className="entry-info">
                                    <div className="entry-date">
                                        <Clock size={12} /> {formatDate(entry.date)}
                                    </div>
                                    <div className="entry-emotion">{entry.emotion || 'Unknown'}</div>
                                </div>
                                <div className="entry-toggle">
                                    {expandedId === entry.id ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                                </div>
                            </div>

                            {expandedId === entry.id && (
                                <div className="history-details glass">
                                    {entry.summary && (
                                        <div className="detail-section">
                                            <h4><MessageSquare size={14} /> Summary</h4>
                                            <p>{entry.summary}</p>
                                        </div>
                                    )}

                                    {entry.topics && entry.topics.length > 0 && (
                                        <div className="detail-section">
                                            <h4>Topics</h4>
                                            <div className="topic-tags">
                                                {entry.topics.map((t, idx) => <span key={idx} className="tag">{t}</span>)}
                                            </div>
                                        </div>
                                    )}

                                    {entry.full_chat && (
                                        <div className="detail-section">
                                            <h4>Conversation History</h4>
                                            <div className="chat-log">
                                                {JSON.parse(entry.full_chat).map((msg, idx) => (
                                                    <div key={idx} className={`log-msg ${msg.role}`}>
                                                        <span className="msg-role">{msg.role === 'user' ? 'You' : 'AI'}:</span>
                                                        <p>{msg.parts[0].text.replace(/^\[.*?\]\s*/, '')}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {!entry.summary && !entry.full_chat && (
                                        <div className="detail-section">
                                            <p className="no-detail">Voice analysis entry. Total transcript: "{entry.transcription}"</p>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default HistoryPanel;
