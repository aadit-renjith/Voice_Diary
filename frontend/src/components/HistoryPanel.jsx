import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './HistoryPanel.css';
import { Calendar, MessageSquare, ChevronDown, ChevronUp, Trash2, Clock, FileText, Sparkles, BookOpen } from 'lucide-react';

const API = "http://localhost:8000";

const emojiMap = {
    happy: '🤩', sad: '😢', angry: '😠', fearful: '😰',
    neutral: '😐', surprised: '😲', calm: '😌', disgust: '🤢'
};

const emotionColor = {
    happy: '#f59e0b', sad: '#60a5fa', angry: '#f87171',
    fearful: '#a78bfa', neutral: '#94a3b8', surprised: '#34d399',
    calm: '#67e8f9', disgust: '#fb923c'
};

const HistoryPanel = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [expandedTranscript, setExpandedTranscript] = useState({});
    const [expandedSummary, setExpandedSummary] = useState({});
    const [expandedTopics, setExpandedTopics] = useState({});
    const [expandedChat, setExpandedChat] = useState({});

    useEffect(() => { fetchHistory(); }, []);

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

    const deleteEntry = async (entryId) => {
        if (!window.confirm("Delete this diary entry?")) return;
        try {
            await axios.delete(`${API}/history/${entryId}`);
            setHistory(prev => prev.filter(entry => entry.id !== entryId));
        } catch (err) {
            console.error("Failed to delete entry:", err);
        }
    };

    const toggle = (setter, id) => setter(prev => ({ ...prev, [id]: !prev[id] }));

    const formatDate = (dateStr) => {
        const d = new Date(dateStr);
        return d.toLocaleString(undefined, {
            weekday: 'short', year: 'numeric', month: 'short',
            day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
    };

    if (loading) return (
        <div className="hp-loading">
            <div className="hp-spinner" />
            <span>Loading your diary...</span>
        </div>
    );

    return (
        <div className="hp-page">
            {/* Header */}
            <div className="hp-header">
                <div className="hp-header-left">
                    <div className="hp-header-icon"><BookOpen size={22} /></div>
                    <div>
                        <h2 className="hp-title">My Diary</h2>
                        <p className="hp-subtitle">{history.length} recorded moment{history.length !== 1 ? 's' : ''}</p>
                    </div>
                </div>
                {history.length > 0 && (
                    <button className="hp-clear-btn" onClick={clearHistory}>
                        <Trash2 size={15} /> Clear All
                    </button>
                )}
            </div>

            {history.length === 0 ? (
                <div className="hp-empty">
                    <div className="hp-empty-icon"><Calendar size={52} /></div>
                    <p className="hp-empty-title">Nothing here yet</p>
                    <p className="hp-empty-sub">Start recording to fill your diary with moments.</p>
                </div>
            ) : (
                <div className="hp-list">
                    {history.map((entry) => {
                        const color = emotionColor[entry.emotion] || '#94a3b8';
                        const emoji = emojiMap[entry.emotion] || '💭';
                        const chatData = entry.full_chat ? (() => { try { return JSON.parse(entry.full_chat); } catch { return null; } })() : null;

                        return (
                            <div key={entry.id} className="hp-card" style={{ '--accent': color }}>
                                {/* Card top strip */}
                                <div className="hp-card-strip" style={{ background: color }} />

                                {/* Card header */}
                                <div className="hp-card-head">
                                    <div className="hp-emotion-badge" style={{ background: `${color}22`, border: `1px solid ${color}55` }}>
                                        <span className="hp-emoji">{emoji}</span>
                                        <span className="hp-emotion-label" style={{ color }}>{entry.emotion || 'Unknown'}</span>
                                    </div>
                                    <div className="hp-card-actions">
                                        <div className="hp-timestamp">
                                            <Clock size={12} />
                                            <span>{formatDate(entry.date)}</span>
                                        </div>
                                        <button
                                            className="hp-entry-delete-btn"
                                            onClick={() => deleteEntry(entry.id)}
                                            title="Delete entry"
                                        >
                                            <Trash2 size={13} />
                                        </button>
                                    </div>
                                </div>

                                {/* Expandable: Transcript */}
                                {entry.transcription && (
                                    <div className="hp-section">
                                        <button className="hp-expand-btn" onClick={() => toggle(setExpandedTranscript, entry.id)}>
                                            <FileText size={14} />
                                            <span>Transcript</span>
                                            {expandedTranscript[entry.id] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                        </button>
                                        {expandedTranscript[entry.id] && (
                                            <div className="hp-expand-body">
                                                <p>{entry.transcription}</p>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Expandable: Summary (only if different from transcription) */}
                                {entry.summary && entry.summary !== entry.transcription && (
                                    <div className="hp-section">
                                        <button className="hp-expand-btn" onClick={() => toggle(setExpandedSummary, entry.id)}>
                                            <Sparkles size={14} />
                                            <span>Summary</span>
                                            {expandedSummary[entry.id] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                        </button>
                                        {expandedSummary[entry.id] && (
                                            <div className="hp-expand-body hp-summary-body">
                                                <p>{entry.summary}</p>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Expandable: Topics */}
                                {entry.topics && entry.topics.length > 0 && (
                                    <div className="hp-section">
                                        <button className="hp-expand-btn" onClick={() => toggle(setExpandedTopics, entry.id)}>
                                            <Sparkles size={14} />
                                            <span>Topics</span>
                                            {expandedTopics[entry.id] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                        </button>
                                        {expandedTopics[entry.id] && (
                                            <div className="hp-topics">
                                                {entry.topics.map((t, i) => (
                                                    <span key={i} className="hp-topic-tag" style={{ color, borderColor: `${color}44` }}>{t}</span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Expandable: Chat log */}
                                {chatData && (
                                    <div className="hp-section">
                                        <button className="hp-expand-btn" onClick={() => toggle(setExpandedChat, entry.id)}>
                                            <MessageSquare size={14} />
                                            <span>Conversation</span>
                                            {expandedChat[entry.id] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                        </button>
                                        {expandedChat[entry.id] && (
                                            <div className="hp-chat-log">
                                                {chatData.map((msg, i) => (
                                                    <div key={i} className={`hp-chat-msg hp-msg-${msg.role}`}>
                                                        <span className="hp-msg-role">{msg.role === 'user' ? 'You' : 'AI'}</span>
                                                        <p>{msg.parts?.[0]?.text?.replace(/^\[.*?\]\s*/, '') || ''}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default HistoryPanel;
