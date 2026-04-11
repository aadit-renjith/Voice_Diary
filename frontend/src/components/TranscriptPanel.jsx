/**
 * TranscriptPanel — shows the real-time speech-to-text output and emotion breakdown.
 * Rendered as a floating panel; enable/disable by importing (or not) in App.jsx.
 * 
 * Props:
 *   transcription  – string, the whisper transcription
 *   audioEmotion   – [label, confidence] | null
 *   textEmotion    – [label, confidence] | null
 *   finalEmotion   – [label, confidence] | null
 */
import React, { useState } from 'react';
import './TranscriptPanel.css';

const pct = (score) => score != null ? `${(score * 100).toFixed(1)}%` : '—';

const emojis = {
    happy: '😊', sad: '😢', angry: '😠', fearful: '😰',
    neutral: '😐', surprised: '😲', calm: '😌', disgust: '🤢',
};

const TranscriptPanel = ({ transcription, audioEmotion, textEmotion, finalEmotion }) => {
    const [collapsed, setCollapsed] = useState(false);

    const hasData = transcription || audioEmotion || textEmotion || finalEmotion;

    if (!hasData) return null;

    return (
        <div className={`transcript-panel ${collapsed ? 'collapsed' : ''}`}>
            <div className="transcript-header" onClick={() => setCollapsed(c => !c)}>
                <span className="transcript-title">🔬 Speech Analysis</span>
                <span className="transcript-toggle">{collapsed ? '▲' : '▼'}</span>
            </div>

            {!collapsed && (
                <div className="transcript-body">
                    {/* Transcription */}
                    {transcription && (
                        <div className="transcript-section">
                            <span className="transcript-label">📝 You said</span>
                            <blockquote className="transcript-quote">"{transcription}"</blockquote>
                        </div>
                    )}

                    {/* Emotion breakdown */}
                    <div className="transcript-emotions">
                        <EmotionRow label="🎙️ Voice" emotion={audioEmotion} />
                        <EmotionRow label="💬 Text" emotion={textEmotion} />
                        <EmotionRow label="✨ Final" emotion={finalEmotion} highlight />
                    </div>
                </div>
            )}
        </div>
    );
};

const EmotionRow = ({ label, emotion, highlight }) => {
    if (!emotion) return (
        <div className={`emotion-row ${highlight ? 'highlight' : ''}`}>
            <span className="emotion-row-label">{label}</span>
            <span className="emotion-row-value muted">—</span>
        </div>
    );

    const [emotionLabel, score] = emotion;
    return (
        <div className={`emotion-row ${highlight ? 'highlight' : ''}`}>
            <span className="emotion-row-label">{label}</span>
            <span className="emotion-row-value">
                {emojis[emotionLabel] || '💭'} {emotionLabel}
            </span>
            <span className="emotion-row-conf">{pct(score)}</span>
        </div>
    );
};

export default TranscriptPanel;
