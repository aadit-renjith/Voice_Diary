import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Loader2, RotateCcw, X, Sparkles, Keyboard } from 'lucide-react';
import axios from 'axios';
import RecordRTC from 'recordrtc';
import './ChatWidget.css';

const API = 'http://localhost:8000';

// Generate a random short session ID
const newSessionId = () => `chat-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

// Typing indicator component
const TypingIndicator = () => (
    <div className="chat-bubble ai-bubble">
        <div className="bubble-avatar">🤖</div>
        <div className="bubble-content typing-indicator">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
        </div>
    </div>
);

const ChatWidget = ({ currentEmotion, onChatComplete }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [sessionId, setSessionId] = useState(newSessionId());
    const [sessionStarted, setSessionStarted] = useState(false);
    const [isComplete, setIsComplete] = useState(false);
    const [summary, setSummary] = useState(null);

    // Voice recording state
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const recorderRef = useRef(null);

    // Text fallback state
    const [showTextInput, setShowTextInput] = useState(false);
    const [textInput, setTextInput] = useState('');

    // Tracked emotion from latest voice (label only)
    const [latestEmotion, setLatestEmotion] = useState(currentEmotion || null);

    const messagesEndRef = useRef(null);

    // Auto-scroll to bottom whenever messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isProcessing]);

    // ── Start a new diary session ─────────────────────────────────────
    const startSession = async () => {
        const sid = newSessionId();
        setSessionId(sid);
        setMessages([]);
        setIsComplete(false);
        setSummary(null);
        setSessionStarted(true);

        try {
            const res = await axios.post(`${API}/chat/start`, { session_id: sid });
            setMessages([{ role: 'ai', content: res.data.reply }]);
        } catch (err) {
            console.error('Failed to start session:', err);
            setMessages([{ role: 'ai', content: "Hey there! 👋 How's your day going?" }]);
        }
    };

    // ── Reset to welcome screen ───────────────────────────────────────
    const resetSession = () => {
        setSessionStarted(false);
        setMessages([]);
        setIsComplete(false);
        setSummary(null);
        setLatestEmotion(currentEmotion || null);
        if (onChatComplete) onChatComplete();
    };

    // ── Send a transcription + emotion to /chat ───────────────────────
    const sendToChat = async (transcription, emotionLabel) => {
        setMessages(prev => [...prev, { role: 'user', content: transcription }]);
        setIsProcessing(true);

        try {
            const chatRes = await axios.post(`${API}/chat`, {
                message: transcription,
                emotion: emotionLabel || latestEmotion || null,
                session_id: sessionId,
            });

            const { reply, is_complete, summary: chatSummary } = chatRes.data;
            setMessages(prev => [...prev, { role: 'ai', content: reply }]);

            if (is_complete) {
                setIsComplete(true);
                setSummary(chatSummary);
                if (onChatComplete) onChatComplete();
            }
        } catch (err) {
            console.error('Chat error:', err);
            setMessages(prev => [...prev, { role: 'system', content: 'Something went wrong. Try again.' }]);
        } finally {
            setIsProcessing(false);
        }
    };

    // ── Voice recording: start ────────────────────────────────────────
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recorderRef.current = new RecordRTC(stream, {
                type: 'audio',
                mimeType: 'audio/wav',
                recorderType: RecordRTC.StereoAudioRecorder,
                numberOfAudioChannels: 1,
                desiredSampRate: 16000,
            });
            recorderRef.current.startRecording();
            setIsRecording(true);
        } catch {
            alert('Microphone access denied.');
        }
    };

    // ── Voice recording: stop, transcribe, send ───────────────────────
    const stopRecording = () => {
        if (!recorderRef.current || !isRecording) return;

        // Flip UI immediately so the stop button disappears right away
        setIsRecording(false);
        setIsProcessing(true);

        recorderRef.current.stopRecording(async () => {
            const blob = recorderRef.current.getBlob();
            recorderRef.current.stream.getTracks().forEach(t => t.stop());

            try {
                const fd = new FormData();
                fd.append('file', blob, 'chat_recording.wav');
                const transcribeRes = await axios.post(`${API}/transcribe`, fd);
                const { transcription, final_emotion } = transcribeRes.data;

                if (!transcription || transcription.trim() === '') {
                    setMessages(prev => [...prev, { role: 'system', content: "Couldn't catch that. Try again." }]);
                    setIsProcessing(false);
                    return;
                }

                const emotionLabel = final_emotion ? final_emotion[0] : null;
                if (emotionLabel) setLatestEmotion(emotionLabel);

                await sendToChat(transcription, emotionLabel);
            } catch (err) {
                console.error('Transcription error:', err);
                setMessages(prev => [...prev, { role: 'system', content: 'Audio processing failed.' }]);
                setIsProcessing(false);
            }
        });
    };

    // ── Text fallback: send ───────────────────────────────────────────
    const sendText = async () => {
        const trimmed = textInput.trim();
        if (!trimmed || isProcessing) return;
        setTextInput('');
        await sendToChat(trimmed, latestEmotion);
    };

    // ── Render: FAB when closed ───────────────────────────────────────
    if (!isOpen) {
        return (
            <button className="chat-fab" onClick={() => setIsOpen(true)}>
                <Sparkles size={18} className="chat-header-icon" />
                <span className="fab-label">Talk to your Diary</span>
            </button>
        );
    }

    // ── Render: full widget ───────────────────────────────────────────
    return (
        <div className="chat-widget">

            {/* Header */}
            <div className="chat-header">
                <div className="chat-header-left">
                    <Sparkles size={18} className="chat-header-icon" />
                    <span className="chat-header-title">Voice Diary Chat</span>
                    {latestEmotion && (
                        <span style={{ fontSize: '0.75rem', opacity: 0.8, marginLeft: 4 }}>
                            · {latestEmotion}
                        </span>
                    )}
                </div>
                <div className="chat-header-right">
                    {sessionStarted && (
                        <button className="chat-header-btn" onClick={resetSession} title="New session">
                            <RotateCcw size={15} />
                        </button>
                    )}
                    <button className="chat-header-btn" onClick={() => setIsOpen(false)} title="Close">
                        <X size={15} />
                    </button>
                </div>
            </div>

            {/* Body */}
            <div className="chat-body">
                {!sessionStarted ? (
                    /* Welcome screen */
                    <div className="chat-welcome">
                        <div className="welcome-emoji">📖</div>
                        <h2 className="welcome-title">Your Voice Diary</h2>
                        <p className="welcome-desc">
                            Speak your heart out. I'll listen, ask questions, and help you reflect on your day.
                        </p>
                        <button className="welcome-start-btn" onClick={startSession}>
                            Start Today's Entry ✨
                        </button>
                    </div>
                ) : (
                    /* Chat messages */
                    <div className="chat-messages">
                        {messages.map((msg, idx) => {
                            if (msg.role === 'system') {
                                return (
                                    <div key={idx} className="chat-bubble system-bubble">
                                        <div className="bubble-content">
                                            <p>{msg.content}</p>
                                        </div>
                                    </div>
                                );
                            }
                            const isAI = msg.role === 'ai';
                            return (
                                <div key={idx} className={`chat-bubble ${isAI ? 'ai-bubble' : 'user-bubble'}`}>
                                    <div className={`bubble-avatar ${isAI ? '' : 'user-avatar'}`}>
                                        {isAI ? '🤖' : '🧑'}
                                    </div>
                                    <div className="bubble-content">
                                        <p>{msg.content}</p>
                                    </div>
                                </div>
                            );
                        })}

                        {isProcessing && <TypingIndicator />}

                        {/* Session complete summary card */}
                        {isComplete && summary && (
                            <div className="chat-summary">
                                <div className="summary-header">
                                    <Sparkles size={14} /> Day captured 🎉
                                </div>
                                <p className="summary-text">{summary}</p>
                                <button className="welcome-start-btn" onClick={resetSession}>
                                    Start New Entry
                                </button>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input area — only visible during an active session that isn't complete */}
            {sessionStarted && !isComplete && (
                <div className="chat-input-area">
                    {isRecording ? (
                        <div className="recording-controls">
                            <div className="recording-indicator">
                                <span className="rec-dot" />
                                <span className="rec-label">Recording...</span>
                            </div>
                            <button className="stop-rec-btn" onClick={stopRecording}>
                                <Square size={16} />
                            </button>
                        </div>
                    ) : isProcessing ? (
                        <div className="processing-controls">
                            <Loader2 size={18} className="spin" />
                            <span className="processing-label">Analysing...</span>
                        </div>
                    ) : showTextInput ? (
                        <div className="text-input-row">
                            <input
                                className="chat-input"
                                placeholder="Type your message..."
                                value={textInput}
                                onChange={e => setTextInput(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && sendText()}
                                disabled={isProcessing}
                            />
                            <button className="mic-input-btn" onClick={() => setShowTextInput(false)} title="Switch to voice">
                                <Mic size={16} />
                            </button>
                            <button className="send-btn" onClick={sendText} disabled={!textInput.trim()}>
                                ➤
                            </button>
                        </div>
                    ) : (
                        <div className="voice-input-row">
                            <button className="record-btn" onClick={startRecording}>
                                <Mic size={18} />
                                Hold to speak
                            </button>
                            <button className="keyboard-toggle-btn" onClick={() => setShowTextInput(true)} title="Type instead">
                                <Keyboard size={16} />
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ChatWidget;