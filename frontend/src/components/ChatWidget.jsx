import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MessageCircle, Mic, Square, Send, X, RotateCcw, Sparkles, Loader2, Keyboard } from 'lucide-react';
import axios from 'axios';
import RecordRTC from 'recordrtc';
import './ChatWidget.css';

const ChatWidget = ({ currentEmotion, onChatComplete }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [hasStarted, setHasStarted] = useState(false);
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isComplete, setIsComplete] = useState(false);
    const [summary, setSummary] = useState(null);
    const [mode, setMode] = useState('guided'); // 'guided' or 'free'
    const [showTextInput, setShowTextInput] = useState(false);
    const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

    const messagesEndRef = useRef(null);
    const recorderRef = useRef(null);
    const inputRef = useRef(null);

    // Auto-scroll to bottom
    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages, scrollToBottom]);

    // Start voice recording
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
        } catch (err) {
            console.error('Mic access denied:', err);
            alert('Microphone access is required to respond via voice.');
        }
    };

    // Stop recording and process the audio
    const stopRecording = () => {
        if (recorderRef.current && isRecording) {
            setIsRecording(false);
            recorderRef.current.stopRecording(async () => {
                const blob = recorderRef.current.getBlob();
                recorderRef.current.stream.getTracks().forEach(t => t.stop());
                await processAudioResponse(blob);
            });
        }
    };

    // Send audio to backend for transcription + emotion, then send to chat
    const processAudioResponse = async (blob) => {
        setIsProcessing(true);
        setIsLoading(true);

        try {
            // Step 1: Transcribe audio + get emotion
            const fd = new FormData();
            fd.append('file', blob, 'chat_recording.wav');
            const transcribeRes = await axios.post('http://localhost:8000/transcribe', fd);

            const { transcription, emotion } = transcribeRes.data;

            if (!transcription || transcription.trim() === '') {
                setMessages(prev => [...prev, {
                    role: 'system',
                    content: "🎤 Couldn't catch that. Please try speaking again, a bit louder or closer to the mic."
                }]);
                setIsProcessing(false);
                setIsLoading(false);
                return;
            }

            // Step 2: Show user message in chat
            const userMsg = { role: 'user', content: transcription };
            setMessages(prev => [...prev, userMsg]);

            // Step 3: Send to chat engine with detected emotion
            const chatRes = await axios.post('http://localhost:8000/chat', {
                message: transcription,
                emotion: emotion || currentEmotion || null,
                session_id: sessionId,
            });

            const aiMsg = { role: 'ai', content: chatRes.data.reply };
            setMessages(prev => [...prev, aiMsg]);

            if (chatRes.data.is_complete) {
                setIsComplete(true);
                setSummary(chatRes.data.summary);
                if (onChatComplete) onChatComplete();
            }
        } catch (err) {
            console.error('Voice processing error:', err);
            setMessages(prev => [...prev, {
                role: 'ai',
                content: "I had trouble processing your recording. Could you try again?"
            }]);
        } finally {
            setIsProcessing(false);
            setIsLoading(false);
        }
    };

    const startChat = async () => {
        setHasStarted(true);
        setIsLoading(true);
        setMessages([]);
        setIsComplete(false);
        setSummary(null);

        try {
            const res = await axios.post('http://localhost:8000/chat/start', {
                session_id: sessionId,
            });

            setMessages([{ role: 'ai', content: res.data.reply }]);
        } catch (err) {
            console.error('Failed to start chat:', err);
            setMessages([{
                role: 'ai',
                content: "Hey there! 👋 How's your day going so far?"
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    // Text fallback send
    const sendTextMessage = async () => {
        const text = inputText.trim();
        if (!text || isLoading) return;

        const userMsg = { role: 'user', content: text };
        setMessages(prev => [...prev, userMsg]);
        setInputText('');
        setIsLoading(true);

        try {
            const res = await axios.post('http://localhost:8000/chat', {
                message: text,
                emotion: currentEmotion || null,
                session_id: sessionId,
            });

            const aiMsg = { role: 'ai', content: res.data.reply };
            setMessages(prev => [...prev, aiMsg]);

            if (res.data.is_complete) {
                setIsComplete(true);
                setSummary(res.data.summary);
                if (onChatComplete) onChatComplete();
            }
        } catch (err) {
            console.error('Chat error:', err);
            setMessages(prev => [...prev, {
                role: 'ai',
                content: "Hmm, I couldn't process that. Could you try again?"
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextMessage();
        }
    };

    const resetChat = () => {
        setMessages([]);
        setHasStarted(false);
        setIsComplete(false);
        setSummary(null);
        setInputText('');
        setShowTextInput(false);
    };

    // Floating action button when closed
    if (!isOpen) {
        return (
            <button
                className="chat-fab"
                onClick={() => setIsOpen(true)}
                id="chat-fab-button"
                aria-label="Open chat"
            >
                <MessageCircle size={24} />
                <span className="fab-label">Talk about your day</span>
            </button>
        );
    }

    return (
        <div className="chat-widget" id="chat-widget">
            {/* Header */}
            <div className="chat-header">
                <div className="chat-header-left">
                    <Sparkles size={18} className="chat-header-icon" />
                    <span className="chat-header-title">Diary Chat</span>
                </div>
                <div className="chat-header-right">
                    {hasStarted && (
                        <button
                            className="chat-header-btn"
                            onClick={resetChat}
                            title="New conversation"
                        >
                            <RotateCcw size={16} />
                        </button>
                    )}
                    <button
                        className="chat-header-btn"
                        onClick={() => setIsOpen(false)}
                        title="Close"
                    >
                        <X size={16} />
                    </button>
                </div>
            </div>

            {/* Mode toggle */}
            {hasStarted && !isComplete && (
                <div className="mode-toggle">
                    <button
                        className={`mode-btn ${mode === 'guided' ? 'active' : ''}`}
                        onClick={() => setMode('guided')}
                    >
                        Guided Q&A
                    </button>
                    <button
                        className={`mode-btn ${mode === 'free' ? 'active' : ''}`}
                        onClick={() => setMode('free')}
                    >
                        Free Talk
                    </button>
                </div>
            )}

            {/* Chat body */}
            <div className="chat-body">
                {!hasStarted ? (
                    // Welcome screen
                    <div className="chat-welcome">
                        <div className="welcome-emoji">🎙️</div>
                        <h3 className="welcome-title">Talk about your day</h3>
                        <p className="welcome-desc">
                            I'll ask you a few questions and you can respond by recording your voice.
                            Let's understand how you're feeling today.
                        </p>
                        <button className="welcome-start-btn" onClick={startChat}>
                            Let's Chat
                        </button>
                    </div>
                ) : (
                    <>
                        {/* Messages */}
                        <div className="chat-messages">
                            {messages.map((msg, idx) => (
                                <div
                                    key={idx}
                                    className={`chat-bubble ${msg.role === 'ai' ? 'ai-bubble' : msg.role === 'system' ? 'system-bubble' : 'user-bubble'}`}
                                >
                                    {msg.role === 'ai' && (
                                        <div className="bubble-avatar">🤖</div>
                                    )}
                                    <div className="bubble-content">
                                        <p>{msg.content}</p>
                                    </div>
                                    {msg.role === 'user' && (
                                        <div className="bubble-avatar user-avatar">🗣️</div>
                                    )}
                                </div>
                            ))}

                            {(isLoading || isProcessing) && (
                                <div className="chat-bubble ai-bubble">
                                    <div className="bubble-avatar">🤖</div>
                                    <div className="bubble-content typing-indicator">
                                        <span className="dot"></span>
                                        <span className="dot"></span>
                                        <span className="dot"></span>
                                    </div>
                                </div>
                            )}

                            <div ref={messagesEndRef} />
                        </div>

                        {/* Summary card */}
                        {isComplete && summary && (
                            <div className="chat-summary">
                                <div className="summary-header">
                                    <Sparkles size={16} />
                                    <span>Today's Summary</span>
                                </div>
                                <p className="summary-text">{summary}</p>
                                <button className="welcome-start-btn" onClick={resetChat}>
                                    Start New Conversation
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>

            {/* Voice-first input area */}
            {hasStarted && !isComplete && (
                <div className="chat-input-area">
                    {isRecording ? (
                        // Recording state — big stop button
                        <div className="recording-controls">
                            <div className="recording-indicator">
                                <span className="rec-dot"></span>
                                <span className="rec-label">Recording... tap to stop</span>
                            </div>
                            <button
                                className="stop-rec-btn"
                                onClick={stopRecording}
                                title="Stop recording"
                            >
                                <Square size={20} />
                            </button>
                        </div>
                    ) : isProcessing ? (
                        // Processing state
                        <div className="processing-controls">
                            <Loader2 size={20} className="spin" />
                            <span className="processing-label">Processing your voice...</span>
                        </div>
                    ) : showTextInput ? (
                        // Text input fallback
                        <div className="text-input-row">
                            <button
                                className="mic-input-btn"
                                onClick={() => setShowTextInput(false)}
                                title="Switch to voice"
                            >
                                <Mic size={18} />
                            </button>
                            <input
                                ref={inputRef}
                                type="text"
                                className="chat-input"
                                placeholder={
                                    mode === 'free'
                                        ? "Say whatever's on your mind..."
                                        : "Type your response..."
                                }
                                value={inputText}
                                onChange={(e) => setInputText(e.target.value)}
                                onKeyDown={handleKeyDown}
                                disabled={isLoading}
                                autoFocus
                            />
                            <button
                                className="send-btn"
                                onClick={sendTextMessage}
                                disabled={!inputText.trim() || isLoading}
                                title="Send message"
                            >
                                <Send size={18} />
                            </button>
                        </div>
                    ) : (
                        // Default: Voice record button
                        <div className="voice-input-row">
                            <button
                                className="record-btn"
                                onClick={startRecording}
                                disabled={isLoading}
                                title="Hold to record your response"
                            >
                                <Mic size={22} />
                                <span>Tap to respond</span>
                            </button>
                            <button
                                className="keyboard-toggle-btn"
                                onClick={() => setShowTextInput(true)}
                                title="Type instead"
                            >
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
