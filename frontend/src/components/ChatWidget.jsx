import React, { useState } from 'react';
import axios from 'axios';

const ChatWidget = ({ currentEmotion }) => {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);

    const sessionId = "session-1";

    const processAudioResponse = async (blob) => {
        setIsProcessing(true);
        setIsLoading(true);

        try {
            const fd = new FormData();
            fd.append('file', blob, 'chat_recording.wav');

            const transcribeRes = await axios.post('http://localhost:8001/transcribe', fd);

            const { transcription, final_emotion } = transcribeRes.data;

            if (!transcription || transcription.trim() === '') {
                setMessages(prev => [...prev, {
                    role: 'system',
                    content: "Couldn't catch that. Try again."
                }]);
                return;
            }

            setMessages(prev => [...prev, { role: 'user', content: transcription }]);

            const chatRes = await axios.post('http://localhost:8001/chat', {
                message: transcription,
                emotion: final_emotion ? final_emotion[0] : currentEmotion || null,
                session_id: sessionId,
            });

            setMessages(prev => [...prev, { role: 'ai', content: chatRes.data.reply }]);

        } catch (err) {
            console.error(err);
        } finally {
            setIsProcessing(false);
            setIsLoading(false);
        }
    };

    return (
        <div style={{ padding: "10px", border: "1px solid #ccc", marginTop: "10px" }}>
            <h3>Chat</h3>

            <div style={{ maxHeight: "200px", overflowY: "auto" }}>
                {messages.map((msg, idx) => (
                    <div key={idx}>
                        <b>{msg.role}:</b> {msg.content}
                    </div>
                ))}
            </div>

            <p style={{ fontSize: "12px", marginTop: "10px" }}>
                (Voice chat active via AudioRecorder)
            </p>
        </div>
    );
};

export default ChatWidget;