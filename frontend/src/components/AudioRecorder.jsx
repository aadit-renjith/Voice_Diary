import React, { useState, useRef } from 'react';
import { Mic, Square, Loader2 } from 'lucide-react';
import axios from 'axios';
import RecordRTC from 'recordrtc';
import './AudioRecorder.css';

const AudioRecorder = ({ onPrediction }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const recorderRef = useRef(null);

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
            alert("Microphone access denied.");
        }
    };

    const stopRecording = () => {
        if (recorderRef.current && isRecording) {
            recorderRef.current.stopRecording(() => {
                const blob = recorderRef.current.getBlob();
                uploadAudio(blob);
                recorderRef.current.stream.getTracks().forEach(t => t.stop());
            });
            setIsRecording(false);
        }
    };

    const uploadAudio = async (blob) => {
        setIsLoading(true);
        try {
            const fd = new FormData();
            fd.append('file', blob, 'recording.wav');
            const res = await axios.post('http://localhost:8000/predict', fd);
            if (onPrediction) onPrediction(res.data.emotion);
        } catch { alert("Analysis failed."); }
        finally { setIsLoading(false); }
    };

    return (
        <div className="recorder">
            {/* Sound wave bars */}
            <div className="wave-container">
                <div className={`wave-bars left ${isRecording ? 'active' : ''}`}>
                    {[...Array(6)].map((_, i) => <span key={i} className="bar green" style={{ animationDelay: `${i * 0.1}s` }}></span>)}
                </div>

                <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isLoading}
                    className={`mic-btn ${isRecording ? 'recording' : ''}`}
                >
                    {isLoading ? <Loader2 size={28} className="spin" /> :
                        isRecording ? <Square size={24} /> :
                            <Mic size={28} />}
                </button>

                <div className={`wave-bars right ${isRecording ? 'active' : ''}`}>
                    {[...Array(6)].map((_, i) => <span key={i} className="bar pink" style={{ animationDelay: `${i * 0.1}s` }}></span>)}
                </div>
            </div>

            <p className="rec-status">
                {isLoading ? "Analyzing..." : isRecording ? "Listening... tap to stop" : "Tap microphone to start"}
            </p>
        </div>
    );
};

export default AudioRecorder;
