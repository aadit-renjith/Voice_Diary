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
        } catch {
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

            const res = await axios.post('http://localhost:8000/transcribe', fd);

            const { transcription, final_emotion, audio_emotion, text_emotion } = res.data;

            if (onPrediction) {
                onPrediction({
                    transcription,
                    finalEmotion: final_emotion,
                    audioEmotion: audio_emotion,
                    textEmotion: text_emotion
                });
            }
        } catch (err) {
            console.error("Upload/analysis error:", err);
            const detail = err?.response?.data?.detail || err?.message || "Unknown error";
            alert(`Analysis failed: ${detail}`);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="recorder">
            <button onClick={isRecording ? stopRecording : startRecording} disabled={isLoading}>
                {isLoading ? <Loader2 className="spin" /> : isRecording ? <Square /> : <Mic />}
            </button>
        </div>
    );
};

export default AudioRecorder;