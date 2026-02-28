import librosa
import numpy as np
import os
import soundfile as sf

def extract_features(file_path):
    """
    Extracts MFCC, Chroma, and Mel Spectrogram features from an audio file.
    Args:
        file_path (str): Path to the audio file.
    Returns:
        np.array: Concatenated feature vector.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        
        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # Chroma
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        
        return np.hstack((mfcc, chroma, mel))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(data_path):
    """
    Loads data from the RAVDESS dataset.
    Args:
        data_path (str): Path to the dataset directory (containing Actor_* folders).
    Returns:
        tuple: (X, y) where X is features and y is labels.
    """
    X, y = [], []
    
    # Emotions in RAVDESS dataset
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Observed emotions
    target_emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised']

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = emotions.get(emotion_code)
                
                if emotion in target_emotions:
                    file_path = os.path.join(root, file)
                    feature = extract_features(file_path)
                    if feature is not None:
                        X.append(feature)
                        y.append(emotion)
                        
    return np.array(X), np.array(y)
