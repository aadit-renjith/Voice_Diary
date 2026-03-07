import librosa
import numpy as np
import os
import soundfile as sf
import random


# ── Audio Augmentation Helpers (training only) ──────────────────────────────

def time_shift(y, sr):
    """Randomly shift the waveform left/right by up to ±20%."""
    shift_max = int(len(y) * 0.2)
    shift = random.randint(-shift_max, shift_max)
    return np.roll(y, shift)


def pitch_shift(y, sr):
    """Randomly shift pitch by -2 to +2 semitones."""
    n_steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def add_noise(y):
    """Add Gaussian white noise at ~20 dB SNR."""
    noise_factor = 0.01
    noise = np.random.randn(len(y)) * noise_factor
    return y + noise


def augment_audio(y, sr):
    """Apply a random subset of augmentations to an audio signal."""
    transforms = [time_shift, pitch_shift, add_noise]
    # Apply each transform with 50% probability (at least one)
    chosen = [t for t in transforms if random.random() < 0.5]
    if not chosen:
        chosen = [random.choice(transforms)]
    for t in chosen:
        if t is add_noise:
            y = t(y)
        else:
            y = t(y, sr)
    return y

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


def extract_features_from_signal(y, sr):
    """
    Extracts MFCC, Chroma, and Mel Spectrogram features from a raw audio signal.
    Same feature set as extract_features(), but operates on a numpy array.
    """
    try:
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        return np.hstack((mfcc, chroma, mel))
    except Exception as e:
        print(f"Error extracting features from signal: {e}")
        return None


def load_data_augmented(data_path, n_augments=3):
    """
    Loads RAVDESS data with augmentation.
    For each file, the original sample is kept and n_augments augmented copies
    are created using random combinations of time-shift, pitch-shift, and noise.
    """
    X, y_labels = [], []

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
    target_emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised']

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = emotions.get(emotion_code)

                if emotion in target_emotions:
                    file_path = os.path.join(root, file)
                    try:
                        # Load raw audio once
                        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

                        # Original sample
                        feature = extract_features_from_signal(audio, sr)
                        if feature is not None:
                            X.append(feature)
                            y_labels.append(emotion)

                        # Augmented copies
                        for _ in range(n_augments):
                            aug_audio = augment_audio(audio.copy(), sr)
                            aug_feature = extract_features_from_signal(aug_audio, sr)
                            if aug_feature is not None:
                                X.append(aug_feature)
                                y_labels.append(emotion)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y_labels)


# ── Deep Learning Feature Extraction ────────────────────────────────────────

MAX_LEN = 130  # ~3 seconds at sr=22050 with hop_length=512

def extract_mel_spectrogram(file_path, max_len=MAX_LEN):
    """
    Extract a 2D mel-spectrogram from an audio file.
    Returns shape (max_len, 128) — padded or truncated to fixed length.
    """
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max).T  # (time, 128)

        # Pad or truncate to fixed length
        if mel_db.shape[0] < max_len:
            pad_width = max_len - mel_db.shape[0]
            mel_db = np.pad(mel_db, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mel_db = mel_db[:max_len, :]

        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_mel_from_signal(y, sr, max_len=MAX_LEN):
    """
    Extract a 2D mel-spectrogram from a raw audio signal.
    Returns shape (max_len, 128).
    """
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max).T  # (time, 128)

        if mel_db.shape[0] < max_len:
            pad_width = max_len - mel_db.shape[0]
            mel_db = np.pad(mel_db, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mel_db = mel_db[:max_len, :]

        return mel_db
    except Exception as e:
        print(f"Error extracting mel from signal: {e}")
        return None


def load_data_dl(data_path, n_augments=3):
    """
    Load RAVDESS data with augmentation for DL models.
    Returns (X, y) where X shape is (n_samples, max_len, 128).
    """
    X, y_labels = [], []

    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    target_emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised']

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = emotions.get(emotion_code)

                if emotion in target_emotions:
                    file_path = os.path.join(root, file)
                    try:
                        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

                        # Original
                        feature = extract_mel_from_signal(audio, sr)
                        if feature is not None:
                            X.append(feature)
                            y_labels.append(emotion)

                        # Augmented
                        for _ in range(n_augments):
                            aug_audio = augment_audio(audio.copy(), sr)
                            aug_feat = extract_mel_from_signal(aug_audio, sr)
                            if aug_feat is not None:
                                X.append(aug_feat)
                                y_labels.append(emotion)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y_labels)


def load_data_dl_speaker_split(data_path, n_augments=3, test_actors=None):
    """
    Load RAVDESS data with SPEAKER-INDEPENDENT split.
    Actors in test_actors go into the test set (no augmentation).
    All other actors go into training (with augmentation).
    This prevents data leakage from same-speaker samples in train & test.
    """
    if test_actors is None:
        test_actors = [21, 22, 23, 24]  # Hold out 4 actors (~17%)

    test_actor_dirs = {f"Actor_{a:02d}" for a in test_actors}

    X_train, y_train = [], []
    X_test, y_test = [], []

    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    target_emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised']

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = emotions.get(emotion_code)

                if emotion in target_emotions:
                    file_path = os.path.join(root, file)

                    # Determine if this actor is in the test set
                    actor_dir = os.path.basename(root)
                    is_test = actor_dir in test_actor_dirs

                    try:
                        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
                        feature = extract_mel_from_signal(audio, sr)

                        if feature is not None:
                            if is_test:
                                # Test: original only, no augmentation
                                X_test.append(feature)
                                y_test.append(emotion)
                            else:
                                # Train: original + augmented
                                X_train.append(feature)
                                y_train.append(emotion)

                                for _ in range(n_augments):
                                    aug_audio = augment_audio(audio.copy(), sr)
                                    aug_feat = extract_mel_from_signal(aug_audio, sr)
                                    if aug_feat is not None:
                                        X_train.append(aug_feat)
                                        y_train.append(emotion)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test))

