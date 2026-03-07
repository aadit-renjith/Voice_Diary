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


def speed_perturb(y, sr):
    """Speed perturbation: 0.9x–1.1x speed (changes duration)."""
    rate = random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y=y, rate=rate)


def augment_audio(y, sr):
    """Apply a random subset of augmentations to an audio signal."""
    transforms = [time_shift, pitch_shift, add_noise, speed_perturb]
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


# ── SpecAugment (applied to feature matrices during training) ───────────────

def spec_augment(features, num_freq_masks=2, freq_mask_width=8,
                 num_time_masks=2, time_mask_width=10):
    """
    Apply SpecAugment: frequency and time masking on 2D feature matrix.
    Args:
        features: (time, freq) array
        num_freq_masks: number of frequency masks to apply
        freq_mask_width: maximum width of each frequency mask
        num_time_masks: number of time masks to apply
        time_mask_width: maximum width of each time mask
    Returns:
        Augmented feature matrix (same shape).
    """
    augmented = features.copy()
    n_time, n_freq = augmented.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_width, n_freq - 1))
        f0 = random.randint(0, n_freq - f)
        augmented[:, f0:f0 + f] = 0.0

    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_width, n_time - 1))
        t0 = random.randint(0, n_time - t)
        augmented[t0:t0 + t, :] = 0.0

    return augmented


# ── Legacy 1D Feature Extraction (kept for backward compatibility) ──────────

def extract_features(file_path):
    """
    Extracts MFCC, Chroma, and Mel Spectrogram features from an audio file.
    Args:
        file_path (str): Path to the audio file.
    Returns:
        np.array: Concatenated feature vector.
    """
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
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
                        feature = extract_features_from_signal(audio, sr)
                        if feature is not None:
                            X.append(feature)
                            y_labels.append(emotion)
                        for _ in range(n_augments):
                            aug_audio = augment_audio(audio.copy(), sr)
                            aug_feature = extract_features_from_signal(aug_audio, sr)
                            if aug_feature is not None:
                                X.append(aug_feature)
                                y_labels.append(emotion)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y_labels)


# ═══════════════════════════════════════════════════════════════════════════
# Deep Learning Multi-Feature Extraction (NEW — v2)
# ═══════════════════════════════════════════════════════════════════════════

MAX_LEN = 130   # ~3 seconds at sr=22050 with hop_length=512
N_MELS = 128
N_MFCC = 13
SR = 22050

# Total features per frame: 128 (mel) + 39 (mfcc+delta+delta2) + 12 (chroma) + 7 (contrast) = 186
N_FEATURES = N_MELS + (N_MFCC * 3) + 12 + 7  # = 186


def _pad_or_truncate(features_2d, max_len=MAX_LEN):
    """Pad with zeros or truncate a (time, freq) array to fixed length."""
    n_time = features_2d.shape[0]
    if n_time < max_len:
        pad_width = max_len - n_time
        features_2d = np.pad(features_2d, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features_2d = features_2d[:max_len, :]
    return features_2d


def extract_multi_features(file_path, max_len=MAX_LEN):
    """
    Extract stacked multi-feature representation from an audio file.
    Features per frame: mel(128) + MFCC+Δ+ΔΔ(39) + chroma(12) + spectral_contrast(7) = 186
    Returns shape (max_len, 186) — padded or truncated.
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=3, offset=0.5)
        return _extract_multi_from_signal(y, sr, max_len)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def _extract_multi_from_signal(y, sr, max_len=MAX_LEN):
    """
    Extract stacked multi-feature representation from a raw audio signal.
    Returns shape (max_len, 186).
    """
    try:
        # 1) Mel spectrogram (log-power)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)  # (128, time)

        # 2) MFCC + delta + delta-delta
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)   # (13, time)
        mfcc_delta = librosa.feature.delta(mfcc)                   # (13, time)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)         # (13, time)

        # 3) Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)           # (12, time)

        # 4) Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)   # (7, time)

        # Stack all features along frequency axis: (186, time)
        stacked = np.vstack([mel_db, mfcc, mfcc_delta, mfcc_delta2, chroma, contrast])
        stacked = stacked.T  # → (time, 186)

        # Pad or truncate to fixed length
        stacked = _pad_or_truncate(stacked, max_len)

        return stacked
    except Exception as e:
        print(f"Error extracting multi-features from signal: {e}")
        return None


# ── Legacy mel-only extraction (kept for backward compatibility) ────────────

def extract_mel_spectrogram(file_path, max_len=MAX_LEN):
    """
    Extract a 2D mel-spectrogram from an audio file.
    Returns shape (max_len, 128) — padded or truncated to fixed length.
    """
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max).T
        return _pad_or_truncate(mel_db, max_len)
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
        mel_db = librosa.power_to_db(mel, ref=np.max).T
        return _pad_or_truncate(mel_db, max_len)
    except Exception as e:
        print(f"Error extracting mel from signal: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Data Loaders for Deep Learning
# ═══════════════════════════════════════════════════════════════════════════

EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised']


def load_data_dl(data_path, n_augments=3):
    """
    Load RAVDESS data with augmentation for DL models.
    Returns (X, y) where X shape is (n_samples, max_len, 128).
    """
    X, y_labels = [], []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = EMOTIONS.get(emotion_code)

                if emotion in TARGET_EMOTIONS:
                    file_path = os.path.join(root, file)
                    try:
                        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

                        feature = extract_mel_from_signal(audio, sr)
                        if feature is not None:
                            X.append(feature)
                            y_labels.append(emotion)

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
        test_actors = [21, 22, 23, 24]

    test_actor_dirs = {f"Actor_{a:02d}" for a in test_actors}

    X_train, y_train = [], []
    X_test, y_test = [], []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = EMOTIONS.get(emotion_code)

                if emotion in TARGET_EMOTIONS:
                    file_path = os.path.join(root, file)
                    actor_dir = os.path.basename(root)
                    is_test = actor_dir in test_actor_dirs

                    try:
                        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
                        feature = extract_mel_from_signal(audio, sr)

                        if feature is not None:
                            if is_test:
                                X_test.append(feature)
                                y_test.append(emotion)
                            else:
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


def load_data_multi_speaker_split(data_path, n_augments=5, test_actors=None,
                                  apply_spec_augment=True):
    """
    Load RAVDESS data with SPEAKER-INDEPENDENT split using multi-feature extraction.
    Uses stacked features: mel + MFCC+Δ+ΔΔ + chroma + spectral_contrast.
    Shape: (n_samples, MAX_LEN, 186).

    Training data: original + n_augments augmented copies (waveform + SpecAugment).
    Test data: original only, no augmentation.
    """
    if test_actors is None:
        test_actors = [21, 22, 23, 24]

    test_actor_dirs = {f"Actor_{a:02d}" for a in test_actors}

    X_train, y_train = [], []
    X_test, y_test = [], []

    total_files = 0
    processed = 0

    # Count files first for progress
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion = EMOTIONS.get(parts[2])
                if emotion in TARGET_EMOTIONS:
                    total_files += 1

    print(f"Found {total_files} target audio files")

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = EMOTIONS.get(emotion_code)

                if emotion in TARGET_EMOTIONS:
                    file_path = os.path.join(root, file)
                    actor_dir = os.path.basename(root)
                    is_test = actor_dir in test_actor_dirs
                    processed += 1

                    if processed % 50 == 0:
                        print(f"  Processing {processed}/{total_files}...")

                    try:
                        audio, sr_loaded = librosa.load(file_path, sr=SR,
                                                        duration=3, offset=0.5)

                        # Extract multi-features from original
                        feature = _extract_multi_from_signal(audio, sr_loaded)

                        if feature is not None:
                            if is_test:
                                # Test: original only, no augmentation
                                X_test.append(feature)
                                y_test.append(emotion)
                            else:
                                # Train: original
                                X_train.append(feature)
                                y_train.append(emotion)

                                # Augmented copies
                                for _ in range(n_augments):
                                    aug_audio = augment_audio(audio.copy(), sr_loaded)
                                    aug_feat = _extract_multi_from_signal(
                                        aug_audio, sr_loaded
                                    )
                                    if aug_feat is not None:
                                        # Optionally apply SpecAugment
                                        if apply_spec_augment:
                                            aug_feat = spec_augment(aug_feat)
                                        X_train.append(aug_feat)
                                        y_train.append(emotion)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    print(f"Processing complete: {processed}/{total_files} files")

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test))
