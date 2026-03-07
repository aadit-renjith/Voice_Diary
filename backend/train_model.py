import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from data_processor import load_data_dl_speaker_split

# ── TensorFlow / Keras imports ───────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    LSTM, Dense, Dropout
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Configuration
DATA_PATH = "../dataset"
MODEL_PATH = "../models/ser_cnn_lstm.keras"
LABEL_ENCODER_PATH = "../models/label_encoder.pkl"
SCALER_PATH = "../models/scaler.pkl"

# Actors 21-24 held out for testing (speaker-independent)
TEST_ACTORS = [21, 22, 23, 24]


def build_model(input_shape, num_classes):
    """Build CNN-LSTM model for SER."""
    model = Sequential([
        # ── CNN Block 1 ──
        Conv1D(64, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(0.001), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # ── CNN Block 2 ──
        Conv1D(128, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # ── LSTM ──
        LSTM(64, return_sequences=False),
        Dropout(0.4),

        # ── Dense ──
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train():
    # ── 1. Load data with speaker-independent split ──────────────────────────
    print(f"Loading data (speaker-independent split, test actors: {TEST_ACTORS})...")
    X_train, y_train_raw, X_test, y_test_raw = load_data_dl_speaker_split(
        DATA_PATH, n_augments=3, test_actors=TEST_ACTORS
    )

    if len(X_train) == 0:
        print("No data found! Check the dataset path.")
        return

    print(f"Train: {len(X_train)} samples (augmented) | Test: {len(X_test)} samples (original only)")
    print(f"Feature shape: {X_train.shape[1:]}")

    print("\nTrain class distribution:")
    unique_tr, counts_tr = np.unique(y_train_raw, return_counts=True)
    for e, c in zip(unique_tr, counts_tr):
        print(f"  {e}: {c}")

    print("\nTest class distribution:")
    unique_te, counts_te = np.unique(y_test_raw, return_counts=True)
    for e, c in zip(unique_te, counts_te):
        print(f"  {e}: {c}")

    # ── 2. Encode labels ─────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    num_classes = len(le.classes_)

    y_train = to_categorical(le.transform(y_train_raw), num_classes=num_classes)
    y_test = to_categorical(le.transform(y_test_raw), num_classes=num_classes)
    print(f"\nClasses: {list(le.classes_)} ({num_classes} total)")

    # ── 3. Feature scaling (fit on train only) ───────────────────────────────
    n_train, n_time, n_freq = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_freq)).reshape(n_train, n_time, n_freq)
    X_test = scaler.transform(X_test.reshape(-1, n_freq)).reshape(n_test, n_time, n_freq)

    # ── 4. Build model ───────────────────────────────────────────────────────
    model = build_model(input_shape=(n_time, n_freq), num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ── 5. Callbacks ─────────────────────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    )

    # ── 6. Train ─────────────────────────────────────────────────────────────
    print("\nTraining CNN-LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Speaker-Independent)")
    print(f"Test actors: {TEST_ACTORS} (never seen during training)")
    print("=" * 60)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Labels: {list(le.classes_)}")

    # ── 8. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    train()
