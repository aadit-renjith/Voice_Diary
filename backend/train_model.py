import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from data_processor import load_data_multi_speaker_split, N_FEATURES

# ── TensorFlow / Keras imports ───────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, Add, Activation,
    Bidirectional, LSTM, Dense, Dropout, Layer, GlobalAveragePooling1D
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

# Hyperparameters
N_AUGMENTS = 5
BATCH_SIZE = 32
EPOCHS = 150
INITIAL_LR = 0.001
LABEL_SMOOTHING = 0.1
L2_REG = 0.0005


# ═══════════════════════════════════════════════════════════════════════════
# Custom Attention Layer
# ═══════════════════════════════════════════════════════════════════════════

class AttentionLayer(Layer):
    """
    Simple additive attention over time steps.
    Input:  (batch, time, features)
    Output: (batch, features) — weighted sum over time.
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x: (batch, time, features)
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, time, 1)
        alpha = tf.nn.softmax(e, axis=1)                 # (batch, time, 1)
        context = tf.reduce_sum(x * alpha, axis=1)        # (batch, features)
        return context

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


# ═══════════════════════════════════════════════════════════════════════════
# Residual CNN Block
# ═══════════════════════════════════════════════════════════════════════════

def residual_block(x, filters, kernel_size=3, pool_size=2, dropout=0.25):
    """
    Residual CNN block: Conv→BN→ReLU→Conv→BN→Add→ReLU→MaxPool→Dropout.
    If input channels != filters, a 1x1 conv is used for the shortcut.
    """
    shortcut = x

    # First conv
    x = Conv1D(filters, kernel_size, padding='same',
               kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second conv
    x = Conv1D(filters, kernel_size, padding='same',
               kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)

    # Shortcut projection if dimensions differ
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Residual add
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    # Pool and dropout
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Model Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_model(input_shape, num_classes):
    """
    Build improved SER model: Residual CNN → Bidirectional LSTM → Attention.

    Architecture:
        Input (MAX_LEN, 186)
        → Res CNN Block 1 (64 filters)
        → Res CNN Block 2 (128 filters)
        → Res CNN Block 3 (256 filters)
        → Bidirectional LSTM (128 units, return_sequences=True)
        → Attention (weighted sum over time)
        → Dense(128) → Dropout(0.4)
        → Dense(num_classes, softmax)
    """
    inputs = Input(shape=input_shape)

    # Residual CNN blocks
    x = residual_block(inputs, 64, dropout=0.2)
    x = residual_block(x, 128, dropout=0.25)
    x = residual_block(x, 256, dropout=0.3)

    # Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=True,
                           kernel_regularizer=l2(L2_REG)))(x)
    x = Dropout(0.3)(x)

    # Attention over time steps
    x = AttentionLayer()(x)

    # Dense classifier
    x = Dense(128, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def train():
    # ── 1. Load data with speaker-independent split ──────────────────────────
    print(f"Loading data (speaker-independent split, test actors: {TEST_ACTORS})...")
    print(f"Using multi-feature extraction (mel + MFCC+Δ+ΔΔ + chroma + spectral_contrast)")
    print(f"Augmentation: {N_AUGMENTS} copies per sample + SpecAugment\n")

    X_train, y_train_raw, X_test, y_test_raw = load_data_multi_speaker_split(
        DATA_PATH, n_augments=N_AUGMENTS, test_actors=TEST_ACTORS,
        apply_spec_augment=True
    )

    if len(X_train) == 0:
        print("No data found! Check the dataset path.")
        return

    print(f"\nTrain: {len(X_train)} samples (augmented) | Test: {len(X_test)} samples (original only)")
    print(f"Feature shape: {X_train.shape[1:]}  (time_steps, features)")

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

    y_train_encoded = le.transform(y_train_raw)
    y_test_encoded = le.transform(y_test_raw)

    y_train = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test = to_categorical(y_test_encoded, num_classes=num_classes)
    print(f"\nClasses: {list(le.classes_)} ({num_classes} total)")

    # ── 3. Compute class weights ─────────────────────────────────────────────
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_encoded),
        y=y_train_encoded
    )
    class_weights = dict(enumerate(class_weights_array))
    print(f"\nClass weights: {class_weights}")

    # ── 4. Feature scaling (fit on train only) ───────────────────────────────
    n_train, n_time, n_freq = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_freq)).reshape(n_train, n_time, n_freq)
    X_test = scaler.transform(X_test.reshape(-1, n_freq)).reshape(n_test, n_time, n_freq)

    # ── 5. Build model ───────────────────────────────────────────────────────
    model = build_model(input_shape=(n_time, n_freq), num_classes=num_classes)

    # Cosine decay with warmup
    warmup_epochs = 5
    total_steps = EPOCHS * (n_train // BATCH_SIZE)
    warmup_steps = warmup_epochs * (n_train // BATCH_SIZE)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6  # minimum LR
    )

    # Wrap with warmup
    class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_schedule, warmup_steps, warmup_lr):
            super().__init__()
            self.base_schedule = base_schedule
            self.warmup_steps = warmup_steps
            self.warmup_lr = warmup_lr

        def __call__(self, step):
            warmup_factor = tf.minimum(
                tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32),
                1.0
            )
            warmup_lr = self.warmup_lr * warmup_factor
            base_lr = self.base_schedule(tf.maximum(step - self.warmup_steps, 0))
            return tf.where(step < self.warmup_steps, warmup_lr, base_lr)

        def get_config(self):
            return {
                "warmup_steps": self.warmup_steps,
                "warmup_lr": self.warmup_lr,
            }

    schedule = WarmupSchedule(lr_schedule, warmup_steps, INITIAL_LR)

    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    model.summary()

    # ── 6. Callbacks ─────────────────────────────────────────────────────────
    # Note: No ReduceLROnPlateau — cosine decay schedule handles LR reduction
    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=1,
        mode='max'
    )

    # ── 7. Train ─────────────────────────────────────────────────────────────
    print("\nTraining improved SER model (Residual CNN + BiLSTM + Attention)...")
    print(f"Batch size: {BATCH_SIZE} | Epochs: {EPOCHS} | Label smoothing: {LABEL_SMOOTHING}")
    print(f"LR schedule: Cosine decay with {warmup_epochs}-epoch warmup\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # ── 8. Evaluate ──────────────────────────────────────────────────────────
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

    # ── 9. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")

    print(f"\n{'=' * 60}")
    print(f"FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
