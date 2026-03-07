"""Quick evaluation script — loads saved model and evaluates on held-out speakers."""
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_processor import load_data_multi_speaker_split
from train_model import AttentionLayer
import tensorflow as tf

# Config (same as train_model.py)
MODEL_PATH = "../models/ser_cnn_lstm.keras"
LABEL_ENCODER_PATH = "../models/label_encoder.pkl"
SCALER_PATH = "../models/scaler.pkl"
DATA_PATH = "../dataset"
TEST_ACTORS = [21, 22, 23, 24]

print("Loading saved model, scaler, and label encoder...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'AttentionLayer': AttentionLayer}
)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

print(f"Loading test data (actors {TEST_ACTORS}, no augmentation)...")
_, _, X_test, y_test_raw = load_data_multi_speaker_split(
    DATA_PATH, n_augments=0, test_actors=TEST_ACTORS,
    apply_spec_augment=False
)
print(f"Test samples: {len(X_test)}")

# Encode & scale
y_test_encoded = le.transform(y_test_raw)
n_test, n_time, n_freq = X_test.shape
X_test_scaled = scaler.transform(X_test.reshape(-1, n_freq)).reshape(n_test, n_time, n_freq)

# Predict
y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Results
print("\n" + "=" * 60)
print("MODEL EVALUATION (Speaker-Independent)")
print(f"Test actors: {TEST_ACTORS} (never seen during training)")
print("=" * 60)
print(f"\nAccuracy: {accuracy_score(y_test_encoded, y_pred) * 100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))
print(f"\nLabels: {list(le.classes_)}")
