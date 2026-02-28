import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_processor import load_data

# Configuration
DATA_PATH = "../dataset"
MODEL_PATH = "../models/pro_ser_model.pkl"

def train():
    print("Loading data...")
    X, y = load_data(DATA_PATH)
    
    if len(X) == 0:
        print("No data found! Check the dataset path.")
        return

    print(f"Data loaded: {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
