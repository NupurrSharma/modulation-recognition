"""
train_classifier.py
Train a simple ML classifier to recognize modulation types (sine, AM, FM, BPSK, QPSK)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
signals = np.load("dataset/signals.npy")
labels = np.load("dataset/labels.npy")

# Convert signals -> FFT features
def extract_features(signals):
    features = []
    for sig in signals:
        fft_mag = np.abs(np.fft.fft(sig))[:len(sig)//2]  # keep half (Nyquist)
        fft_mag = fft_mag / np.max(fft_mag)              # normalize
        features.append(fft_mag[:500])  # take first 500 bins to reduce size
    return np.array(features)

X = extract_features(signals)
y = labels

print("Feature matrix shape:", X.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Overall Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(clf, "modulation_classifier.pkl")
print("\n🎉 Model saved as modulation_classifier.pkl")
