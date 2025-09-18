"""
test_noise_robustness.py
Evaluate CNN performance under noisy conditions (SNR robustness)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from train_cnn import CNNModClassifier, make_spectrogram
from noise_utils import add_awgn

# =====================
# Load model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModClassifier(num_classes=5).to(device)
model.load_state_dict(torch.load("cnn_classifier.pth", map_location=device))
model.eval()

# =====================
# Load dataset
# =====================
signals = np.load("dataset/signals.npy")
labels = np.load("dataset/labels.npy")

# =====================
# Test SNR levels
# =====================
snr_values = [20, 10, 0, -5]   # in dB
accuracies = []

for snr in snr_values:
    correct, total = 0, 0
    for i in range(500):  # test on 500 random samples
        sig = add_awgn(signals[i], snr)
        spec = make_spectrogram(sig)
        spec_tensor = torch.tensor(spec[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(spec_tensor)
            _, pred = torch.max(output, 1)
            if pred.item() == labels[i]:
                correct += 1
            total += 1

    acc = 100 * correct / total
    accuracies.append(acc)
    print(f"SNR {snr} dB → Accuracy: {acc:.2f}%")

# =====================
# Plot results
# =====================
plt.plot(snr_values, accuracies, marker='o', linewidth=2)
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("CNN Robustness to Noise")
plt.grid(True)
plt.show()
