"""
visualize_gradcam.py
Apply Grad-CAM on CNN classifier for spectrogram explainability
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from gradcam import GradCAM
from train_cnn import CNNModClassifier, make_spectrogram

# =====================
# Load model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModClassifier(num_classes=5).to(device)
model.load_state_dict(torch.load("cnn_classifier.pth", map_location=device))
model.eval()

# =====================
# Load one test signal
# =====================
signals = np.load("dataset/signals.npy")
labels = np.load("dataset/labels.npy")

idx = 123  # pick any test signal
signal = signals[idx]
label = labels[idx]

spec = make_spectrogram(signal)
spec_tensor = torch.tensor(spec[np.newaxis, np.newaxis, :, :]).to(device)

# =====================
# Grad-CAM
# =====================
gradcam = GradCAM(model, model.conv2)  # pick target layer
cam = gradcam.generate(spec_tensor)

# =====================
# Visualization
# =====================
plt.figure(figsize=(10, 4))
plt.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
plt.imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
plt.title(f"True Label: {label}")
plt.colorbar(label="Grad-CAM intensity")
plt.show()
