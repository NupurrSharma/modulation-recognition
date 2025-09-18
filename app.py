# app.py
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_cnn import CNNModClassifier
from gradcam import GradCAM
from noise_utils import add_awgn
import pandas as pd

st.set_page_config(page_title="Modulation Recognition", layout="wide")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("Controls")
dataset = st.sidebar.selectbox("Select Dataset", ["synthetic", "radioml"])
uploaded_file = st.sidebar.file_uploader("Upload Signal (.npy)", type=["npy"])
add_noise = st.sidebar.checkbox("Add Noise")
snr = st.sidebar.slider("SNR (dB)", min_value=0, max_value=50, value=20)
gradcam_alpha = st.sidebar.slider("Grad-CAM Transparency", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# -----------------------------
# Load Model & Class Mapping
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dataset == "synthetic":
    model_path = "cnn_classifier_synthetic.pth"
    class_mapping = {0:"BPSK", 1:"QPSK", 2:"8PSK", 3:"16QAM", 4:"64QAM"}
    class_desc = {
        "BPSK":"Binary Phase Shift Keying",
        "QPSK":"Quadrature Phase Shift Keying",
        "8PSK":"8-Phase PSK",
        "16QAM":"16-level Quadrature Amplitude Modulation",
        "64QAM":"64-level QAM"
    }
else:
    model_path = "cnn_classifier_radioml.pth"
    class_mapping = {0:"BPSK",1:"QPSK",2:"8PSK",3:"16QAM",4:"64QAM",
                     5:"AM-DSB",6:"AM-SSB",7:"FM",8:"GFSK",9:"CPFSK",10:"16APSK"}
    class_desc = {
        "BPSK":"Binary Phase Shift Keying",
        "QPSK":"Quadrature Phase Shift Keying",
        "8PSK":"8-Phase PSK",
        "16QAM":"16-level Quadrature Amplitude Modulation",
        "64QAM":"64-level QAM",
        "AM-DSB":"Amplitude Modulation Double Sideband",
        "AM-SSB":"Amplitude Modulation Single Sideband",
        "FM":"Frequency Modulation",
        "GFSK":"Gaussian Frequency Shift Keying",
        "CPFSK":"Continuous Phase Frequency Shift Keying",
        "16APSK":"16-level Amplitude Phase Shift Keying"
    }

model = CNNModClassifier(len(class_mapping)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

st.title("📡 Modulation Recognition & Explainable AI")

# -----------------------------
# Process Uploaded Signal
# -----------------------------
if uploaded_file is not None:
    signal = np.load(uploaded_file)

    if add_noise:
        signal = add_awgn(signal, snr_db=snr)

    # Convert to spectrogram
    spectrogram = np.abs(np.fft.fftshift(np.fft.fft(signal)))
    size = int(np.ceil(np.sqrt(len(spectrogram))))
    spectrogram = np.pad(spectrogram, (0, size*size - len(spectrogram)), mode='constant')
    spectrogram = spectrogram.reshape(size, size)

    # -----------------------------
    # Layout: Raw + Spectrogram
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Signal")
        fig_sig, ax_sig = plt.subplots(figsize=(8,2))
        ax_sig.plot(signal)
        ax_sig.set_xlabel("Samples")
        ax_sig.set_ylabel("Amplitude")
        st.pyplot(fig_sig)

    with col2:
        st.subheader("Spectrogram")
        fig_spec, ax_spec = plt.subplots(figsize=(6,4))
        cax = ax_spec.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        fig_spec.colorbar(cax, ax=ax_spec, orientation='vertical', label='Magnitude')
        st.pyplot(fig_spec)

    # -----------------------------
    # Prediction
    # -----------------------------
    input_tensor = torch.tensor(spectrogram[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).cpu().item()
        predicted_class = class_mapping.get(predicted_idx, "Unknown")

    st.markdown("### Predicted Modulation Class")
    st.markdown(f"<h1 style='color:blue;'>{predicted_class}</h1>", unsafe_allow_html=True)

    # -----------------------------
    # Grad-CAM Heatmap
    # -----------------------------
    grad_cam = GradCAM(model, target_layer="conv2")
    heatmap = grad_cam.generate_heatmap(input_tensor)

    st.subheader("Grad-CAM Overlay")
    fig_cam, ax_cam = plt.subplots(figsize=(6,4))
    ax_cam.imshow(spectrogram, cmap='viridis', alpha=1.0)
    hm = ax_cam.imshow(heatmap, cmap='jet', alpha=gradcam_alpha)
    fig_cam.colorbar(hm, ax=ax_cam, orientation='vertical', label='Importance')
    st.pyplot(fig_cam)

    # -----------------------------
    # Class Description Table
    # -----------------------------
    st.subheader("Class Descriptions")
    class_table = pd.DataFrame(list(class_desc.items()), columns=["Class", "Description"])
    st.table(class_table)

    st.success("✅ Prediction & Grad-CAM generated successfully!")

else:
    st.info("Please upload a `.npy` signal file to start.")
