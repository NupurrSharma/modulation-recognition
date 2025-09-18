"""
dataset_generator.py
Generates synthetic modulated signals + noise, saves them with labels for AI training.
"""

import numpy as np
from scipy.signal import square
import os

# Sampling settings
FS = 8000        # Sampling frequency
T = 1.0          # Duration (seconds)
N = int(FS * T)  # Samples per signal
t = np.arange(N) / FS

# Dataset settings
CLASSES = ["sine", "am", "fm", "bpsk", "qpsk"]
SAMPLES_PER_CLASS = 2000   # Adjust based on how big you want dataset

def sine_wave():
    f = np.random.randint(50, 400)       # random frequency
    return np.sin(2 * np.pi * f * t)

def am_wave():
    fc = np.random.randint(500, 1500)    # carrier freq
    fm = np.random.randint(10, 100)      # modulating freq
    ka = np.random.uniform(0.3, 1.0)     # modulation index
    m = np.sin(2 * np.pi * fm * t)
    return (1 + ka * m) * np.cos(2 * np.pi * fc * t)

def fm_wave():
    fc = np.random.randint(500, 1500)
    fm = np.random.randint(10, 100)
    beta = np.random.uniform(2, 8)       # modulation index
    return np.cos(2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t))

def bpsk_wave():
    fc = np.random.randint(500, 1500)
    symbols = np.random.choice([1, -1], size=N//100)  # random bits
    signal = np.repeat(symbols, 100)  # upsample
    return signal * np.cos(2 * np.pi * fc * t)

def qpsk_wave():
    fc = np.random.randint(500, 1500)
    bits = np.random.choice([0, 1], size=(N//100, 2))  # pairs
    symbols = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)     # map to constellation
    signal = np.repeat(symbols, 100)                   # upsample
    return np.real(signal * np.exp(1j*2*np.pi*fc*t))

def add_awgn(sig, snr_db):
    sig_power = np.mean(sig**2)
    snr_lin = 10**(snr_db/10)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power) * np.random.randn(len(sig))
    return sig + noise

def generate_dataset():
    signals = []
    labels = []

    for idx, cls in enumerate(CLASSES):
        for _ in range(SAMPLES_PER_CLASS):
            if cls == "sine":
                sig = sine_wave()
            elif cls == "am":
                sig = am_wave()
            elif cls == "fm":
                sig = fm_wave()
            elif cls == "bpsk":
                sig = bpsk_wave()
            elif cls == "qpsk":
                sig = qpsk_wave()
            else:
                continue

            # Add random noise
            snr_db = np.random.uniform(0, 20)   # 0–20 dB SNR
            sig_noisy = add_awgn(sig, snr_db)

            # Normalize (helps ML training)
            sig_noisy = sig_noisy / np.max(np.abs(sig_noisy))

            signals.append(sig_noisy)
            labels.append(idx)

    signals = np.array(signals)
    labels = np.array(labels)

    # Save
    os.makedirs("dataset", exist_ok=True)
    np.save("dataset/signals.npy", signals)
    np.save("dataset/labels.npy", labels)

    print(f"✅ Dataset created: {signals.shape} signals, {len(CLASSES)} classes")

if __name__ == "__main__":
    generate_dataset()
