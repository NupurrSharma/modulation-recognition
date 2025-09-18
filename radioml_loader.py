"""
radioml_loader.py
Loader and preprocessor for RadioML 2016.10a dataset
"""

import pickle
import numpy as np
from signal_utils import make_spectrogram


def load_radioml_dataset(file_path, snr_list=None, max_samples_per_class=1000):
    """
    Load RadioML 2016.10a dataset and preprocess to spectrograms.

    Args:
        file_path (str): Path to RML2016.10a_dict.pkl
        snr_list (list, optional): List of SNRs to include (e.g., [0, 10, 20]).
        max_samples_per_class (int): Limit per (modulation, SNR) pair.

    Returns:
        X (np.array): Spectrogram dataset [N, H, W]
        y (np.array): Labels [N]
        class_names (list): List of modulation classes
    """
    # =====================
    # Load dataset
    # =====================
    with open(file_path, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    X, y = [], []
    class_names = []
    label_map = {}

    idx = 0
    for (mod, snr), samples in dataset.items():
        # Filter by SNR if needed
        if snr_list and snr not in snr_list:
            continue

        if mod not in label_map:
            label_map[mod] = idx
            class_names.append(mod)
            idx += 1

        # Limit samples per class for speed
        samples = samples[:max_samples_per_class]

        for s in samples:
            iq_signal = s[0] + 1j * s[1]  # convert I/Q to complex
            spec = make_spectrogram(iq_signal)
            X.append(spec)
            y.append(label_map[mod])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, class_names
