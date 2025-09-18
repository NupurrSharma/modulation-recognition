"""
noise_utils.py
Utility functions for adding noise to signals
"""

import numpy as np

def add_awgn(signal, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.

    Parameters:
        signal (np.array): Input signal
        snr_db (float): Desired Signal-to-Noise Ratio in dB

    Returns:
        np.array: Noisy signal
    """
    # Signal power
    sig_power = np.mean(np.abs(signal) ** 2)

    # SNR (linear scale)
    snr_linear = 10 ** (snr_db / 10)

    # Noise power
    noise_power = sig_power / snr_linear

    # Generate Gaussian noise
    noise = np.sqrt(noise_power / 2) * np.random.randn(*signal.shape)

    return signal + noise
