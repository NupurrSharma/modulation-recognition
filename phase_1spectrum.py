"""
phase1_spectrum.py
Generate example signals (sine, square, AM, FM), add noise, and plot time-domain and FFT spectra.
Run: python phase1_spectrum.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth

def make_time(fs, T):
    N = int(fs * T)
    t = np.arange(N) / fs
    return t

def sine(t, f, A=1.0, phase=0.0):
    return A * np.sin(2 * np.pi * f * t + phase)

def square_wave(t, f, A=1.0):
    return A * square(2 * np.pi * f * t)

def am_signal(t, fc, fm, ka=0.5, Am=1.0):
    """
    AM: s(t) = [1 + ka * m(t)] * cos(2π fc t)
    where m(t) = Am*sin(2π fm t) (normalized)
    """
    m = Am * np.sin(2 * np.pi * fm * t)
    return (1.0 + ka * m) * np.cos(2 * np.pi * fc * t)

def fm_signal(t, fc, fm, beta=5.0, Am=1.0):
    """
    Narrowband style: s(t) = Am * cos(2π fc t + beta * sin(2π fm t))
    Here beta = Δf / fm (modulation index).
    """
    return Am * np.cos(2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t))

def add_awgn(sig, snr_db):
    """Add white Gaussian noise to achieve a target SNR (dB)."""
    sig_power = np.mean(sig**2)
    snr_lin = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power) * np.random.randn(*sig.shape)
    return sig + noise

def plot_time(signal, t, title, t_zoom=None):
    plt.figure(figsize=(9,3))
    plt.plot(t, signal, linewidth=0.8)
    plt.title(title + " — Time domain")
    plt.xlabel("Time [s]")
    plt.grid(True)
    if t_zoom is not None:
        plt.xlim(0, t_zoom)
    plt.tight_layout()
    plt.show()

def plot_fft(signal, fs, title, window=True, single_sided=True, zero_pad=0):
    N = len(signal)
    if window:
        win = np.hamming(N)
    else:
        win = np.ones(N)
    sig_win = signal * win
    Nfft = N + int(zero_pad)
    fft_vals = np.fft.fft(sig_win, n=Nfft)
    fft_freqs = np.fft.fftfreq(Nfft, d=1.0/fs)
    mag = np.abs(fft_vals) / N  # normalize by N to get amplitude scale

    if single_sided:
        half = Nfft // 2 + 1
        freqs = fft_freqs[:half]
        mag_ss = mag[:half].copy()
        # double non-DC/non-Nyquist bins to get single-sided amplitude
        if half > 2:
            mag_ss[1:-1] *= 2.0
        plt.figure(figsize=(9,4))
        plt.plot(freqs, mag_ss, linewidth=1)
        plt.title(title + " — Single-sided Amplitude Spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(9,4))
        plt.plot(np.fft.fftshift(fft_freqs), np.fft.fftshift(mag), linewidth=1)
        plt.title(title + " — Two-sided Spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def demo():
    fs = 8000        # sampling freq (Hz)
    T  = 1.0         # duration (s)
    t = make_time(fs, T)

    # ---- signals ----
    s_sine = sine(t, f=50, A=1.0)
    s_square = square_wave(t, f=50, A=1.0)
    s_am = am_signal(t, fc=1000, fm=50, ka=0.7, Am=1.0)
    s_fm = fm_signal(t, fc=1000, fm=50, beta=5.0, Am=1.0)

    # ---- plot time (zoomed) and spectrum ----
    plot_time(s_sine, t, "Pure Sine 50 Hz", t_zoom=0.02)
    plot_fft(s_sine, fs, "Pure Sine 50 Hz", window=False)

    plot_time(s_square, t, "Square wave 50 Hz (first 20 ms)", t_zoom=0.02)
    plot_fft(s_square, fs, "Square 50 Hz (harmonics)")

    plot_time(s_am, t, "AM: fc=1 kHz, fm=50 Hz (first 5 ms)", t_zoom=0.005)
    plot_fft(s_am, fs, "AM (carrier + sidebands)")

    plot_time(s_fm, t, "FM: fc=1 kHz, fm=50 Hz (first 5 ms)", t_zoom=0.005)
    plot_fft(s_fm, fs, "FM (multiple sidebands)")

    # ---- add noise and inspect ----
    s_am_noisy = add_awgn(s_am, snr_db=5)
    plot_fft(s_am_noisy, fs, "AM noisy (SNR=5 dB)")

if __name__ == "__main__":
    demo()
