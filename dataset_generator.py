#!/usr/bin/env python3
"""
dataset_generator.py

Robust generator for synthetic modulated signals + AWGN.
Writes signals incrementally to a disk-backed numpy memmap to avoid using lots of RAM.

Usage:
    python dataset_generator.py        # uses defaults
    python dataset_generator.py --samples-per-class 500 --out dataset_small
"""

import numpy as np
import os
import argparse

# -----------------------
# Settings (defaults)
# -----------------------
FS = 8000        # sampling frequency (Hz)
T = 1.0          # duration (seconds)
N = int(FS * T)  # number of samples per signal

DEFAULT_CLASSES = ["sine", "am", "fm", "bpsk", "qpsk"]

# -----------------------
# Signal generators
# -----------------------
def sine_wave(rng):
    f = rng.integers(50, 400)
    t = np.arange(N) / FS
    return np.sin(2 * np.pi * f * t)

def am_wave(rng):
    fc = rng.integers(500, 1500)
    fm = rng.integers(10, 100)
    ka = rng.uniform(0.3, 1.0)
    t = np.arange(N) / FS
    m = np.sin(2 * np.pi * fm * t)
    return (1 + ka * m) * np.cos(2 * np.pi * fc * t)

def fm_wave(rng):
    fc = rng.integers(500, 1500)
    fm = rng.integers(10, 100)
    beta = rng.uniform(2, 8)
    t = np.arange(N) / FS
    return np.cos(2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t))

def bpsk_wave(rng, upsample=100):
    fc = rng.integers(500, 1500)
    # generate symbol stream of length ceil(N/upsample)
    n_symbols = int(np.ceil(N / upsample))
    symbols = rng.choice([1.0, -1.0], size=n_symbols)
    waveform = np.repeat(symbols, upsample)[:N]
    t = np.arange(N) / FS
    return waveform * np.cos(2 * np.pi * fc * t)

def qpsk_wave(rng, upsample=100):
    fc = rng.integers(500, 1500)
    n_symbols = int(np.ceil(N / upsample))
    bits = rng.integers(0, 2, size=(n_symbols, 2))
    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    waveform = np.repeat(symbols, upsample)[:N]
    t = np.arange(N) / FS
    complex_signal = waveform * np.exp(1j * 2 * np.pi * fc * t)
    # We'll return the real part (you could return complex if you want)
    return np.real(complex_signal)

# -----------------------
# AWGN helper
# -----------------------
def add_awgn(sig, snr_db, rng):
    """Additive White Gaussian Noise. Handles real or complex `sig`."""
    # signal power: use magnitude-squared for complex or real
    sig_power = np.mean(np.abs(sig)**2)
    if sig_power == 0 or not np.isfinite(sig_power):
        # signal is all zeros or degenerate; just return noise
        snr_lin = 10**(snr_db/10)
        noise_power = 1.0 / snr_lin
    else:
        snr_lin = 10**(snr_db/10)
        noise_power = sig_power / snr_lin

    if np.iscomplexobj(sig):
        noise = (rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig))) * np.sqrt(noise_power/2)
    else:
        noise = rng.standard_normal(len(sig)) * np.sqrt(noise_power)
    return sig + noise

# -----------------------
# Main dataset generation
# -----------------------
def generate_dataset(out_dir="dataset",
                     classes=DEFAULT_CLASSES,
                     samples_per_class=500,
                     snr_min=0.0,
                     snr_max=20.0,
                     seed=None,
                     dtype=np.float32):
    """
    Generate dataset and save to disk-backed memmap.
    - out_dir: directory to place dataset files (created if needed)
    - classes: list of class names (strings, must match generator functions)
    - samples_per_class: samples to generate per class
    - snr_min, snr_max: random SNR range (dB)
    - seed: RNG seed (int) or None
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    generators = {
        "sine": sine_wave,
        "am": am_wave,
        "fm": fm_wave,
        "bpsk": bpsk_wave,
        "qpsk": qpsk_wave
    }

    # validate classes
    for c in classes:
        if c not in generators:
            raise ValueError(f"Unknown class '{c}'. Allowed: {list(generators.keys())}")

    total = int(len(classes) * samples_per_class)
    print(f"Generating {total} signals ({samples_per_class} per class), each {N} samples -> total elements {total*N}")

    # create memmap for signals
    signals_path = os.path.join(out_dir, "signals.npy")
    labels_path = os.path.join(out_dir, "labels.npy")

    # If file exists, overwrite safely
    if os.path.exists(signals_path):
        print(f"Overwriting existing {signals_path}")
        os.remove(signals_path)
    if os.path.exists(labels_path):
        os.remove(labels_path)

    signals_mem = np.lib.format.open_memmap(signals_path, mode="w+", dtype=dtype, shape=(total, N))
    labels = np.empty((total,), dtype=np.int16)

    idx = 0
    try:
        for cls_idx, cls in enumerate(classes):
            gen = generators[cls]
            for s in range(samples_per_class):
                # draw random SNR in range
                snr_db = float(rng.uniform(snr_min, snr_max))

                # generate clean signal
                if cls in ("bpsk", "qpsk"):
                    sig = gen(rng)  # bpsk/qpsk accept rng
                else:
                    sig = gen(rng)

                # add noise
                sig_noisy = add_awgn(sig, snr_db, rng)

                # normalize safely
                maxv = np.max(np.abs(sig_noisy))
                if maxv == 0 or not np.isfinite(maxv):
                    sig_norm = sig_noisy.astype(dtype)
                else:
                    sig_norm = (sig_noisy / maxv).astype(dtype)

                signals_mem[idx, :] = sig_norm
                labels[idx] = cls_idx
                idx += 1

                # lightweight progress print every 500 samples
                if (idx % 500) == 0 or idx == total:
                    print(f"Generated {idx}/{total} signals...")

    except KeyboardInterrupt:
        print("Generation interrupted by user. Partial dataset saved.")

    # flush to disk
    del signals_mem
    np.save(labels_path, labels)

    print(f"✅ Dataset created: signals -> {signals_path}, labels -> {labels_path}")
    return signals_path, labels_path

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic modulation dataset.")
    parser.add_argument("--out", type=str, default="dataset", help="Output directory")
    parser.add_argument("--samples-per-class", type=int, default=500, help="Samples per class")
    parser.add_argument("--snr-min", type=float, default=0.0, help="Min SNR in dB")
    parser.add_argument("--snr-max", type=float, default=20.0, help="Max SNR in dB")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args()

    generate_dataset(out_dir=args.out,
                     classes=DEFAULT_CLASSES,
                     samples_per_class=args.samples_per_class,
                     snr_min=args.snr_min,
                     snr_max=args.snr_max,
                     seed=args.seed)
