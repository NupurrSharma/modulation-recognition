import numpy as np

def generate_sine_wave(length=128, freq=5):
    t = np.linspace(0, 1, length)
    return np.sin(2 * np.pi * freq * t)

def generate_square_wave(length=128, freq=5):
    t = np.linspace(0, 1, length)
    return np.sign(np.sin(2 * np.pi * freq * t))

def generate_noise(length=128):
    return np.random.randn(length)

def save_demo_signals():
    np.save("sine.npy", generate_sine_wave())
    np.save("square.npy", generate_square_wave())
    np.save("noise.npy", generate_noise())
    print("✅ Demo signals saved: sine.npy, square.npy, noise.npy")

if __name__ == "__main__":
    save_demo_signals()
