import numpy as np
from scipy.signal import spectrogram
from skimage.transform import resize

def make_spectrogram(signal, nfft=128, noverlap=64, target_size=(32, 32)):
    """
    Converts 1D signal to 2D spectrogram for CNN input.
    """
    f, t, Sxx = spectrogram(signal, nperseg=nfft, noverlap=noverlap)
    Sxx = np.log1p(Sxx)  # log scale
    Sxx_resized = resize(Sxx, target_size, mode='reflect', anti_aliasing=True)
    Sxx_resized = Sxx_resized / np.max(Sxx_resized)
    return Sxx_resized
