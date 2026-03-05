import numpy as np
from scipy.signal import stft
import pywt
from typing import Tuple

def compute_stft(x: np.ndarray, fs: float, win_sec: float, overlap_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = int(max(16, win_sec * fs))
    noverlap = int(overlap_frac * nperseg)
    f, t, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    P = np.abs(Z) ** 2
    return f, t, P

def compute_cwt_morlet(x: np.ndarray, fs: float, fmin: float=1.0, fmax: float=45.0, num_freqs: int=60, w0: float=6.0):
    # Morlet CWT via PyWavelets. Convert target freqs -> scales using central frequency.
    freqs = np.linspace(fmin, fmax, num_freqs)
    wavelet = pywt.ContinuousWavelet(f"cmor{w0}-1.0")
    cf = pywt.central_frequency(wavelet)
    scales = cf * fs / freqs
    coef, freqs_out = pywt.cwt(x, scales, wavelet, sampling_period=1/fs)
    power = np.abs(coef) ** 2
    # coef shape: (n_scales, n_time)
    t = np.arange(power.shape[1]) / fs
    return freqs, t, power
