import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, detrend as sp_detrend
from .config import PreprocessConfig
from .io import EEGRecording

def _bandpass(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sfreq
    lo = l_freq / nyq
    hi = h_freq / nyq
    b, a = butter(order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)

def _notch(x: np.ndarray, sfreq: float, f0: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=q, fs=sfreq)
    return filtfilt(b, a, x, axis=-1)

def _winsorize_mad(x: np.ndarray, z: float) -> np.ndarray:
    # Robust per-channel clipping using MAD (basic artifact handling)
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True) + 1e-12
    lo = med - z * 1.4826 * mad
    hi = med + z * 1.4826 * mad
    return np.clip(x, lo, hi)

def preprocess(rec: EEGRecording, cfg: PreprocessConfig) -> EEGRecording:
    y = rec.data.astype(np.float64, copy=True)

    if cfg.detrend:
        y = sp_detrend(y, axis=-1, type="linear")

    # basic artifact handling (winsorize)
    if cfg.clip_mad_z is not None and cfg.clip_mad_z > 0:
        y = _winsorize_mad(y, cfg.clip_mad_z)

    y = _bandpass(y, rec.sfreq, cfg.l_freq, cfg.h_freq, order=4)

    # notch mains + harmonic if inside band
    for f0 in [cfg.mains, 2*cfg.mains]:
        if cfg.l_freq < f0 < cfg.h_freq:
            y = _notch(y, rec.sfreq, f0=f0, q=cfg.notch_q)

    if cfg.reref_average and y.shape[0] > 1:
        y = y - np.mean(y, axis=0, keepdims=True)

    return EEGRecording(
        time=rec.time.copy(),
        data=y.astype(np.float32),
        sfreq=rec.sfreq,
        ch_names=list(rec.ch_names),
        labels=None if rec.labels is None else rec.labels.copy(),
    )
