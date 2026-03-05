import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
from .config import Bands, WindowConfig
from .io import EEGRecording, majority_label

def bandpower_welch(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs*2)))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return float("nan")
    return float(trapezoid(pxx[mask], f[mask]))

def extract_features(rec: EEGRecording, bands: Bands, win: WindowConfig) -> pd.DataFrame:
    fs = float(rec.sfreq)
    win_n = int(win.window_sec * fs)
    hop_n = int(win.hop_sec * fs)
    n = rec.data.shape[1]
    rows = []

    for start in range(0, max(1, n - win_n + 1), hop_n):
        end = start + win_n
        seg = rec.data[:, start:end]
        t0 = float(rec.time[start])
        t1 = float(rec.time[end-1])

        x = np.mean(seg, axis=0)

        row = {"t_start": t0, "t_end": t1}
        row["bp_delta"] = bandpower_welch(x, fs, *bands.delta)
        row["bp_theta"] = bandpower_welch(x, fs, *bands.theta)
        row["bp_alpha"] = bandpower_welch(x, fs, *bands.alpha)
        row["bp_beta"]  = bandpower_welch(x, fs, *bands.beta)
        row["bp_gamma"] = bandpower_welch(x, fs, *bands.gamma)

        eps = 1e-12
        row["beta_alpha"] = row["bp_beta"] / (row["bp_alpha"] + eps)
        row["theta_beta"] = row["bp_theta"] / (row["bp_beta"] + eps)

        if rec.labels is not None:
            row["label"] = majority_label(rec.labels[start:end])
        else:
            row["label"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)

    def norm_log(v):
        v = np.log(v + 1e-12)
        lo, hi = np.percentile(v[np.isfinite(v)], [5, 95])
        return np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    for c in ["bp_delta","bp_theta","bp_alpha","bp_beta","bp_gamma","beta_alpha","theta_beta"]:
        df[c + "_norm"] = norm_log(df[c].to_numpy(dtype=float))

    return df
