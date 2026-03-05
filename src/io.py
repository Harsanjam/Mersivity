import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EEGRecording:
    time: np.ndarray              # (n_samples,)
    data: np.ndarray              # (n_ch, n_samples)
    sfreq: float
    ch_names: List[str]
    labels: Optional[np.ndarray]  # (n_samples,) or None

def load_csv(path: str) -> EEGRecording:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column in seconds.")
    time = df["time"].to_numpy(dtype=float)

    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time column; cannot infer sampling rate.")
    sfreq = 1.0 / dt

    labels = None
    if "label" in df.columns:
        labels = df["label"].astype(str).to_numpy()

    ch_cols = [c for c in df.columns if c not in ("time", "label")]
    if len(ch_cols) < 1:
        raise ValueError("CSV must include at least one channel column besides time/label.")
    data = df[ch_cols].to_numpy(dtype=float).T  # (n_ch, n_samples)
    return EEGRecording(time=time, data=data, sfreq=float(sfreq), ch_names=ch_cols, labels=labels)

def majority_label(window_labels: np.ndarray) -> str:
    vals = [str(v).strip().lower() for v in window_labels if str(v).strip() != ""]
    if not vals:
        return ""
    return max(set(vals), key=vals.count)
