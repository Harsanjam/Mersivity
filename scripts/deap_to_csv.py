#!/usr/bin/env python3
"""Convert a DEAP subject file (Python preprocessed .dat) to a long-form CSV.

DEAP (Koelstra et al., 2012) provides per-trial valence/arousal ratings (1-9).
This script converts trials into a single continuous time series with a per-sample label:
- calm: arousal <= calm_thr
- stress: arousal >= stress_thr
- otherwise: unlabeled

Usage:
python scripts/deap_to_csv.py --deap_file /path/to/s01.dat --out_csv data/deap_s01.csv
"""

import argparse, os, pickle
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deap_file", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--calm_thr", type=float, default=4.0)
    ap.add_argument("--stress_thr", type=float, default=6.0)
    ap.add_argument("--eeg_only", action="store_true", help="Keep only EEG channels (first 32).")
    args = ap.parse_args()

    with open(args.deap_file, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    data = obj["data"]  # shape: (n_trials, n_channels, n_samples)
    labels = obj["labels"]  # shape: (n_trials, 4) -> valence, arousal, dominance, liking

    if args.eeg_only:
        data = data[:, :32, :]

    sfreq = 128.0
    n_trials, n_ch, n_samp = data.shape

    rows = []
    t_offset = 0.0
    for tr in range(n_trials):
        arousal = float(labels[tr, 1])
        if arousal <= args.calm_thr:
            lab = "calm"
        elif arousal >= args.stress_thr:
            lab = "stress"
        else:
            lab = ""

        x = data[tr].astype(np.float32)  # (n_ch, n_samp)
        t = (np.arange(n_samp) / sfreq) + t_offset
        t_offset = float(t[-1] + 1/sfreq)

        df = pd.DataFrame({"time": t})
        for c in range(n_ch):
            df[f"ch{c+1}"] = x[c]
        df["label"] = lab
        df["trial"] = tr
        df["arousal"] = arousal
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
