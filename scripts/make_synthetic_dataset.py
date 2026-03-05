#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

def make_synth(seconds=120, sfreq=256, n_ch=4, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    labels = np.where(t < (seconds/2), "calm", "stress")

    alpha_f, beta_f, theta_f = 10.0, 20.0, 6.0
    data = np.zeros((n_ch, n), dtype=np.float32)

    for ch in range(n_ch):
        phase = rng.uniform(0, 2*np.pi)
        alpha_amp = np.where(labels=="calm", 20.0, 8.0)
        beta_amp  = np.where(labels=="calm",  7.0, 20.0)
        theta_amp = np.where(labels=="calm", 10.0, 11.0)
        sig = (
            alpha_amp*np.sin(2*np.pi*alpha_f*t + phase) +
            beta_amp*np.sin(2*np.pi*beta_f*t + 0.5*phase) +
            theta_amp*np.sin(2*np.pi*theta_f*t + 1.3*phase) +
            rng.normal(0, 10.0, size=n)
        )
        data[ch] = sig.astype(np.float32)
    return t, data, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--sfreq", type=float, default=256.0)
    ap.add_argument("--channels", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t, data, labels = make_synth(seconds=args.seconds, sfreq=int(args.sfreq), n_ch=args.channels, seed=args.seed)

    df = pd.DataFrame({"time": t})
    for ch in range(data.shape[0]):
        df[f"ch{ch+1}"] = data[ch]
    df["label"] = labels
    df.to_csv(args.out, index=False)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
