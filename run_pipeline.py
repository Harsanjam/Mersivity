#!/usr/bin/env python3
import argparse, os, json, time
import numpy as np

from src.config import Bands, PreprocessConfig, WindowConfig, TFConfig, SonificationConfig
from src.io import load_csv
from src.preprocess import preprocess
from src.features import extract_features
from src.timefreq import compute_stft, compute_cwt_morlet
from src.sonify import render_sonification
from src.plots import (
    plot_psd_before_after, plot_bandpowers, plot_confusion_matrix,
    plot_tf, plot_latency
)

def parse_args():
    ap = argparse.ArgumentParser(description="Milestones 1&2: preprocess -> features -> TF compare -> classify -> sonify.")
    ap.add_argument("--input", required=True, help="CSV with time + channels + optional label.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mains", type=float, default=60.0)
    ap.add_argument("--window_sec", type=float, default=2.0)
    ap.add_argument("--hop_sec", type=float, default=0.25)
    ap.add_argument("--no_reref", action="store_true")
    ap.add_argument("--skip_ml", action="store_true")
    ap.add_argument("--tf_compare", action="store_true", help="Generate STFT and Wavelet plots on a representative segment.")
    ap.add_argument("--benchmark", action="store_true", help="Measure compute time per hop (responsiveness).")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.perf_counter()
    rec0 = load_csv(args.input)

    pre_cfg = PreprocessConfig(mains=args.mains, reref_average=(not args.no_reref))
    rec = preprocess(rec0, pre_cfg)
    plot_psd_before_after(rec0, rec, os.path.join(args.outdir, "psd_before_after.png"))

    bands = Bands()
    win = WindowConfig(window_sec=args.window_sec, hop_sec=args.hop_sec)

    # Benchmark responsiveness: time feature extraction per hop (approx)
    lat_ms = []
    if args.benchmark:
        fs = rec.sfreq
        win_n = int(win.window_sec * fs)
        hop_n = int(win.hop_sec * fs)
        n = rec.data.shape[1]
        for start in range(0, max(1, n - win_n + 1), hop_n):
            end = start + win_n
            seg = rec.data[:, start:end]
            x = np.mean(seg, axis=0)
            t_a = time.perf_counter()
            # compute only key features (alpha/theta/beta + ratio) to emulate streaming
            from src.features import bandpower_welch
            _ = bandpower_welch(x, fs, 4, 8)
            _a = bandpower_welch(x, fs, 8, 13)
            _b = bandpower_welch(x, fs, 13, 30)
            _ = _b / (_a + 1e-12)
            lat_ms.append((time.perf_counter() - t_a) * 1000.0)
        plot_latency(lat_ms, os.path.join(args.outdir, "latency_ms.png"))

    df = extract_features(rec, bands=bands, win=win)
    df.to_csv(os.path.join(args.outdir, "features_windows.csv"), index=False)
    plot_bandpowers(df, os.path.join(args.outdir, "bandpowers.png"))

    # Time-frequency comparison: STFT vs Wavelet on representative segment
    tf_stats = {}
    if args.tf_compare:
        tf_cfg = TFConfig()
        fs = rec.sfreq
        seg_n = int(tf_cfg.segment_sec * fs)
        seg_n = min(seg_n, rec.data.shape[1])
        x = np.mean(rec.data[:, :seg_n], axis=0)

        t_a = time.perf_counter()
        f_s, t_s, P_s = compute_stft(x, fs, win_sec=tf_cfg.stft_win_sec, overlap_frac=tf_cfg.stft_overlap)
        tf_stats["stft_ms"] = (time.perf_counter() - t_a) * 1000.0
        plot_tf(f_s, t_s, P_s, os.path.join(args.outdir, "tf_stft.png"), "STFT time-frequency (Fourier)")

        t_a = time.perf_counter()
        f_w, t_w, P_w = compute_cwt_morlet(x, fs, fmin=1.0, fmax=45.0, num_freqs=80, w0=tf_cfg.wavelet_w0)
        tf_stats["wavelet_ms"] = (time.perf_counter() - t_a) * 1000.0
        plot_tf(f_w, t_w, P_w, os.path.join(args.outdir, "tf_wavelet.png"), "Wavelet scalogram (Morlet CWT)")

    # Sonification
    out_wav = os.path.join(args.outdir, "sonification.wav")
    son_cfg = SonificationConfig()
    render_sonification(df, out_wav=out_wav, cfg=son_cfg, hop_sec=args.hop_sec)

    metrics = {
        "input": args.input,
        "sfreq": float(rec.sfreq),
        "n_channels": int(rec.data.shape[0]),
        "n_windows": int(len(df)),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "benchmark": {
            "enabled": bool(args.benchmark),
            "mean_ms_per_hop": float(np.mean(lat_ms)) if lat_ms else None,
            "p95_ms_per_hop": float(np.percentile(lat_ms, 95)) if lat_ms else None,
            "real_time_factor": float((args.hop_sec*1000.0) / (np.mean(lat_ms)+1e-9)) if lat_ms else None
        },
        "tf_compare": tf_stats
    }

    # Classification (baseline)
    if (not args.skip_ml) and ("label" in df.columns) and (df["label"].astype(str).str.strip() != "").any():
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

        y = df["label"].astype(str).str.strip().str.lower().to_numpy()
        mask = np.isin(y, ["calm", "stress", "stressed"])
        y = np.where(y == "stressed", "stress", y)
        df2 = df.loc[mask].copy()
        y = y[mask]

        if len(df2) >= 30 and len(set(y)) >= 2:
            X = df2[[
                "bp_delta","bp_theta","bp_alpha","bp_beta","bp_gamma",
                "beta_alpha","theta_beta"
            ]].to_numpy(dtype=float)

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            clf = Pipeline([("scaler", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=2000))])
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)

            acc = float(accuracy_score(yte, pred))
            f1 = float(f1_score(yte, pred, average="macro"))
            cm = confusion_matrix(yte, pred, labels=["calm","stress"])
            plot_confusion_matrix(cm, ["calm","stress"], os.path.join(args.outdir, "confusion_matrix.png"))

            metrics.update({
                "accuracy": acc,
                "f1_macro": f1,
                "label_counts": {k: int((y==k).sum()) for k in set(y)},
                "classification_report": classification_report(yte, pred, output_dict=True),
                "confusion_matrix": cm.tolist()
            })
        else:
            metrics["ml_note"] = "Not enough labeled windows (need >=30 windows and both classes)."

    metrics["total_runtime_s"] = float(time.perf_counter() - t0)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone. Outputs:", args.outdir)

if __name__ == "__main__":
    main()
