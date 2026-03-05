import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from .io import EEGRecording

def plot_psd_before_after(before: EEGRecording, after: EEGRecording, out_png: str, fmin=0.5, fmax=45.0):
    def mean_psd(rec):
        psds = []
        for ch in rec.data:
            f, p = welch(ch, fs=rec.sfreq, nperseg=min(len(ch), int(rec.sfreq*2)))
            m = (f >= fmin) & (f <= fmax)
            psds.append(p[m])
        return f[m], np.mean(np.vstack(psds), axis=0)

    f1, p1 = mean_psd(before)
    f2, p2 = mean_psd(after)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.semilogy(f1, p1, label="before")
    ax.semilogy(f2, p2, label="after")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD sanity check: before vs after preprocessing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_bandpowers(df, out_png: str):
    t = 0.5*(df["t_start"].to_numpy() + df["t_end"].to_numpy())
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(t, df["bp_theta"], label="theta")
    ax.plot(t, df["bp_alpha"], label="alpha")
    ax.plot(t, df["bp_beta"], label="beta")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bandpower")
    ax.set_title("Bandpower over time (windowed Welch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_confusion_matrix(cm, labels, out_png: str):
    fig, ax = plt.subplots(figsize=(4.5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_tf(f, t, P, out_png: str, title: str, fmax: float=45.0):
    fig, ax = plt.subplots(figsize=(9,4))
    m = (f <= fmax)
    f2 = f[m]
    P2 = P[m, :]
    im = ax.pcolormesh(t, f2, 10*np.log10(P2 + 1e-12), shading='auto')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_latency(lat_ms, out_png: str):
    lat_ms = np.asarray(lat_ms)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(lat_ms)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Compute time (ms)")
    ax.set_title("Pipeline responsiveness: compute time per hop (ms)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
