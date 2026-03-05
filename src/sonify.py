import numpy as np
import soundfile as sf
from .config import SonificationConfig

def _fade_envelope(n: int, sr: int, fade_ms: float) -> np.ndarray:
    fade_n = int(sr * (fade_ms / 1000.0))
    fade_n = max(1, min(fade_n, n//2))
    env = np.ones(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    env[:fade_n] = ramp
    env[-fade_n:] = ramp[::-1]
    return env

def _quantize_major(freq_hz: float) -> float:
    scale = (0,2,4,5,7,9,11)
    midi = 69 + 12*np.log2(freq_hz / 440.0)
    base = np.floor(midi/12)*12
    candidates = np.array([base+s for s in scale] + [base+12+s for s in scale])
    m = candidates[np.argmin(np.abs(candidates - midi))]
    return float(440.0 * (2 ** ((m - 69) / 12)))

def _quantize_minor(freq_hz: float) -> float:
    scale = (0,2,3,5,7,8,10)
    midi = 69 + 12*np.log2(freq_hz / 440.0)
    base = np.floor(midi/12)*12
    candidates = np.array([base+s for s in scale] + [base+12+s for s in scale])
    m = candidates[np.argmin(np.abs(candidates - midi))]
    return float(440.0 * (2 ** ((m - 69) / 12)))

def render_sonification(df, out_wav: str, cfg: SonificationConfig, hop_sec: float):
    sr = int(cfg.sr)
    hop_n = int(sr * hop_sec)
    if hop_n <= 0:
        raise ValueError("hop_sec too small.")

    env = _fade_envelope(hop_n, sr, cfg.fade_ms)
    chunks = []
    phase = 0.0

    for _, r in df.iterrows():
        alpha = float(r.get("bp_alpha_norm", 0.0))
        theta = float(r.get("bp_theta_norm", 0.0))
        bright = float(r.get("beta_alpha_norm", 0.0))
        total = float(
            r.get("bp_delta_norm",0.0)+r.get("bp_theta_norm",0.0)+r.get("bp_alpha_norm",0.0)+
            r.get("bp_beta_norm",0.0)+r.get("bp_gamma_norm",0.0)
        )
        amp = np.clip(0.10 + 0.18*(total/5.0), 0.06, 0.30)

        base = cfg.base_hz + cfg.range_hz * alpha
        label = str(r.get("label","")).strip().lower()
        f0 = _quantize_minor(base) if label in ("stress","stressed") else _quantize_major(base)

        t = np.arange(hop_n, dtype=np.float32) / sr
        vib_depth = (cfg.vib_depth_cents * theta) / 1200.0
        vib = 2 ** (vib_depth * np.sin(2*np.pi*cfg.vib_rate_hz*t))
        f = f0 * vib

        phase_inc = 2*np.pi * f / sr
        ph = phase + np.cumsum(phase_inc)
        phase = float(ph[-1] % (2*np.pi))

        tone = np.sin(ph).astype(np.float32)
        sat = np.tanh(1.6 * tone)
        tone = (1.0 - bright) * tone + bright * sat

        noise = np.random.randn(hop_n).astype(np.float32)
        y = tone + (cfg.noise_gain * bright) * noise

        y *= env
        y *= (amp * cfg.master_gain)
        y = np.tanh(y)

        chunks.append(y)

    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(out_wav, audio, sr)
