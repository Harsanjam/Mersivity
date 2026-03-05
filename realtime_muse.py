#!/usr/bin/env python3
"""
Mersivity - Real-Time Muse S EEG System
Milestones 1, 2 & 3: Live preprocessing → feature extraction → sonification → SWIM visualization

Architecture mirrors the offline pipeline (src/) but adapted for streaming:
  - Circular buffer with sliding window (same Welch bandpower as features.py)
  - Running percentile normalizer (replaces offline norm_log — no future data leakage)
  - Real-time audio synthesis (same mapping logic as sonify.py)
  - SWIM particle visualizer (from visualize_swim.py, fixed & enhanced)

Usage:
    # Terminal 1
    muselsl stream

    # Terminal 2
    python realtime_muse.py

    # Optional flags
    python realtime_muse.py --width 1920 --height 1080 --fps 60 --calibrate 30
"""

import numpy as np
import time
import argparse
import math
import threading
import queue
import socket
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import pygame
import pygame.sndarray
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.integrate import trapezoid
from pylsl import StreamInlet, resolve_byprop

# ============================================================================
# CONFIGURATION  (mirrors src/config.py)
# ============================================================================

@dataclass
class RTConfig:
    # --- LSL / hardware ---
    SFREQ: float = 256.0          # Muse S sample rate
    N_CHANNELS: int = 4           # TP9, AF7, AF8, TP10

    # --- Preprocessing (mirrors PreprocessConfig) ---
    L_FREQ: float = 1.0
    H_FREQ: float = 45.0
    MAINS: float = 60.0           # 50 Hz for Europe
    NOTCH_Q: float = 30.0
    CLIP_MAD_Z: float = 6.0

    # --- Feature extraction (mirrors WindowConfig) ---
    WINDOW_SEC: float = 2.0
    HOP_SEC: float = 0.25         # update rate = 4 Hz

    # --- EEG bands (mirrors Bands) ---
    DELTA: Tuple = (1.0, 4.0)
    THETA: Tuple = (4.0, 8.0)
    ALPHA: Tuple = (8.0, 13.0)
    BETA:  Tuple = (13.0, 30.0)
    GAMMA: Tuple = (30.0, 45.0)

    # --- Normaliser ---
    NORM_HISTORY: int = 200       # how many windows to keep for running percentiles
    NORM_LO_PCT: float = 5.0
    NORM_HI_PCT: float = 95.0

    # --- Audio (mirrors SonificationConfig) ---
    AUDIO_SR: int = 44100
    BASE_HZ: float = 220.0
    RANGE_HZ: float = 660.0
    VIB_RATE_HZ: float = 5.0
    VIB_DEPTH_CENTS: float = 25.0
    NOISE_GAIN: float = 0.10
    MASTER_GAIN: float = 0.25
    FADE_MS: float = 10.0

    # --- Visualisation ---
    VIZ_WIDTH: int = 1200
    VIZ_HEIGHT: int = 800
    FPS: int = 60
    N_PARTICLES: int = 60

    # --- Calibration ---
    CALIBRATE_SEC: float = 20.0   # collect data silently before display

    # --- Unity UDP bridge ---
    UNITY_UDP_ENABLED: bool = True
    UNITY_IP: str = "127.0.0.1"
    UNITY_PORT: int = 9000


CFG = RTConfig()


# ============================================================================
# SIGNAL PROCESSING  (mirrors preprocess.py)
# ============================================================================

class RTPreprocessor:
    """Per-sample bandpass + notch filter using IIR in direct-form II.
    scipy filtfilt needs a block; we use lfilter with zi (causal, real-time safe).
    """

    def __init__(self, sfreq: float = CFG.SFREQ):
        from scipy.signal import lfilter_zi, lfilter
        self._lfilter = lfilter
        nyq = 0.5 * sfreq

        # Bandpass
        self._bp_b, self._bp_a = butter(4, [CFG.L_FREQ / nyq, CFG.H_FREQ / nyq], btype='bandpass')
        # Notch(es)
        self._notch_filters = []
        for f0 in [CFG.MAINS, 2 * CFG.MAINS]:
            if CFG.L_FREQ < f0 < CFG.H_FREQ:
                b, a = iirnotch(w0=f0, Q=CFG.NOTCH_Q, fs=sfreq)
                self._notch_filters.append((b, a))

        n_ch = CFG.N_CHANNELS
        self._bp_zi  = [lfilter_zi(self._bp_b,  self._bp_a)  * 0 for _ in range(n_ch)]
        self._notch_zi = [[lfilter_zi(b, a) * 0 for b, a in self._notch_filters]
                          for _ in range(n_ch)]

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Filter one sample (n_ch,) → filtered (n_ch,)."""
        out = np.empty(CFG.N_CHANNELS, dtype=np.float32)
        for ch in range(CFG.N_CHANNELS):
            x = np.array([sample[ch]], dtype=np.float64)

            # notch
            for fi, (b, a) in enumerate(self._notch_filters):
                x, self._notch_zi[ch][fi] = self._lfilter(b, a, x, zi=self._notch_zi[ch][fi])

            # bandpass
            x, self._bp_zi[ch] = self._lfilter(self._bp_b, self._bp_a, x, zi=self._bp_zi[ch])
            out[ch] = x[0]
        return out


# ============================================================================
# FEATURE EXTRACTION  (mirrors features.py bandpower_welch + norm_log)
# ============================================================================

def bandpower_welch(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(pxx[mask], f[mask]))


class RunningPercentileNorm:
    """Online version of features.py norm_log.
    Keeps a rolling history of log-bandpower values and normalises
    using running [5th, 95th] percentiles — no future data needed.
    """

    def __init__(self, history_len: int = CFG.NORM_HISTORY):
        self._history: deque = deque(maxlen=history_len)

    def push_and_norm(self, raw: float) -> float:
        log_val = float(np.log(max(raw, 1e-12)))
        self._history.append(log_val)
        if len(self._history) < 10:
            return 0.5   # not enough data yet
        arr = np.array(self._history)
        lo = float(np.percentile(arr, CFG.NORM_LO_PCT))
        hi = float(np.percentile(arr, CFG.NORM_HI_PCT))
        return float(np.clip((log_val - lo) / (hi - lo + 1e-12), 0.0, 1.0))

    @property
    def ready(self) -> bool:
        return len(self._history) >= 10


class EEGFeatureExtractor:
    """Sliding window feature extractor driven by the sample buffer."""

    def __init__(self, sfreq: float = CFG.SFREQ):
        self.sfreq = sfreq
        self.win_n  = int(CFG.WINDOW_SEC * sfreq)
        self.hop_n  = int(CFG.HOP_SEC    * sfreq)
        self._buf   = [deque(maxlen=self.win_n) for _ in range(CFG.N_CHANNELS)]
        self._since_update = 0

        # One normaliser per band feature
        self._norms = {k: RunningPercentileNorm() for k in
                       ['delta', 'theta', 'alpha', 'beta', 'gamma', 'beta_alpha', 'theta_beta']}

        # Exposed features (normalised 0–1)
        self.alpha_norm      = 0.5
        self.beta_norm       = 0.5
        self.theta_norm      = 0.5
        self.beta_alpha_norm = 0.5
        self.theta_beta_norm = 0.5
        self.ready           = False

    def push(self, filtered_sample: np.ndarray) -> bool:
        """Add one filtered sample. Returns True when features were updated."""
        for ch in range(CFG.N_CHANNELS):
            self._buf[ch].append(filtered_sample[ch])
        self._since_update += 1

        if self._since_update >= self.hop_n and len(self._buf[0]) == self.win_n:
            self._update()
            self._since_update = 0
            return True
        return False

    def _update(self):
        data  = np.array([list(b) for b in self._buf], dtype=np.float32)
        # average-reference across channels (mirrors preprocess.py reref_average)
        data  = data - data.mean(axis=0, keepdims=True)
        x     = data.mean(axis=0)

        bp = {
            'delta':      bandpower_welch(x, self.sfreq, *CFG.DELTA),
            'theta':      bandpower_welch(x, self.sfreq, *CFG.THETA),
            'alpha':      bandpower_welch(x, self.sfreq, *CFG.ALPHA),
            'beta':       bandpower_welch(x, self.sfreq, *CFG.BETA),
            'gamma':      bandpower_welch(x, self.sfreq, *CFG.GAMMA),
        }
        bp['beta_alpha'] = bp['beta']  / (bp['alpha'] + 1e-12)
        bp['theta_beta'] = bp['theta'] / (bp['beta']  + 1e-12)

        for k, v in bp.items():
            norm = self._norms[k].push_and_norm(v)
            if k == 'alpha':      self.alpha_norm      = norm
            elif k == 'beta':     self.beta_norm       = norm
            elif k == 'theta':    self.theta_norm      = norm
            elif k == 'beta_alpha': self.beta_alpha_norm = norm
            elif k == 'theta_beta': self.theta_beta_norm = norm

        self.ready = self._norms['alpha'].ready


# ============================================================================
# AUDIO SYNTHESIS  (mirrors sonify.py render_sonification logic)
# ============================================================================

def _fade_envelope(n: int, fade_ms: float, sr: int) -> np.ndarray:
    fade_n = max(1, min(int(sr * fade_ms / 1000.0), n // 2))
    env = np.ones(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    env[:fade_n]  = ramp
    env[-fade_n:] = ramp[::-1]
    return env


def _quantize_scale(freq_hz: float, minor: bool = False) -> float:
    """Snap frequency to major or natural minor scale (mirrors sonify.py)."""
    scale = (0, 2, 3, 5, 7, 8, 10) if minor else (0, 2, 4, 5, 7, 9, 11)
    midi  = 69 + 12 * np.log2(max(freq_hz, 1.0) / 440.0)
    base  = np.floor(midi / 12) * 12
    cands = np.array([base + s for s in scale] + [base + 12 + s for s in scale])
    m     = cands[np.argmin(np.abs(cands - midi))]
    return float(440.0 * (2 ** ((m - 69) / 12)))


class RTAudioSynth:
    """Generates one hop-sized audio chunk from current EEG features.
    Mirrors sonify.py but stateful (phase continuity between hops).
    """

    def __init__(self):
        pygame.mixer.pre_init(frequency=CFG.AUDIO_SR, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        self._phase = 0.0
        self._hop_n = int(CFG.AUDIO_SR * CFG.HOP_SEC)
        self._env   = _fade_envelope(self._hop_n, CFG.FADE_MS, CFG.AUDIO_SR)

    def synthesise(self,
                   alpha_norm: float,
                   theta_norm: float,
                   beta_alpha_norm: float,
                   theta_beta_norm: float,
                   is_stress: bool = False) -> np.ndarray:
        n  = self._hop_n
        sr = CFG.AUDIO_SR
        t  = np.arange(n, dtype=np.float32) / sr

        # Pitch: alpha drives base frequency (calm → higher pitch)
        base = CFG.BASE_HZ + CFG.RANGE_HZ * alpha_norm
        f0   = _quantize_scale(base, minor=is_stress)

        # Vibrato: theta drives depth
        vib_depth = (CFG.VIB_DEPTH_CENTS * theta_norm) / 1200.0
        vib = 2.0 ** (vib_depth * np.sin(2 * np.pi * CFG.VIB_RATE_HZ * t))

        # Phase-continuous synthesis
        phase_inc = 2 * np.pi * (f0 * vib) / sr
        ph = self._phase + np.cumsum(phase_inc)
        self._phase = float(ph[-1] % (2 * np.pi))

        tone = np.sin(ph).astype(np.float32)

        # Brightness / saturation: beta_alpha drives timbre
        bright = beta_alpha_norm
        sat    = np.tanh(1.6 * tone)
        tone   = (1.0 - bright) * tone + bright * sat

        # Noise texture
        noise = np.random.randn(n).astype(np.float32)
        y = tone + CFG.NOISE_GAIN * bright * noise

        # Amplitude: louder when more relaxed (total power proxy)
        total = (alpha_norm + theta_norm + beta_alpha_norm + theta_beta_norm) / 4.0
        amp   = np.clip(0.10 + 0.18 * total, 0.06, 0.30)

        y = y * self._env * amp * CFG.MASTER_GAIN
        y = np.tanh(y)
        return y

    def play(self, mono: np.ndarray):
        stereo = np.stack([mono, mono], axis=1)
        pcm    = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
        try:
            sound = pygame.sndarray.make_sound(pcm)
            sound.play()
        except Exception:
            pass


# ============================================================================
# SWIM VISUALISER  (visualize_swim.py ported + fixed + enhanced for live use)
# ============================================================================

@dataclass
class Particle:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    color: Tuple[int, int, int] = (100, 150, 255)
    size: float = 3.0
    trail: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def update(self, dt: float, fx: float, fy: float):
        self.vx = self.vx * 0.98 + fx * dt
        self.vy = self.vy * 0.98 + fy * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.trail.append((int(self.x), int(self.y)))

    def draw(self, screen: pygame.Surface):
        trail = list(self.trail)
        n = len(trail)
        if n > 1:
            for i in range(1, n):
                a = int(255 * i / n)
                col = (max(30, int(self.color[0] * a / 255)),
                       max(30, int(self.color[1] * a / 255)),
                       max(30, int(self.color[2] * a / 255)))
                thick = max(1, int(self.size * a / 255))
                try:
                    pygame.draw.line(screen, col, trail[i - 1], trail[i], thick)
                except Exception:
                    pass
        try:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), max(1, int(self.size)))
        except Exception:
            pass


class SWIMVisualizer:
    """SWIM-inspired rotary particle flow — live EEG driven."""

    CALM_COLORS   = [(100, 150, 255), (150, 200, 255), (200, 220, 255)]
    STRESS_COLORS = [(255, 100, 100), (255, 150,  50), (255, 200, 100)]

    def __init__(self, width: int = CFG.VIZ_WIDTH, height: int = CFG.VIZ_HEIGHT):
        self.width  = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Mersivity — Real-Time EEG · SWIM Multimodal Feedback")
        self.clock  = pygame.time.Clock()

        self._rotation = 0.0
        self._radius   = 200.0

        # EEG state (updated from main loop)
        self.alpha = 0.5
        self.beta  = 0.5
        self.theta = 0.5
        self.beta_alpha = 0.5
        self.is_stress  = False
        self.calibrating = True

        self.particles: List[Particle] = []
        self._init_particles(CFG.N_PARTICLES)

    def _init_particles(self, n: int):
        self.particles = []
        for i in range(n):
            angle = (i / n) * 2 * math.pi
            x = self.width  / 2 + self._radius * math.cos(angle)
            y = self.height / 2 + self._radius * math.sin(angle)
            color = self.CALM_COLORS[i % len(self.CALM_COLORS)]
            self.particles.append(Particle(x=x, y=y, color=color))

    def update(self, dt: float):
        # Rotation speed: alpha drives calm slow rotation
        rot_speed = 0.3 + 1.7 * self.alpha
        self._rotation += rot_speed * dt

        # Orbit radius: beta drives expansion (stress = bigger, more chaotic)
        target_r = 120.0 + 200.0 * self.beta
        self._radius += (target_r - self._radius) * 0.05

        cx, cy = self.width / 2, self.height / 2
        target_colors = self.STRESS_COLORS if self.is_stress else self.CALM_COLORS

        for i, p in enumerate(self.particles):
            offset = (i / len(self.particles)) * 2 * math.pi
            angle  = self._rotation + offset
            spiral = 1.0 + 0.12 * self.theta * math.sin(angle * 2)

            tx = cx + self._radius * spiral * math.cos(angle)
            ty = cy + self._radius * spiral * math.sin(angle)

            force_mag = 80.0 + 250.0 * self.beta
            fx = (tx - p.x) * force_mag * 0.001
            fy = (ty - p.y) * force_mag * 0.001
            p.update(dt, fx, fy)

            # Trail length: theta drives longer trails (meditation)
            p.trail = deque(p.trail, maxlen=int(15 + 45 * self.theta))

            # Size: alpha drives particle size
            p.size = 1.5 + 4.0 * self.alpha

            # Colour lerp
            tc = target_colors[i % len(target_colors)]
            p.color = tuple(int(p.color[j] * 0.95 + tc[j] * 0.05) for j in range(3))

    def draw(self):
        # Gradient background
        if self.is_stress:
            bg_top, bg_bot = (40, 20, 20), (60, 25, 25)
        else:
            bg_top, bg_bot = (18, 18, 38), (25, 30, 60)

        # Fast fill then draw gradient strips
        self.screen.fill(bg_top)
        step = 4
        for y in range(0, self.height, step):
            b = y / self.height
            c = tuple(int(bg_top[i] * (1 - b) + bg_bot[i] * b) for i in range(3))
            pygame.draw.rect(self.screen, c, (0, y, self.width, step))

        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Central ring
        ring_col = self.STRESS_COLORS[0] if self.is_stress else self.CALM_COLORS[0]
        pygame.draw.circle(self.screen, ring_col,
                           (self.width // 2, self.height // 2),
                           int(15 + 25 * self.alpha), 3)

        # HUD
        self._draw_hud()
        pygame.display.flip()

    def _draw_hud(self):
        font  = pygame.font.Font(None, 34)
        sfont = pygame.font.Font(None, 24)

        if self.calibrating:
            msg = font.render("Calibrating… put on headset and relax", True, (220, 200, 100))
            self.screen.blit(msg, (self.width // 2 - msg.get_width() // 2, self.height // 2 + 80))

        title = font.render("Mersivity — Real-Time EEG", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))

        state_str = "STRESS" if self.is_stress else "CALM"
        state_col = (255, 120, 80) if self.is_stress else (120, 200, 255)
        self.screen.blit(sfont.render(f"State: {state_str}", True, state_col), (20, 58))

        features = [
            ("Alpha  (Relaxation)", self.alpha,      (100, 150, 255)),
            ("Beta   (Alertness)",  self.beta,       (255, 150, 100)),
            ("Theta  (Meditation)", self.theta,      (150, 255, 150)),
            ("β/α ratio",           self.beta_alpha, (255, 200, 100)),
        ]
        y = 90
        for name, val, col in features:
            self.screen.blit(sfont.render(name, True, (190, 190, 190)), (20, y))
            pygame.draw.rect(self.screen, (50, 50, 50),      (20, y + 18, 180, 12))
            pygame.draw.rect(self.screen, col,               (20, y + 18, int(180 * val), 12))
            pygame.draw.rect(self.screen, (100, 100, 100),   (20, y + 18, 180, 12), 1)
            self.screen.blit(sfont.render(f"{val:.2f}", True, (200, 200, 200)), (208, y + 16))
            y += 38

        self.screen.blit(sfont.render("ESC: quit", True, (130, 130, 130)), (20, self.height - 30))


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class MuseRealTimeApp:
    """Orchestrates LSL stream → preprocessor → feature extractor → audio + viz."""

    def __init__(self):
        print("\n" + "=" * 60)
        print("  Mersivity — Real-Time Muse S EEG System")
        print("  Milestones 1 · 2 · 3")
        print("=" * 60)

        pygame.init()

        self.preprocessor = RTPreprocessor(sfreq=CFG.SFREQ)
        self.extractor    = EEGFeatureExtractor(sfreq=CFG.SFREQ)
        self.synth        = RTAudioSynth()
        self.viz          = SWIMVisualizer(width=CFG.VIZ_WIDTH, height=CFG.VIZ_HEIGHT)

        self._audio_q: queue.Queue = queue.Queue(maxsize=8)
        self._running = False
        self._calib_start: Optional[float] = None
        self._last_audio_t = 0.0
        self._unity_socket: Optional[socket.socket] = None

        if CFG.UNITY_UDP_ENABLED:
            self._unity_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"✓ Unity UDP bridge: {CFG.UNITY_IP}:{CFG.UNITY_PORT}")

        print("✓ All components initialised")

    # ------------------------------------------------------------------
    def _connect(self) -> Optional[StreamInlet]:
        print("\nSearching for Muse EEG stream…")
        print("(Make sure  muselsl stream  is running in another terminal)\n")
        try:
            streams = resolve_byprop('type', 'EEG', timeout=12)
        except Exception as e:
            print(f"❌ LSL resolve error: {e}")
            return None

        if not streams:
            print("❌ No EEG stream found.")
            print("   → Terminal 1:  muselsl stream")
            return None

        inlet = StreamInlet(streams[0], max_buflen=2)
        info  = streams[0]
        print(f"✓ Connected  : {info.name()}")
        print(f"✓ Channels   : {info.channel_count()}")
        print(f"✓ Sample rate: {info.nominal_srate()} Hz")
        return inlet

    # ------------------------------------------------------------------
    def _audio_worker(self):
        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.15)
                self.synth.play(chunk)
            except queue.Empty:
                continue

    # ------------------------------------------------------------------
    def _classify_state(self) -> bool:
        """Simple threshold: high beta/alpha ratio → stress."""
        return self.extractor.beta_alpha_norm > 0.65

    # ------------------------------------------------------------------
    def run(self):
        inlet = self._connect()
        if inlet is None:
            pygame.quit()
            return

        print(f"\n⏳ Calibrating for {int(CFG.CALIBRATE_SEC)}s — please put on the headset and relax…")
        print("   Press ESC at any time to quit.\n")

        self._running   = True
        self._calib_start = time.time()

        audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        audio_thread.start()

        try:
            while self._running:
                # ── pygame events ──────────────────────────────────────
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self._running = False

                # ── ingest all pending samples ─────────────────────────
                while True:
                    sample, _ = inlet.pull_sample(timeout=0.0)
                    if sample is None:
                        break
                    raw = np.array(sample[:CFG.N_CHANNELS], dtype=np.float32)
                    filtered = self.preprocessor.process_sample(raw)
                    self.extractor.push(filtered)

                # ── calibration gate ───────────────────────────────────
                elapsed_calib = time.time() - self._calib_start
                calibrating   = elapsed_calib < CFG.CALIBRATE_SEC
                self.viz.calibrating = calibrating

                # ── update viz state ───────────────────────────────────
                self.viz.alpha      = self.extractor.alpha_norm
                self.viz.beta       = self.extractor.beta_norm
                self.viz.theta      = self.extractor.theta_norm
                self.viz.beta_alpha = self.extractor.beta_alpha_norm
                self.viz.is_stress  = (not calibrating) and self._classify_state()

                # ── stream EEG features to Unity (alpha,beta,theta) ───
                if self._unity_socket is not None:
                    msg = (
                        f"{self.extractor.alpha_norm:.6f},"
                        f"{self.extractor.beta_norm:.6f},"
                        f"{self.extractor.theta_norm:.6f}"
                    )
                    try:
                        self._unity_socket.sendto(msg.encode("utf-8"), (CFG.UNITY_IP, CFG.UNITY_PORT))
                    except OSError:
                        pass

                # ── audio synthesis (every HOP_SEC) ───────────────────
                now = time.time()
                if (not calibrating) and self.extractor.ready and \
                   (now - self._last_audio_t >= CFG.HOP_SEC):
                    chunk = self.synth.synthesise(
                        alpha_norm      = self.extractor.alpha_norm,
                        theta_norm      = self.extractor.theta_norm,
                        beta_alpha_norm = self.extractor.beta_alpha_norm,
                        theta_beta_norm = self.extractor.theta_beta_norm,
                        is_stress       = self.viz.is_stress,
                    )
                    try:
                        self._audio_q.put_nowait(chunk)
                    except queue.Full:
                        pass
                    self._last_audio_t = now

                # ── draw ───────────────────────────────────────────────
                dt = self.viz.clock.tick(CFG.FPS) / 1000.0
                if not calibrating or self.extractor.ready:
                    self.viz.update(dt)
                self.viz.draw()

        except KeyboardInterrupt:
            print("\n⚡ Interrupted")
        except Exception as e:
            import traceback
            print(f"\n❌ Runtime error: {e}")
            traceback.print_exc()
        finally:
            self._running = False
            if self._unity_socket is not None:
                self._unity_socket.close()
            print("\n✓ Shutting down…")
            pygame.quit()
            print("✓ Done.\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Mersivity real-time EEG — Milestones 1–3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  muselsl stream                          # Terminal 1
  python realtime_muse.py                 # Terminal 2 (defaults)
  python realtime_muse.py --calibrate 30 --width 1920 --height 1080
        """
    )
    ap.add_argument('--width',     type=int,   default=CFG.VIZ_WIDTH,      help='Window width')
    ap.add_argument('--height',    type=int,   default=CFG.VIZ_HEIGHT,     help='Window height')
    ap.add_argument('--fps',       type=int,   default=CFG.FPS,            help='Target FPS')
    ap.add_argument('--calibrate', type=float, default=CFG.CALIBRATE_SEC,  help='Calibration period (s)')
    ap.add_argument('--mains',     type=float, default=CFG.MAINS,          help='Powerline frequency (50 or 60 Hz)')
    ap.add_argument('--unity-ip',  type=str,   default=CFG.UNITY_IP,       help='Unity UDP target IP')
    ap.add_argument('--unity-port', type=int,  default=CFG.UNITY_PORT,     help='Unity UDP target port')
    ap.add_argument('--no-unity',  action='store_true',                     help='Disable Unity UDP streaming')
    args = ap.parse_args()

    CFG.VIZ_WIDTH     = args.width
    CFG.VIZ_HEIGHT    = args.height
    CFG.FPS           = args.fps
    CFG.CALIBRATE_SEC = args.calibrate
    CFG.MAINS         = args.mains
    CFG.UNITY_IP      = args.unity_ip
    CFG.UNITY_PORT    = args.unity_port
    CFG.UNITY_UDP_ENABLED = not args.no_unity

    MuseRealTimeApp().run()


if __name__ == "__main__":
    main()
