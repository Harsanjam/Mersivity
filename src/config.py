from dataclasses import dataclass

@dataclass(frozen=True)
class Bands:
    delta: tuple = (1.0, 4.0)
    theta: tuple = (4.0, 8.0)
    alpha: tuple = (8.0, 13.0)
    beta:  tuple = (13.0, 30.0)
    gamma: tuple = (30.0, 45.0)

@dataclass(frozen=True)
class PreprocessConfig:
    l_freq: float = 1.0
    h_freq: float = 45.0
    mains: float = 60.0
    notch_q: float = 30.0
    reref_average: bool = True
    # basic artifact handling
    clip_mad_z: float = 6.0  # winsorize beyond this robust z-score
    detrend: bool = True

@dataclass(frozen=True)
class WindowConfig:
    window_sec: float = 2.0
    hop_sec: float = 0.25

@dataclass(frozen=True)
class TFConfig:
    segment_sec: float = 10.0
    stft_win_sec: float = 2.0
    stft_overlap: float = 0.75  # fraction overlap
    wavelet_w0: float = 6.0     # morlet width

@dataclass(frozen=True)
class SonificationConfig:
    sr: int = 44100
    base_hz: float = 220.0
    range_hz: float = 660.0
    noise_gain: float = 0.10
    vib_rate_hz: float = 5.0
    vib_depth_cents: float = 25.0
    master_gain: float = 0.25
    fade_ms: float = 10.0
