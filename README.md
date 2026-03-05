# Multimodal Mersivity (Milestones 1–3) — Offline + Real-Time EEG Pipeline

This repository implements a **multimodal EEG processing pipeline** for the *Mersivity* project.

The system supports:

- **Milestone 1:** Offline EEG preprocessing and spectral analysis
- **Milestone 2:** EEG sonification pipeline and responsiveness benchmarking
- **Milestone 3:** Real-time EEG streaming from a Muse headset with Unity visualization**

The project demonstrates a **complete brain-computer interface (BCI) pipeline** from EEG acquisition to real-time audiovisual feedback.

---

## System Architecture

```text
Muse EEG Headset
        ↓
Lab Streaming Layer (LSL)
        ↓
Python Signal Processing
        ↓
Feature Extraction (Alpha / Beta / Theta)
        ↓
UDP Streaming
        ↓
Unity Visualization + Sonification
```

This architecture enables **real-time neuroadaptive interaction**.

---

## Milestone Overview

### Milestone 1 — Dataset Processing

Reproducible EEG preprocessing and spectral analysis.

Includes:

- filtering + artifact reduction
- power spectral density analysis
- band power feature extraction
- **STFT vs Wavelet comparison**

Outputs:

```text
psd_before_after.png
bandpowers.png
tf_stft.png
tf_wavelet.png
```

---

### Milestone 2 — Sonification

Mapping EEG features to audio parameters.

Pipeline:

```text
EEG features
    ↓
audio synthesis
    ↓
neurofeedback soundscape
```

Outputs:

```text
sonification.wav
latency_ms.png
metrics.json
features_windows.csv
```

Benchmarks responsiveness of the sonification system.

---

### Milestone 3 — Real-Time Muse EEG Interface

Milestone 3 extends the pipeline to **live EEG streaming from a Muse headset**.

The system:

- acquires EEG via **Muse**
- streams data using **Lab Streaming Layer**
- extracts **alpha, beta, theta band features**
- classifies brain states
- streams features to **Unity in real time**

Example console output:

```text
Alpha: 0.31
Beta: 0.54
Theta: 0.22
State: STRESS
```

These features drive **real-time visualization and audio feedback**.

---

## Project Structure

```text
Mersivity/
│
├── run_pipeline.py
├── realtime_muse.py
│
├── data/
│   └── sample_stress_calm.csv
│
├── scripts/
│   ├── make_synthetic_dataset.py
│   └── deap_to_csv.py
│
├── outputs/
│
├── report/
│   └── main.tex
│
└── requirements.txt
```

---

## Quickstart (Offline Pipeline — Milestones 1 & 2)

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Generate synthetic dataset:

```bash
python scripts/make_synthetic_dataset.py \
  --out data/sample_stress_calm.csv \
  --seconds 120 \
  --sfreq 256
```

Run pipeline:

```bash
python run_pipeline.py \
  --input data/sample_stress_calm.csv \
  --outdir outputs/demo \
  --mains 60 \
  --tf_compare \
  --benchmark
```

Outputs:

```text
outputs/demo/psd_before_after.png
outputs/demo/bandpowers.png
outputs/demo/tf_stft.png
outputs/demo/tf_wavelet.png
outputs/demo/latency_ms.png
outputs/demo/confusion_matrix.png
outputs/demo/sonification.wav
outputs/demo/metrics.json
outputs/demo/features_windows.csv
```

---

## Using a Public Dataset (DEAP)

DEAP provides EEG recordings with arousal ratings.

Stress proxy labels:

```text
calm   = arousal ≤ calm_thr
stress = arousal ≥ stress_thr
```

Convert dataset:

```bash
python scripts/deap_to_csv.py \
  --deap_file /path/to/data_preprocessed_python/s01.dat \
  --out_csv data/deap_s01.csv \
  --calm_thr 4.0 \
  --stress_thr 6.0
```

Run pipeline:

```bash
python run_pipeline.py \
  --input data/deap_s01.csv \
  --outdir outputs/deap_s01 \
  --mains 50 \
  --tf_compare \
  --benchmark
```

---

## Real-Time Muse Setup (Milestone 3)

### Install Muse streaming tools

```bash
pip install muselsl
```

### Start EEG stream

Turn on the Muse headset and run:

```bash
muselsl stream
```

This broadcasts EEG via **Lab Streaming Layer (LSL)**.

### Run the real-time pipeline

```bash
python realtime_muse.py
```

The system performs:

```text
EEG acquisition
signal filtering
band power extraction
brain state detection
UDP streaming to Unity
real-time sonification
```

---

## Unity Visualization

Unity receives EEG features via **UDP**.

Data format:

```text
alpha,beta,theta
```

Example:

```text
0.23,0.61,0.18
```

Unity maps the features to visual parameters:

| EEG Feature | Unity Mapping   |
|-------------|-----------------|
| Alpha       | sphere scaling  |
| Beta        | sphere rotation |
| Theta       | color modulation |

Pipeline:

```text
Python EEG pipeline
        ↓
UDP (port 9000)
        ↓
Unity EEGReceiver
        ↓
Real-time visualization
```

---

## CSV Format

Required columns:

```text
time
ch1
ch2
...
```

Optional:

```text
label (calm / stress)
```

---

## LaTeX Report

Compile the milestone report:

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```

The report includes:

- preprocessing pipeline
- STFT vs Wavelet comparison
- sonification methodology
- real-time system architecture
- Unity neuroadaptive visualization

---

## Key Contributions

This project demonstrates:

- reproducible EEG preprocessing
- multimodal neurofeedback
- real-time brain-computer interaction
- neuroadaptive XR visualization

---

## Future Work

Possible extensions:

- Meta Quest VR neurofeedback
- adaptive XR environments
- deep learning EEG classification
- multimodal biosignal integration
