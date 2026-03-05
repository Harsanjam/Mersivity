# Multimodal Mersivity (Milestones 1 & 2) — Offline EEG Pipeline

This repo is a **milestone-ready** baseline for:
- **Milestone 1:** dataset handling + reproducible preprocessing + band features + **STFT vs Wavelet comparison**
- **Milestone 2:** a **sonification pipeline** (EEG features → audio parameters) + **responsiveness benchmarking**

It runs fully offline on a CSV EEG recording. A synthetic demo dataset is included so you can run immediately.

---

## Quickstart (runs immediately)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python scripts/make_synthetic_dataset.py --out data/sample_stress_calm.csv --seconds 120 --sfreq 256
python run_pipeline.py --input data/sample_stress_calm.csv --outdir outputs/demo --mains 60 --tf_compare --benchmark
```

Outputs:
- `outputs/demo/psd_before_after.png`
- `outputs/demo/bandpowers.png`
- `outputs/demo/tf_stft.png`
- `outputs/demo/tf_wavelet.png`
- `outputs/demo/latency_ms.png`
- `outputs/demo/confusion_matrix.png` (if labels are sufficient)
- `outputs/demo/sonification.wav`
- `outputs/demo/metrics.json`
- `outputs/demo/features_windows.csv`

---

## Using a public dataset (DEAP — calm vs stress proxy)
DEAP provides EEG + arousal ratings per trial. Define:
- calm: low arousal (<= calm_thr)
- stress: high arousal (>= stress_thr)

1) Download DEAP (manual step due to licensing / distribution).
2) Convert a subject file:
```bash
python scripts/deap_to_csv.py   --deap_file /path/to/data_preprocessed_python/s01.dat   --out_csv data/deap_s01.csv   --calm_thr 4.0 --stress_thr 6.0
```
3) Run the same pipeline:
```bash
python run_pipeline.py --input data/deap_s01.csv --outdir outputs/deap_s01 --mains 50 --tf_compare --benchmark
```

---

## CSV format
Required columns:
- `time` seconds
- one or more channel columns `ch1`, `ch2`, ...
Optional:
- `label` in {calm, stress}

---

## LaTeX report (Milestones 1 & 2)
A milestone-ready report lives in `report/`:
```bash
cd report
pdflatex main.tex
pdflatex main.tex
```
