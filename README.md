# BioSentinel

Open-source ML anomaly detection for biomedical time-series data (cold-chain sensors, lab instruments).

---

## The Problem

Cold-chain equipment and laboratory instruments stream sensor readings constantly. A refrigerator storing vaccines or insulin can fail slowly over hours. A tampered sensor can hide a contaminated batch behind fake-normal readings. By the time a manual check catches either failure, the product is usually already ruined.

Simple threshold systems catch sharp spikes but miss sustained slow failures. Advanced commercial systems exist but are proprietary and unauditable. BioSentinel is a fully open and reproducible baseline that any researcher or student can inspect, run, and extend.

---

## Dataset

BioSentinel is evaluated on the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) `machine_temperature_system_failure` dataset:

- 22,695 rows, one reading every 5 minutes
- Date range: December 2, 2013 to February 19, 2014
- Four labeled anomaly windows covering equipment failure events
- 2,268 labeled anomaly points (10.0% of all readings)

The dataset is not included in this repository (the `data/` folder is gitignored). To use it:

```bash
mkdir data
curl -L -o data/machine_temp.csv \
  "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
```

Or download that URL manually and save the file as `data/machine_temp.csv`.

---

## Methodology

Seven detectors are built in order of complexity. Each step is motivated by the specific failure of the one before it.

| Step | Detector | Why it was built |
|------|----------|-----------------|
| 1 | 3-Sigma (Global) | Statistical baseline — anything more than 3 standard deviations from the global mean is flagged. Auditable in two lines of arithmetic. |
| 2 | Rolling 3-Sigma | Uses a local rolling mean instead of the global mean. Expected to adapt better to gradual drift. |
| 3 | EWMA | Exponentially weighted baseline with span=5000 (~17 days). Adapts slowly enough to stay stable through a 48-hour failure. |
| 4 | Hampel Filter | Robust statistical filter using rolling median and MAD instead of mean and std. Known to handle sudden spikes well. |
| 5 | LOF | Local Outlier Factor compares local density to neighbors. Designed to find isolated outliers in arbitrary feature space. |
| 6 | IsolationForest | Tree-based method that isolates points using random cuts. Global distribution, immune to the window-adaptation problem. |
| 7 | IsolationForest + Rolling | Adds one engineered feature: `sustained_dev = value - rolling_mean(360)`. During a 48-hour failure, this stays strongly negative even when the raw temperature looks like a normal dip. **Best-performing model.** |

---

## Results

All metrics are computed against the four NAB ground-truth anomaly windows. No cherry-picking. Detectors that failed are included and explained.

| Detector | Precision | Recall | F1 |
|----------|-----------|--------|----|
| 3-Sigma (Global) | 0.991 | 0.202 | 0.336 |
| Rolling 3-Sigma (w=1000) | 0.283 | 0.088 | 0.134 |
| EWMA (span=5000) | 0.586 | 0.513 | 0.547 |
| Hampel Filter (w=144) | 0.107 | 0.033 | 0.050 |
| LOF (k=20) | 0.109 | 0.131 | 0.119 |
| IsolationForest (raw) | 0.471 | 0.566 | 0.514 |
| **IsolationForest + Rolling (w=360)** | **0.568** | **0.682** | **0.619** |

**Note on the recall ceiling:** 38.8% of labeled anomaly points fall within the normal temperature range because NAB anomaly windows start before the temperature drops. No raw-value-only detector can exceed roughly 61.2% recall. The IsolationForest + Rolling model achieves recall=0.682 by using `sustained_dev` as a second feature, giving the model temporal context the raw value cannot provide.

**Precision-Recall AUC scores** (ranking ability across all thresholds):

| Detector | AUC-PR |
|----------|--------|
| 3-Sigma | 0.604 |
| EWMA | 0.418 |
| LOF | 0.112 |
| IsolationForest (raw) | 0.559 |
| IsolationForest + Rolling | **0.610** |

---

## Hyperparameter Tuning

Both key hyperparameters were swept systematically rather than guessed.

**Contamination sweep** (0.02 to 0.30, long_window fixed at 360): F1 peaks at contamination=0.12. The true anomaly rate is 10.0%, but IsolationForest's internal threshold calibration is slightly conservative, so setting contamination slightly above the true rate yields better precision-recall balance.

**Window sweep** (24 to 576 rows, contamination fixed at 0.12): F1 peaks at 360 rows = 30 hours. The February failure lasts roughly 48 hours. A 30-hour baseline does not fully adapt into the failure, keeping `sustained_dev` clearly negative throughout. Shorter windows normalize the failure; longer ones lose resolution at the onset.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10 or later.

---

## Running the CLI

The main entry point is `biosentinel.py detect`. It loads a CSV file, runs the chosen detector, and prints the flagged timestamps sorted by anomaly score.

**Basic usage — print the top 20 highest-scoring anomalies:**

```bash
python biosentinel.py detect data/machine_temp.csv
```

**Choose a different detector:**

```bash
python biosentinel.py detect data/machine_temp.csv --detector ewma
python biosentinel.py detect data/machine_temp.csv --detector 3sigma
python biosentinel.py detect data/machine_temp.csv --detector iforest
```

**Print all flagged timestamps (not just top 20):**

```bash
python biosentinel.py detect data/machine_temp.csv --all
```

**Export results to CSV:**

```bash
python biosentinel.py detect data/machine_temp.csv --all --export flagged.csv
```

**Save a detection plot:**

```bash
python biosentinel.py detect data/machine_temp.csv --plot out.png
```

**Evaluate against ground-truth windows (computes precision, recall, F1):**

```bash
python biosentinel.py detect data/machine_temp.csv \
  --labels "2013-12-10 06:25:00" "2013-12-12 05:35:00" \
           "2013-12-15 17:50:00" "2013-12-17 17:00:00" \
           "2014-01-27 14:20:00" "2014-01-29 13:30:00" \
           "2014-02-07 14:55:00" "2014-02-09 14:05:00"
```

**All options:**

```
python biosentinel.py detect --help

  --detector     3sigma | rolling-3sigma | ewma | hampel | lof | iforest | rolling-iforest
  --contamination  float  (default: 0.12)
  --window         int    rolling window rows for rolling-3sigma / hampel (default: 1000)
  --span           int    EWMA span in rows (default: 5000)
  --long-window    int    sustained-deviation window for rolling-iforest (default: 360)
  --n-neighbors    int    LOF neighbors (default: 20)
  --top            int    print top N flagged timestamps (default: 20)
  --all                   print every flagged timestamp
  --labels         START1 END1 START2 END2 ...  (for scoring)
  --plot           OUTPUT.png
  --export         OUTPUT.csv
```

---

## Running the Streamlit App

The interactive app lets you switch between all 7 detectors and explore the signal across 5 tabs: Detection Plot, Anomaly Overlap, Feature View, Anomaly Score, and Detector Comparison.

```bash
streamlit run app.py
```

---

## Running the Analysis Scripts

```bash
# Compare all 7 detectors and print the full table
python compare.py

# Hyperparameter sweep — generates data/tune_sweep.png
python tune.py

# Precision-Recall curves — generates data/pr_curve.png
python pr_curve.py

# Explore the raw signal
python explore.py

# Regenerate all website images
# Note: run tune.py and pr_curve.py first — generate_web_images.py copies
# data/tune_sweep.png and data/pr_curve.png into docs/images/
python generate_web_images.py
```

The figures in `docs/images/` are pre-generated and committed, so a fresh clone does not need to re-run these to use the site.

---

## Project Structure

```
biosentinel/
├── biosentinel.py          # All 7 detectors, evaluation helpers, CLI
├── app.py                  # Streamlit interactive demo
├── compare.py              # Run all detectors side by side
├── tune.py                 # Hyperparameter sweep
├── pr_curve.py             # Precision-Recall curves
├── explore.py              # Signal visualization
├── evaluate.py             # Baseline evaluation
├── generate_web_images.py  # Reproducible figure generation for website
├── requirements.txt
├── data/                   # Not included — download NAB dataset here
│   └── machine_temp.csv
└── website/                # Static portfolio site
    ├── index.html
    └── images/
```

---

## Limitations

This project is honest about what it is and what it is not.

**Single dataset.** All results are from one NAB time-series. The detector tuning, the optimal window size, and the observed recall ceiling are specific to this dataset and this failure type. Performance on other sensors or failure modes is unknown.

**Hand-designed feature.** The `sustained_dev` feature was designed specifically for sustained temperature drops. Other failure modes (oscillating failures, sensor drift, intermittent faults) would likely require different engineered features.

**Point-based labels.** NAB labels are per-reading (binary: anomaly or not). Real pharmaceutical quality control often requires batch-level or window-level decisions: not just "is this point anomalous" but "is this production run compromised." Point-based F1 is a useful proxy but not the same thing.

**No cross-validation.** The entire dataset is used for both training and evaluation. IsolationForest is an unsupervised method (it does not use the labels during training), so strict train/test separation is less critical, but the hyperparameter tuning did use the labels to pick contamination and window size.

**Not validated on real pharmaceutical data.** This project is a research prototype and educational tool. It has not been tested on actual pharmaceutical cold-chain sensor data, regulatory-grade label standards, or production monitoring infrastructure.

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgements

Dataset: [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) (BSD 3-Clause License).
