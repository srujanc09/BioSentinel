"""
Precision-Recall curves for BioSentinel detectors.

Binary F1 at a single threshold is easy to game — a model can inflate F1
by tuning exactly one number. PR curves show performance across ALL thresholds
and the AUC-PR summarizes the overall ranking ability.

For detectors that produce continuous scores (IsolationForest decision_function,
LOF negative_outlier_factor, z-scores) we sweep every unique score as a threshold
and compute precision and recall at each point.

Saves: data/pr_curve.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from biosentinel import load_data

WINDOWS = [
    ("2013-12-10 06:25:00", "2013-12-12 05:35:00"),
    ("2013-12-15 17:50:00", "2013-12-17 17:00:00"),
    ("2014-01-27 14:20:00", "2014-01-29 13:30:00"),
    ("2014-02-07 14:55:00", "2014-02-09 14:05:00"),
]

BG    = "#050d1a"
BG2   = "#0a1628"
CYAN  = "#00d4ff"
BLUE  = "#3b82f6"
GREEN = "#22c55e"
TEAL  = "#0d9488"
AMBER = "#f59e0b"
PURPLE= "#a855f7"
GRAY  = "#94a3b8"
WHITE = "#f1f5f9"


def make_labels(df):
    df = df.copy()
    df["label"] = 0
    for s, e in WINDOWS:
        df.loc[s:e, "label"] = 1
    return df


def dark_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": "#1e3a5f", "axes.labelcolor": GRAY,
        "axes.titlecolor": WHITE, "xtick.color": GRAY, "ytick.color": GRAY,
        "axes.grid": True, "grid.color": "#1e3a5f", "grid.alpha": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": BG2, "legend.edgecolor": "#1e3a5f",
        "legend.labelcolor": GRAY,
    })


dark_style()
df_base = make_labels(load_data("data/machine_temp.csv"))
y_true  = df_base["label"].values
v       = df_base["value"].values

# ── Score functions ────────────────────────────────────────────────────────

print("Computing anomaly scores for all detectors...")

# 1. 3-Sigma — score = |z-score|
mean = v.mean(); std = v.std()
scores_3sigma = np.abs((v - mean) / std)
print("  3-sigma done")

# 2. EWMA — score = |EWMA z-score| with span=5000
ewm_mean = pd.Series(v).ewm(span=5000, min_periods=1).mean().values
ewm_std  = pd.Series(v).ewm(span=5000, min_periods=1).std().fillna(std).values
scores_ewma = np.abs((v - ewm_mean) / ewm_std)
print("  EWMA done")

# 3. LOF — negative_outlier_factor_ (already negative, negate so higher = more anomalous)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.12)
lof.fit_predict(v.reshape(-1, 1))
scores_lof = -lof.negative_outlier_factor_
print("  LOF done")

# 4. IsolationForest raw — decision_function (negate so higher = more anomalous)
clf_if = IsolationForest(contamination=0.12, random_state=42, n_estimators=200)
clf_if.fit(v.reshape(-1, 1))
scores_if = -clf_if.decision_function(v.reshape(-1, 1))
print("  IsolationForest (raw) done")

# 5. IsolationForest + Rolling
sustained_dev = (
    pd.Series(v) - pd.Series(v).rolling(360, min_periods=1).mean()
).values
X = np.column_stack([v, sustained_dev])
clf_ifr = IsolationForest(contamination=0.12, random_state=42, n_estimators=200)
clf_ifr.fit(X)
scores_ifr = -clf_ifr.decision_function(X)
print("  IsolationForest + Rolling done")


# ── PR Curves ──────────────────────────────────────────────────────────────

detectors = [
    ("3-Sigma (Global)",          scores_3sigma, BLUE),
    ("EWMA (span=5000)",          scores_ewma,   AMBER),
    ("LOF (k=20)",                scores_lof,    PURPLE),
    ("IsolationForest (Raw)",     scores_if,     TEAL),
    ("IsolationForest + Rolling (w=360)", scores_ifr, CYAN),
]

fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
ax.set_facecolor(BG2)

# Random baseline (area = fraction of positive class)
pos_rate = y_true.mean()
ax.plot([0, 1], [pos_rate, pos_rate], color=GRAY, linewidth=1.2,
        linestyle="--", label=f"Random baseline  (AUC={pos_rate:.3f})")

for name, scores, color in detectors:
    prec, rec, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(rec, prec)
    ax.plot(rec, prec, color=color, linewidth=2.0, label=f"{name}  (AUC={auc_pr:.3f})")
    print(f"  {name:35s}  AUC-PR = {auc_pr:.3f}")

# Mark the theoretical recall ceiling
ax.axvline(0.612, color=GRAY, linewidth=1.0, linestyle=":", alpha=0.7)
ax.text(0.618, 0.95, "Recall ceiling\n61.2%", color=GRAY, fontsize=8, va="top")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title(
    "Precision-Recall Curves — All Detectors\n"
    "AUC-PR summarises performance across all thresholds",
    fontsize=12, pad=14
)
ax.set_xlim(0, 1.01)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc="upper right")
ax.tick_params(colors=GRAY)
for sp in ax.spines.values():
    sp.set_edgecolor("#1e3a5f")

plt.tight_layout()
fig.savefig("data/pr_curve.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved data/pr_curve.png")
