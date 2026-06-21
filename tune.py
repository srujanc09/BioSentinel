"""
Hyperparameter sensitivity analysis for BioSentinel.

Sweeps two key parameters:
  1. contamination (0.02 to 0.30) with long_window fixed at 144
  2. long_window (24 to 576 rows = 2h to 48h) with contamination fixed at 0.12

For each parameter value we compute Precision, Recall, and F1 against the
four NAB ground-truth windows. Saves two plots:
  data/tune_contamination.png
  data/tune_window.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from biosentinel import load_data, detect_iforest_rolling, score, f1

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
RED   = "#ef4444"
AMBER = "#f59e0b"
GRAY  = "#94a3b8"
WHITE = "#f1f5f9"


def apply_labels(df):
    df = df.copy()
    df["label"] = 0
    for s, e in WINDOWS:
        df.loc[s:e, "label"] = 1
    return df


def evaluate(df_with_labels):
    tp, fp, fn_c, p, r = score(df_with_labels)
    return p, r, f1(p, r)


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


# ── Sweep 1: contamination ─────────────────────────────────────────────────
print("Sweeping contamination (0.02 → 0.30)...")
cont_values = np.arange(0.02, 0.31, 0.01)
cont_results = []
for c in cont_values:
    df = load_data("data/machine_temp.csv")
    df = detect_iforest_rolling(df, contamination=round(float(c), 2), long_window=144)
    df = apply_labels(df)
    p, r, f = evaluate(df)
    cont_results.append((round(float(c), 2), p, r, f))
    print(f"  c={c:.2f}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

best_c     = max(cont_results, key=lambda x: x[3])
cont_arr   = np.array(cont_results)
print(f"\nBest contamination: {best_c[0]}  (F1={best_c[3]:.3f})")

# ── Sweep 2: long_window ───────────────────────────────────────────────────
print("\nSweeping long_window (24 → 576 rows = 2h → 48h)...")
window_values = list(range(24, 577, 24))
win_results   = []
for w in window_values:
    df = load_data("data/machine_temp.csv")
    df = detect_iforest_rolling(df, contamination=0.12, long_window=w)
    df = apply_labels(df)
    p, r, f = evaluate(df)
    win_results.append((w, p, r, f))
    print(f"  w={w:>4}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

best_w   = max(win_results, key=lambda x: x[3])
win_arr  = np.array(win_results)
print(f"\nBest long_window: {int(best_w[0])} rows = {int(best_w[0])//12}h  (F1={best_w[3]:.3f})")

# ── Plot ───────────────────────────────────────────────────────────────────
dark_style()
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.subplots_adjust(wspace=0.38)

# Panel 1 — contamination sweep
ax = axes[0]
ax.set_facecolor(BG2)
ax.plot(cont_arr[:, 0], cont_arr[:, 1], color=BLUE,  linewidth=2.0, label="Precision")
ax.plot(cont_arr[:, 0], cont_arr[:, 2], color=AMBER, linewidth=2.0, label="Recall")
ax.plot(cont_arr[:, 0], cont_arr[:, 3], color=GREEN, linewidth=2.5, label="F1", zorder=5)
ax.axvline(best_c[0], color=CYAN, linewidth=1.4, linestyle="--", alpha=0.8)
ax.axvline(0.10,      color=GRAY, linewidth=1.0, linestyle=":",  alpha=0.6)
ax.text(best_c[0] + 0.005, 0.08, f"opt c={best_c[0]}", color=CYAN, fontsize=9)
ax.text(0.10 + 0.005,       0.02, "true rate\n10.0%", color=GRAY, fontsize=8)
ax.annotate(f"F1={best_c[3]:.3f}",
            xy=(best_c[0], best_c[3]),
            xytext=(best_c[0] + 0.04, best_c[3] - 0.05),
            arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.2),
            fontsize=9, color=CYAN)
ax.set_xlabel("Contamination parameter", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Contamination Sweep\n(long_window fixed at 144)", fontsize=12, pad=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.tick_params(colors=GRAY)
for sp in ax.spines.values():
    sp.set_edgecolor("#1e3a5f")

# Panel 2 — long_window sweep
ax2 = axes[1]
ax2.set_facecolor(BG2)
hours = win_arr[:, 0] / 12
ax2.plot(hours, win_arr[:, 1], color=BLUE,  linewidth=2.0, label="Precision")
ax2.plot(hours, win_arr[:, 2], color=AMBER, linewidth=2.0, label="Recall")
ax2.plot(hours, win_arr[:, 3], color=GREEN, linewidth=2.5, label="F1", zorder=5)
best_h = best_w[0] / 12
ax2.axvline(best_h, color=CYAN, linewidth=1.4, linestyle="--", alpha=0.8)
ax2.text(best_h + 0.5, 0.08, f"opt {int(best_w[0])} rows\n= {int(best_h)}h", color=CYAN, fontsize=9)
ax2.annotate(f"F1={best_w[3]:.3f}",
             xy=(best_h, best_w[3]),
             xytext=(best_h + 5, best_w[3] - 0.05),
             arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.2),
             fontsize=9, color=CYAN)
ax2.set_xlabel("Sustained-deviation window (hours)", fontsize=11)
ax2.set_ylabel("Score", fontsize=11)
ax2.set_title("Window Size Sweep\n(contamination fixed at 0.12)", fontsize=12, pad=12)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=10)
ax2.tick_params(colors=GRAY)
for sp in ax2.spines.values():
    sp.set_edgecolor("#1e3a5f")

fig.suptitle(
    "Hyperparameter Sensitivity — IsolationForest + Rolling Features",
    fontsize=13, color=WHITE, y=1.02
)
fig.savefig("data/tune_sweep.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved data/tune_sweep.png")
