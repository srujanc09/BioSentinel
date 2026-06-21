"""
Runs all BioSentinel detectors against NAB ground-truth windows and prints
a comparison table, then saves a multi-panel plot to data/compare_plot.png.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from biosentinel import (
    load_data, detect_3sigma, detect_rolling_3sigma,
    detect_ewma, detect_hampel, detect_lof,
    detect_iforest, detect_iforest_rolling,
    score, f1,
)

WINDOWS = [
    ("2013-12-10 06:25:00", "2013-12-12 05:35:00"),
    ("2013-12-15 17:50:00", "2013-12-17 17:00:00"),
    ("2014-01-27 14:20:00", "2014-01-29 13:30:00"),
    ("2014-02-07 14:55:00", "2014-02-09 14:05:00"),
]

BG    = "#050d1a"
BG2   = "#0a1628"
GOLD  = "#fbbf24"
GRAY  = "#94a3b8"
WHITE = "#f1f5f9"


def apply_labels(df):
    df = df.copy()
    df["label"] = 0
    for s, e in WINDOWS:
        df.loc[s:e, "label"] = 1
    return df


def run(name, fn, *args):
    df = load_data("data/machine_temp.csv")
    df = fn(df, *args)
    df = apply_labels(df)
    tp, fp, fn_c, p, r = score(df)
    return {
        "Detector":  name,
        "Flagged":   int(df["anomaly"].sum()),
        "TP": int(tp), "FP": int(fp), "FN": int(fn_c),
        "Precision": round(p, 3),
        "Recall":    round(r, 3),
        "F1":        round(f1(p, r), 3),
        "_df": df,
    }


results = [
    run("3-Sigma (Global)",             detect_3sigma),
    run("Rolling 3-Sigma (w=1000)",     detect_rolling_3sigma,  1000),
    run("EWMA (span=5000)",             detect_ewma,             5000),
    run("Hampel Filter (w=144)",        detect_hampel,           144),
    run("LOF (k=20)",                   detect_lof,              20, 0.12),
    run("IsolationForest (raw)",        detect_iforest,          0.12),
    run("IsolationForest + Rolling",    detect_iforest_rolling,  0.12, 360),
]

table = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in results])
print("\n=== BioSentinel — Full Detector Comparison ===\n")
print(table.to_string(index=False))
print(f"\nRecall ceiling for value-only detectors: 61.2%")
print(f"Best F1: {table['F1'].max():.3f}  ({table.loc[table['F1'].idxmax(), 'Detector']})")

# ── Plot ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "figure.facecolor": BG,
    "axes.facecolor": BG2, "axes.edgecolor": "#1e3a5f",
    "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.grid": True, "grid.color": "#1e3a5f", "grid.alpha": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.facecolor": BG2, "legend.edgecolor": "#1e3a5f",
})

colors = ["#3b82f6", "#f59e0b", "#0d9488", "#a855f7", "#ec4899", "#22c55e", "#00d4ff"]
n = len(results)
fig = plt.figure(figsize=(16, n * 3.0 + 1), facecolor=BG)
gs  = gridspec.GridSpec(n, 1, hspace=0.72)

for i, result in enumerate(results):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor(BG2)
    df = result["_df"]
    for s, e in WINDOWS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=GOLD, alpha=0.10)
    ax.plot(df.index, df["value"], linewidth=0.5, color="#2d4a6e", zorder=1)
    fl = df[df["anomaly"]]
    ax.scatter(fl.index, fl["value"], color=colors[i], s=5, zorder=3, edgecolors="none")
    ax.set_ylabel("°C", fontsize=8, color=GRAY)
    ax.set_title(
        f"{result['Detector']}   "
        f"P={result['Precision']:.3f}  R={result['Recall']:.3f}  F1={result['F1']:.3f}"
        + ("  ← best F1" if result["F1"] == table["F1"].max() else ""),
        fontsize=9, color=WHITE, pad=5,
    )
    ax.tick_params(labelsize=7, colors=GRAY)
    ax.set_ylim(bottom=0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3a5f")
    if i == 0:
        handles = [mpatches.Patch(color=GOLD, alpha=0.4, label="Anomaly window"),
                   mpatches.Patch(color=colors[i], label="Flagged")]
        ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

fig.suptitle(
    "All Seven Detectors vs Ground Truth (yellow)  |  Recall ceiling 61.2%",
    fontsize=12, y=1.005, color=WHITE,
)
fig.savefig("data/compare_plot.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nPlot saved to data/compare_plot.png")
