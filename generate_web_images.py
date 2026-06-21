"""Generate high-quality images for the BioSentinel portfolio website."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from biosentinel import (
    load_data, detect_3sigma, detect_rolling_3sigma,
    detect_ewma, detect_hampel, detect_lof,
    detect_iforest, detect_iforest_rolling,
)

WINDOWS = [
    ("2013-12-10 06:25:00", "2013-12-12 05:35:00"),
    ("2013-12-15 17:50:00", "2013-12-17 17:00:00"),
    ("2014-01-27 14:20:00", "2014-01-29 13:30:00"),
    ("2014-02-07 14:55:00", "2014-02-09 14:05:00"),
]

CYAN   = "#00d4ff"
BLUE   = "#3b82f6"
TEAL   = "#0d9488"
RED    = "#ef4444"
AMBER  = "#f59e0b"
GREEN  = "#22c55e"
PURPLE = "#a855f7"
GOLD   = "#fbbf24"
BG     = "#050d1a"
BG2    = "#0a1628"
CARD   = "#0f2744"
GRAY   = "#94a3b8"
WHITE  = "#f1f5f9"


def dark_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": "#1e3a5f", "axes.labelcolor": GRAY,
        "axes.titlecolor": WHITE, "xtick.color": GRAY, "ytick.color": GRAY,
        "grid.color": "#1e3a5f", "grid.linestyle": "--", "grid.alpha": 0.45,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": CARD, "legend.edgecolor": "#1e3a5f",
        "legend.labelcolor": GRAY, "text.color": WHITE,
    })


def shade_windows(ax):
    for i, (s, e) in enumerate(WINDOWS):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=GOLD, alpha=0.12,
                   label="Anomaly window" if i == 0 else "")


def style_ax(ax):
    ax.tick_params(colors=GRAY)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3a5f")


dark_style()


# ── 1. Raw signal overview ─────────────────────────────────────────────────
print("1. signal_overview...")
df = load_data("data/machine_temp.csv")
fig, ax = plt.subplots(figsize=(16, 5))
shade_windows(ax)
ax.plot(df.index, df["value"], linewidth=0.9, color=CYAN, alpha=0.85)
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title("Machine Temperature Sensor — 22,695 readings at 5-minute intervals (Dec 2013 – Feb 2014)",
             fontsize=13, pad=14)
ax.set_ylim(bottom=0)
style_ax(ax)
handles = [mpatches.Patch(color=CYAN, label="Temperature"),
           mpatches.Patch(color=GOLD, alpha=0.5, label="Known anomaly window")]
ax.legend(handles=handles, loc="lower right", fontsize=9)
fig.tight_layout(pad=1.5)
fig.savefig("docs/images/signal_overview.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("   done")


# ── 2. 3-Sigma annotated ──────────────────────────────────────────────────
print("2. detector_3sigma...")
dark_style()
df2 = load_data("data/machine_temp.csv")
mean = df2["value"].mean()
std  = df2["value"].std()
df2["anomaly"] = ((df2["value"] - mean) / std).abs() > 3

fig, ax = plt.subplots(figsize=(16, 5))
shade_windows(ax)
ax.plot(df2.index, df2["value"], linewidth=0.8, color=BLUE, alpha=0.8, label="Temperature")
fl = df2[df2["anomaly"]]
ax.scatter(fl.index, fl["value"], color=RED, s=20, zorder=5, label=f"Flagged ({len(fl):,} pts)", edgecolors="none")
ax.axhline(mean + 3*std, color=AMBER, linewidth=1.3, linestyle="--", alpha=0.8, label="3-sigma bounds")
ax.axhline(mean - 3*std, color=AMBER, linewidth=1.3, linestyle="--", alpha=0.8)
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title("3-Sigma (Global)  —  Precision: 0.991  |  Recall: 0.202  |  F1: 0.336", fontsize=13, pad=14)
ax.annotate("Catches sharp transient crash",
            xy=(pd.Timestamp("2013-12-15 20:00"), 5),
            xytext=(pd.Timestamp("2013-12-22"), 20),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
            fontsize=9, color=GRAY)
ax.annotate("Misses sustained 48-hour failure",
            xy=(pd.Timestamp("2014-02-08"), 50),
            xytext=(pd.Timestamp("2014-01-14"), 30),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
            fontsize=9, color=RED)
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(bottom=0)
style_ax(ax)
fig.tight_layout(pad=1.5)
fig.savefig("docs/images/detector_3sigma.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("   done")


# ── 3. Best detector: IF + rolling (w=360) ────────────────────────────────
print("3. detector_iforest_rolling...")
dark_style()
df4 = load_data("data/machine_temp.csv")
df4 = detect_iforest_rolling(df4, contamination=0.12, long_window=360)

fig, ax = plt.subplots(figsize=(16, 5))
shade_windows(ax)
ax.plot(df4.index, df4["value"], linewidth=0.8, color=CYAN, alpha=0.75, label="Temperature")
fl4 = df4[df4["anomaly"]]
ax.scatter(fl4.index, fl4["value"], color=RED, s=8, zorder=5, label=f"Flagged ({len(fl4):,} pts)", edgecolors="none")
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title(
    "IsolationForest + Rolling (w=360)  —  Precision: 0.568  |  Recall: 0.682  |  F1: 0.619",
    fontsize=13, pad=14)
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(bottom=0)
style_ax(ax)
fig.tight_layout(pad=1.5)
fig.savefig("docs/images/detector_iforest_rolling.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("   done")


# ── 4. Four-detector comparison (key detectors) ───────────────────────────
print("4. comparison_all...")
dark_style()
detectors_cmp = [
    ("3-Sigma (Global)",              detect_3sigma,         [],           BLUE,   "P=0.991  R=0.202  F1=0.336"),
    ("EWMA (span=5000)",              detect_ewma,           [5000],       AMBER,  "P=0.586  R=0.513  F1=0.547"),
    ("IsolationForest (Raw)",         detect_iforest,        [0.12],       TEAL,   "P=0.471  R=0.566  F1=0.514"),
    ("IsolationForest + Rolling",     detect_iforest_rolling,[0.12, 360],  CYAN,   "P=0.568  R=0.682  F1=0.619  ← best"),
]
fig = plt.figure(figsize=(16, 14), facecolor=BG)
gs  = gridspec.GridSpec(4, 1, hspace=0.72)
for i, (name, fn, args, color, metrics) in enumerate(detectors_cmp):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor(BG2)
    dfx = load_data("data/machine_temp.csv")
    dfx = fn(dfx, *args)
    shade_windows(ax)
    ax.plot(dfx.index, dfx["value"], linewidth=0.5, color="#2d4a6e", zorder=1)
    fl = dfx[dfx["anomaly"]]
    ax.scatter(fl.index, fl["value"], color=color, s=6, zorder=3, edgecolors="none")
    ax.set_ylabel("°C", fontsize=9, color=GRAY)
    ax.set_title(f"{name}   —   {metrics}", fontsize=10, color=WHITE, pad=6)
    ax.tick_params(labelsize=8, colors=GRAY)
    ax.set_ylim(bottom=0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3a5f")
    if i == 0:
        handles = [mpatches.Patch(color=GOLD, alpha=0.4, label="Anomaly window"),
                   mpatches.Patch(color=color, label="Flagged")]
        ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
fig.suptitle("All Detectors vs Ground Truth (yellow)  |  Recall ceiling 61.2% for value-only methods",
             fontsize=12, y=1.005, color=WHITE)
fig.savefig("docs/images/comparison_all.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("   done")


# ── 5. Feature insight ────────────────────────────────────────────────────
print("5. feature_insight...")
dark_style()
df5 = load_data("data/machine_temp.csv")
v   = df5["value"]
sustained_dev = v - v.rolling(360, min_periods=1).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True, facecolor=BG)
fig.subplots_adjust(hspace=0.52)
for ax in (ax1, ax2):
    ax.set_facecolor(BG2)
    shade_windows(ax)
    style_ax(ax)

ax1.plot(df5.index, v, linewidth=0.8, color=BLUE, alpha=0.85)
ax1.set_ylabel("Temperature (°C)", fontsize=11)
ax1.set_title("Raw signal — the 48-hour Feb failure sits at 40-60°C, hard to separate from brief dips", fontsize=11)
ax1.set_ylim(bottom=0)

ax2.plot(df5.index, sustained_dev, linewidth=0.8, color=CYAN, alpha=0.85)
ax2.axhline(0, color="#2d4a6e", linewidth=1.0, linestyle="--")
ax2.set_ylabel("Deviation from\n30h baseline (°C)", fontsize=10)
ax2.set_title("sustained_dev = value - rolling_mean(360)  —  Feb failure drops sharply and stays low",
              fontsize=11)

fig.savefig("docs/images/feature_insight.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("   done")


# ── 6. Metrics bar chart ──────────────────────────────────────────────────
print("6. metrics_bar...")
dark_style()
labels_b  = ["3-Sigma\n(Global)", "EWMA\n(span=5000)", "IsolationForest\n(Raw)", "IF + Rolling\n(w=360, best)"]
precision = [0.991, 0.586, 0.471, 0.568]
recall    = [0.202, 0.513, 0.566, 0.682]
f1_scores = [0.336, 0.547, 0.514, 0.619]
colors_b  = [BLUE, AMBER, TEAL, CYAN]

x = np.arange(len(labels_b))
w = 0.25
fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
ax.set_facecolor(BG2)
bars_p = ax.bar(x - w, precision, w, label="Precision", color=BLUE,  alpha=0.85, zorder=3)
bars_r = ax.bar(x,     recall,    w, label="Recall",    color=CYAN,  alpha=0.85, zorder=3)
bars_f = ax.bar(x + w, f1_scores, w, label="F1 Score",  color=GREEN, alpha=0.85, zorder=3)
for bars in (bars_p, bars_r, bars_f):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, color=GRAY)
ax.set_xticks(x)
ax.set_xticklabels(labels_b, fontsize=10, color=WHITE)
ax.set_ylim(0, 1.14)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Precision, Recall, and F1 — All Four Detectors", fontsize=13, pad=14)
ax.legend(fontsize=10)
ax.axvline(x=0.5, color="#1e3a5f", linewidth=1.5, linestyle=":", alpha=0.8)
ax.text(0.55,  1.09, "ML methods →", color=GRAY, fontsize=8)
ax.text(-0.5, 1.09, "← Statistical", color=GRAY, fontsize=8)
ax.tick_params(colors=GRAY)
for sp in ax.spines.values():
    sp.set_edgecolor("#1e3a5f")
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout(pad=1.5)
fig.savefig("docs/images/metrics_bar.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("   done")


# ── 7. February failure zoom ──────────────────────────────────────────────
print("7. feb_failure_zoom...")
dark_style()
df_3s = load_data("data/machine_temp.csv")
df_3s = detect_3sigma(df_3s)
df_if = load_data("data/machine_temp.csv")
df_if = detect_iforest_rolling(df_if, contamination=0.12, long_window=360)

feb_s, feb_e = "2014-02-04", "2014-02-13"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, facecolor=BG)
fig.subplots_adjust(hspace=0.52)
for ax in (ax1, ax2):
    ax.set_facecolor(BG2)
    ax.axvspan(pd.Timestamp("2014-02-07 14:55"), pd.Timestamp("2014-02-09 14:05"),
               color=GOLD, alpha=0.18, label="Anomaly window")
    style_ax(ax)

sl1 = df_3s.loc[feb_s:feb_e]
ax1.plot(sl1.index, sl1["value"], linewidth=1.3, color=BLUE, alpha=0.8, label="Temperature")
fl1 = sl1[sl1["anomaly"]]
ax1.scatter(fl1.index, fl1["value"], color=RED, s=40, zorder=5, label=f"Flagged ({len(fl1)} pts)")
ax1.set_ylabel("Temperature (°C)", fontsize=11)
ax1.set_title("3-Sigma (Global)  —  catches only the lowest extremes, misses most of the failure", fontsize=11)
ax1.legend(fontsize=9, loc="lower left")

sl2 = df_if.loc[feb_s:feb_e]
ax2.plot(sl2.index, sl2["value"], linewidth=1.3, color=CYAN, alpha=0.8, label="Temperature")
fl2 = sl2[sl2["anomaly"]]
ax2.scatter(fl2.index, fl2["value"], color=RED, s=40, zorder=5, label=f"Flagged ({len(fl2)} pts)")
ax2.set_ylabel("Temperature (°C)", fontsize=11)
ax2.set_title("IsolationForest + Rolling (w=360)  —  catches the full sustained failure period", fontsize=11)
ax2.legend(fontsize=9, loc="lower left")

fig.suptitle("Zoomed In: February 7-9 Machine Failure Event", fontsize=13, color=WHITE, y=1.01)
fig.savefig("docs/images/feb_failure_zoom.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("   done")


# ── 8. Tuning sweep (copy from data/) ─────────────────────────────────────
import shutil
shutil.copy("data/tune_sweep.png", "docs/images/tune_sweep.png")
print("8. tune_sweep.png copied")

shutil.copy("data/pr_curve.png", "docs/images/pr_curve.png")
print("9. pr_curve.png copied")


print("\nAll images done.")
