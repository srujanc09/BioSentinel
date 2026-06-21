import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from biosentinel import (
    load_data, detect_3sigma, detect_rolling_3sigma,
    detect_ewma, detect_hampel, detect_lof,
    detect_iforest, detect_iforest_rolling,
    score, f1,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioSentinel",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #050d1a; }
  section[data-testid="stSidebar"] { background: #081525; border-right: 1px solid rgba(0,212,255,.15); }
  html, body, [class*="css"] { color: #f1f5f9; font-family: 'Inter', sans-serif; }
  label, .stRadio label { color: #94a3b8 !important; }
  h1, h2, h3 { color: #f1f5f9 !important; }
  p, span, div { color: #94a3b8; }
  .stSlider > div > div > div { background: rgba(0,212,255,.15) !important; }
  .stTextArea textarea { background: #081525; border: 1px solid rgba(0,212,255,.2); color: #f1f5f9; }
  .stFileUploader { background: #081525; border: 1px solid rgba(0,212,255,.15); border-radius: 8px; }
  [data-testid="stMetric"] { background: #0d1f3c; border: 1px solid rgba(0,212,255,.12); border-radius: 10px; padding: 16px; }
  [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.6rem !important; }
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
  .stRadio > div label { background: #0d1f3c; border: 1px solid rgba(255,255,255,.07); border-radius: 8px; padding: 8px 14px; transition: border-color .2s; }
  .stRadio > div label:hover { border-color: rgba(0,212,255,.4); }
  .streamlit-expanderHeader { background: #0d1f3c; border: 1px solid rgba(0,212,255,.12); border-radius: 8px; color: #f1f5f9 !important; }
  hr { border-color: rgba(0,212,255,.1); }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #050d1a; }
  ::-webkit-scrollbar-thumb { background: rgba(0,212,255,.3); border-radius: 3px; }
  .stCaption, [data-testid="stCaptionContainer"] { color: #64748b !important; font-size: .78rem; }
  div[data-baseweb="tab-list"] { gap: 4px; background: #081525; border-radius: 8px; padding: 4px; }
  button[data-baseweb="tab"] { background: transparent !important; color: #64748b !important; border-radius: 6px !important; }
  button[data-baseweb="tab"][aria-selected="true"] { background: rgba(0,212,255,.12) !important; color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
NAB_WINDOWS = [
    ("2013-12-10 06:25:00", "2013-12-12 05:35:00"),
    ("2013-12-15 17:50:00", "2013-12-17 17:00:00"),
    ("2014-01-27 14:20:00", "2014-01-29 13:30:00"),
    ("2014-02-07 14:55:00", "2014-02-09 14:05:00"),
]

C = {
    "bg":    "#050d1a", "bg2":  "#081525", "card": "#0d1f3c",
    "cyan":  "#00d4ff", "blue": "#3b82f6", "teal": "#0d9488",
    "red":   "#ef4444", "gold": "#fbbf24", "gray": "#94a3b8",
    "green": "#22c55e", "amb":  "#f59e0b", "text": "#f1f5f9",
    "pur":   "#a855f7",
}

DETECTOR_INFO = {
    "3-Sigma (Global)": {
        "desc": "Flags anything beyond 3 standard deviations from the global mean. Near-perfect precision but misses sustained failures because the global std is inflated by the anomaly period itself.",
        "color": C["blue"], "best_for": "brief transient spikes",
    },
    "EWMA": {
        "desc": "Exponentially weighted baseline with slow adaptation (span=5000). Recent readings count slightly more than old ones, but the very large span keeps the baseline stable across multi-day events.",
        "color": C["amb"], "best_for": "gradual drift detection",
    },
    "Hampel Filter": {
        "desc": "Robust median-based filter: flags points where |value - rolling_median| > 3 × 1.4826 × MAD. Designed for impulsive spikes. Adapts into sustained failures (same failure mode as rolling 3-sigma).",
        "color": C["pur"], "best_for": "isolated spike removal",
    },
    "LOF": {
        "desc": "Local Outlier Factor compares the local density around each point to that of its neighbors. Struggles here because the sustained failure creates a dense anomalous cluster that looks 'normal' locally.",
        "color": "#ec4899", "best_for": "clustered multi-dimensional anomalies",
    },
    "IsolationForest": {
        "desc": "Builds 200 random isolation trees. Points isolated in fewer cuts are anomalies. No hard threshold. Better recall than any statistical method on the same raw data.",
        "color": C["teal"], "best_for": "global distribution anomalies",
    },
    "IsolationForest + Rolling (Best)": {
        "desc": "Adds sustained_dev = value − rolling_mean(360 rows = 30h) as a second feature. During sustained failures, current readings stay far below the slow-moving baseline, creating a distinct 2D signature that breaks through the value-only recall ceiling.",
        "color": C["cyan"], "best_for": "sustained multi-hour failures ← this dataset",
    },
}


def dark_plt():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "figure.facecolor": C["bg"], "axes.facecolor": C["bg2"],
        "axes.edgecolor": "#1e3a5f", "axes.labelcolor": C["gray"],
        "axes.titlecolor": C["text"], "xtick.color": C["gray"],
        "ytick.color": C["gray"], "axes.grid": True,
        "grid.color": "#1e3a5f", "grid.alpha": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": C["card"], "legend.edgecolor": "#1e3a5f",
        "legend.labelcolor": C["gray"],
    })


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 10px">
  <span style="font-size:2.2rem;font-weight:800;letter-spacing:-.03em;color:#f1f5f9">
    🧬 Bio<span style="color:#00d4ff">Sentinel</span>
  </span>
  <p style="color:#94a3b8;font-size:.95rem;margin-top:8px;max-width:560px;margin-left:auto;margin-right:auto">
    Biomedical anomaly detection — upload any sensor CSV to get started,
    or use the machine_temp.csv demo file.
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr style="margin:0 0 28px">', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="color:#00d4ff;font-weight:700;font-size:.74rem;letter-spacing:.1em;text-transform:uppercase">Data Source</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type="csv",
                                help="Must have 'timestamp' and 'value' columns")

    st.markdown("---")
    st.markdown('<p style="color:#00d4ff;font-weight:700;font-size:.74rem;letter-spacing:.1em;text-transform:uppercase">Detector</p>', unsafe_allow_html=True)
    detector = st.radio("Choose detector", list(DETECTOR_INFO.keys()), label_visibility="collapsed")

    # Params
    contamination = 0.12
    window = 1000
    span = 5000
    long_window = 360
    n_neighbors = 20

    if detector == "EWMA":
        span = st.slider("EWMA span (rows)", 100, 10000, 5000, step=100)
        st.caption("5000 rows ≈ 17 days at 5-min intervals. Very slow adaptation keeps the baseline stable across long failures.")
    elif detector == "Hampel Filter":
        window = st.slider("Window (rows)", 24, 576, 144, step=24)
        st.caption("Hampel uses centered rolling median and MAD. Good for isolated spikes, but adapts into sustained failures.")
    elif detector == "LOF":
        n_neighbors = st.slider("k (neighbors)", 5, 100, 20, step=5)
        contamination = st.slider("Contamination", 0.01, 0.30, 0.12, step=0.01)
    elif detector in ("IsolationForest", "IsolationForest + Rolling (Best)"):
        contamination = st.slider("Contamination", 0.01, 0.30, 0.12, step=0.01)
        if detector == "IsolationForest + Rolling (Best)":
            long_window = st.slider("Sustained-deviation window (rows)", 48, 576, 360, step=24,
                                    help="Tuning sweep found F1 peaks at 360 rows = 30 hours.")
            st.caption(f"{long_window} rows = {long_window//12}h at 5-min intervals. Optimal found at 360 (30h) via hyperparameter sweep.")

    st.markdown("---")
    st.markdown('<p style="color:#00d4ff;font-weight:700;font-size:.74rem;letter-spacing:.1em;text-transform:uppercase">Ground Truth Labels</p>', unsafe_allow_html=True)
    use_nab = st.checkbox("Use NAB machine_temp labels", value=True)
    default_text = "\n".join(f"{s}, {e}" for s, e in NAB_WINDOWS) if use_nab else ""
    windows_text = st.text_area("Custom windows (one per line: START, END)", value=default_text, height=120)

if uploaded is None:
    st.info("Upload a CSV file above to run the detector.")
    with st.expander("What is this?"):
        st.markdown("""
**BioSentinel** detects anomalies in biomedical sensor time-series (cold-chain sensors, lab instruments, HVAC in pharma facilities).

Upload any CSV with `timestamp` and `value` columns. The tool runs the selected detector and computes precision, recall, and F1 against any ground-truth windows you provide.

**Detectors available:**
| Detector | Best for | F1 on NAB |
|---|---|---|
| 3-Sigma (Global) | Brief transient spikes | 0.336 |
| EWMA | Gradual drift | 0.547 |
| Hampel Filter | Isolated outliers | 0.050 |
| LOF | Multi-dim cluster anomalies | 0.119 |
| IsolationForest | Global distribution | 0.514 |
| **IF + Rolling** | **Sustained failures** | **0.619** |

Dataset used for evaluation: [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) — machine_temperature_system_failure
        """)
    st.stop()

# ── Load ───────────────────────────────────────────────────────────────────
df = load_data(io.StringIO(uploaded.read().decode("utf-8")))

# ── Data info ──────────────────────────────────────────────────────────────
st.markdown('<p style="color:#00d4ff;font-weight:700;font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px">Dataset Overview</p>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows",     f"{len(df):,}")
c2.metric("Start",    str(df.index.min())[:10])
c3.metric("End",      str(df.index.max())[:10])
c4.metric("Mean",     f"{df['value'].mean():.2f}")
c5.metric("Std Dev",  f"{df['value'].std():.2f}")
c6.metric("Range",    f"{df['value'].min():.1f} – {df['value'].max():.1f}")
st.markdown('<hr style="margin:20px 0">', unsafe_allow_html=True)

# ── Run detector ───────────────────────────────────────────────────────────
detector_fn_map = {
    "3-Sigma (Global)":              (detect_3sigma,         []),
    "EWMA":                          (detect_ewma,           [span]),
    "Hampel Filter":                 (detect_hampel,         [window]),
    "LOF":                           (detect_lof,            [n_neighbors, contamination]),
    "IsolationForest":               (detect_iforest,        [contamination]),
    "IsolationForest + Rolling (Best)":(detect_iforest_rolling,[contamination, long_window]),
}
fn, fn_args = detector_fn_map[detector]
df = fn(df, *fn_args)
dlabel = detector

# ── Parse labels ───────────────────────────────────────────────────────────
has_labels = False
windows_parsed = []
if windows_text.strip():
    try:
        for line in windows_text.strip().splitlines():
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2:
                windows_parsed.append((parts[0], parts[1]))
        if windows_parsed:
            df["label"] = 0
            for s, e in windows_parsed:
                df.loc[s:e, "label"] = 1
            has_labels = True
    except Exception as ex:
        st.warning(f"Could not parse windows: {ex}")

# ── Results header ─────────────────────────────────────────────────────────
info = DETECTOR_INFO[detector]
st.markdown(
    f'<div style="background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.15);'
    f'border-left:3px solid {info["color"]};border-radius:8px;padding:14px 18px;margin-bottom:18px">'
    f'<span style="color:{info["color"]};font-weight:700;font-size:.8rem">{dlabel}</span>'
    f'<p style="margin:6px 0 0;font-size:.87rem;color:#94a3b8">{info["desc"]}</p>'
    f'</div>',
    unsafe_allow_html=True
)

flagged_count = int(df["anomaly"].sum())
flagged_pct   = 100 * df["anomaly"].mean()

if has_labels:
    tp, fp, fn_c, precision, recall = score(df)
    f1_val = f1(precision, recall)
    cols = st.columns(8)
    cols[0].metric("Flagged",   f"{flagged_count:,}")
    cols[1].metric("Flag rate", f"{flagged_pct:.1f}%")
    cols[2].metric("TP",        f"{tp:,}")
    cols[3].metric("FP",        f"{fp:,}")
    cols[4].metric("FN",        f"{fn_c:,}")
    cols[5].metric("Precision", f"{precision:.3f}")
    cols[6].metric("Recall",    f"{recall:.3f}")
    cols[7].metric("F1",        f"{f1_val:.3f}")

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    col_p, col_r, col_f = st.columns(3)
    with col_p:
        st.caption(f"Precision ({precision:.3f})")
        st.progress(float(precision))
    with col_r:
        st.caption(f"Recall ({recall:.3f})  — ceiling 61.2% for value-only")
        st.progress(float(recall))
    with col_f:
        st.caption(f"F1 Score ({f1_val:.3f})")
        st.progress(float(f1_val))
else:
    c1, c2 = st.columns(2)
    c1.metric("Flagged",   f"{flagged_count:,}")
    c2.metric("Flag rate", f"{flagged_pct:.1f}%")

st.markdown('<hr style="margin:18px 0">', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
dark_plt()
tabs = st.tabs(["Detection Plot", "Anomaly Overlap", "Feature View", "Anomaly Score", "Detector Comparison"])

with tabs[0]:
    fig, ax = plt.subplots(figsize=(16, 4.5), facecolor=C["bg"])
    ax.set_facecolor(C["bg2"])
    if has_labels:
        for s, e in windows_parsed:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=C["gold"], alpha=0.15)
    ax.plot(df.index, df["value"], linewidth=0.7, color=info["color"], alpha=0.8, label="Signal")
    flagged = df[df["anomaly"]]
    ax.scatter(flagged.index, flagged["value"], color=C["red"], s=8, zorder=5,
               label=f"Flagged ({len(flagged):,})", edgecolors="none")
    ax.set_ylabel("Value", color=C["gray"])
    ax.set_title(f"{dlabel}" + (" — yellow = ground truth" if has_labels else ""), color=C["text"])
    ax.legend(fontsize=9)
    ax.tick_params(colors=C["gray"])
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3a5f")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tabs[1]:
    if has_labels:
        fig, ax = plt.subplots(figsize=(16, 3), facecolor=C["bg"])
        ax.set_facecolor(C["bg2"])
        ax.fill_between(df.index, df["label"],               alpha=0.40, color=C["gold"], label="Ground truth")
        ax.fill_between(df.index, df["anomaly"].astype(int), alpha=0.55, color=C["red"],  label="Flagged")
        ax.set_ylabel("Anomaly (0/1)", color=C["gray"])
        ax.set_title("Flag overlap with ground truth", color=C["text"])
        ax.legend(fontsize=9)
        ax.tick_params(colors=C["gray"])
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e3a5f")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Paste anomaly windows in the sidebar to see the overlap view.")

with tabs[2]:
    v = df["value"]
    lw = long_window if detector == "IsolationForest + Rolling (Best)" else 360
    sustained_dev = v - v.rolling(lw, min_periods=1).mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True, facecolor=C["bg"])
    fig.subplots_adjust(hspace=0.50)
    for ax in (ax1, ax2):
        ax.set_facecolor(C["bg2"])
        ax.tick_params(colors=C["gray"])
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e3a5f")
        if has_labels:
            for s, e in windows_parsed:
                ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=C["gold"], alpha=0.12)
    ax1.plot(df.index, v, linewidth=0.7, color=C["blue"], alpha=0.85)
    ax1.set_ylabel("Signal", color=C["gray"])
    ax1.set_title("Raw signal", color=C["text"])
    ax2.plot(df.index, sustained_dev, linewidth=0.7, color=C["cyan"], alpha=0.85)
    ax2.axhline(0, color="#2d4a6e", linewidth=0.8, linestyle="--")
    ax2.set_ylabel(f"Dev from\n{lw}-row baseline", color=C["gray"])
    ax2.set_title(f"sustained_dev = value - rolling_mean({lw}) — sustained failures stay negative for hours", color=C["text"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tabs[3]:
    if "score" in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True, facecolor=C["bg"])
        fig.subplots_adjust(hspace=0.50)
        for ax in (ax1, ax2):
            ax.set_facecolor(C["bg2"])
            ax.tick_params(colors=C["gray"])
            for sp in ax.spines.values():
                sp.set_edgecolor("#1e3a5f")
            if has_labels:
                for s, e in windows_parsed:
                    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=C["gold"], alpha=0.12)
        ax1.plot(df.index, df["value"], linewidth=0.7, color=info["color"], alpha=0.8)
        ax1.set_ylabel("Signal", color=C["gray"])
        ax1.set_title("Raw signal", color=C["text"])
        ax2.plot(df.index, df["score"], linewidth=0.7, color=C["cyan"], alpha=0.8)
        threshold_val = df.loc[df["anomaly"], "score"].min() if df["anomaly"].any() else None
        if threshold_val is not None:
            ax2.axhline(threshold_val, color=C["red"], linewidth=1.2, linestyle="--", alpha=0.7,
                        label=f"Decision threshold ({threshold_val:.3f})")
            ax2.legend(fontsize=9)
        ax2.set_ylabel("Anomaly score\n(higher = more anomalous)", color=C["gray"])
        ax2.set_title("Continuous anomaly score — not just binary flag/not-flag", color=C["text"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("The anomaly score is the continuous output before applying the threshold. "
                   "High scores indicate points the model finds most unusual. "
                   "The decision threshold is where the model switches from 'normal' to 'anomaly'.")
    else:
        st.info("Anomaly scores not available for this detector.")

with tabs[4]:
    ALL_RESULTS = [
        ("3-Sigma (Global)",            0.991, 0.202, 0.336, C["blue"]),
        ("EWMA (span=5000)",            0.586, 0.513, 0.547, C["amb"]),
        ("Hampel Filter (w=144)",       0.107, 0.033, 0.050, C["pur"]),
        ("LOF (k=20)",                  0.109, 0.131, 0.119, "#ec4899"),
        ("IsolationForest (Raw)",       0.471, 0.566, 0.514, C["teal"]),
        ("IsolationForest + Rolling",   0.568, 0.682, 0.619, C["cyan"]),
    ]
    import numpy as _np
    dark_plt()
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=C["bg"])
    ax.set_facecolor(C["bg2"])
    names = [r[0] for r in ALL_RESULTS]
    precs = [r[1] for r in ALL_RESULTS]
    recs  = [r[2] for r in ALL_RESULTS]
    f1s   = [r[3] for r in ALL_RESULTS]
    x = _np.arange(len(names))
    w = 0.25
    ax.bar(x - w, precs, w, label="Precision", color=C["blue"],  alpha=0.85, zorder=3)
    ax.bar(x,     recs,  w, label="Recall",    color=C["cyan"],  alpha=0.85, zorder=3)
    ax.bar(x + w, f1s,   w, label="F1",        color=C["green"], alpha=0.85, zorder=3)
    for i, (p, r, f_) in enumerate(zip(precs, recs, f1s)):
        ax.text(x[i]-w, p+.015, f"{p:.2f}", ha="center", va="bottom", fontsize=7, color=C["gray"])
        ax.text(x[i],   r+.015, f"{r:.2f}", ha="center", va="bottom", fontsize=7, color=C["gray"])
        ax.text(x[i]+w, f_+.015,f"{f_:.2f}",ha="center", va="bottom", fontsize=7, color=C["gray"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, color=C["text"], rotation=12, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color=C["gray"])
    ax.set_title("All Detectors — Full Comparison", color=C["text"])
    ax.legend(fontsize=9)
    ax.tick_params(colors=C["gray"])
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3a5f")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.caption(
        "Recall ceiling of 61.2% applies to value-only detectors (methods that only see the raw reading). "
        "IsolationForest + Rolling (w=360) achieves recall=0.682, breaking through this ceiling because "
        "the sustained_dev feature provides contextual information beyond the raw value."
    )

# ── How it works ───────────────────────────────────────────────────────────
with st.expander("How each detector works"):
    for name, info_d in DETECTOR_INFO.items():
        st.markdown(
            f"**{name}** — best for: *{info_d['best_for']}*\n\n{info_d['desc']}\n"
        )
        st.markdown("---")
    st.markdown(
        "**Recall ceiling (61.2%)** — 38.8% of labeled anomaly points in the NAB dataset "
        "have temperatures in the normal range because anomaly windows start before the temperature drops. "
        "No raw-value-only detector can flag a reading that looks identical to a normal one. "
        "The sustained_dev feature breaks this ceiling by adding temporal context."
    )
