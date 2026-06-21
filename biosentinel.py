import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


# ── Detectors ─────────────────────────────────────────────────────────────────

def detect_3sigma(df):
    mean = df["value"].mean()
    std  = df["value"].std()
    df["anomaly"] = ((df["value"] - mean) / std).abs() > 3
    df["score"]   = ((df["value"] - mean) / std).abs()
    return df


def detect_rolling_3sigma(df, window=1000):
    roll         = df["value"].rolling(window, min_periods=1)
    rolling_mean = roll.mean()
    rolling_std  = roll.std().fillna(df["value"].std())
    z            = ((df["value"] - rolling_mean) / rolling_std).abs()
    df["anomaly"] = z > 3
    df["score"]   = z
    return df


def detect_ewma(df, span=5000, threshold=2.0):
    ewm_mean = df["value"].ewm(span=span, min_periods=1).mean()
    ewm_std  = df["value"].ewm(span=span, min_periods=1).std().fillna(df["value"].std())
    z        = ((df["value"] - ewm_mean) / ewm_std).abs()
    df["anomaly"] = z > threshold
    df["score"]   = z
    return df


def detect_hampel(df, window=144, n_sigma=3):
    """
    Hampel filter: flag points where |value - rolling_median| > n_sigma * k * MAD.
    k=1.4826 makes MAD equivalent to std for Gaussian data.
    Uses a centered window so it can see both past and future context.
    """
    k   = 1.4826
    v   = df["value"]
    med = v.rolling(window, center=True, min_periods=1).median()
    mad = (v - med).abs().rolling(window, center=True, min_periods=1).median()
    threshold = n_sigma * k * mad
    dev = (v - med).abs()
    df["anomaly"] = dev > threshold
    df["score"]   = (dev / (threshold + 1e-9))
    return df


def detect_lof(df, n_neighbors=20, contamination=0.12):
    """
    Local Outlier Factor: compares the local density of each point to its neighbors.
    Points in sparser regions than their neighbors get high LOF scores (anomalies).
    """
    v   = df["value"].values.reshape(-1, 1)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    df["anomaly"] = clf.fit_predict(v) == -1
    df["score"]   = -clf.negative_outlier_factor_
    return df


def detect_iforest(df, contamination=0.12):
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    df["anomaly"] = clf.fit_predict(df[["value"]]) == -1
    df["score"]   = -clf.decision_function(df[["value"]])
    return df


def detect_iforest_rolling(df, contamination=0.12, long_window=144):
    """
    IsolationForest with a sustained-deviation feature.
    sustained_dev = value - rolling_mean(long_window).
    During a sustained failure the current reading stays far below the slow-moving
    baseline, giving the model an explicit signal that a threshold cannot see.
    """
    v             = df["value"]
    sustained_dev = v - v.rolling(long_window, min_periods=1).mean()
    X             = pd.DataFrame({"value": v, "sustained_dev": sustained_dev}, index=df.index)
    clf           = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    clf.fit(X)
    df["anomaly"] = clf.predict(X) == -1
    df["score"]   = -clf.decision_function(X)
    return df


# ── Evaluation ────────────────────────────────────────────────────────────────

def score(df):
    tp        = (df["anomaly"] & (df["label"] == 1)).sum()
    fp        = (df["anomaly"] & (df["label"] == 0)).sum()
    fn        = (~df["anomaly"] & (df["label"] == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return int(tp), int(fp), int(fn), float(precision), float(recall)


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# ── Plotting ──────────────────────────────────────────────────────────────────

_STYLE = {
    "BG":   "#050d1a", "BG2":  "#0a1628",
    "CYAN": "#00d4ff", "RED":  "#ef4444",
    "GOLD": "#fbbf24", "GRAY": "#94a3b8",
    "TEXT": "#f1f5f9", "BLUE": "#3b82f6",
}


def _apply_dark():
    plt.rcParams.update({
        "font.family": "sans-serif", "figure.facecolor": _STYLE["BG"],
        "axes.facecolor": _STYLE["BG2"], "axes.edgecolor": "#1e3a5f",
        "axes.labelcolor": _STYLE["GRAY"], "xtick.color": _STYLE["GRAY"],
        "ytick.color": _STYLE["GRAY"], "axes.grid": True,
        "grid.color": "#1e3a5f", "grid.alpha": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": _STYLE["BG2"], "legend.edgecolor": "#1e3a5f",
    })


def save_plot(df, output_path, detector_name):
    _apply_dark()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, facecolor=_STYLE["BG"])

    ax1.plot(df.index, df["value"], linewidth=0.7, color=_STYLE["CYAN"], alpha=0.8)
    flagged = df[df["anomaly"]]
    ax1.scatter(flagged.index, flagged["value"], color=_STYLE["RED"], s=8,
                zorder=5, label=f"Flagged ({len(flagged):,})", edgecolors="none")
    ax1.set_ylabel("Value", color=_STYLE["GRAY"])
    ax1.set_title(detector_name, color=_STYLE["TEXT"])
    ax1.legend(fontsize=9)
    ax1.set_facecolor(_STYLE["BG2"])

    if "label" in df.columns:
        ax2.fill_between(df.index, df["label"], alpha=0.35, color=_STYLE["GOLD"], label="Ground truth")
    ax2.fill_between(df.index, df["anomaly"].astype(int), alpha=0.5, color=_STYLE["RED"], label="Flagged")
    ax2.set_ylabel("Anomaly (0/1)", color=_STYLE["GRAY"])
    ax2.legend(fontsize=9)
    ax2.set_facecolor(_STYLE["BG2"])

    if "score" in df.columns:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df.index, df["score"], linewidth=0.6, color=_STYLE["CYAN"],
                      alpha=0.45, label="Anomaly score")
        ax2_twin.set_ylabel("Score (higher = more anomalous)", color=_STYLE["CYAN"], fontsize=8)
        ax2_twin.tick_params(colors=_STYLE["CYAN"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=_STYLE["BG"], bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_detector_args(p):
    p.add_argument(
        "--detector",
        choices=["3sigma", "rolling-3sigma", "ewma", "hampel", "lof", "iforest", "rolling-iforest"],
        default="rolling-iforest",
        help="Detector to use (default: rolling-iforest)",
    )
    p.add_argument("--contamination", type=float, default=0.12,
                   help="Expected anomaly fraction for IsolationForest/LOF (default: 0.12)")
    p.add_argument("--window",      type=int, default=1000,
                   help="Rolling window rows for rolling-3sigma and hampel (default: 1000)")
    p.add_argument("--span",        type=int, default=5000,
                   help="EWMA span in rows (default: 5000)")
    p.add_argument("--long-window", type=int, default=360,
                   help="Sustained-deviation window for rolling-iforest (default: 360 = 30h at 5-min intervals)")
    p.add_argument("--n-neighbors", type=int, default=20,
                   help="LOF neighbor count (default: 20)")


def _build_detector(args, df):
    detector_map = {
        "3sigma":          (detect_3sigma,         [],                                      "3-Sigma (Global)"),
        "rolling-3sigma":  (detect_rolling_3sigma,  [args.window],                          f"Rolling 3-Sigma (w={args.window})"),
        "ewma":            (detect_ewma,            [args.span],                            f"EWMA (span={args.span})"),
        "hampel":          (detect_hampel,          [args.window],                          f"Hampel Filter (w={args.window})"),
        "lof":             (detect_lof,             [args.n_neighbors, args.contamination], f"LOF (k={args.n_neighbors})"),
        "iforest":         (detect_iforest,         [args.contamination],                   f"IsolationForest (c={args.contamination})"),
        "rolling-iforest": (detect_iforest_rolling, [args.contamination, args.long_window],
                            f"IsolationForest + Rolling (c={args.contamination}, w={args.long_window})"),
    }
    fn, fn_args, name = detector_map[args.detector]
    return fn(df, *fn_args), name


def main():
    parser = argparse.ArgumentParser(
        prog="biosentinel",
        description="BioSentinel — anomaly detection for biomedical time-series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python biosentinel.py detect data/machine_temp.csv\n"
            "  python biosentinel.py detect data/machine_temp.csv --detector ewma\n"
            "  python biosentinel.py detect data/machine_temp.csv --all --export flagged.csv\n"
            "  python biosentinel.py detect data/machine_temp.csv --plot out.png\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── detect ────────────────────────────────────────────────────────────────
    p_det = sub.add_parser(
        "detect",
        help="Run a detector and print flagged timestamps",
    )
    p_det.add_argument("csv", help="CSV file with 'timestamp' and 'value' columns")
    _add_detector_args(p_det)
    p_det.add_argument("--top",   type=int, default=20,
                       help="Print top N flagged timestamps by anomaly score (default: 20)")
    p_det.add_argument("--all",   action="store_true",
                       help="Print all flagged timestamps, not just --top")
    p_det.add_argument("--labels", nargs="+", metavar="TIMESTAMP",
                       help="Ground-truth anomaly windows: START1 END1 START2 END2 ...")
    p_det.add_argument("--plot",   metavar="OUTPUT.png",
                       help="Save a two-panel detection plot to this path")
    p_det.add_argument("--export", metavar="OUTPUT.csv",
                       help="Save full results (anomaly flag + score column) to this CSV")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # ── load ──────────────────────────────────────────────────────────────────
    df = load_data(args.csv)
    print(f"\nLoaded {len(df):,} rows from {args.csv}")
    print(f"Date range: {df.index.min()}  to  {df.index.max()}")
    print(f"Mean: {df['value'].mean():.2f}   Std: {df['value'].std():.2f}   "
          f"Min: {df['value'].min():.2f}   Max: {df['value'].max():.2f}")

    df, detector_name = _build_detector(args, df)

    flagged_ct = int(df["anomaly"].sum())
    print(f"\nDetector:       {detector_name}")
    print(f"Points flagged: {flagged_ct:,} / {len(df):,}  ({100 * df['anomaly'].mean():.2f}%)")

    # ── print flagged timestamps ───────────────────────────────────────────────
    flagged_df = df[df["anomaly"]].sort_values("score", ascending=False)
    to_show = flagged_df if args.all else flagged_df.head(args.top)
    label   = f"all {flagged_ct:,}" if args.all else f"top {args.top} by anomaly score"

    print(f"\nFlagged timestamps ({label}):")
    print(f"  {'Timestamp':<26}  {'Value':>8}  {'Score':>8}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*8}")
    for ts, row in to_show.iterrows():
        print(f"  {str(ts):<26}  {row['value']:>8.2f}  {row['score']:>8.4f}")

    # ── optional: evaluate against ground-truth labels ─────────────────────────
    if args.labels:
        if len(args.labels) % 2 != 0:
            print("\nERROR: --labels must be pairs of START END timestamps.")
            return
        df["label"] = 0
        for start, end in zip(args.labels[0::2], args.labels[1::2]):
            df.loc[start:end, "label"] = 1

        tp, fp, fn_c, p, r = score(df)
        f1_val        = f1(p, r)
        total_anomaly = int(df["label"].sum())

        print(f"\nGround truth:   {total_anomaly:,} labeled anomaly points")
        print(f"TP: {tp:,}  |  FP: {fp:,}  |  FN: {fn_c:,}")
        print(f"Precision:  {p:.3f}")
        print(f"Recall:     {r:.3f}")
        print(f"F1:         {f1_val:.3f}")

        ceiling = 0.612
        print(f"\nRecall ceiling for value-only detectors: {ceiling:.1%}")
        if r > 0:
            print(f"This detector reaches {r / ceiling:.1%} of the theoretical ceiling.")

    # ── optional: export ──────────────────────────────────────────────────────
    if args.export:
        df.to_csv(args.export)
        print(f"\nResults saved to {args.export}")

    # ── optional: plot ────────────────────────────────────────────────────────
    if args.plot:
        save_plot(df, args.plot, detector_name)


if __name__ == "__main__":
    main()
