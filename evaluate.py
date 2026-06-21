import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/machine_temp.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp").sort_index()

windows = [
    ("2013-12-10 06:25:00", "2013-12-12 05:35:00"),
    ("2013-12-15 17:50:00", "2013-12-17 17:00:00"),
    ("2014-01-27 14:20:00", "2014-01-29 13:30:00"),
    ("2014-02-07 14:55:00", "2014-02-09 14:05:00"),
]

df["label"] = 0
for start, end in windows:
    df.loc[start:end, "label"] = 1

mean = df["value"].mean()
std = df["value"].std()
df["anomaly"] = ((df["value"] - mean) / std).abs() > 3

tp = ((df["anomaly"] == True) & (df["label"] == 1)).sum()
fp = ((df["anomaly"] == True) & (df["label"] == 0)).sum()
fn = ((df["anomaly"] == False) & (df["label"] == 1)).sum()

precision = tp / (tp + fp)
recall    = tp / (tp + fn)

print(f"True anomaly points (inside windows): {df['label'].sum()}")
print(f"Points flagged by 3-sigma:             {df['anomaly'].sum()}")
print(f"\nTP: {tp}  |  FP: {fp}  |  FN: {fn}")
print(f"\nPrecision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

for start, end in windows:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="yellow", alpha=0.3)
ax1.plot(df.index, df["value"], linewidth=0.6, color="steelblue")
flagged = df[df["anomaly"]]
ax1.scatter(flagged.index, flagged["value"], color="red", s=10, zorder=5, label="3-sigma flag")
ax1.set_ylabel("Temperature")
ax1.set_title("3-Sigma Flags vs Ground-Truth Windows (yellow)")
ax1.legend()

ax2.fill_between(df.index, df["label"], alpha=0.4, color="gold", label="Ground truth")
ax2.fill_between(df.index, df["anomaly"].astype(int), alpha=0.4, color="red", label="3-sigma flag")
ax2.set_ylabel("Anomaly (0/1)")
ax2.set_title(f"Overlap  —  Precision: {precision:.3f}  |  Recall: {recall:.3f}")
ax2.legend()

plt.tight_layout()
plt.savefig("data/evaluate_plot.png", dpi=150)
plt.show()
