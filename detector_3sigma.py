import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/machine_temp.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp")

mean = df["value"].mean()
std = df["value"].std()

df["zscore"] = (df["value"] - mean) / std

df["anomaly"] = df["zscore"].abs() > 3

print(f"Mean:  {mean:.2f}")
print(f"Std:   {std:.2f}")
print(f"Upper threshold: {mean + 3 * std:.2f}")
print(f"Lower threshold: {mean - 3 * std:.2f}")
print(f"\nAnomalies flagged: {df['anomaly'].sum()} / {len(df)} ({100 * df['anomaly'].mean():.2f}%)")

anomalies = df[df["anomaly"]]

plt.figure(figsize=(14, 4))
plt.plot(df.index, df["value"], linewidth=0.6, color="steelblue", label="Temperature")
plt.scatter(anomalies.index, anomalies["value"], color="red", s=10, zorder=5, label="Flagged anomaly")
plt.axhline(mean + 3 * std, color="orange", linewidth=0.8, linestyle="--", label="3-sigma bounds")
plt.axhline(mean - 3 * std, color="orange", linewidth=0.8, linestyle="--")
plt.title("3-Sigma Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.tight_layout()
plt.savefig("data/3sigma_plot.png", dpi=150)
plt.show()
