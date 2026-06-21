import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/machine_temp.csv", parse_dates=["timestamp"])

df = df.set_index("timestamp")

print(df.head())
print(f"\nShape: {df.shape}")
print(f"Date range: {df.index.min()}  →  {df.index.max()}")

plt.figure(figsize=(14, 4))
plt.plot(df.index, df["value"], linewidth=0.6, color="steelblue")
plt.title("Machine Temperature Over Time")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.tight_layout()
plt.savefig("data/machine_temp_plot.png", dpi=150)
plt.show()
