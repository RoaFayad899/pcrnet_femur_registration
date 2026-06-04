import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# PATHS
# ==========================================================

log_file = "/home/roa.fayad/pcrnet_checkpoints_geodesic_translation/training_log.csv"

output_figure = (
    "/home/roa.fayad/pcrnet_checkpoints_geodesic_translation/loss_curves.png"
)

# ==========================================================
# LOAD CSV
# ==========================================================

df = pd.read_csv(log_file)

print(df.head())


# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["train_loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Geodesic + Translation Loss")

plt.title("PCRNet Training and Validation Loss")


# ==========================================================
# BETTER Y AXIS SCALE
# ==========================================================

max_visible = max(
    df["train_loss"].quantile(0.95),
    df["val_loss"].quantile(0.95)
)

plt.ylim(0, max_visible * 1.1)

# ==========================================================
# STYLE
# ==========================================================

plt.grid(True)
plt.legend()


# ==========================================================
# SAVE
# ==========================================================

plt.savefig(
    output_figure,
    dpi=300,
    bbox_inches="tight"
)

print("\nFigure saved to:")
print(output_figure)

plt.show()