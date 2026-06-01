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
plt.ylabel("Loss")

plt.title("PCRNet Training and Validation Loss")


# ==========================================================
# Y AXIS SCALE
# ==========================================================

max_loss = max(
    df["train_loss"].max(),
    df["val_loss"].max()
)

plt.yticks(
    range(
        0,
        int(max_loss) + 20,
        10   # change to 20 if you want larger spacing
    )
)


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