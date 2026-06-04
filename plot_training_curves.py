import pandas as pd
import matplotlib.pyplot as plt


log_file = "/home/roa.fayad/pcrnet_checkpoints_geodesic_translation/training_log.csv"

output_figure = (
    "/home/roa.fayad/pcrnet_checkpoints_geodesic_translation/loss_curves.png"
)


df = pd.read_csv(log_file)

print(df.head())


# Remove first epoch from plot because it is an outlier
df_plot = df[df["epoch"] > 1]


plt.figure(figsize=(8, 5))

plt.plot(
    df_plot["epoch"],
    df_plot["train_loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    df_plot["epoch"],
    df_plot["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Geodesic + Translation Loss")
plt.title("PCRNet Training and Validation Loss")

# better y-axis scaling
max_visible = max(
    df_plot["train_loss"].quantile(0.95),
    df_plot["val_loss"].quantile(0.95)
)

plt.ylim(0, max_visible * 1.1)

plt.grid(True)
plt.legend()

plt.savefig(
    output_figure,
    dpi=300,
    bbox_inches="tight"
)

print("\nFigure saved to:")
print(output_figure)

plt.show()