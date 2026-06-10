import os
import csv
import torch
from torch.utils.data import DataLoader

from pcrnet.data_utils import FemurPCRNetDataset
from pcrnet.models.pcrnet_6Drepresentation import iPCRNet      ################
from pcrnet.losses.geodesic_translation_loss import GeodesicTranslationLoss

from torch.utils.tensorboard import SummaryWriter
# ==========================================================
# SETTINGS
# ==========================================================

dataset_dir = "/home/roa.fayad/pcrnet_dataset_partial_fragment_to_full_femur"  #dataset_dir = r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur"

checkpoint_dir = "/home/roa.fayad/pcrnet_checkpoints_overfit_one_sample_msegeodesic_6drepresentation_800samples_iter1_lambda500"   ###r"C:\data_unibas\pcrnet_checkpoints_chamfer"
os.makedirs(checkpoint_dir, exist_ok=True)

log_file = os.path.join(checkpoint_dir, "training_log.csv")

tensorboard_dir = os.path.join(checkpoint_dir, "tensorboard")
tb_writer = SummaryWriter(log_dir=tensorboard_dir)

epochs = 1000  ####100 or 2
batch_size = 32  #######16 or 2
learning_rate = 1e-4 #############
max_iteration = 1 ##########8 or 1

save_every = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# DATASETS AND LOADERS
# ==========================================================

train_dataset = FemurPCRNetDataset(
    dataset_dir=dataset_dir,
    split="train"
)

val_dataset = FemurPCRNetDataset(
    dataset_dir=dataset_dir,
    split="val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# ==========================================================
# MODEL, LOSS, OPTIMIZER
# ==========================================================

model = iPCRNet().to(device)

criterion = GeodesicTranslationLoss(lambda_translation= 500)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ==========================================================
# LOG FILE
# ==========================================================

with open(log_file, mode="w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "learning_rate",
        "max_iteration"
    ])


best_val_loss = float("inf")


# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(epochs):

    print(f"\n========== EPOCH {epoch + 1}/{epochs} ==========")

    # --------------------------
    # TRAINING
    # --------------------------

    model.train()
    train_loss_total = 0.0

    for batch_idx, batch in enumerate(train_loader):

        source = batch["source"].to(device)
        target = batch["target"].to(device)

        R_gt = batch["R_gt"].to(device)
        t_gt = batch["t_gt"].to(device)

        optimizer.zero_grad()

        result = model(
            template=target,
            source=source,
            max_iteration=max_iteration
        )

        R_pred = result["est_R"]
        t_pred = result["est_t"]

        loss_dict = criterion(
            R_pred,
            t_pred,
            R_gt,
            t_gt
        )

        loss = loss_dict["total_loss"]

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Train batch {batch_idx}/{len(train_loader)} "
                f"| loss = {loss.item():.6f}"
            )

    avg_train_loss = train_loss_total / len(train_loader)

    # --------------------------
    # VALIDATION
    # --------------------------

    model.eval()
    val_loss_total = 0.0

    with torch.no_grad():

        for batch in val_loader:

            source = batch["source"].to(device)
            target = batch["target"].to(device)

            R_gt = batch["R_gt"].to(device)
            t_gt = batch["t_gt"].to(device)

            result = model(
                template=target,
                source=source,
                max_iteration=max_iteration
            )

            R_pred = result["est_R"]
            t_pred = result["est_t"]

            loss_dict = criterion(
                R_pred,
                t_pred,
                R_gt,
                t_gt
            )

            loss = loss_dict["total_loss"]

            val_loss_total += loss.item()

    avg_val_loss = val_loss_total / len(val_loader)

    print(f"Average train loss: {avg_train_loss:.6f}")
    print(f"Average val loss:   {avg_val_loss:.6f}")

    # --------------------------
    # WRITE LOG
    # --------------------------

    current_lr = optimizer.param_groups[0]["lr"]

    tb_writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
    tb_writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
    tb_writer.add_scalar("Learning_rate", current_lr, epoch + 1)

    with open(log_file, mode="a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            current_lr,
            max_iteration
        ])

    # --------------------------
    # SAVE BEST CHECKPOINT
    # --------------------------

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        best_path = os.path.join(
            checkpoint_dir,
            "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "max_iteration": max_iteration,
            },
            best_path
        )

        print("Saved best model:", best_path)

    # --------------------------
    # SAVE PERIODIC CHECKPOINT
    # --------------------------

    if (epoch + 1) % save_every == 0:

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"pcrnet_epoch_{epoch + 1}.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "max_iteration": max_iteration,
            },
            checkpoint_path
        )

        print("Saved checkpoint:", checkpoint_path)


print("\nDONE: training finished.")
print("Best validation loss:", best_val_loss)
print("Checkpoints saved in:", checkpoint_dir)
print("Log file:", log_file)

tb_writer.close()