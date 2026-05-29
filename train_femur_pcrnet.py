import os
import torch
from torch.utils.data import DataLoader

from pcrnet.data_utils import FemurPCRNetDataset
from pcrnet.models.pcrnet import iPCRNet
from pcrnet.losses.chamfer_distance import ChamferDistanceLoss


# ==========================================================
# SETTINGS FOR LOCAL TEST ONLY
# ==========================================================

dataset_dir = r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur"

epochs = 2
batch_size = 2
learning_rate = 1e-3
max_iteration = 1

checkpoint_dir = r"C:\data_unibas\pcrnet_checkpoints_test"
os.makedirs(checkpoint_dir, exist_ok=True)

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
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ==========================================================
# MODEL, LOSS, OPTIMIZER
# ==========================================================

model = iPCRNet().to(device)

criterion = ChamferDistanceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(epochs):

    print(f"\n========== EPOCH {epoch + 1}/{epochs} ==========")

    # --------------------------
    # TRAIN
    # --------------------------

    model.train()
    train_loss_total = 0.0

    for batch_idx, batch in enumerate(train_loader):

        source = batch["source"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        result = model(
            template=target,
            source=source,
            max_iteration=max_iteration
        )

        transformed_source = result["transformed_source"]

        loss = criterion(
            target,
            transformed_source
        )

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()

        if batch_idx % 20 == 0:
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

            result = model(
                template=target,
                source=source,
                max_iteration=max_iteration
            )

            transformed_source = result["transformed_source"]

            val_loss = criterion(
                target,
                transformed_source
            )

            val_loss_total += val_loss.item()

    avg_val_loss = val_loss_total / len(val_loader)

    print(f"Average train loss: {avg_train_loss:.6f}")
    print(f"Average val loss:   {avg_val_loss:.6f}")

    # --------------------------
    # SAVE CHECKPOINT
    # --------------------------

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


print("\nDONE: tiny local training test finished.")