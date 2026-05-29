import torch
import numpy as np
from torch.utils.data import DataLoader

from pcrnet.data_utils import FemurPCRNetDataset
from pcrnet.models.pcrnet import iPCRNet
from pcrnet.losses.chamfer_distance import ChamferDistanceLoss


# ==========================================================
# PATHS
# ==========================================================

dataset_dir = "/home/roa.fayad/pcrnet_dataset_partial_fragment_to_full_femur"

checkpoint_path = "/home/roa.fayad/pcrnet_checkpoints_chamfer/best_model.pth"


# ==========================================================
# SETTINGS
# ==========================================================

batch_size = 16
max_iteration = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# HELPERS
# ==========================================================

def rotation_error_degrees(R_pred, R_gt):
    """
    Computes rotation error in degrees between predicted and GT rotation.
    R_pred, R_gt: [B, 3, 3]
    """
    R_diff = torch.bmm(R_pred, R_gt.transpose(1, 2))

    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle = torch.acos(cos_angle)
    angle_deg = angle * 180.0 / np.pi

    return angle_deg


def translation_error_mm(t_pred, t_gt):
    """
    Computes translation error in mm.
    t_pred: [B, 1, 3]
    t_gt:   [B, 3]
    """
    t_pred = t_pred.squeeze(1)
    return torch.linalg.norm(t_pred - t_gt, dim=1)


# ==========================================================
# DATASET
# ==========================================================

test_dataset = FemurPCRNetDataset(
    dataset_dir=dataset_dir,
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print("test dataset:", len(test_dataset), "samples")


# ==========================================================
# MODEL
# ==========================================================

model = iPCRNet().to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint:")
print("epoch:", checkpoint["epoch"])
print("train_loss:", checkpoint["train_loss"])
print("val_loss:", checkpoint["val_loss"])
print("max_iteration:", checkpoint["max_iteration"])


# ==========================================================
# LOSS
# ==========================================================

criterion = ChamferDistanceLoss()


# ==========================================================
# EVALUATION
# ==========================================================

chamfer_before_all = []
chamfer_after_all = []
rotation_errors_all = []
translation_errors_all = []

with torch.no_grad():

    for batch in test_loader:

        source = batch["source"].to(device)
        target = batch["target"].to(device)

        R_gt = batch["R_gt"].to(device)
        t_gt = batch["t_gt"].to(device)

        # Chamfer before registration
        loss_before = criterion(target, source)

        # Model prediction
        result = model(
            template=target,
            source=source,
            max_iteration=max_iteration
        )

        transformed_source = result["transformed_source"]
        R_pred = result["est_R"]
        t_pred = result["est_t"]

        # Chamfer after registration
        loss_after = criterion(target, transformed_source)

        # Transformation errors
        rot_err = rotation_error_degrees(R_pred, R_gt)
        trans_err = translation_error_mm(t_pred, t_gt)

        chamfer_before_all.append(loss_before.item())
        chamfer_after_all.append(loss_after.item())
        rotation_errors_all.extend(rot_err.cpu().numpy())
        translation_errors_all.extend(trans_err.cpu().numpy())


# ==========================================================
# RESULTS
# ==========================================================

rotation_errors_all = np.array(rotation_errors_all)
translation_errors_all = np.array(translation_errors_all)

print("\n========== TEST RESULTS ==========")

print(f"Chamfer before registration: {np.mean(chamfer_before_all):.6f}")
print(f"Chamfer after registration:  {np.mean(chamfer_after_all):.6f}")

print("\nRotation error [degrees]:")
print(f"Mean:   {rotation_errors_all.mean():.6f}")
print(f"Median: {np.median(rotation_errors_all):.6f}")
print(f"Std:    {rotation_errors_all.std():.6f}")

print("\nTranslation error [mm]:")
print(f"Mean:   {translation_errors_all.mean():.6f}")
print(f"Median: {np.median(translation_errors_all):.6f}")
print(f"Std:    {translation_errors_all.std():.6f}")