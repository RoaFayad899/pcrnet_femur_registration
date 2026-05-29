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
    Rotation error in degrees.
    R_pred, R_gt: [B, 3, 3]
    """
    R_diff = torch.bmm(R_pred, R_gt.transpose(1, 2))

    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle = torch.acos(cos_angle)
    return angle * 180.0 / np.pi


def translation_error_mm(t_pred, t_gt):
    """
    t_pred: [B, 1, 3] or [B, 3]
    t_gt:   [B, 3]
    """
    if t_pred.ndim == 3:
        t_pred = t_pred.squeeze(1)

    return torch.linalg.norm(t_pred - t_gt, dim=1)


def invert_transform(R, t):
    """
    Invert rigid transform.

    If:
        x2 = R x1 + t

    then inverse:
        x1 = R_inv x2 + t_inv

    R: [B, 3, 3]
    t: [B, 3]
    """
    R_inv = R.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)

    return R_inv, t_inv

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
rotation_errors_gt_all = []
translation_errors_gt_all = []

rotation_errors_inv_gt_all = []
translation_errors_inv_gt_all = []

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

        # ----------------------------------------------------------
        # Transformation errors against stored GT
        # ----------------------------------------------------------

        rot_err_gt = rotation_error_degrees(R_pred, R_gt)
        trans_err_gt = translation_error_mm(t_pred, t_gt)

        # ----------------------------------------------------------
        # Transformation errors against inverse GT
        # ----------------------------------------------------------

        R_gt_inv, t_gt_inv = invert_transform(R_gt, t_gt)

        rot_err_inv_gt = rotation_error_degrees(R_pred, R_gt_inv)
        trans_err_inv_gt = translation_error_mm(t_pred, t_gt_inv)

        # Store
        chamfer_before_all.append(loss_before.item())
        chamfer_after_all.append(loss_after.item())

        rotation_errors_gt_all.extend(rot_err_gt.cpu().numpy())
        translation_errors_gt_all.extend(trans_err_gt.cpu().numpy())

        rotation_errors_inv_gt_all.extend(rot_err_inv_gt.cpu().numpy())
        translation_errors_inv_gt_all.extend(trans_err_inv_gt.cpu().numpy())


# ==========================================================
# RESULTS
# ==========================================================

rotation_errors_gt_all = np.array(rotation_errors_gt_all)
translation_errors_gt_all = np.array(translation_errors_gt_all)

rotation_errors_inv_gt_all = np.array(rotation_errors_inv_gt_all)
translation_errors_inv_gt_all = np.array(translation_errors_inv_gt_all)

print("\n========== TEST RESULTS ==========")

print(f"Chamfer before registration: {np.mean(chamfer_before_all):.6f}")
print(f"Chamfer after registration:  {np.mean(chamfer_after_all):.6f}")

print("\n========== AGAINST STORED GT ==========")

print("\nRotation error [degrees]:")
print(f"Mean:   {rotation_errors_gt_all.mean():.6f}")
print(f"Median: {np.median(rotation_errors_gt_all):.6f}")
print(f"Std:    {rotation_errors_gt_all.std():.6f}")

print("\nTranslation error [mm]:")
print(f"Mean:   {translation_errors_gt_all.mean():.6f}")
print(f"Median: {np.median(translation_errors_gt_all):.6f}")
print(f"Std:    {translation_errors_gt_all.std():.6f}")

print("\n========== AGAINST INVERSE GT ==========")

print("\nRotation error [degrees]:")
print(f"Mean:   {rotation_errors_inv_gt_all.mean():.6f}")
print(f"Median: {np.median(rotation_errors_inv_gt_all):.6f}")
print(f"Std:    {rotation_errors_inv_gt_all.std():.6f}")

print("\nTranslation error [mm]:")
print(f"Mean:   {translation_errors_inv_gt_all.mean():.6f}")
print(f"Median: {np.median(translation_errors_inv_gt_all):.6f}")
print(f"Std:    {translation_errors_inv_gt_all.std():.6f}")