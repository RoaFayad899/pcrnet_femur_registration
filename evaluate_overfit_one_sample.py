import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from pcrnet.data_utils import FemurPCRNetDataset
from pcrnet.models.pcrnet_6Drepresentation import iPCRNet   #######################
#from pcrnet.losses.geodesic_translation_loss import GeodesicTranslationLoss
from pcrnet.losses.one_sided_chamfer_distance import OneSidedChamferDistanceLoss

# ==========================================================
# PATHS
# ==========================================================

dataset_dir = "/home/roa.fayad/pcrnet_dataset_partial_fragment_to_full_femur"
#checkpoint_path = "/home/roa.fayad/pcrnet_checkpoints_overfit_one_sample/best_model.pth"    #############
checkpoint_path = "/home/roa.fayad/pcrnet_checkpoints_overfit_one_sample_chamfer/best_model.pth"
#checkpoint_dir = "/home/roa.fayad/pcrnet_checkpoints_overfit_one_sample_chamfer_iter30/best_model.pth"
# ==========================================================
# SETTINGS
# ==========================================================

batch_size = 1      #####16, 32
max_iteration = 1  #######8, 30
lambda_translation = 10 #######1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# HELPERS
# ==========================================================

def rotation_error_degrees(R_pred, R_gt):
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)

    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    theta_rad = torch.acos(cos_theta)
    theta_deg = theta_rad * 180.0 / np.pi

    return theta_rad, theta_deg


def translation_error_mm(t_pred, t_gt, normalization_scale):

    if t_pred.ndim == 3:
        t_pred = t_pred.squeeze(1)

    error_normalized = torch.linalg.norm(
        t_pred - t_gt,
        dim=1
    )

    error_mm = error_normalized * normalization_scale

    return error_mm


def translation_mse_per_sample(t_pred, t_gt):
    if t_pred.ndim == 3:
        t_pred = t_pred.squeeze(1)

    return torch.mean((t_pred - t_gt) ** 2, dim=1)


# ==========================================================
# DATASET
# ==========================================================

test_dataset = FemurPCRNetDataset(
    dataset_dir=dataset_dir,
    split="train"
)

sample0 = np.load(test_dataset.files[0])
normalization_scale = float(sample0["normalization_scale"])

one_sample_dataset = Subset(test_dataset, [0])

test_loader = DataLoader(
    one_sample_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("overfit sample dataset:", len(one_sample_dataset), "sample")


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

# criterion = GeodesicTranslationLoss(
#     lambda_translation=lambda_translation)

criterion = OneSidedChamferDistanceLoss()            ###################

# ==========================================================
# EVALUATION
# ==========================================================

total_losses = []
rotation_losses_rad = []
rotation_errors_deg = []
translation_mse_values = []
translation_errors_mm = []

with torch.no_grad():

    for batch in test_loader:

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

        if len(translation_errors_mm) == 0:
            print("\n========== DEBUG TRANSLATION ==========")
            print("t_pred first 3:")
            print(t_pred[:3])

            print("\nt_gt first 3:")
            print(t_gt[:3])

        # loss_dict = criterion(            ######################
        #     R_pred,
        #     t_pred,
        #     R_gt,
        #     t_gt
        # )
        #
        # total_loss = loss_dict["total_loss"]

        transformed_source = result["transformed_source"]

        total_loss = criterion(
            target,
            transformed_source
        )

        # rotation_loss = loss_dict["rotation_loss"]          #################
        # translation_loss = loss_dict["translation_loss"]

        rot_rad, rot_deg = rotation_error_degrees(R_pred, R_gt)
        trans_err = translation_error_mm(
            t_pred,
            t_gt,
            normalization_scale)

        trans_mse = translation_mse_per_sample(t_pred, t_gt)

        batch_size_actual = source.shape[0]

        total_losses.extend([total_loss.item()] * batch_size_actual)
        rotation_losses_rad.extend(rot_rad.cpu().numpy())
        rotation_errors_deg.extend(rot_deg.cpu().numpy())
        translation_mse_values.extend(trans_mse.cpu().numpy())
        translation_errors_mm.extend(trans_err.cpu().numpy())


# ==========================================================
# RESULTS
# ==========================================================

total_losses = np.array(total_losses)
rotation_losses_rad = np.array(rotation_losses_rad)
rotation_errors_deg = np.array(rotation_errors_deg)
translation_mse_values = np.array(translation_mse_values)
translation_errors_mm = np.array(translation_errors_mm)

#print("\n========== GEODESIC + TRANSLATION MSE EVALUATION ==========")    #############
print("\n========== ONE-SIDED CHAMFER OVERFIT EVALUATION ==========")

#print(f"Lambda translation: {lambda_translation}")    ################

print("\nTotal loss:")
print(f"Mean:   {total_losses.mean():.6f}")
print(f"Median: {np.median(total_losses):.6f}")
print(f"Std:    {total_losses.std():.6f}")

print("\nRotation geodesic loss [radians]:")
print(f"Mean:   {rotation_losses_rad.mean():.6f}")
print(f"Median: {np.median(rotation_losses_rad):.6f}")
print(f"Std:    {rotation_losses_rad.std():.6f}")

print("\nRotation error [degrees]:")
print(f"Mean:   {rotation_errors_deg.mean():.6f}")
print(f"Median: {np.median(rotation_errors_deg):.6f}")
print(f"Std:    {rotation_errors_deg.std():.6f}")

print("\nTranslation MSE:")
print(f"Mean:   {translation_mse_values.mean():.6f}")
print(f"Median: {np.median(translation_mse_values):.6f}")
print(f"Std:    {translation_mse_values.std():.6f}")

print("\nTranslation error [mm]:")
print(f"Mean:   {translation_errors_mm.mean():.6f}")
print(f"Median: {np.median(translation_errors_mm):.6f}")
print(f"Std:    {translation_errors_mm.std():.6f}")