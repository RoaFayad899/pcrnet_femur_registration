import torch
from torch.utils.data import DataLoader

from pcrnet.data_utils import FemurPCRNetDataset
from pcrnet.models.pcrnet import iPCRNet
from pcrnet.losses.geodesic_translation_loss import GeodesicTranslationLoss


# ==========================================================
# DATASET
# ==========================================================

dataset = FemurPCRNetDataset(
    dataset_dir= "/home/roa.fayad/pcrnet_dataset_partial_fragment_to_full_femur",    ### r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur"
    split="train")

# dataset = FemurPCRNetDataset(
#     dataset_dir= r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur",    ### r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur"
#     split="train")


loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

batch = next(iter(loader))

source = batch['source']
target = batch['target']

print("\n========== INPUT ==========")
print("source:", source.shape)
print("target:", target.shape)


# ==========================================================
# MODEL
# ==========================================================

model = iPCRNet()


# ==========================================================
# FORWARD PASS
# ==========================================================

result = model(target, source, max_iteration=8)

transformed_source = result['transformed_source']

print("\n========== OUTPUT ==========")
print("transformed source:", transformed_source.shape)
print("estimated rotation:", result['est_R'].shape)
print("estimated translation:", result['est_t'].shape)


# ==========================================================
# LOSS
# ==========================================================

criterion = GeodesicTranslationLoss(lambda_translation=0.01)

loss_dict = criterion(
    result["est_R"],
    result["est_t"],
    batch["R_gt"],
    batch["t_gt"]
)

print("\n========== LOSS ==========")
print("total loss:", loss_dict["total_loss"].item())
print("rotation loss:", loss_dict["rotation_loss"].item())
print("translation loss:", loss_dict["translation_loss"].item())


print("\n========== GT SCALE CHECK ==========")
print("t_gt first sample:", batch["t_gt"][0])
print("est_t first sample:", result["est_t"][0])