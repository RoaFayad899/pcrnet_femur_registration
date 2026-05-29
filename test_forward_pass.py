import torch
from torch.utils.data import DataLoader

from pcrnet.data_utils import FemurPCRNetDataset
from pcrnet.models.pcrnet import iPCRNet
from pcrnet.losses.chamfer_distance import ChamferDistanceLoss


# ==========================================================
# DATASET
# ==========================================================

dataset = FemurPCRNetDataset(
    dataset_dir=r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur",
    split="train"
)

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

criterion = ChamferDistanceLoss()
loss = criterion(target, transformed_source)

print("\n========== LOSS ==========")
print("Chamfer loss:", loss.item())