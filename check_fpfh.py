from torch.utils.data import DataLoader
from pcrnet.data_utils.femur_dataset_fpfh import FemurPCRNetDatasetFPFH

dataset = FemurPCRNetDatasetFPFH(
    dataset_dir="/home/roa.fayad/pcrnet_dataset_partial_fragment_to_full_femur_large_fpfh",
    split="train"
)

loader = DataLoader(dataset, batch_size=4)

batch = next(iter(loader))

source = batch["source"]

print("XYZ min:", source[:, :, :3].min())
print("XYZ max:", source[:, :, :3].max())

print("FPFH min:", source[:, :, 3:].min())
print("FPFH max:", source[:, :, 3:].max())

print("FPFH mean:", source[:, :, 3:].mean())
print("FPFH std:", source[:, :, 3:].std())