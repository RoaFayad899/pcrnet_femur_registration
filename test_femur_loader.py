from torch.utils.data import DataLoader

from pcrnet.data_utils import FemurPCRNetDataset


dataset_dir = r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur"


train_dataset = FemurPCRNetDataset(
    dataset_dir,
    split="train"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)


batch = next(iter(train_loader))


print("\n========== BATCH INFO ==========")

print("source shape:", batch["source"].shape)
print("target shape:", batch["target"].shape)

print("R_gt shape:", batch["R_gt"].shape)
print("t_gt shape:", batch["t_gt"].shape)
print("T_gt shape:", batch["T_gt"].shape)

print("\nsource dtype:", batch["source"].dtype)
print("target dtype:", batch["target"].dtype)