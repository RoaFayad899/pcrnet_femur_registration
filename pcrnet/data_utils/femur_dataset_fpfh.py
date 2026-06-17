import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class FemurPCRNetDatasetFPFH(Dataset):
    def __init__(self, dataset_dir, split="train", train_ratio=0.8, val_ratio=0.1):
        self.dataset_dir = dataset_dir

        all_files = sorted(glob.glob(os.path.join(dataset_dir, "sample_*.npz")))

        if len(all_files) == 0:
            raise RuntimeError(f"No .npz files found in: {dataset_dir}")

        n_total = len(all_files)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        if split == "train":
            self.files = all_files[:n_train]
        elif split == "val":
            self.files = all_files[n_train:n_train + n_val]
        elif split == "test":
            self.files = all_files[n_train + n_val:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        print(f"{split} dataset: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        source_xyz = data["source"].astype(np.float32)          # [N, 3]
        target_xyz = data["target"].astype(np.float32)          # [N, 3]

        source_fpfh = data["source_fpfh"].astype(np.float32)    # [N, 33]
        target_fpfh = data["target_fpfh"].astype(np.float32)    # [N, 33]


        source_fpfh = source_fpfh / 200.0
        target_fpfh = target_fpfh / 200.0

        source_input = np.concatenate([source_xyz, source_fpfh], axis=1).astype(np.float32)
        target_input = np.concatenate([target_xyz, target_fpfh], axis=1).astype(np.float32)

        T_gt = data["T_gt"].astype(np.float32)
        R_gt = data["R_gt"].astype(np.float32)
        t_gt = data["t_gt"].astype(np.float32)

        sample = {
            # input used by PointNet
            "source": torch.from_numpy(source_input),
            "target": torch.from_numpy(target_input),

            # xyz only, needed later for transformation/visualization
            "source_xyz": torch.from_numpy(source_xyz),
            "target_xyz": torch.from_numpy(target_xyz),

            "T_gt": torch.from_numpy(T_gt),
            "R_gt": torch.from_numpy(R_gt),
            "t_gt": torch.from_numpy(t_gt),
        }

        return sample