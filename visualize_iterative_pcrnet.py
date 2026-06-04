import torch
import numpy as np
import open3d as o3d

from pcrnet.data_utils import FemurPCRNetDataset
from pcrnet.models.pcrnet import iPCRNet


# ==========================================================
# PATHS
# ==========================================================

dataset_dir = "/home/roa.fayad/pcrnet_dataset_partial_fragment_to_full_femur"
checkpoint_path = "/home/roa.fayad/pcrnet_checkpoints_geodesic_translation/best_model.pth"


# ==========================================================
# SETTINGS
# ==========================================================

sample_index = 0
max_iterations = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# LOAD TEST SAMPLE
# ==========================================================

test_dataset = FemurPCRNetDataset(
    dataset_dir=dataset_dir,
    split="test"
)

sample = test_dataset[sample_index]

source = sample["source"].unsqueeze(0).to(device)
target = sample["target"].unsqueeze(0).to(device)

source_np = sample["source"].numpy()
target_np = sample["target"].numpy()

print("\nLoaded test sample:", sample_index)
print("source:", source.shape)
print("target:", target.shape)


# ==========================================================
# LOAD MODEL
# ==========================================================

model = iPCRNet().to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nLoaded best model:")
print("epoch:", checkpoint["epoch"])
print("train_loss:", checkpoint["train_loss"])
print("val_loss:", checkpoint["val_loss"])


# ==========================================================
# OPEN3D HELPER
# ==========================================================

def make_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd


target_pcd = make_point_cloud(target_np, [0.2, 0.6, 1.0])
source_initial_pcd = make_point_cloud(source_np, [1.0, 0.1, 0.1])


# ==========================================================
# WINDOW 0: INITIAL MISALIGNMENT
# ==========================================================

o3d.visualization.draw_geometries(
    [target_pcd, source_initial_pcd],
    window_name="Iteration 0: red source before registration, blue target",
)


# ==========================================================
# ITERATIVE VISUALIZATION
# ==========================================================

with torch.no_grad():

    for iteration in range(1, max_iterations + 1):

        result = model(
            template=target,
            source=source,
            max_iteration=iteration
        )

        transformed_source = result["transformed_source"]
        transformed_source_np = transformed_source.squeeze(0).cpu().numpy()

        transformed_pcd = make_point_cloud(
            transformed_source_np,
            [0.1, 0.9, 0.2]
        )

        print(f"\nIteration {iteration}")
        print("Estimated translation:")
        print(result["est_t"].squeeze().cpu().numpy())

        o3d.visualization.draw_geometries(
            [target_pcd, transformed_pcd],
            window_name=f"Iteration {iteration}: green transformed source, blue target",
        )