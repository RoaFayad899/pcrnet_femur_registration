import os
import torch
import numpy as np
import open3d as o3d

from pcrnet.data_utils import FemurPCRNetDataset
from pcrnet.models.pcrnet_6Drepresentation import iPCRNet  ##############


dataset_dir = "/home/roa.fayad/pcrnet_dataset_partial_fragment_to_full_femur_large"
#checkpoint_dir = "/home/roa.fayad/pcrnet_checkpoints_overfit_one_sample/best_model.pth"  #################
#checkpoint_dir = "/home/roa.fayad/pcrnet_checkpoints_geodesic_translation/best_model.pth"
#checkpoint_dir = "/home/roa.fayad/pcrnet_checkpoints_overfit_one_sample_chamfer/best_model.pth"
#checkpoint_dir = "/home/roa.fayad/pcrnet_checkpoints_overfit_one_sample_chamfer_iter30/best_model.pth"
#checkpoint_dir = "/home/roa.fayad/pcrnet_checkpoints_6d_chamfer_overfit_iter5_2/best_model.pth"
checkpoint_dir = "/home/roa.fayad/pcrnet_checkpoints_msegeodesic_6drepresentation_1600samples_iter1_large/best_model.pth"


#output_dir = "/home/roa.fayad/pcrnet_iterative_visualization"  ####################
#output_dir = "/home/roa.fayad/pcrnet_iterative_visualization_overfit_chamfer"
#output_dir = "/home/roa.fayad/pcrnet_iterative_visualization_overfit_chamfer_itr30"
output_dir = "/home/roa.fayad/pcrnet_visualization_msegeodesic_6drepresentation_1600samples_iter1_large"

os.makedirs(output_dir, exist_ok=True)

sample_index = 0
max_iterations = 1 ##########8, 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


test_dataset = FemurPCRNetDataset(
    dataset_dir=dataset_dir,
    split="test"                  ######split="test"                     #############################
)

sample = test_dataset[sample_index]

source = sample["source"].unsqueeze(0).to(device)
target = sample["target"].unsqueeze(0).to(device)

source_np = sample["source"].numpy()
target_np = sample["target"].numpy()

print("\nLoaded test sample:", sample_index)
print("source:", source.shape)
print("target:", target.shape)


model = iPCRNet().to(device)

checkpoint = torch.load(
    checkpoint_dir,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nLoaded best model:")
print("epoch:", checkpoint["epoch"])
print("train_loss:", checkpoint["train_loss"])
print("val_loss:", checkpoint["val_loss"])


def make_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd


# Save target once
target_pcd = make_point_cloud(target_np, [0.2, 0.6, 1.0])
target_path = os.path.join(output_dir, "target_blue.ply")
o3d.io.write_point_cloud(target_path, target_pcd)

# Save initial source
source_initial_pcd = make_point_cloud(source_np, [1.0, 0.1, 0.1])
source_initial_path = os.path.join(output_dir, "iteration_00_source_red.ply")
o3d.io.write_point_cloud(source_initial_path, source_initial_pcd)
T_gt = sample["T_gt"].unsqueeze(0).to(device)

source_h = torch.cat(
    [
        source,
        torch.ones(source.shape[0], source.shape[1], 1).to(device)
    ],
    dim=2
)

source_gt = torch.bmm(
    source_h,
    T_gt.transpose(1, 2)
)[:, :, :3]

source_gt_np = source_gt.squeeze(0).cpu().numpy()

source_gt_pcd = make_point_cloud(source_gt_np, [1.0, 1.0, 0.0])

source_gt_path = os.path.join(output_dir, "source_after_GT_yellow.ply")
o3d.io.write_point_cloud(source_gt_path, source_gt_pcd)

print(source_gt_path)

print("\nSaved:")
print(target_path)
print(source_initial_path)


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

        output_path = os.path.join(
            output_dir,
            f"iteration_{iteration:02d}_source_transformed_green.ply"
        )

        o3d.io.write_point_cloud(output_path, transformed_pcd)

        print(f"\nIteration {iteration}")
        print("Estimated translation:")
        print(result["est_t"].squeeze().cpu().numpy())
        print("Saved:", output_path)


print("\nDONE.")
print("Files saved in:", output_dir)