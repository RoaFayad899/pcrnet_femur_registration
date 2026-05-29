import os
import json
import numpy as np
import nibabel as nib
import open3d as o3d
from scipy.ndimage import label
from skimage import measure


# ============================================================
# SETTINGS
# ============================================================

file_path = r"C:\data_unibas\Healthy-Total-Body-CTs-001.nii"

output_dir = r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur"
os.makedirs(output_dir, exist_ok=True)

N_TOTAL_SAMPLES = 1000
N_SAMPLES_PER_FRAGMENT = N_TOTAL_SAMPLES // 2

N_SOURCE_POINTS = 1024
N_TARGET_POINTS = 1024

NOISE_SCALE = 0.01
RANDOM_SEED = 42

FRACTURE_GAP_SIZE = 2.0
FRACTURE_ANGLE_DEG = 15.0
PARTIAL_KEEP_PERCENTILE = 55

MAX_SOURCE_ROT_DEG = 6.0
MAX_TARGET_ROT_DEG = 8.0

SOURCE_TRANSLATION_RANGE = (-6.0, 6.0)
TARGET_TRANSLATION_RANGE = (-10.0, 10.0)

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# BASIC HELPERS
# ============================================================

def make_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def rotation_matrix_from_axis_angle(axis, angle_deg):
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
    return R


def random_rotation_matrix(max_angle_deg, rng):
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)

    angle = rng.uniform(-max_angle_deg, max_angle_deg)

    R = rotation_matrix_from_axis_angle(axis, angle)
    return R, axis, angle


def random_translation(translation_range, rng):
    return rng.uniform(
        translation_range[0],
        translation_range[1],
        size=3
    )


def apply_rigid(points, center, R, t):
    points_centered = points - center
    points_rotated = points_centered @ R.T
    return points_rotated + center + t


def build_transform_matrix(center, R, t):
    """
    apply_rigid uses:
        p' = (p - center) @ R.T + center + t

    Equivalent homogeneous transform:
        p' = R p + [center + t - R center]
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center + t - R @ center
    return T


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def sample_fixed_number(points, n_points, rng):
    points = np.asarray(points)

    if len(points) >= n_points:
        idx = rng.choice(len(points), size=n_points, replace=False)
    else:
        idx = rng.choice(len(points), size=n_points, replace=True)

    return points[idx]


def add_multiplicative_noise(points, noise_scale, rng):
    noise = rng.normal(
        loc=1.0,
        scale=noise_scale,
        size=points.shape
    )
    return points * noise


# ============================================================
# LOAD CT SEGMENTATION
# ============================================================

img = nib.load(file_path)
data = img.get_fdata()

voxel_size_mm = np.array(img.header.get_zooms()[:3], dtype=np.float64)

print("Voxel size mm:", voxel_size_mm)
print("Image shape:", data.shape)

femur = (data == 15)

labeled_femur, num_components = label(femur)
print("Number of connected femur components:", num_components)

one_femur = (labeled_femur == 1)


# ============================================================
# FULL INTACT FEMUR SURFACE
# ============================================================

full_vertices_mm, full_faces, _, _ = measure.marching_cubes(
    one_femur,
    level=0.5,
    spacing=voxel_size_mm
)

full_femur_points = full_vertices_mm.copy()

print("Full intact femur points:", len(full_femur_points))


# ============================================================
# CREATE PARTIAL LONGITUDINAL SURFACE
# ============================================================

def extract_longitudinal_half_mesh(vertices, faces, keep_percentile=55):
    center = vertices.mean(axis=0)
    vertices_centered = vertices - center

    cov = np.cov(vertices_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    long_axis = eigenvectors[:, np.argmax(eigenvalues)]
    long_axis = long_axis / np.linalg.norm(long_axis)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, long_axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    cut_direction = np.cross(long_axis, ref)
    cut_direction = cut_direction / np.linalg.norm(cut_direction)

    side_projection = (vertices - center) @ cut_direction
    threshold = np.percentile(side_projection, keep_percentile)

    keep_vertices_mask = side_projection <= threshold
    keep_faces_mask = keep_vertices_mask[faces].all(axis=1)

    kept_faces_old = faces[keep_faces_mask]
    kept_vertex_indices = np.unique(kept_faces_old)

    new_vertices = vertices[kept_vertex_indices]

    old_to_new = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(kept_vertex_indices)
    }

    new_faces = np.array(
        [[old_to_new[idx] for idx in face] for face in kept_faces_old],
        dtype=np.int32
    )

    return new_vertices, new_faces, long_axis, cut_direction


partial_vertices, partial_faces, _, _ = extract_longitudinal_half_mesh(
    full_vertices_mm,
    full_faces,
    keep_percentile=PARTIAL_KEEP_PERCENTILE
)

print("Partial surface points:", len(partial_vertices))


# ============================================================
# DEFINE FRACTURE PLANE
# ============================================================

center_full = full_femur_points.mean(axis=0)
points_centered = full_femur_points - center_full

cov = np.cov(points_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

long_axis = eigenvectors[:, np.argmax(eigenvalues)]
long_axis = long_axis / np.linalg.norm(long_axis)

projections = points_centered @ long_axis
mid_proj = 0.5 * (projections.min() + projections.max())
plane_point = center_full + mid_proj * long_axis

ref = np.array([1.0, 0.0, 0.0])
if abs(np.dot(ref, long_axis)) > 0.9:
    ref = np.array([0.0, 1.0, 0.0])

perp = np.cross(long_axis, ref)
perp = perp / np.linalg.norm(perp)

fracture_angle_rad = np.deg2rad(FRACTURE_ANGLE_DEG)

plane_normal = (
    np.cos(fracture_angle_rad) * long_axis
    + np.sin(fracture_angle_rad) * perp
)

plane_normal = plane_normal / np.linalg.norm(plane_normal)

print("Long axis:", long_axis)
print("Plane normal:", plane_normal)


# ============================================================
# SPLIT PARTIAL SURFACE INTO TWO FRACTURED PARTIAL FRAGMENTS
# ============================================================

def create_fractured_partial_fragments(points, gap_size):
    signed_distances = (points - plane_point) @ plane_normal

    frag1 = points[signed_distances <= 0]
    frag2 = points[signed_distances > 0]

    half_gap = gap_size / 2.0

    frag1_gap = frag1 - half_gap * plane_normal
    frag2_gap = frag2 + half_gap * plane_normal

    return frag1_gap, frag2_gap


frag1_clean, frag2_clean = create_fractured_partial_fragments(
    partial_vertices,
    gap_size=FRACTURE_GAP_SIZE
)

print("Fragment 1 partial points:", len(frag1_clean))
print("Fragment 2 partial points:", len(frag2_clean))


# ============================================================
# DATASET GENERATION - CORRECTED LOGIC
# ============================================================

metadata = []
sample_id = 0

for fragment_id, fragment_clean in [
    (1, frag1_clean),
    (2, frag2_clean)
]:

    for local_id in range(N_SAMPLES_PER_FRAGMENT):

        # ====================================================
        # 1. GLOBAL TRANSFORM
        # Same transform applied to target and source base
        # ====================================================

        R_global, axis_global, angle_global = random_rotation_matrix(
            MAX_TARGET_ROT_DEG,
            rng
        )

        t_global = random_translation(
            TARGET_TRANSLATION_RANGE,
            rng
        )

        # ====================================================
        # 2. TARGET
        # Clean whole intact femur + global transform only
        # No noise
        # ====================================================

        target_clean = full_femur_points.copy()
        center_target = target_clean.mean(axis=0)

        target_transformed = apply_rigid(
            target_clean,
            center_target,
            R_global,
            t_global
        )

        T_global_target = build_transform_matrix(
            center_target,
            R_global,
            t_global
        )

        # ====================================================
        # 3. SOURCE BASE
        # Same anatomical/global pose as target
        # ====================================================

        source_base_clean = fragment_clean.copy()
        center_source_base = source_base_clean.mean(axis=0)

        source_base_transformed = apply_rigid(
            source_base_clean,
            center_source_base,
            R_global,
            t_global
        )

        T_global_source = build_transform_matrix(
            center_source_base,
            R_global,
            t_global
        )

        # ====================================================
        # 4. ADD NOISE TO SOURCE ONLY
        # ====================================================

        source_noisy = add_multiplicative_noise(
            source_base_transformed,
            noise_scale=NOISE_SCALE,
            rng=rng
        )

        # ====================================================
        # 5. EXTRA LOCAL SOURCE PERTURBATION
        # This creates the registration problem
        # ====================================================

        R_extra, axis_extra, angle_extra = random_rotation_matrix(
            MAX_SOURCE_ROT_DEG,
            rng
        )

        t_extra = random_translation(
            SOURCE_TRANSLATION_RANGE,
            rng
        )

        center_extra = source_noisy.mean(axis=0)

        source_transformed = apply_rigid(
            source_noisy,
            center_extra,
            R_extra,
            t_extra
        )

        T_extra = build_transform_matrix(
            center_extra,
            R_extra,
            t_extra
        )

        # ====================================================
        # 6. GROUND TRUTH
        # This aligns perturbed source back to target pose
        # ====================================================

        T_gt = invert_transform(T_extra)

        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]

        # ====================================================
        # 7. FIXED NUMBER OF POINTS
        # ====================================================

        source_sampled = sample_fixed_number(
            source_transformed,
            N_SOURCE_POINTS,
            rng
        )

        target_sampled = sample_fixed_number(
            target_transformed,
            N_TARGET_POINTS,
            rng
        )

        # ====================================================
        # 8. SAVE SAMPLE
        # ====================================================

        filename = f"sample_{sample_id:06d}_frag{fragment_id}.npz"
        filepath = os.path.join(output_dir, filename)

        np.savez_compressed(
            filepath,

            source=source_sampled.astype(np.float32),
            target=target_sampled.astype(np.float32),

            R_gt=R_gt.astype(np.float32),
            t_gt=t_gt.astype(np.float32),
            T_gt=T_gt.astype(np.float32),

            T_global_target=T_global_target.astype(np.float32),
            T_global_source=T_global_source.astype(np.float32),
            T_extra=T_extra.astype(np.float32),

            R_global=R_global.astype(np.float32),
            t_global=t_global.astype(np.float32),

            R_extra=R_extra.astype(np.float32),
            t_extra=t_extra.astype(np.float32),

            fragment_id=np.array(fragment_id, dtype=np.int32),
            sample_id=np.array(sample_id, dtype=np.int32),
            noise_scale=np.array(NOISE_SCALE, dtype=np.float32),

            global_rotation_angle_deg=np.array(angle_global, dtype=np.float32),
            extra_source_rotation_angle_deg=np.array(angle_extra, dtype=np.float32)
        )

        metadata.append({
            "sample_id": sample_id,
            "filename": filename,
            "fragment_id": fragment_id,
            "noise_scale": NOISE_SCALE,

            "global_rotation_angle_deg": float(angle_global),
            "global_translation": t_global.tolist(),

            "extra_source_rotation_angle_deg": float(angle_extra),
            "extra_source_translation": t_extra.tolist(),

            "source_points": N_SOURCE_POINTS,
            "target_points": N_TARGET_POINTS
        })

        sample_id += 1


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = os.path.join(output_dir, "metadata.json")

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nDONE.")
print("Dataset saved to:", output_dir)
print("Total samples:", sample_id)
print("Fragment 1 samples:", N_SAMPLES_PER_FRAGMENT)
print("Fragment 2 samples:", N_SAMPLES_PER_FRAGMENT)
print("Target: clean whole femur with global transform")
print("Source: same global transform + noise + extra source perturbation")
print("Ground truth: inverse of extra source perturbation")



########################################################################################################################
########################################################################################################################
########################################################################################################################
# ============================================================
# CLEAR MESH VISUALIZATION OF ONE EXAMPLE
# corrected logic: same global transform + extra source transform
# ============================================================

print("\n========== CLEAR MESH VISUALIZATION OF ONE EXAMPLE ==========")

def make_open3d_mesh(vertices, faces, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def split_partial_mesh_into_fracture_fragments(vertices, faces, gap_size):
    signed_distances = (vertices - plane_point) @ plane_normal

    mask_frag1 = signed_distances <= 0
    mask_frag2 = signed_distances > 0

    faces_frag1_old = faces[mask_frag1[faces].all(axis=1)]
    faces_frag2_old = faces[mask_frag2[faces].all(axis=1)]

    def remap_mesh(faces_old):
        old_indices = np.unique(faces_old)
        new_vertices = vertices[old_indices]

        old_to_new = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(old_indices)
        }

        new_faces = np.array(
            [[old_to_new[idx] for idx in face] for face in faces_old],
            dtype=np.int32
        )

        return new_vertices, new_faces

    v1, f1 = remap_mesh(faces_frag1_old)
    v2, f2 = remap_mesh(faces_frag2_old)

    half_gap = gap_size / 2.0

    v1_gap = v1 - half_gap * plane_normal
    v2_gap = v2 + half_gap * plane_normal

    return v1_gap, f1, v2_gap, f2


# ------------------------------------------------------------
# Choose fragment for visualization
# ------------------------------------------------------------

VIS_FRAGMENT_ID = 1   # change to 2 if you want fragment 2

v1_gap, f1, v2_gap, f2 = split_partial_mesh_into_fracture_fragments(
    partial_vertices,
    partial_faces,
    gap_size=FRACTURE_GAP_SIZE
)

if VIS_FRAGMENT_ID == 1:
    source_vertices_clean = v1_gap
    source_faces = f1
else:
    source_vertices_clean = v2_gap
    source_faces = f2


# ------------------------------------------------------------
# 1. Generate one global transform
# same for target and source base
# ------------------------------------------------------------

R_global, axis_global, angle_global = random_rotation_matrix(
    MAX_TARGET_ROT_DEG,
    rng
)

t_global = random_translation(
    TARGET_TRANSLATION_RANGE,
    rng
)


# ------------------------------------------------------------
# 2. Target mesh = whole intact femur + global transform
# ------------------------------------------------------------

center_target = full_vertices_mm.mean(axis=0)

target_vertices_global = apply_rigid(
    full_vertices_mm,
    center_target,
    R_global,
    t_global
)

target_mesh = make_open3d_mesh(
    target_vertices_global,
    full_faces,
    [0.2, 0.6, 1.0]   # blue
)


# ------------------------------------------------------------
# 3. Source mesh base = same global transform
# no noise for mesh visualization to keep it clean
# ------------------------------------------------------------

center_source_base = source_vertices_clean.mean(axis=0)

source_vertices_global = apply_rigid(
    source_vertices_clean,
    center_source_base,
    R_global,
    t_global
)


# ------------------------------------------------------------
# 4. Extra source perturbation
# this is the misalignment PCRNet should learn to correct
# ------------------------------------------------------------

R_extra, axis_extra, angle_extra = random_rotation_matrix(
    MAX_SOURCE_ROT_DEG,
    rng
)

t_extra = random_translation(
    SOURCE_TRANSLATION_RANGE,
    rng
)

center_extra = source_vertices_global.mean(axis=0)

source_vertices_misaligned = apply_rigid(
    source_vertices_global,
    center_extra,
    R_extra,
    t_extra
)

source_mesh = make_open3d_mesh(
    source_vertices_misaligned,
    source_faces,
    [1.0, 0.2, 0.1]   # red
)


# ------------------------------------------------------------
# Optional: wireframe target for clarity
# ------------------------------------------------------------

target_wire = o3d.geometry.LineSet.create_from_triangle_mesh(target_mesh)
target_wire.paint_uniform_color([0.0, 0.0, 0.8])


# ------------------------------------------------------------
# Print info
# ------------------------------------------------------------

print("\nTARGET")
print("Clean whole femur")
print("Global rotation angle deg:", angle_global)
print("Global translation mm:", t_global)

print("\nSOURCE")
print("Fragment:", VIS_FRAGMENT_ID)
print("Same global transform as target")
print("Extra source rotation angle deg:", angle_extra)
print("Extra source translation mm:", t_extra)

print("\nGround truth for registration = inverse of extra source transform")


# ------------------------------------------------------------
# Visualize
# ------------------------------------------------------------

o3d.visualization.draw_geometries(
    [target_mesh, target_wire, source_mesh],
    window_name="Corrected dataset mesh example: source vs target",
    mesh_show_back_face=True
)