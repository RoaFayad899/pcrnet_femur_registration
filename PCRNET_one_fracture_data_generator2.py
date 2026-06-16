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

output_dir = r"C:\data_unibas\pcrnet_dataset_partial_fragment_to_full_femur_large_fpfh"
os.makedirs(output_dir, exist_ok=True)

N_TOTAL_SAMPLES = 5000
N_SAMPLES_PER_FRAGMENT = N_TOTAL_SAMPLES // 2

N_SOURCE_POINTS = 1024
N_TARGET_POINTS = 1024

NOISE_SCALE = 0.01
RANDOM_SEED = 42


FRACTURE_ANGLE_DEG = 15.0
PARTIAL_KEEP_PERCENTILE = 55

MAX_SOURCE_ROT_DEG = 45.0
MAX_TARGET_ROT_DEG = 180.0

FRACTURE_GAP_SIZE = 0.01

SOURCE_TRANSLATION_RANGE = (-0.10, 0.10)

TARGET_TRANSLATION_MM = 10.0



rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# BASIC HELPERS
# ============================================================

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

def apply_transform(points, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t

def build_transform_matrix(center, R, t):
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


def make_open3d_mesh(vertices, faces, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


##Normalization
def normalize_points(points, center, scale):
    return (points - center) / scale


def denormalize_points(points_normalized, center, scale):
    return points_normalized * scale + center


def compute_fpfh_features(points, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30
        )
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100
        )
    )

    return np.asarray(fpfh.data).T.astype(np.float32)  # [N, 33]

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

full_points_mm = full_vertices_mm.copy()

normalization_center = full_points_mm.mean(axis=0)

normalization_scale = np.max(
    np.linalg.norm(
        full_points_mm - normalization_center,
        axis=1
    )
)

TARGET_TRANSLATION_RANGE = (
    -TARGET_TRANSLATION_MM / normalization_scale,
     TARGET_TRANSLATION_MM / normalization_scale
)

full_vertices = normalize_points(
    full_vertices_mm,
    normalization_center,
    normalization_scale
)

full_points = full_vertices.copy()
common_center = full_points.mean(axis=0)

full_femur_points = full_vertices.copy()
common_center_full_femur = common_center.copy()

print("Full intact femur points:", len(full_femur_points))
print("Common global center:", common_center_full_femur)


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
    full_vertices,
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
# DATASET GENERATION
# Normalized + matrix-based version
# ============================================================

metadata = []
sample_id = 0

for fragment_id, fragment_clean in [
    (1, frag1_clean),
    (2, frag2_clean)
]:

    for local_id in range(N_SAMPLES_PER_FRAGMENT):

        # ====================================================
        # 1. GLOBAL TRANSFORM MATRIX
        # ====================================================

        R_global, axis_global, angle_global = random_rotation_matrix(
            MAX_TARGET_ROT_DEG,
            rng
        )

        t_global = random_translation(
            TARGET_TRANSLATION_RANGE,
            rng
        )

        common_center = common_center_full_femur.copy()

        T_global = build_transform_matrix(
            common_center,
            R_global,
            t_global
        )

        # ====================================================
        # 2. TARGET
        # Whole intact femur + global transform matrix
        # ====================================================

        target_clean = full_femur_points.copy()

        target_transformed = apply_transform(
            target_clean,
            T_global
        )

        T_global_target = T_global.copy()

        # ====================================================
        # 3. SOURCE BASE
        # Partial fractured fragment + same global matrix
        # ====================================================

        source_base_clean = fragment_clean.copy()

        source_base_transformed = apply_transform(
            source_base_clean,
            T_global
        )

        T_global_source = T_global.copy()

        # ====================================================
        # 4. ADD NOISE TO SOURCE ONLY
        # ====================================================

        source_noisy = add_multiplicative_noise(
            source_base_transformed,
            noise_scale=NOISE_SCALE,
            rng=rng
        )

        # ====================================================
        # 5. EXTRA SOURCE PERTURBATION MATRIX
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

        T_extra = build_transform_matrix(
            center_extra,
            R_extra,
            t_extra
        )

        source_transformed = apply_transform(
            source_noisy,
            T_extra
        )

        # ====================================================
        # 6. GROUND TRUTH
        # GT maps perturbed source back to source_noisy pose
        # ====================================================

        T_gt = invert_transform(T_extra)

        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]

        T_source_total = T_extra @ T_global_source
        T_target_total = T_global_target.copy()

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

        source_fpfh = compute_fpfh_features(source_sampled)
        target_fpfh = compute_fpfh_features(target_sampled)

        # ====================================================
        # 8. SAVE SAMPLE
        # ====================================================

        filename = f"sample_{sample_id:06d}_frag{fragment_id}.npz"
        filepath = os.path.join(output_dir, filename)

        np.savez_compressed(
            filepath,

            source=source_sampled.astype(np.float32),
            target=target_sampled.astype(np.float32),

            source_fpfh=source_fpfh.astype(np.float32),
            target_fpfh=target_fpfh.astype(np.float32),

            R_gt=R_gt.astype(np.float32),
            t_gt=t_gt.astype(np.float32),
            T_gt=T_gt.astype(np.float32),

            T_global_target=T_global_target.astype(np.float32),
            T_global_source=T_global_source.astype(np.float32),
            T_extra=T_extra.astype(np.float32),

            T_source_total=T_source_total.astype(np.float32),
            T_target_total=T_target_total.astype(np.float32),

            R_global=R_global.astype(np.float32),
            t_global=t_global.astype(np.float32),

            R_extra=R_extra.astype(np.float32),
            t_extra=t_extra.astype(np.float32),

            fragment_id=np.array(fragment_id, dtype=np.int32),
            sample_id=np.array(sample_id, dtype=np.int32),
            noise_scale=np.array(NOISE_SCALE, dtype=np.float32),

            global_rotation_angle_deg=np.array(angle_global, dtype=np.float32),
            extra_source_rotation_angle_deg=np.array(angle_extra, dtype=np.float32),

            normalization_center=normalization_center.astype(np.float32),
            normalization_scale=np.array(normalization_scale, dtype=np.float32)
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
            "target_points": N_TARGET_POINTS,

            "normalization_scale": float(normalization_scale)
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
print("Target: clean whole femur with global transform around full femur center")
print("Source: same global transform around full femur center + noise + extra source perturbation")
print("Ground truth: inverse of extra source perturbation")


########################################################################################################################
# ============================================================
# CLEAR MESH VISUALIZATION OF ONE EXAMPLE
# Normalized + matrix-based version
# ============================================================

print("\n========== CLEAR MESH VISUALIZATION OF ONE EXAMPLE ==========")


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


VIS_FRAGMENT_ID = 1

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


# ============================================================
# GLOBAL TRANSFORM MATRIX
# ============================================================

R_global, axis_global, angle_global = random_rotation_matrix(
    MAX_TARGET_ROT_DEG,
    rng
)

t_global = random_translation(
    TARGET_TRANSLATION_RANGE,
    rng
)

common_center_vis = common_center_full_femur.copy()

T_global_vis = build_transform_matrix(
    common_center_vis,
    R_global,
    t_global
)

target_vertices_global = apply_transform(
    full_femur_points,
    T_global_vis
)

source_vertices_global = apply_transform(
    source_vertices_clean,
    T_global_vis
)


# ============================================================
# EXTRA SOURCE PERTURBATION MATRIX
# ============================================================

R_extra, axis_extra, angle_extra = random_rotation_matrix(
    MAX_SOURCE_ROT_DEG,
    rng
)

t_extra = random_translation(
    SOURCE_TRANSLATION_RANGE,
    rng
)

center_extra = source_vertices_global.mean(axis=0)

T_extra_vis = build_transform_matrix(
    center_extra,
    R_extra,
    t_extra
)

source_vertices_misaligned = apply_transform(
    source_vertices_global,
    T_extra_vis
)

T_gt_vis = invert_transform(T_extra_vis)

source_vertices_corrected = apply_transform(
    source_vertices_misaligned,
    T_gt_vis
)


# ============================================================
# NUMERICAL CHECK
# ============================================================

print("source_fpfh shape:", source_fpfh.shape)
print("target_fpfh shape:", target_fpfh.shape)

alignment_error = np.linalg.norm(
    source_vertices_corrected - source_vertices_global,
    axis=1
)

print("\n========== VISUALIZATION GT CHECK ==========")
print("Mean point error corrected vs source_global:", alignment_error.mean())
print("Max point error corrected vs source_global:", alignment_error.max())


# ============================================================
# MESHES
# ============================================================

target_mesh = make_open3d_mesh(
    target_vertices_global,
    full_faces,
    [0.2, 0.6, 1.0]
)

source_misaligned_mesh = make_open3d_mesh(
    source_vertices_misaligned,
    source_faces,
    [1.0, 0.2, 0.1]
)

source_corrected_mesh = make_open3d_mesh(
    source_vertices_corrected,
    source_faces,
    [0.1, 0.9, 0.2]
)

target_wire = o3d.geometry.LineSet.create_from_triangle_mesh(target_mesh)
target_wire.paint_uniform_color([0.0, 0.0, 0.8])


print("\nTARGET")
print("Clean whole femur, normalized coordinates")
print("Global rotation angle deg:", angle_global)
print("Global translation normalized units:", t_global)

print("\nSOURCE")
print("Fragment:", VIS_FRAGMENT_ID)
print("Same global transform around normalized full femur center")
print("Extra source rotation angle deg:", angle_extra)
print("Extra source translation normalized units:", t_extra)

print("\nExpected:")
print("Red = misaligned source")
print("Green = corrected source using GT inverse")
print("Blue = target")


o3d.visualization.draw_geometries(
    [target_mesh, target_wire, source_misaligned_mesh, source_corrected_mesh],
    window_name="Dataset mesh example: red misaligned, green corrected, blue target",
    mesh_show_back_face=True
)