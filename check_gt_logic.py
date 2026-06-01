import numpy as np
import nibabel as nib
import open3d as o3d
from scipy.ndimage import label
from skimage import measure


file_path = r"C:\data_unibas\Healthy-Total-Body-CTs-001.nii"

FRACTURE_GAP_SIZE = 2.0
FRACTURE_ANGLE_DEG = 15.0
PARTIAL_KEEP_PERCENTILE = 55

MAX_SOURCE_ROT_DEG = 6.0
MAX_TARGET_ROT_DEG = 8.0
SOURCE_TRANSLATION_RANGE = (-6.0, 6.0)
TARGET_TRANSLATION_RANGE = (-10.0, 10.0)

rng = np.random.default_rng(42)


def rotation_matrix_from_axis_angle(axis, angle_deg):
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def random_rotation_matrix(max_angle_deg):
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = rng.uniform(-max_angle_deg, max_angle_deg)
    return rotation_matrix_from_axis_angle(axis, angle), angle


def random_translation(translation_range):
    return rng.uniform(translation_range[0], translation_range[1], size=3)


def apply_rigid(points, center, R, t):
    return (points - center) @ R.T + center + t


def build_transform_matrix(center, R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center + t - R @ center
    return T


def apply_transform(points, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def translation_mse(t_pred, t_gt):
    """
    Mean squared error between two translation vectors.
    """
    return np.mean((t_pred - t_gt) ** 2)


def rotation_geodesic_distance(R_pred, R_gt):
    """
    Geodesic distance between two rotation matrices, in radians and degrees.
    """
    R_diff = R_pred.T @ R_gt

    trace = np.trace(R_diff)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta_rad = np.arccos(cos_theta)
    theta_deg = np.rad2deg(theta_rad)

    return theta_rad, theta_deg

def make_mesh(vertices, faces, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


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

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_vertex_indices)}

    new_faces = np.array(
        [[old_to_new[idx] for idx in face] for face in kept_faces_old],
        dtype=np.int32
    )

    return new_vertices, new_faces


def split_mesh_by_fracture(vertices, faces, plane_point, plane_normal, gap_size):
    signed = (vertices - plane_point) @ plane_normal

    mask_frag1 = signed <= 0
    faces_frag1_old = faces[mask_frag1[faces].all(axis=1)]

    old_indices = np.unique(faces_frag1_old)
    new_vertices = vertices[old_indices]

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(old_indices)}

    new_faces = np.array(
        [[old_to_new[idx] for idx in face] for face in faces_frag1_old],
        dtype=np.int32
    )

    new_vertices = new_vertices - (gap_size / 2.0) * plane_normal

    return new_vertices, new_faces


# ==========================================================
# LOAD FEMUR
# ==========================================================

img = nib.load(file_path)
data = img.get_fdata()
voxel_size_mm = np.array(img.header.get_zooms()[:3], dtype=np.float64)

femur = data == 15
labeled_femur, _ = label(femur)
one_femur = labeled_femur == 1

full_vertices, full_faces, _, _ = measure.marching_cubes(
    one_femur,
    level=0.5,
    spacing=voxel_size_mm
)

full_points = full_vertices.copy()
common_center = full_points.mean(axis=0)

partial_vertices, partial_faces = extract_longitudinal_half_mesh(
    full_vertices,
    full_faces,
    keep_percentile=PARTIAL_KEEP_PERCENTILE
)

# fracture plane
center_full = full_points.mean(axis=0)
points_centered = full_points - center_full

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

angle_rad = np.deg2rad(FRACTURE_ANGLE_DEG)
plane_normal = np.cos(angle_rad) * long_axis + np.sin(angle_rad) * perp
plane_normal = plane_normal / np.linalg.norm(plane_normal)

source_clean_vertices, source_faces = split_mesh_by_fracture(
    partial_vertices,
    partial_faces,
    plane_point,
    plane_normal,
    FRACTURE_GAP_SIZE
)

# ==========================================================
# TRANSFORMS
# ==========================================================

R_global, angle_global = random_rotation_matrix(MAX_TARGET_ROT_DEG)
t_global = random_translation(TARGET_TRANSLATION_RANGE)

target_global = apply_rigid(
    full_vertices,
    common_center,
    R_global,
    t_global
)

source_global = apply_rigid(
    source_clean_vertices,
    common_center,
    R_global,
    t_global
)

T_global = build_transform_matrix(
    common_center,
    R_global,
    t_global
)

R_extra, angle_extra = random_rotation_matrix(MAX_SOURCE_ROT_DEG)
t_extra = random_translation(SOURCE_TRANSLATION_RANGE)

center_extra = source_global.mean(axis=0)

source_misaligned = apply_rigid(
    source_global,
    center_extra,
    R_extra,
    t_extra
)

T_extra = build_transform_matrix(
    center_extra,
    R_extra,
    t_extra
)

T_gt = invert_transform(T_extra)

R_gt = T_gt[:3, :3]
t_gt = T_gt[:3, 3]

R_extra_inv = T_extra[:3, :3].T
t_extra_inv = -R_extra_inv @ T_extra[:3, 3]

rot_geo_rad, rot_geo_deg = rotation_geodesic_distance(R_gt, R_extra_inv)
trans_mse = translation_mse(t_gt, t_extra_inv)

print("\n========== GT ROTATION / TRANSLATION LOSS CHECK ==========")
print("Rotation geodesic distance between T_gt and inverse(T_extra):")
print("Radians:", rot_geo_rad)
print("Degrees:", rot_geo_deg)

print("\nTranslation MSE between T_gt and inverse(T_extra):")
print(trans_mse)


source_corrected = apply_transform(
    source_misaligned,
    T_gt
)

alignment_error = np.linalg.norm(source_corrected - source_global, axis=1)

print("\n========== GT NUMERICAL PROOF ==========")
print("Mean point error source_corrected vs source_global:", alignment_error.mean())
print("Max point error source_corrected vs source_global:", alignment_error.max())

def one_sided_chamfer_source_to_target(a, b):
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(a.astype(np.float64))

    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(b.astype(np.float64))

    dists_a_to_b = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    return np.mean(dists_a_to_b ** 2)

cd_before = one_sided_chamfer_source_to_target(source_misaligned, target_global)
cd_after_gt = one_sided_chamfer_source_to_target(source_corrected, target_global)

print("\n========== CHAMFER CHECK ==========")
print("Chamfer before GT correction:", cd_before)
print("Chamfer after GT correction: ", cd_after_gt)

# ==========================================================
# PRINT CHECK
# ==========================================================

print("\nGlobal transform:")
print("rotation deg:", angle_global)
print("translation:", t_global)

print("\nExtra source perturbation:")
print("rotation deg:", angle_extra)
print("translation:", t_extra)

print("\nExpected:")
print("Red = misaligned source")
print("Green = source after GT inverse")
print("Blue = target")
print("Green should go back onto the target pose/fracture location.")

# ==========================================================
# VISUALIZATION
# ==========================================================

target_mesh = make_mesh(target_global, full_faces, [0.2, 0.6, 1.0])
source_misaligned_mesh = make_mesh(source_misaligned, source_faces, [1.0, 0.1, 0.1])
source_corrected_mesh = make_mesh(source_corrected, source_faces, [0.1, 0.9, 0.2])

target_wire = o3d.geometry.LineSet.create_from_triangle_mesh(target_mesh)
target_wire.paint_uniform_color([0.0, 0.0, 0.8])

o3d.visualization.draw_geometries(
    [target_mesh, target_wire, source_misaligned_mesh, source_corrected_mesh],
    window_name="GT visual check: red misaligned, green corrected, blue target",
    mesh_show_back_face=True
)