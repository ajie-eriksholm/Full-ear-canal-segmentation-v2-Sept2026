import os
import sys
import warnings
import numpy as np
import nibabel as nib
import torch
import csv
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import pyvista as pv
import traceback
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform
from totalsegmentator.python_api import totalsegmentator

# Configure PyVista for off-screen rendering
pv.set_plot_theme("document")
pv.global_theme.background = 'white'

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Default configuration (can be overridden by command-line arguments)
Processed_scans_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Processed-Data"
landmark_detection_model_path = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/best_model_2025-12-05_13-16-35.pth"
output_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Output_no_alignment"
output_transform_dir = os.path.join(output_dir, "Logs", "transform_logs")

# Debug mode - process only one specific scan
debug_mode = False  # Set to False to process all scans
debug_scan_name = "sub01_pituitary_CT_resampled_256.nii.gz"  # Specific scan to process in debug mode

# No eyes mode - use mandible segmentation instead of eye landmarks for plane fitting
no_eyes = False  # Set to True if scans don't include eyes (uses mandible top points instead of landmarks 12-13)
skip_alignment = False  # Set to True to skip alignment step (useful for testing landmark detection only)


CUDA_WARNING_PATTERNS = [
    r".*cuda capability.*",
    r".*Please install PyTorch with a following CUDA.*",
    r".*is not compatible with the current PyTorch installation.*",
]


def suppress_incompatible_cuda_warnings():
    """Suppress noisy warnings emitted for unsupported GPUs during CPU fallback."""
    for pattern in CUDA_WARNING_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern, category=UserWarning)

# Device configuration - check for GPU compatibility
def get_device():
    """Determine the best available device, checking GPU compatibility."""
    with warnings.catch_warnings():
        suppress_incompatible_cuda_warnings()
        if torch.cuda.is_available():
            try:
                # PyTorch 2.x requires compute capability >= 7.0 for this install.
                major, minor = torch.cuda.get_device_capability()
                compute_capability = float(f"{major}.{minor}")
                if compute_capability >= 7.0:
                    print(f"GPU detected and compatible - using CUDA (compute capability: {compute_capability})")
                    return torch.device('cuda')

                print(f"GPU compute capability {compute_capability} < 7.0 (required by PyTorch 2.x)")
                print("Falling back to CPU")
                return torch.device('cpu')
            except Exception as e:
                print(f"Error checking GPU compatibility: {e}")
                print("Falling back to CPU")
                return torch.device('cpu')

        print("No GPU available - using CPU")
        return torch.device('cpu')

DEVICE = get_device()

# Landmark configuration
landmark_ids = [8, 9, 10, 11, 12, 13]
num_landmarks = len(landmark_ids)

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')


def strip_nifti_extension(filename):
    """Strip .nii or .nii.gz extension from a filename."""
    for ext in NIFTI_EXTENSIONS:
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def extract_scan_name_from_resampled(filename):
    """Extract scan name from <scan>_CT_resampled_256(.nii|.nii.gz)."""
    stem = strip_nifti_extension(filename)
    suffix = '_CT_resampled_256'
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def is_resampled_scan(filename):
    """Return True for <scan>_CT_resampled_256(.nii|.nii.gz)."""
    return strip_nifti_extension(filename).endswith('_CT_resampled_256')


# === Model Architecture ===
class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet3D(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=16):
        super().__init__()
        f = base_features
        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)

        self.pool = torch.nn.MaxPool3d(2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # Decoder: channel sizes reflect concatenations
        self.dec3 = DoubleConv(f * 8 + f * 4, f * 4)  # input channels after concat
        self.dec2 = DoubleConv(f * 4 + f * 2, f * 2)
        self.dec1 = DoubleConv(f * 2 + f, f)

        self.out_conv = torch.nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape expected (B, C, D, H, W)
        e1 = self.enc1(x)              # B, f, ...
        e2 = self.enc2(self.pool(e1))  # B, f*2, ...
        e3 = self.enc3(self.pool(e2))  # B, f*4, ...
        e4 = self.enc4(self.pool(e3))  # B, f*8, ...

        d3 = self.up(e4)
        # ensure same spatial dims (in case of odd sizes) by center crop or interpolation - here we rely on upsample
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out


# === Helper Functions ===
def initialize_transform_json(scan_path, dimensions, affine, step_number=1):
    """Initialize the transform JSON with original scan metadata for P2."""
    scan_name = extract_scan_name_from_resampled(os.path.basename(scan_path))
    
    transform_data = {
        "scan_name": scan_name,
        "original_scan_path": scan_path,
        "processing_timestamp": datetime.now().isoformat(),
        "processing_stage": "P2_Landmark_Detection_and_Alignment",
        "input_metadata": {
            "dimensions": [int(x) for x in dimensions],
            "affine_matrix": [[float(x) for x in row] for row in affine],
            "origin_mm": [float(x) for x in affine[:3, 3]],
            "voxel_spacing_mm": [float(np.abs(affine[i, i])) for i in range(3)]
        },
        "transformations": [],
        "starting_step": step_number
    }
    
    return transform_data

def save_transform_json(transform_data, scan_name, output_dir):
    """Save the transform JSON to file."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{scan_name}_transform_log_P2.json")
    with open(json_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    return json_path

def compute_plane_normal(points):
    """Compute normal vector of a plane defined by 3 points."""
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def align_to_horizontal(scan_points, landmark_positions):
    """
    Align scan points so that the Frankfort plane becomes horizontal.
    Steps:
    1. Rotate original normal to Z-axis (negative direction).
    2. Correct left-right orientation to X-axis.
    """
    orig_normal = compute_plane_normal(landmark_positions[:3])
    target_normal = np.array([0, 0, -1])  # Z-axis downward

    center = landmark_positions.mean(axis=0)
    scan_points_centered = scan_points - center

    # First rotation: make Frankfort plane horizontal
    rot_to_horizontal, _ = R.align_vectors([target_normal], [orig_normal])
    rotated_landmarks = rot_to_horizontal.apply(landmark_positions - center)

    # Print landmark positions for verification
    print(f"  Landmark 12 (after horizontal): {rotated_landmarks[2]}")
    print(f"  Landmark 13 (after horizontal): {rotated_landmarks[3]}")
    
    # Second rotation: correct left-right orientation
    left_right_vector = rotated_landmarks[3] - rotated_landmarks[2]
    lr_distance = np.linalg.norm(left_right_vector)
    left_right_vector /= lr_distance
    
    # PROJECT onto XY plane to ensure it's truly horizontal
    left_right_xy = left_right_vector.copy()
    left_right_xy[2] = 0  # Zero out Z component
    lr_xy_norm = np.linalg.norm(left_right_xy)
    
    print(f"  Left-Right vector: {left_right_vector}")
    print(f"  L-R distance: {lr_distance:.2f}")
    print(f"  Z-component: {left_right_vector[2]:.4f}")
    print(f"  XY projection norm: {lr_xy_norm:.4f}")
    
    # Handle degenerate case: if vector is already very close to X-axis or too vertical
    if lr_xy_norm < 0.1:  # Almost vertical after first rotation
        print("  WARNING: Left-right vector is nearly vertical! Skipping L-R alignment.")
        final_rotation = rot_to_horizontal
        z_rotation_degrees = 0.0
    else:
        # Normalize the XY projection
        left_right_xy /= lr_xy_norm
        target_lr = np.array([1, 0, 0])  # X-axis
        
        # Pick the closer X direction (+X or -X) to avoid a ~180° flip
        dot_product = np.dot(left_right_xy, target_lr)
        if dot_product < 0:
            target_lr = -target_lr          # accept -X as the LR axis
            dot_product = -dot_product       # now angle is measured to -X
            print("  LR vector closer to -X; targeting -X to avoid mirroring.")
        
        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        print(f"  Angle to target X-axis: {angle_deg:.2f}°")
        
        if angle_deg < 1.0:  # Already aligned within 1 degree
            print("  Left-right vector already aligned to X-axis, skipping rotation.")
            final_rotation = rot_to_horizontal
            z_rotation_degrees = 0.0
        else:
            # Compute rotation around Z-axis using angle and cross product
            # More stable than align_vectors for this case
            cross = np.cross(left_right_xy, target_lr)
            if cross[2] > 0:  # Rotate counter-clockwise
                angle_rad = np.arccos(dot_product)
            else:  # Rotate clockwise
                angle_rad = -np.arccos(dot_product)
            
            rot_z = R.from_euler('z', angle_rad)
            print(f"  Applying Z-rotation: {np.degrees(angle_rad):.2f}°")
            
            # Combine rotations (apply horizontal first, then LR correction)
            final_rotation = rot_z * rot_to_horizontal
            z_rotation_degrees = np.degrees(angle_rad)
    
    # Apply rotation to scan points
    aligned_scan = final_rotation.apply(scan_points_centered) + center

    return aligned_scan, final_rotation, orig_normal, center, abs(z_rotation_degrees)

def load_nii(nii_path):
    """Load a NIfTI file and return as tensor."""
    nii = nib.load(nii_path)
    img_data = nii.get_fdata()
    img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)  # Add channel dim
    return img_tensor, nii.affine


def segment_craniofacial_structures(input_path, output_dir, scan_name):
    """
    Segment craniofacial structures using TotalSegmentator.
    Reference: Wasserthal et al. (2023) TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images.
               Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024
    
    Args:
        input_path: Path to the input NIfTI scan
        output_dir: Directory to save segmentation results
        scan_name: Name of the scan for output files
    
    Returns:
        Dictionary with segmentation masks for each structure
    """
    print(f"  Running TotalSegmentator for craniofacial structures...")
    
    # Determine device for segmentation
    seg_device = 'gpu' if DEVICE.type == 'cuda' else 'cpu'

    original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

    try:
        if seg_device == 'cpu':
            # Hide unsupported GPUs from TotalSegmentator subprocesses to avoid repeated CUDA warnings.
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        with warnings.catch_warnings():
            suppress_incompatible_cuda_warnings()
            # Available structures: mandible, teeth_lower, skull, head, sinus_maxillary, sinus_frontal, teeth_upper
            segmentation_img = totalsegmentator(input_path, task='craniofacial_structures', device=seg_device)
    finally:
        if original_cuda_visible_devices is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
    
    if segmentation_img is None:
        raise ValueError("Craniofacial segmentation failed or returned no output.")
    
    # Save segmentation
    seg_output_path = os.path.join(output_dir, f"{scan_name}_craniofacial_segmented.nii.gz")
    nib.save(segmentation_img, seg_output_path)
    print(f"  Craniofacial segmentation saved to {seg_output_path}")
    
    return segmentation_img, seg_output_path


def extract_mandible_top_points(segmentation_data, affine, scan_name):
    """
    Extract the top (superior) points of the mandible on left and right sides.
    These points will be used as replacements for eye landmarks (lm12, lm13) in no_eyes mode.
    
    Args:
        segmentation_data: 3D numpy array of segmentation labels
        affine: Affine transformation matrix
        scan_name: Name of the scan (for logging)
    
    Returns:
        Tuple of (left_top_point, right_top_point) in world coordinates (mm)
    """
    # TotalSegmentator label for mandible is typically 1
    # Check for mandible label - it might vary by version
    unique_labels = np.unique(segmentation_data)
    print(f"  Unique segmentation labels found: {unique_labels}")
    
    # Try common mandible label IDs
    mandible_label = None
    for label_id in [1, 126]:  # Common labels for mandible
        if label_id in unique_labels:
            mandible_label = label_id
            break
    
    if mandible_label is None:
        raise ValueError(f"Mandible label not found in segmentation. Available labels: {unique_labels}")
    
    print(f"  Using mandible label: {mandible_label}")
    
    # Extract mandible mask
    mandible_mask = (segmentation_data == mandible_label)
    mandible_coords = np.argwhere(mandible_mask)
    
    if mandible_coords.size == 0:
        raise ValueError("No mandible voxels found in segmentation")
    
    print(f"  Found {len(mandible_coords)} mandible voxels")
    
    # Convert voxel coordinates to world coordinates
    mandible_coords_homogeneous = np.hstack([mandible_coords[:, [2, 1, 0]], np.ones((len(mandible_coords), 1))])
    mandible_world = (affine @ mandible_coords_homogeneous.T).T[:, :3]
    
    # Apply the same coordinate correction as for landmarks: (x,y,z) -> (z,-y,-x)
    mandible_world_corrected = np.column_stack([
        mandible_world[:, 2],   # z -> x
        -mandible_world[:, 1],  # -y -> y  
        -mandible_world[:, 0]   # -x -> z
    ])
    
    print(f"  Mandible coordinate ranges:")
    print(f"    X (left-right): [{mandible_world_corrected[:, 0].min():.2f}, {mandible_world_corrected[:, 0].max():.2f}]")
    print(f"    Y (anterior-posterior): [{mandible_world_corrected[:, 1].min():.2f}, {mandible_world_corrected[:, 1].max():.2f}]")
    print(f"    Z (inferior-superior): [{mandible_world_corrected[:, 2].min():.2f}, {mandible_world_corrected[:, 2].max():.2f}]")
    
    # Find the center point (approximate midline) using mean of X coordinate
    center_x = np.mean(mandible_world_corrected[:, 0])
    print(f"  Midline X-coordinate (mean): {center_x:.2f}")
    
    # Split into left and right halves based on X coordinate
    # In anatomical coordinates: negative X = left, positive X = right
    left_mask = mandible_world_corrected[:, 0] < center_x
    right_mask = mandible_world_corrected[:, 0] >= center_x
    
    left_points = mandible_world_corrected[left_mask]
    right_points = mandible_world_corrected[right_mask]
    
    if len(left_points) == 0 or len(right_points) == 0:
        raise ValueError("Could not split mandible into left and right halves")
    
    print(f"  Left side: {len(left_points)} points")
    print(f"  Right side: {len(right_points)} points")
    
    # For each side, find the topmost (maximum Z) point
    # Z-axis is superior-inferior, so max Z is the top
    left_top_idx = np.argmax(left_points[:, 2])
    right_top_idx = np.argmax(right_points[:, 2])
    
    left_top_point = left_points[left_top_idx]
    right_top_point = right_points[right_top_idx]
    
    print(f"  Left mandible top point (X < {center_x:.2f}): ({left_top_point[0]:.2f}, {left_top_point[1]:.2f}, {left_top_point[2]:.2f})")
    print(f"  Right mandible top point (X >= {center_x:.2f}): ({right_top_point[0]:.2f}, {right_top_point[1]:.2f}, {right_top_point[2]:.2f})")
    
    # Verify that points are on opposite sides
    if left_top_point[0] >= right_top_point[0]:
        print(f"  WARNING: Left point X ({left_top_point[0]:.2f}) should be < Right point X ({right_top_point[0]:.2f})")
    else:
        print(f"  ✓ Verification passed: Left point X < Right point X")
    
    return left_top_point, right_top_point


def visualize_landmarks_with_scan(scan_data, all_landmarks, output_png_path, threshold=-300):
    """
    Visualize the scan mask with landmarks in three views and save as PNG.
    Used for scans with large rotation angles.
    
    Args:
        scan_data: 3D numpy array of the scan
        all_landmarks: Array of landmark positions (6, 3)
        output_png_path: Path to save the PNG image
        threshold: Threshold for creating the scan mask (default: -300)
    """
    # Create mask based on threshold
    mask_points = np.argwhere(scan_data > threshold).astype(np.float32)
    
    # Define landmark names and colors
    landmark_names = ['L8', 'L9', 'L10', 'L11', 'L12', 'L13']
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    
    # Create plotter with 3 subplots (1 row, 3 columns)
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2880, 1080])
    
    views = [
        ('Axial (Top-Down)', 'xy', 0),
        ('Sagittal (Side)', 'yz', 90),
        ('Coronal (Front)', 'xz', 0)
    ]
    
    for idx, (title, view, azimuth) in enumerate(views):
        plotter.subplot(0, idx)
        
        # Add scan points as point cloud
        grid = pv.PolyData(mask_points)
        plotter.add_mesh(grid, color='lightgray', opacity=0.3, point_size=2)
        
        # Add each landmark as a sphere
        for i, (landmark, name, color) in enumerate(zip(all_landmarks, landmark_names, colors)):
            sphere = pv.Sphere(radius=3, center=landmark)
            plotter.add_mesh(sphere, color=color, label=name if idx == 2 else None)
            
            # Add text label near the landmark
            plotter.add_point_labels(
                [landmark], 
                [name], 
                font_size=16, 
                text_color=color,
                point_size=1,
                shape_opacity=0
            )
        
        # Add legend only to the last subplot
        if idx == 2:
            plotter.add_legend(bcolor='white', face='rectangle', size=(0.2, 0.2))
        
        # Set camera position based on view
        plotter.camera_position = view
        if azimuth != 0:
            plotter.camera.azimuth = azimuth
        plotter.camera.zoom(1.3)
        
        # Add title for this view
        plotter.add_text(title, position='upper_edge', font_size=14, color='black')
    
    # Save screenshot
    plotter.screenshot(output_png_path)
    plotter.close()
    
    print(f"  ⚠️  Visualization saved to {output_png_path}")

def visualize_with_pyvista(scan_path, landmarks_world, landmarks_voxel, affine, output_dir, scan_name):
    """
    Create PyVista visualizations with landmarks.
    Save 3 PNG images: axial, sagittal, coronal views.
    """
    # Load scan data
    nii = nib.load(scan_path)
    img_data = nii.get_fdata()
    
    # Create PyVista grid using wrap
    grid = pv.wrap(img_data)
    grid.spacing = np.abs(np.diag(affine)[:3])
    grid.origin = affine[:3, 3]
    
    # Get scan bounds for camera positioning
    bounds = grid.bounds
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    
    # Create visualizations for 3 views
    views = {
        'axial': {'position': (center[0], center[1], center[2] + 300), 'viewup': (0, 1, 0)},
        'sagittal': {'position': (center[0] + 300, center[1], center[2]), 'viewup': (0, 0, 1)},
        'coronal': {'position': (center[0], center[1] - 300, center[2]), 'viewup': (0, 0, 1)}
    }
    
    for view_name, view_params in views.items():
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        
        # Add volume with opacity
        plotter.add_volume(grid, opacity='sigmoid', cmap='bone', shade=True)
        
        # Add landmarks as spheres
        for i, lm_id in enumerate(sorted(landmarks_world.keys())):
            point = landmarks_world[lm_id]
            sphere = pv.Sphere(radius=4, center=point)
            plotter.add_mesh(sphere, color='red', label=f'LM {lm_id}')
        
        # Set camera
        plotter.camera_position = [view_params['position'], center, view_params['viewup']]
        plotter.camera.zoom(0.6)
        
        # Save screenshot
        output_path = os.path.join(output_dir, f"{scan_name}_{view_name}_view.png")
        plotter.screenshot(output_path)
        plotter.close()
        
        print(f"Saved {view_name} view to {output_path}")


# ─── Head-muscles constants ────────────────────────────────────────────────────
# Label IDs from TotalSegmentator head_muscles task (map_to_binary.py)
HEAD_MUSCLES_LABEL_IDS = {
    'masseter_right': 1,
    'masseter_left': 2,
    'lateral_pterygoid_right': 5,
    'lateral_pterygoid_left': 6,
}

# Outlier detection (centroid-based): a point is flagged as an outlier if its
# distance to the 6-point centroid exceeds (median_distance × ratio_threshold).
# Muscle top points are expected to cluster more tightly around the centroid
# than the eye-canal landmarks (lm10 / lm11), so we use a stricter ratio for
# muscles and a looser ratio for lm10 / lm11.
OUTLIER_CENTROID_RATIO_MUSCLE = 1.8     # stricter — muscles must stay close
OUTLIER_CENTROID_RATIO_LANDMARK = 2.0   # tightened (was 3.0): catch grossly mislocalized lm10/lm11
LANDMARK_LABELS_FOR_OUTLIER = ('lm10', 'lm11')

# LM10 / LM11 anatomical distance range (corrected world coordinates, mm).
# Used by check_lm10_lm11_symmetry to reject scans where one ear-canal
# landmark is grossly mislocalized. Uses ONLY the 3D Euclidean distance
# between the two points, which is invariant to head tilt/rotation in the
# scanner — so a tilted-but-correct head still passes.
# Adult ear-canal-to-ear-canal distance is typically 120-150 mm; we widen
# this range to be safe against pediatric/anatomical variation.
LM10_LM11_MIN_DIST_MM = 80.0
LM10_LM11_MAX_DIST_MM = 200.0


def fit_plane_svd(points):
    """
    Fit a plane through N points (N >= 3) using SVD (least squares).

    Returns:
        normal   – unit normal vector of the best-fit plane
        centroid – centroid of the input points (lies on the plane)
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]          # right singular vector with smallest singular value
    normal /= np.linalg.norm(normal)
    return normal, centroid


def check_lm10_lm11_symmetry(lm10, lm11,
                              min_dist=LM10_LM11_MIN_DIST_MM,
                              max_dist=LM10_LM11_MAX_DIST_MM):
    """
    Validate that the 3D Euclidean distance between LM10 and LM11 is within
    an anatomically plausible range.

    Distance is rotation-invariant, so a tilted-but-correctly-localized head
    still passes — we explicitly do NOT check Y/Z component agreement,
    because head misalignment in the scanner is exactly what FH alignment is
    designed to correct.

    Args:
        lm10, lm11 – (3,) arrays in corrected world coordinates (mm)

    Returns:
        bad_reasons – list of human-readable failure strings (empty if OK)
        info        – dict with the measured distance for logging
    """
    dist = float(np.linalg.norm(lm10 - lm11))
    info = {
        'distance_mm': dist,
        'min_dist_mm': min_dist,
        'max_dist_mm': max_dist,
    }

    bad = []
    if dist < min_dist:
        bad.append(f"|LM10 - LM11| = {dist:.1f} mm < {min_dist:.1f} mm (too close)")
    elif dist > max_dist:
        bad.append(f"|LM10 - LM11| = {dist:.1f} mm > {max_dist:.1f} mm (too far)")

    if bad:
        for reason in bad:
            print(f"  WARNING: LM10/LM11 distance — {reason}")
    else:
        print(f"  OK: LM10/LM11 distance = {dist:.1f} mm")

    return bad, info


def detect_and_remove_outliers(points, point_labels, landmark_labels=LANDMARK_LABELS_FOR_OUTLIER,
                                max_rounds=2):
    """
    Centroid-based outlier detection.

    Compute the centroid of all input points, then measure each point's distance
    to that centroid. A point is flagged as an outlier if its distance exceeds
    (median_distance × ratio), where the ratio depends on the point type:
        - lm10 / lm11 (eye-canal landmarks): looser threshold
          (OUTLIER_CENTROID_RATIO_LANDMARK)
        - all other points (muscle top points): stricter threshold
          (OUTLIER_CENTROID_RATIO_MUSCLE)

    Runs up to *max_rounds* rounds. In each round the single worst offender
    (largest distance/median ratio relative to its own threshold) is removed,
    and the centroid is recomputed for the next round.

    Args:
        points           – (N, 3) numpy array  (N >= 4)
        point_labels     – list of N string labels
        landmark_labels  – set/tuple of labels that use the looser threshold
                           (and that the caller treats as fatal if removed)
        max_rounds       – maximum number of outlier-removal rounds

    Returns:
        filtered_points   – array after removal
        filtered_labels   – labels after removal
        removed_labels    – list of labels removed (in order)
        all_round_info    – list of dicts, one per round, with keys
                            'distances' (dict label→float),
                            'centroid' (list of 3 floats),
                            'median_distance' (float),
                            'removed' (label or None)
    """
    current_points = points.copy()
    current_labels = list(point_labels)
    removed_labels = []
    all_round_info = []

    landmark_label_set = set(landmark_labels)

    for _round in range(max_rounds):
        n = len(current_points)
        if n < 4:
            break  # need at least 4 points to be meaningful

        centroid = current_points.mean(axis=0)
        distances = np.linalg.norm(current_points - centroid, axis=1)
        median_distance = float(np.median(distances))

        round_info = {
            'distances': {lbl: float(d) for lbl, d in zip(current_labels, distances)},
            'centroid': [float(c) for c in centroid],
            'median_distance': median_distance,
            'removed': None,
        }

        if median_distance < 1e-6:
            all_round_info.append(round_info)
            break

        # Per-point excess ratio = distance / (median * threshold_for_this_point).
        # Values > 1 indicate the point exceeds its own allowed multiple.
        excess = np.zeros(n)
        for i, lbl in enumerate(current_labels):
            ratio = (OUTLIER_CENTROID_RATIO_LANDMARK
                     if lbl in landmark_label_set
                     else OUTLIER_CENTROID_RATIO_MUSCLE)
            excess[i] = distances[i] / (median_distance * ratio)

        if excess.max() <= 1.0:
            all_round_info.append(round_info)
            break  # no outlier detected — done

        outlier_idx = int(np.argmax(excess))
        outlier_label = current_labels[outlier_idx]
        outlier_ratio_used = (OUTLIER_CENTROID_RATIO_LANDMARK
                              if outlier_label in landmark_label_set
                              else OUTLIER_CENTROID_RATIO_MUSCLE)
        round_info['removed'] = outlier_label
        all_round_info.append(round_info)

        removed_labels.append(outlier_label)

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[outlier_idx] = False
        current_points = current_points[keep_mask]
        current_labels = [lbl for i, lbl in enumerate(current_labels) if i != outlier_idx]

        print(f"    Round {_round + 1}: removed '{outlier_label}' "
              f"(distance {distances[outlier_idx]:.2f} mm, "
              f"{distances[outlier_idx] / median_distance:.2f}x median, "
              f"threshold {outlier_ratio_used:.1f}x)")

        # If the removed label is a landmark, no need to continue
        if outlier_label in landmark_label_set:
            break

    return current_points, current_labels, removed_labels, all_round_info


def check_muscle_horizontal_containment(lm10, lm11, muscle_points, muscle_labels):
    """
    Verify that all muscle top points lie within the horizontal (X-axis) span
    defined by LM10 and LM11. Anatomically the masseter / lateral pterygoid
    bellies should sit between the two ear-canal landmarks along the
    left-right axis. A muscle whose X-coordinate falls outside the
    [min(lm10.x, lm11.x), max(lm10.x, lm11.x)] range (plus a tolerance) is
    flagged.

    Args:
        lm10, lm11      – (3,) arrays in corrected world coordinates
        muscle_points   – (M, 3) array of muscle top-point coordinates
        muscle_labels   – list of M string labels (for logging)

    Returns:
        bad_muscles – list of muscle label strings that fail the containment
                      check (may be empty)
        info        – dict with debug details
    """
    x_lo = float(min(lm10[0], lm11[0]))
    x_hi = float(max(lm10[0], lm11[0]))
    x_range = x_hi - x_lo
    # Allow 20 pct of the LM10/LM11 X-range as tolerance
    tolerance = 0.20 * x_range if x_range > 1e-3 else 5.0

    bad = []
    info = {
        'lm10_x': float(lm10[0]),
        'lm11_x': float(lm11[0]),
        'landmark_x_min': x_lo,
        'landmark_x_max': x_hi,
        'tolerance': float(tolerance),
        'muscles': {},
    }

    for label, pt in zip(muscle_labels, muscle_points):
        x = float(pt[0])
        outside = x < (x_lo - tolerance) or x > (x_hi + tolerance)
        info['muscles'][label] = {'x': x, 'outside': outside}
        if outside:
            bad.append(label)
            print(f"  WARNING: {label} X={x:.2f} is OUTSIDE landmark X-range "
                  f"[{x_lo:.2f}, {x_hi:.2f}] +/- {tolerance:.2f}")
        else:
            print(f"  OK: {label} X={x:.2f} within landmark X-range "
                  f"[{x_lo:.2f}, {x_hi:.2f}] +/- {tolerance:.2f}")

    return bad, info


def render_frankfort_landmarks_visualization(scan_data, points_6pt, point_labels,
                                             output_path,
                                             removed_labels=None,
                                             flagged_labels=None,
                                             status_text=None,
                                             threshold=-300):
    """
    Render the 6-point Frankfort-plane landmark visualization (3 views).

    Always produces an output image, regardless of whether the scan ultimately
    passes or fails the outlier / containment / lateral-extent checks.

    Args:
        scan_data        – 3D numpy array of the (resampled) scan
        points_6pt       – (6, 3) array of candidate plane points in corrected
                           world coordinates
        point_labels     – list of 6 string labels matching points_6pt order
        output_path      – PNG path to save
        removed_labels   – iterable of labels removed by outlier detection
                           (rendered in gray and tagged "OUTLIER")
        flagged_labels   – iterable of labels that triggered a check failure
                           (rendered in gray and tagged "FLAGGED")
        status_text      – optional banner string drawn at the top of the image
        threshold        – HU threshold for the gray scan point cloud
    """
    removed_set = set(removed_labels or [])
    flagged_set = set(flagged_labels or [])

    # Map of internal label -> (display name, color). Any label not in this
    # map (shouldn't happen) falls back to the raw label / 'gray'.
    LABEL_DISPLAY = {
        'lm10':                    ('LM10',          'yellow'),
        'lm11':                    ('LM11',          'orange'),
        'masseter_right':          ('Masseter R',    'cyan'),
        'masseter_left':           ('Masseter L',    'magenta'),
        'lateral_pterygoid_right': ('Lat.Pteryg. R', 'lime'),
        'lateral_pterygoid_left':  ('Lat.Pteryg. L', 'red'),
    }

    vis_views = [
        ('Axial (Top-Down)', 'xy', 0),
        ('Sagittal (Side)', 'yz', 90),
        ('Coronal (Front)', 'xz', 0),
    ]

    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2880, 1080])
    mask_points_vis = np.argwhere(scan_data > threshold).astype(np.float32)

    for idx, (title, view, azimuth) in enumerate(vis_views):
        plotter.subplot(0, idx)

        if mask_points_vis.size:
            grid_vis = pv.PolyData(mask_points_vis)
            plotter.add_mesh(grid_vis, color='lightgray', opacity=0.2, point_size=2)

        for i, landmark in enumerate(points_6pt):
            lbl = point_labels[i]
            name, color = LABEL_DISPLAY.get(lbl, (lbl, 'gray'))
            is_outlier = lbl in removed_set
            is_flagged = lbl in flagged_set
            display_color = 'gray' if (is_outlier or is_flagged) else color

            sphere = pv.Sphere(radius=4, center=landmark)
            plotter.add_mesh(sphere, color=display_color,
                             label=name if idx == 2 else None)

            tag = ''
            if is_outlier and is_flagged:
                tag = ' (OUTLIER+FLAGGED)'
            elif is_outlier:
                tag = ' (OUTLIER)'
            elif is_flagged:
                tag = ' (FLAGGED)'
            label_text = f"{name}{tag}"
            plotter.add_point_labels(
                [landmark], [label_text],
                font_size=16, text_color=display_color,
                point_size=1, shape_opacity=0, bold=True,
            )

        if idx == 2:
            plotter.add_legend(bcolor='white', face='rectangle', size=(0.25, 0.35))

        plotter.camera_position = view
        if azimuth != 0:
            plotter.camera.azimuth = azimuth
        plotter.camera.zoom(1.3)
        plotter.add_text(title, position='upper_edge', font_size=14, color='black')

        if status_text and idx == 0:
            plotter.add_text(status_text, position='lower_edge',
                             font_size=12, color='red')

    plotter.screenshot(output_path)
    plotter.close()


def check_landmarks_further_than_muscles(points, labels,
                                         landmark_labels=LANDMARK_LABELS_FOR_OUTLIER):
    """
    Require that the kept lm10 / lm11 landmarks lie further from the centroid
    of the point set than every kept muscle point. Anatomically the ear-canal
    landmarks should be the most lateral points among the 6 reference points;
    if a muscle ends up further from the centroid than lm10 or lm11, that
    landmark is suspect.

    Args:
        points         – (N, 3) array of points kept after outlier removal
        labels         – list of N string labels
        landmark_labels – set/tuple of labels to validate

    Returns:
        bad_landmarks – list of landmark labels that are closer to the
                        centroid than the furthest kept muscle point
        info          – dict with distances and the muscle reference distance
    """
    if len(points) < 2:
        return [], {'note': 'too few points to evaluate'}

    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)

    landmark_label_set = set(landmark_labels)
    muscle_distances = [d for lbl, d in zip(labels, distances)
                        if lbl not in landmark_label_set]

    info = {
        'centroid': [float(c) for c in centroid],
        'distances': {lbl: float(d) for lbl, d in zip(labels, distances)},
    }

    if not muscle_distances:
        info['note'] = 'no muscle points present after filtering'
        return [], info

    max_muscle_distance = float(max(muscle_distances))
    info['max_muscle_distance'] = max_muscle_distance

    bad = []
    for lbl, d in zip(labels, distances):
        if lbl not in landmark_label_set:
            continue
        if d <= max_muscle_distance:
            bad.append(lbl)
            print(f"  WARNING: {lbl} centroid distance {d:.2f} mm is NOT greater "
                  f"than max muscle distance {max_muscle_distance:.2f} mm")
        else:
            print(f"  OK: {lbl} centroid distance {d:.2f} mm > max muscle "
                  f"distance {max_muscle_distance:.2f} mm")

    return bad, info


def segment_head_muscles(input_path, output_dir, scan_name):
    """
    Segment head muscles using TotalSegmentator head_muscles task.

    Relevant labels: masseter_right (1), masseter_left (2),
                     lateral_pterygoid_right (5), lateral_pterygoid_left (6).

    Reference: Wasserthal et al. (2023) TotalSegmentator: Robust Segmentation
               of 104 Anatomic Structures in CT Images.
               Radiology: Artificial Intelligence.
               https://doi.org/10.1148/ryai.230024

    Returns:
        Tuple of (segmentation NIfTI image, output path string)
    """
    print(f"  Running TotalSegmentator for head muscles (masseter + lateral pterygoid)...")

    seg_device = 'gpu' if DEVICE.type == 'cuda' else 'cpu'
    original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

    try:
        if seg_device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        with warnings.catch_warnings():
            suppress_incompatible_cuda_warnings()
            segmentation_img = totalsegmentator(input_path, task='head_muscles', device=seg_device)
    finally:
        if original_cuda_visible_devices is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices

    if segmentation_img is None:
        raise ValueError("Head muscles segmentation failed or returned no output.")

    seg_output_path = os.path.join(output_dir, f"{scan_name}_head_muscles_segmented.nii.gz")
    nib.save(segmentation_img, seg_output_path)
    print(f"  Head muscles segmentation saved to {seg_output_path}")

    return segmentation_img, seg_output_path


def extract_muscle_top_points(segmentation_data, affine, scan_name):
    """
    Extract the topmost (superior) voxel of each of the four target muscles:
        masseter_right, masseter_left,
        lateral_pterygoid_right, lateral_pterygoid_left.

    Missing muscles are tolerated: if TotalSegmentator does not produce a
    label for one or more muscles on a given scan, that muscle is returned
    as ``None``. The caller is responsible for verifying that enough points
    remain to fit the Frankfort plane and to define a left/right axis.

    Applies the same coordinate correction used elsewhere:
        nibabel world (x, y, z) → corrected (z, -y, -x).

    Returns:
        Tuple of (masseter_right_top, masseter_left_top,
                  lateral_pterygoid_right_top, lateral_pterygoid_left_top)
        where each entry is either a (3,) corrected world-coordinate array
        in mm or ``None`` if the muscle was missing/empty.
    """
    unique_labels = np.unique(segmentation_data)
    print(f"  Unique head_muscles segmentation labels: {unique_labels}")

    target_muscles = {
        'masseter_right':          HEAD_MUSCLES_LABEL_IDS['masseter_right'],
        'masseter_left':           HEAD_MUSCLES_LABEL_IDS['masseter_left'],
        'lateral_pterygoid_right': HEAD_MUSCLES_LABEL_IDS['lateral_pterygoid_right'],
        'lateral_pterygoid_left':  HEAD_MUSCLES_LABEL_IDS['lateral_pterygoid_left'],
    }

    muscle_top_points = {}

    for muscle_name, label_id in target_muscles.items():
        if label_id not in unique_labels:
            print(f"  WARNING: label {label_id} ({muscle_name}) not present "
                  f"in head_muscles segmentation — skipping this muscle.")
            muscle_top_points[muscle_name] = None
            continue

        mask = (segmentation_data == label_id)
        voxel_coords = np.argwhere(mask)   # (N, 3): (i, j, k) nibabel index order

        if voxel_coords.size == 0:
            print(f"  WARNING: {muscle_name} (label {label_id}) has 0 voxels — skipping.")
            muscle_top_points[muscle_name] = None
            continue

        print(f"  {muscle_name}: {len(voxel_coords)} voxels found")

        # Convert (i, j, k) → world coordinates via affine
        # nibabel affine expects column order (x=k, y=j, z=i)
        voxel_homogeneous = np.hstack([
            voxel_coords[:, [2, 1, 0]],          # reorder (i,j,k) → (k,j,i)=(x,y,z)
            np.ones((len(voxel_coords), 1))
        ])
        world_coords = (affine @ voxel_homogeneous.T).T[:, :3]

        # Coordinate correction: (x, y, z) → (z, -y, -x)
        world_corrected = np.column_stack([
            world_coords[:, 2],    # z  → new x
            -world_coords[:, 1],   # -y → new y
            -world_coords[:, 0],   # -x → new z
        ])

        # Top point = maximum corrected-Z (superior direction)
        top_idx = np.argmax(world_corrected[:, 2])
        top_point = world_corrected[top_idx]
        muscle_top_points[muscle_name] = top_point

        print(f"    Top point: ({top_point[0]:.2f}, {top_point[1]:.2f}, {top_point[2]:.2f})")

    return (
        muscle_top_points['masseter_right'],
        muscle_top_points['masseter_left'],
        muscle_top_points['lateral_pterygoid_right'],
        muscle_top_points['lateral_pterygoid_left'],
    )


def align_to_horizontal_multipoint(scan_points, plane_points, right_points, left_points):
    """
    Align scan points so that the Frankfort plane becomes horizontal,
    using SVD-based plane fitting for robustness with N >= 3 reference points.

    Steps:
    1. Fit plane normal via SVD over all plane_points.
    2. Rotate so the plane normal aligns with the negative Z-axis (horizontal).
    3. Correct left-right orientation using centroids of right_points vs left_points.

    Args:
        scan_points   – (N, 3) array of scan mask points to transform
        plane_points  – (M, 3) reference points for plane fitting (M >= 3)
        right_points  – (K, 3) anatomical right-side reference points (K >= 1)
        left_points   – (K, 3) anatomical left-side reference points (K >= 1)

    Returns:
        Same 5-tuple as align_to_horizontal:
        (aligned_scan, final_rotation, orig_normal, center, z_rotation_degrees)
    """
    orig_normal, center = fit_plane_svd(plane_points)
    target_normal = np.array([0, 0, -1])

    # SVD normal sign is arbitrary – orient it towards the target so that
    # R.align_vectors computes a small corrective rotation instead of a
    # ~180° flip that would mirror the volume.
    if np.dot(orig_normal, target_normal) < 0:
        orig_normal = -orig_normal

    scan_points_centered = scan_points - center

    # First rotation: make Frankfort plane horizontal
    rot_to_horizontal, _ = R.align_vectors([target_normal], [orig_normal])

    # Rotate right/left reference points with the first rotation
    right_rotated = rot_to_horizontal.apply(right_points - center)
    left_rotated  = rot_to_horizontal.apply(left_points  - center)

    right_centroid = right_rotated.mean(axis=0)
    left_centroid  = left_rotated.mean(axis=0)

    print(f"  Right muscle centroid (after horizontal rotation): {right_centroid}")
    print(f"  Left muscle centroid  (after horizontal rotation): {left_centroid}")

    # Compute LR direction: from left centroid to right centroid
    left_right_vector = right_centroid - left_centroid
    lr_distance = np.linalg.norm(left_right_vector)
    left_right_vector /= lr_distance

    # Project onto XY plane to ensure it is truly horizontal
    left_right_xy = left_right_vector.copy()
    left_right_xy[2] = 0
    lr_xy_norm = np.linalg.norm(left_right_xy)

    print(f"  Left-Right vector: {left_right_vector}")
    print(f"  L-R distance: {lr_distance:.2f}")
    print(f"  Z-component: {left_right_vector[2]:.4f}")
    print(f"  XY projection norm: {lr_xy_norm:.4f}")

    if lr_xy_norm < 0.1:
        print("  WARNING: Left-right vector is nearly vertical! Skipping L-R alignment.")
        final_rotation = rot_to_horizontal
        z_rotation_degrees = 0.0
    else:
        left_right_xy /= lr_xy_norm
        target_lr = np.array([1, 0, 0])

        # Pick the closer X direction (+X or -X) to avoid a ~180° flip
        dot_product = np.dot(left_right_xy, target_lr)
        if dot_product < 0:
            target_lr = -target_lr
            dot_product = -dot_product
            print("  LR vector closer to -X; targeting -X to avoid mirroring.")

        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        print(f"  Angle to target X-axis: {angle_deg:.2f}°")

        if angle_deg < 1.0:
            print("  Left-right vector already aligned to X-axis, skipping rotation.")
            final_rotation = rot_to_horizontal
            z_rotation_degrees = 0.0
        else:
            cross = np.cross(left_right_xy, target_lr)
            if cross[2] > 0:
                angle_rad = np.arccos(dot_product)
            else:
                angle_rad = -np.arccos(dot_product)

            rot_z = R.from_euler('z', angle_rad)
            print(f"  Applying Z-rotation: {np.degrees(angle_rad):.2f}°")

            final_rotation = rot_z * rot_to_horizontal
            z_rotation_degrees = np.degrees(angle_rad)

    aligned_scan = final_rotation.apply(scan_points_centered) + center

    return aligned_scan, final_rotation, orig_normal, center, abs(z_rotation_degrees)


# === Main Processing Pipeline ===
class _AlignmentSkipped(Exception):
    """Raised intentionally when skip_alignment=True to exit the alignment try-block cleanly."""
    pass


class _LandmarkOutlierError(Exception):
    """Raised when lm10 or lm11 is detected as a plane outlier, indicating a landmark error."""
    pass


def process_scans():
    """Main function to process all scans with landmark detection."""
    
    # Create output directories
    landmarks_output_dir = os.path.join(output_dir, "Preprocessing", "P2_Landmarks", "heatmaps")
    visualizations_output_dir = os.path.join(output_dir, "Logs", "P2_FH_landmark_visualizations")
    aligned_landmarks_dir = os.path.join(output_dir, "Preprocessing", "P2_Landmarks")
    aligned_npy_dir = os.path.join(output_dir, "Preprocessing", "P2_Landmarks", "aligned_npy")
    os.makedirs(landmarks_output_dir, exist_ok=True)
    os.makedirs(visualizations_output_dir, exist_ok=True)
    os.makedirs(aligned_landmarks_dir, exist_ok=True)
    os.makedirs(aligned_npy_dir, exist_ok=True)
    
    # Load model
    print("Loading landmark detection model...")
    model = UNet3D(in_channels=1, out_channels=num_landmarks, base_features=16)
    checkpoint = torch.load(landmark_detection_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {landmark_detection_model_path}")
    
    # Find all resampled scans
    nii_files = []
    for root, dirs, files in os.walk(Processed_scans_dir):
        for file in files:
            if is_resampled_scan(file):
                nii_files.append(os.path.join(root, file))
    
    nii_files = sorted(nii_files)
    
    # Filter scans if debug mode is enabled
    if debug_mode:
        nii_files = [path for path in nii_files if os.path.basename(path) == debug_scan_name]
        if not nii_files:
            print(f"ERROR: Debug scan '{debug_scan_name}' not found in {Processed_scans_dir}")
            return
        print(f"\n{'='*60}")
        print(f"DEBUG MODE ENABLED - Processing only: {debug_scan_name}")
        print(f"{'='*60}\n")
    
    print(f"\nFound {len(nii_files)} scans to process")
    
    if len(nii_files) == 0:
        print("No scans found matching pattern '*_CT_resampled_256.nii[.gz]'")
        return
    
    # Store all predictions for CSV export
    all_predictions = []
    
    # Store all aligned landmarks for CSV export
    all_aligned_landmarks = []
    
    # Store flagged scans with large rotation angles
    flagged_scans = []
    
    # Process each scan
    for nii_path in tqdm(nii_files, desc="Processing scans"):
        # Extract scan/patient name
        filename = os.path.basename(nii_path)
        scan_name = extract_scan_name_from_resampled(filename)
        
        print(f"\n{'='*60}")
        print(f"Processing: {scan_name}")
        print(f"{'='*60}")
        
        # Create output directory for this scan
        scan_landmarks_dir = os.path.join(landmarks_output_dir, scan_name)
        scan_viz_dir = os.path.join(visualizations_output_dir, scan_name)
        os.makedirs(scan_landmarks_dir, exist_ok=True)
        os.makedirs(scan_viz_dir, exist_ok=True)
        
        # Load and preprocess scan
        img_tensor, affine = load_nii(nii_path)
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension: (1, 1, D, H, W)
        
        # Initialize transform logging (continuing from P1 which ended at step 8)
        img_shape = img_tensor.shape[2:]  # Get spatial dimensions (D, H, W)
        transform_data = initialize_transform_json(nii_path, img_shape, affine, step_number=9)
        
        # Inference
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # Shape: (num_landmarks, D, H, W)
        
        # Process predictions and compute landmark locations
        landmark_locations_world = {}
        landmark_voxel_pred = {}
        
        for i, lm_id in enumerate(landmark_ids):
            pred_heatmap = probs[i]
            
            # Save heatmap as NIfTI
            heatmap_path = os.path.join(scan_landmarks_dir, f"pred_heatmap_landmark{lm_id}.nii.gz")
            nib.save(nib.Nifti1Image(pred_heatmap.astype(np.float32), affine=affine), heatmap_path)
            
            # Apply threshold and compute centroid
            threshold = 0.5
            mask = pred_heatmap >= threshold
            
            if np.any(mask):
                coords = np.argwhere(mask)
                weights = pred_heatmap[mask]
                centroid = np.average(coords, axis=0, weights=weights)
                voxel_idx = tuple(np.round(centroid).astype(int))
            else:
                voxel_idx = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
            
            landmark_voxel_pred[lm_id] = voxel_idx
            
            # Convert voxel index to world coordinates (mm)
            # voxel_idx is (z, y, x) from numpy, need to convert to (x, y, z, 1) for affine
            voxel_h = np.append(voxel_idx[::-1], 1.0)
            world_coords = affine @ voxel_h
            
            # Apply coordinate correction: predicted as (-X, -Y, Z), should be (Z, Y, X)
            # Flip signs of x and y, then reorder to (z, y, x)
            x, y, z = world_coords[0], world_coords[1], world_coords[2]
            corrected_coords = np.array([z, -y, -x])
            
            landmark_locations_world[lm_id] = corrected_coords
        
        # Record landmark detection in transform JSON
        landmarks_dict = {}
        for lm_id in landmark_ids:
            landmarks_dict[f"landmark_{lm_id}"] = {
                "world_mm": [float(x) for x in landmark_locations_world[lm_id]],
                "voxel": [int(x) for x in landmark_voxel_pred[lm_id]]
            }
        
        transform_data["transformations"].append({
            "step": 9,
            "operation": "landmark_detection",
            "parameters": {
                "model_path": landmark_detection_model_path,
                "num_landmarks": num_landmarks,
                "landmark_ids": landmark_ids,
                "threshold": 0.5,
                "coordinate_correction": "Applied transformation: (x,y,z) -> (z,-y,-x)"
            },
            "detected_landmarks": landmarks_dict,
            "output_files": {
                "landmarks_txt": os.path.join(scan_landmarks_dir, "predicted_landmarks.txt"),
                "landmarks_json": os.path.join(scan_landmarks_dir, "predicted_landmarks.json")
            }
        })
        
        # Save predicted landmark coordinates
        coords_file = os.path.join(scan_landmarks_dir, "predicted_landmarks.txt")
        with open(coords_file, "w") as f:
            f.write(f"Scan: {scan_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for lm_id in landmark_ids:
                coords_mm = landmark_locations_world[lm_id]
                coords_vx = landmark_voxel_pred[lm_id]
                f.write(f"Landmark {lm_id}:\n")
                f.write(f"  World (mm): ({coords_mm[0]:.3f}, {coords_mm[1]:.3f}, {coords_mm[2]:.3f})\n")
                f.write(f"  Voxel: ({coords_vx[0]}, {coords_vx[1]}, {coords_vx[2]})\n\n")
        
        # Save as JSON
        json_data = {
            'scan_name': scan_name,
            'timestamp': datetime.now().isoformat(),
            'landmarks': {}
        }
        for lm_id in landmark_ids:
            json_data['landmarks'][str(lm_id)] = {
                'world_mm': landmark_locations_world[lm_id].tolist(),
                'voxel': [int(v) for v in landmark_voxel_pred[lm_id]]
            }
        
        json_file = os.path.join(scan_landmarks_dir, "predicted_landmarks.json")
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)
        
        # Print predicted landmark locations to console
        print(f"\n📍 Predicted Landmark Locations:")
        print(f"{'='*60}")
        for lm_id in landmark_ids:
            coords_mm = landmark_locations_world[lm_id]
            coords_vx = landmark_voxel_pred[lm_id]
            print(f"Landmark {lm_id}:")
            print(f"  World (mm): ({coords_mm[0]:8.3f}, {coords_mm[1]:8.3f}, {coords_mm[2]:8.3f})")
            print(f"  Voxel:      ({coords_vx[0]:3d}, {coords_vx[1]:3d}, {coords_vx[2]:3d})")
        print(f"{'='*60}\n")
        
        # Store for CSV
        prediction_row = {'scan_name': scan_name}
        for lm_id in landmark_ids:
            coords_mm = landmark_locations_world[lm_id]
            coords_vx = landmark_voxel_pred[lm_id]
            prediction_row[f'landmark_{lm_id}_mm_x'] = f"{coords_mm[0]:.3f}"
            prediction_row[f'landmark_{lm_id}_mm_y'] = f"{coords_mm[1]:.3f}"
            prediction_row[f'landmark_{lm_id}_mm_z'] = f"{coords_mm[2]:.3f}"
            prediction_row[f'landmark_{lm_id}_vx_x'] = coords_vx[0]
            prediction_row[f'landmark_{lm_id}_vx_y'] = coords_vx[1]
            prediction_row[f'landmark_{lm_id}_vx_z'] = coords_vx[2]
        all_predictions.append(prediction_row)
        
        # === Scan Alignment ===
        try:
            if skip_alignment:
                print(f"  skip_alignment=True — skipping Frankfort plane alignment for {scan_name}")
                transform_data["transformations"].append({
                    "step": 10,
                    "operation": "frankfort_plane_alignment",
                    "status": "SKIPPED",
                    "reason": "skip_alignment=True"
                })
                raise _AlignmentSkipped(scan_name)

            print(f"\nAligning scan {scan_name} based on Frankfort plane...")
            # Prepare landmarks array (all 6 landmarks: 8-13)
            all_landmarks = np.array([landmark_locations_world[lm_id] for lm_id in landmark_ids])
            
            # Choose plane fitting method based on no_eyes configuration
            right_muscle_points = None   # set only in no_eyes mode (for LR alignment)
            left_muscle_points  = None
            if no_eyes:
                print(f"  No-eyes mode: Using lm10, lm11 + masseter/lateral-pterygoid top points (6 pts)")

                # Segment head muscles
                segmentation_img, seg_path = segment_head_muscles(
                    nii_path, scan_landmarks_dir, scan_name
                )
                segmentation_data = segmentation_img.get_fdata()

                # Extract topmost voxel for each of the four target muscles
                # (any muscle missing from the segmentation is returned as None)
                masseter_right_top, masseter_left_top, lat_pteryg_right_top, lat_pteryg_left_top = \
                    extract_muscle_top_points(segmentation_data, affine, scan_name)

                # Build the candidate point set, skipping any muscles that
                # TotalSegmentator failed to produce. LM10/LM11 are always
                # included; we require at least one muscle on each side so
                # the LR-direction step still has a meaningful left vs. right
                # reference.
                _muscle_candidates = [
                    ('masseter_right',          masseter_right_top),
                    ('masseter_left',           masseter_left_top),
                    ('lateral_pterygoid_right', lat_pteryg_right_top),
                    ('lateral_pterygoid_left',  lat_pteryg_left_top),
                ]
                _missing_muscles = [name for name, pt in _muscle_candidates if pt is None]
                _present_muscles = [(name, pt) for name, pt in _muscle_candidates if pt is not None]

                _right_present = any(name.endswith('_right') for name, _ in _present_muscles)
                _left_present  = any(name.endswith('_left')  for name, _ in _present_muscles)

                if _missing_muscles:
                    print(f"  Missing muscle segmentations: {_missing_muscles}")
                if len(_present_muscles) < 2:
                    raise _LandmarkOutlierError(
                        f"Only {len(_present_muscles)} muscle(s) segmented "
                        f"({[n for n, _ in _present_muscles]}); need at least 2 "
                        f"to fit a Frankfort plane with LM10/LM11. Scan: '{scan_name}'."
                    )
                if not (_right_present and _left_present):
                    raise _LandmarkOutlierError(
                        f"Muscle segmentation missing on one side "
                        f"(right={_right_present}, left={_left_present}); cannot "
                        f"determine left-right axis. Scan: '{scan_name}'."
                    )

                point_labels_6pt = ['lm10', 'lm11'] + [name for name, _ in _present_muscles]
                points_6pt = np.array(
                    [landmark_locations_world[10], landmark_locations_world[11]]
                    + [pt for _, pt in _present_muscles]
                )

                print(f"\n  {len(points_6pt)}-point FH plane candidates:")
                for _lbl, _pt in zip(point_labels_6pt, points_6pt):
                    print(f"    {_lbl}: ({_pt[0]:.2f}, {_pt[1]:.2f}, {_pt[2]:.2f})")

                fh_plane_vis_path = os.path.join(scan_viz_dir, f"{scan_name}_frankfort_plane_landmarks.png")
                scan_volume = img_tensor.cpu().numpy()[0, 0]

                # Collect failures across all checks; we render the visualization
                # before raising, so the user always gets an image for review.
                failure_reasons = []        # list of human-readable strings
                flagged_labels = set()      # labels to highlight in the image
                removed_labels = []
                all_round_info = []
                filtered_points = points_6pt
                filtered_labels = list(point_labels_6pt)
                lateral_info = {}
                symmetry_info = {}

                # --- Check 0: LM10 / LM11 anatomical distance ---
                # Catches grossly mislocalized ear-canal landmarks via the
                # rotation-invariant 3D distance between them. Tilted heads
                # still pass; only anatomically impossible spacings fail.
                print(f"\n  LM10/LM11 distance check:")
                bad_symmetry, symmetry_info = check_lm10_lm11_symmetry(
                    landmark_locations_world[10], landmark_locations_world[11],
                )
                if bad_symmetry:
                    # Cannot determine which of LM10/LM11 is wrong from the
                    # distance test alone, so flag both for the visualization.
                    flagged_labels.update(['lm10', 'lm11'])
                    failure_reasons.append(
                        f"LM10/LM11 distance check failed: {'; '.join(bad_symmetry)}."
                    )

                # --- Check 1: Horizontal containment of muscles within LM10/LM11 ---
                # Only check muscles that were actually segmented.
                muscle_lbls_array = [name for name, _ in _present_muscles]
                muscle_pts_array  = np.array([pt for _, pt in _present_muscles])
                print(f"\n  Horizontal containment check (X-axis): muscles inside LM10/LM11 span")
                bad_muscles_containment, containment_info = check_muscle_horizontal_containment(
                    landmark_locations_world[10], landmark_locations_world[11],
                    muscle_pts_array, muscle_lbls_array,
                )
                if bad_muscles_containment:
                    flagged_labels.update(bad_muscles_containment)
                    failure_reasons.append(
                        f"Muscle(s) {bad_muscles_containment} failed horizontal containment "
                        f"(X outside LM10/LM11 span)."
                    )

                # --- Check 2: Centroid-based outlier detection (up to 2 rounds) ---
                if not failure_reasons:
                    print(f"\n  Centroid-based outlier detection:")
                    filtered_points, filtered_labels, removed_labels, all_round_info = \
                        detect_and_remove_outliers(points_6pt, point_labels_6pt)

                    if all_round_info:
                        first_round = all_round_info[0]
                        first_distances = first_round['distances']
                        print(f"\n  Centroid distances - round 1 "
                              f"(median = {first_round['median_distance']:.2f} mm):")
                        for _lbl in point_labels_6pt:
                            if _lbl in first_distances:
                                print(f"    {_lbl}: {first_distances[_lbl]:.2f} mm")

                    removed_landmarks = [lbl for lbl in removed_labels if lbl in ('lm10', 'lm11')]
                    if removed_landmarks:
                        flagged_labels.update(removed_landmarks)
                        detail_parts = []
                        for ri, rinfo in enumerate(all_round_info):
                            if rinfo['removed']:
                                d_val = rinfo['distances'].get(rinfo['removed'], 0.0)
                                detail_parts.append(
                                    f"round {ri+1}: removed '{rinfo['removed']}' "
                                    f"(distance {d_val:.2f} mm, median {rinfo['median_distance']:.2f} mm)"
                                )
                        failure_reasons.append(
                            f"Landmark(s) {removed_landmarks} detected as centroid outlier(s). "
                            f"{'; '.join(detail_parts)}."
                        )

                    if removed_labels and not removed_landmarks:
                        for rl in removed_labels:
                            print(f"  Outlier '{rl}' excluded from plane fitting.")

                # --- Check 3: lm10 / lm11 must lie further from centroid than
                #     every kept muscle point (they should be the most lateral).
                if not failure_reasons:
                    print(f"\n  Lateral-extent check (lm10/lm11 further than muscles):")
                    bad_lateral, lateral_info = check_landmarks_further_than_muscles(
                        filtered_points, filtered_labels,
                    )
                    if bad_lateral:
                        flagged_labels.update(bad_lateral)
                        failure_reasons.append(
                            f"Landmark(s) {bad_lateral} are not further from centroid than "
                            f"the muscle points (max muscle distance "
                            f"{lateral_info.get('max_muscle_distance', float('nan')):.2f} mm)."
                        )

                # Always render the 6-point landmark visualization, even if the
                # scan will be rejected below. This gives the user an image for
                # post-hoc inspection of skipped scans.
                print(f"  Creating Frankfort plane landmarks visualization...")
                status_text = None
                if failure_reasons:
                    status_text = "SCAN FLAGGED: " + " | ".join(failure_reasons)
                try:
                    render_frankfort_landmarks_visualization(
                        scan_volume, points_6pt, point_labels_6pt,
                        fh_plane_vis_path,
                        removed_labels=removed_labels,
                        flagged_labels=flagged_labels,
                        status_text=status_text,
                    )
                    print(f"  ✓ Frankfort plane landmarks visualization saved to {fh_plane_vis_path}")
                except Exception as _vis_err:
                    print(f"  WARNING: Failed to render Frankfort visualization: {_vis_err}")

                # Now raise if any check failed (image already saved).
                if failure_reasons:
                    raise _LandmarkOutlierError(
                        f"{' '.join(failure_reasons)} "
                        f"Symmetry details: {symmetry_info}. "
                        f"Containment details: {containment_info}. "
                        f"Lateral details: {lateral_info}. "
                        f"Scan: '{scan_name}'."
                    )

                landmarks_for_plane = filtered_points

                # Right / left muscle points for LR alignment (after outlier removal)
                _right_labels = {'masseter_right', 'lateral_pterygoid_right'}
                _left_labels  = {'masseter_left',  'lateral_pterygoid_left'}
                right_muscle_points = np.array([
                    p for lbl, p in zip(filtered_labels, filtered_points)
                    if lbl in _right_labels
                ])
                left_muscle_points = np.array([
                    p for lbl, p in zip(filtered_labels, filtered_points)
                    if lbl in _left_labels
                ])

                # Fallback: if all muscle points on one side were removed as outliers
                if len(right_muscle_points) == 0 or len(left_muscle_points) == 0:
                    print("  WARNING: Insufficient muscle points for LR alignment after outlier removal.")
                    print("           Falling back to lm10 / lm11 for left-right direction.")
                    right_muscle_points = np.array([landmark_locations_world[10]])
                    left_muscle_points  = np.array([landmark_locations_world[11]])

                # Record in transform JSON
                transform_data["transformations"].append({
                    "step": "9b",
                    "operation": "head_muscles_segmentation",
                    "parameters": {
                        "task": "head_muscles",
                        "reference": (
                            "Wasserthal et al. (2023) TotalSegmentator: Robust Segmentation of "
                            "104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. "
                            "https://doi.org/10.1148/ryai.230024"
                        ),
                        "structures": [
                            "masseter_right", "masseter_left",
                            "lateral_pterygoid_right", "lateral_pterygoid_left",
                        ],
                        "muscle_top_points_mm": {
                            name: ([float(x) for x in pt] if pt is not None else None)
                            for name, pt in [
                                ('masseter_right',          masseter_right_top),
                                ('masseter_left',           masseter_left_top),
                                ('lateral_pterygoid_right', lat_pteryg_right_top),
                                ('lateral_pterygoid_left',  lat_pteryg_left_top),
                            ]
                        },
                        "missing_muscles": _missing_muscles,
                        "outlier_detection": {
                            "method": "centroid_distance",
                            "threshold_ratio_muscle": OUTLIER_CENTROID_RATIO_MUSCLE,
                            "threshold_ratio_landmark": OUTLIER_CENTROID_RATIO_LANDMARK,
                            "removed_outliers": removed_labels,
                            "lm10_lm11_symmetry_check": symmetry_info,
                            "muscle_containment_check": containment_info,
                            "lateral_extent_check": lateral_info,
                            "rounds": [
                                {
                                    "distances_mm": rinfo['distances'],
                                    "centroid_mm": rinfo['centroid'],
                                    "median_distance_mm": rinfo['median_distance'],
                                    "removed": rinfo['removed'],
                                }
                                for rinfo in all_round_info
                            ],
                        },
                    },
                    "output_file": seg_path,
                    "visualization": fh_plane_vis_path,
                })

            else:
                # Standard mode: Extract landmarks 10-13 (indices 2-5) for plane computation
                print(f"  Standard mode: Using landmarks 10-13 for Frankfort plane")
                landmarks_for_plane = all_landmarks[2:6]

            # Get mask points (threshold = 0 for resampled scans)
            mask_points = np.argwhere(img_tensor.cpu().numpy()[0, 0] > 0).astype(np.float32)

            # Align scan points — use SVD multipoint alignment in no_eyes mode
            if no_eyes:
                aligned_points, rotation, orig_normal, center, z_rotation_angle = \
                    align_to_horizontal_multipoint(
                        mask_points, landmarks_for_plane,
                        right_muscle_points, left_muscle_points,
                    )
            else:
                aligned_points, rotation, orig_normal, center, z_rotation_angle = \
                    align_to_horizontal(mask_points, landmarks_for_plane)

            corrected_normal = np.array([0, 0, -1])
            angles = rotation.as_euler('xyz', degrees=True)

            print(f"  Original Frankfort Plane Normal: {orig_normal}")
            print(f"  Corrected Plane Normal: {corrected_normal}")
            print(f"  Plane Center: {center}")
            print(f"  Rotation applied (degrees): {angles}")
            print(f"  Z-axis rotation: {z_rotation_angle:.2f}°")
            
            # Always create visualization of predicted landmarks
            print(f"  Creating landmark visualization...")
            vis_path = os.path.join(scan_viz_dir, f"{scan_name}_landmarks_visualization.png")
            visualize_landmarks_with_scan(
                img_tensor.cpu().numpy()[0, 0], 
                all_landmarks, 
                vis_path, 
                threshold=-300
            )
            print(f"  ✓ Visualization saved to {vis_path}")
            
            # Check if large rotation should be flagged (angle > 10 degrees)
            rotation_threshold = 10.0
            needs_flagging = abs(z_rotation_angle) > rotation_threshold
            
            if needs_flagging:
                print(f"  ⚠️  Large rotation detected ({abs(z_rotation_angle):.2f}°)!")
                # Add to flagged scans
                flagged_scans.append({
                    'scan_name': scan_name,
                    'z_rotation_angle_degrees': abs(z_rotation_angle),
                    'warning': 'Large rotation angle detected - possible landmark prediction error'
                })
            else:
                print(f"  ✓ Rotation within acceptable range ({abs(z_rotation_angle):.2f}°)")
            
            # Save rotated volume
            rotation_matrix = rotation.as_matrix()
            if np.linalg.det(rotation_matrix) < 0:
                print("  Reflection detected! Fixing handedness...")
                rotation_matrix[:, 2] *= -1
                rotation = R.from_matrix(rotation_matrix)
            
            inverse_rotation = np.linalg.inv(rotation_matrix)
            offset = center - inverse_rotation @ center
            
            scan_data = img_tensor.cpu().numpy()[0, 0]
            rotated_volume = affine_transform(
                scan_data,
                inverse_rotation,
                offset=offset,
                order=1,
                cval=-1000
            )
            
            # Save aligned scan in the same directory as input scan
            patient_dir = os.path.dirname(nii_path)
            aligned_scan_path = os.path.join(patient_dir, f"{scan_name}_aligned.nii.gz")
            aligned_img = nib.Nifti1Image(rotated_volume, affine)
            nib.save(aligned_img, aligned_scan_path)
            print(f"  Aligned scan saved to {aligned_scan_path}")
            
            # Align all landmarks using the same rotation
            aligned_landmarks = rotation.apply(all_landmarks - center) + center
            
            # Save aligned landmarks as .npy in centralized output directory
            aligned_lm_path = os.path.join(aligned_npy_dir, f"{scan_name}_lm_aligned.npy")
            np.save(aligned_lm_path, aligned_landmarks)
            print(f"  Aligned landmarks saved to {aligned_lm_path}")
            
            # Add to CSV list
            aligned_lm_row = {'scan_name': scan_name}
            for i, lm_id in enumerate(landmark_ids):
                aligned_lm_row[f'landmark_{lm_id}_x'] = aligned_landmarks[i, 0]
                aligned_lm_row[f'landmark_{lm_id}_y'] = aligned_landmarks[i, 1]
                aligned_lm_row[f'landmark_{lm_id}_z'] = aligned_landmarks[i, 2]
            all_aligned_landmarks.append(aligned_lm_row)
            
            # Record alignment in transform JSON
            aligned_landmarks_dict = {}
            for i, lm_id in enumerate(landmark_ids):
                aligned_landmarks_dict[f"landmark_{lm_id}"] = {
                    "aligned_world_mm": [float(x) for x in aligned_landmarks[i]]
                }
            
            # Determine alignment method description
            if no_eyes:
                alignment_method_desc = (
                    "Frankfort plane SVD (lm10, lm11 + masseter right/left + "
                    "lateral pterygoid right/left — 6 points, outlier detection applied)"
                )
            else:
                alignment_method_desc = "Frankfort plane (landmarks 10-13)"
            
            transform_data["transformations"].append({
                "step": 10,
                "operation": "frankfort_plane_alignment",
                "parameters": {
                    "alignment_method": alignment_method_desc,
                    "no_eyes_mode": no_eyes,
                    "original_plane_normal": [float(x) for x in orig_normal],
                    "target_plane_normal": [float(x) for x in corrected_normal],
                    "rotation_center_mm": [float(x) for x in center],
                    "rotation_matrix": [[float(x) for x in row] for row in rotation_matrix],
                    "rotation_angles_xyz_degrees": [float(x) for x in angles],
                    "z_rotation_angle_degrees": float(z_rotation_angle),
                    "interpolation_order": 1,
                    "fill_value": -1000,
                    "flagged_for_review": bool(needs_flagging)
                },
                "aligned_landmarks": aligned_landmarks_dict,
                "output_files": {
                    "aligned_scan": aligned_scan_path,
                    "aligned_landmarks_npy": aligned_lm_path
                }
            })
            
        except _AlignmentSkipped:
            pass  # skip_alignment=True — alignment intentionally skipped
        except _LandmarkOutlierError as e:
            print(f"\n{'!'*60}")
            print(f"SCAN SKIPPED — landmark outlier detected:")
            print(f"  {e}")
            print(f"{'!'*60}\n")
            # Log the skipped scan to a persistent file
            skipped_file = os.path.join(output_dir, "skipped_scans.txt")
            with open(skipped_file, 'a') as sf:
                sf.write(f"{scan_name}\t{datetime.now().isoformat()}\t{e}\n")
            print(f"  Logged to {skipped_file}")
            # Save transform JSON for this scan
            transform_data["transformations"].append({
                "step": 10,
                "operation": "frankfort_plane_alignment",
                "status": "SKIPPED_OUTLIER",
                "error": str(e),
            })
            try:
                save_transform_json(transform_data, scan_name, output_transform_dir)
            except Exception:
                pass
            continue  # skip this scan, proceed with remaining scans
        except Exception as e:
            print(f"ERROR: Alignment failed for {scan_name}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            
            # Record alignment failure in transform JSON
            transform_data["transformations"].append({
                "step": 10,
                "operation": "frankfort_plane_alignment",
                "status": "FAILED",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e)
                }
            })
        
        # Save transform JSON
        try:
            json_path = save_transform_json(transform_data, scan_name, output_transform_dir)
            print(f"  Transform log saved to: {json_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save transform JSON: {e}")
        
        print(f"✓ Completed {scan_name}: {len(landmark_locations_world)} landmarks detected and aligned")
    
    # Save CSV with all predictions
    csv_file = os.path.join(aligned_landmarks_dir, "all_landmark_predictions.csv")
    if all_predictions:
        fieldnames = ['scan_name']
        for lm_id in landmark_ids:
            fieldnames.extend([
                f'landmark_{lm_id}_mm_x', f'landmark_{lm_id}_mm_y', f'landmark_{lm_id}_mm_z',
                f'landmark_{lm_id}_vx_x', f'landmark_{lm_id}_vx_y', f'landmark_{lm_id}_vx_z'
            ])
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_predictions)
        
        print(f"\n✓ Saved all predictions to CSV: {csv_file}")
    
    # Save CSV with all aligned landmarks
    aligned_csv_file = os.path.join(aligned_landmarks_dir, "all_aligned_landmarks.csv")
    if all_aligned_landmarks:
        fieldnames_aligned = ['scan_name']
        for lm_id in landmark_ids:
            fieldnames_aligned.extend([
                f'landmark_{lm_id}_x', f'landmark_{lm_id}_y', f'landmark_{lm_id}_z'
            ])
        
        with open(aligned_csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_aligned)
            writer.writeheader()
            writer.writerows(all_aligned_landmarks)
        
        print(f"✓ Saved all aligned landmarks to CSV: {aligned_csv_file}")
    
    # Save flagged scans to CSV
    if flagged_scans:
        flagged_csv_path = os.path.join(visualizations_output_dir, "flagged_large_rotations.csv")
        flagged_df_data = sorted(flagged_scans, key=lambda x: x['z_rotation_angle_degrees'], reverse=True)
        
        with open(flagged_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['scan_name', 'z_rotation_angle_degrees', 'warning'])
            writer.writeheader()
            writer.writerows(flagged_df_data)
        
        print(f"\n⚠️  Flagged scans with large rotations saved to {flagged_csv_path}")
        print(f"Total flagged scans: {len(flagged_scans)}")
    else:
        print("\n✓ No scans with rotation angle > 10° detected!")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Scans processed: {len(nii_files)}")
    print(f"Total landmarks detected: {len(nii_files) * num_landmarks}")
    print(f"\nOutputs saved to:")
    print(f"  - Heatmaps: {landmarks_output_dir}")
    print(f"  - Aligned NPY: {aligned_npy_dir}")
    print(f"  - Aligned Landmarks: {aligned_landmarks_dir}")
    print(f"  - Visualizations: {visualizations_output_dir}")
    print(f"  - Predictions CSV: {csv_file}")
    if all_aligned_landmarks:
        print(f"  - Aligned Landmarks CSV: {aligned_csv_file}")
    if flagged_scans:
        print(f"  - Flagged Scans CSV: {flagged_csv_path}")
    print(f"{'='*60}\n")


# === Argument Parser ===
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='P2 Preprocessing: Landmark detection and alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--processed_scans_dir', type=str,
                        default=r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Processed-Data",
                        help='Directory containing processed scans from P1')
    parser.add_argument('--output_dir', type=str,
                        default=r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Output_no_alignment",
                        help='Directory for outputs (logs, landmarks)')
    parser.add_argument('--landmark_model', type=str,
                        default=r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/best_model_2025-12-05_13-16-35.pth",
                        help='Path to landmark detection model')
    parser.add_argument('--no_eyes', type=str, default='False',
                        choices=['True', 'False'],
                        help='Use mandible top points instead of eye landmarks')
    parser.add_argument('--skip_alignment', type=str, default='False',
                        choices=['True', 'False'],
                        help='Skip alignment step (landmark detection only)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with command-line arguments
    Processed_scans_dir = args.processed_scans_dir
    output_dir = args.output_dir
    landmark_detection_model_path = args.landmark_model
    no_eyes = (args.no_eyes == 'True')
    skip_alignment = (args.skip_alignment == 'True')
    
    # Update derived paths
    output_transform_dir = os.path.join(output_dir, "Logs", "transform_logs")
    
    print(f"\n{'='*60}")
    print(f"LANDMARK DETECTION PIPELINE - P2 Final Preprocessing")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Input directory: {Processed_scans_dir}")
    print(f"Model path: {landmark_detection_model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Landmarks to detect: {landmark_ids}")
    print(f"No-eyes mode: {no_eyes}")
    if no_eyes:
        print(f"  → Using mandible top points instead of eye landmarks (12-13)")
        print(f"  → Reference: Wasserthal et al. (2023) TotalSegmentator")
    print(f"Skip alignment: {skip_alignment}")
    print(f"{'='*60}\n")
    
    process_scans()
