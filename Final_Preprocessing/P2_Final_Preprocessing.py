import os
import sys
import numpy as np
import nibabel as nib
import torch
import csv
import json
from datetime import datetime
from tqdm import tqdm
import pyvista as pv
import traceback
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform

# Configure PyVista for off-screen rendering
pv.set_plot_theme("document")
pv.global_theme.background = 'white'

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configuration
Processed_scans_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Processed_data"
landmark_detection_model_path = r"/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/best_model_2025-12-05_13-16-35.pth"
output_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output"
output_transform_dir = os.path.join(output_dir, "transform_logs")


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Landmark configuration
landmark_ids = [8, 9, 10, 11, 12, 13]
num_landmarks = len(landmark_ids)


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
    scan_name = os.path.basename(scan_path).replace('_CT_resampled_256.nii.gz', '')
    
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
        
        # Check if already aligned
        dot_product = np.dot(left_right_xy, target_lr)
        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        print(f"  Angle to X-axis: {angle_deg:.2f}°")
        
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


# === Main Processing Pipeline ===
def process_scans():
    """Main function to process all scans with landmark detection."""
    
    # Create output directories
    landmarks_output_dir = os.path.join(output_dir, "Landmarks")
    visualizations_output_dir = os.path.join(output_dir, "Visualizations")
    aligned_scans_dir = os.path.join(output_dir, "Aligned_Scans")
    aligned_landmarks_dir = os.path.join(output_dir, "Aligned_Landmarks")
    os.makedirs(landmarks_output_dir, exist_ok=True)
    os.makedirs(visualizations_output_dir, exist_ok=True)
    os.makedirs(aligned_scans_dir, exist_ok=True)
    os.makedirs(aligned_landmarks_dir, exist_ok=True)
    
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
            if file.endswith('_CT_resampled_256.nii.gz'):
                nii_files.append(os.path.join(root, file))
    
    nii_files = sorted(nii_files)
    print(f"\nFound {len(nii_files)} scans to process")
    
    if len(nii_files) == 0:
        print("No scans found matching pattern '*_CT_resampled_256.nii.gz'")
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
        scan_name = filename.replace('_CT_resampled_256.nii.gz', '')
        
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
        print(f"\nAligning scan {scan_name} based on Frankfort plane...")
        try:
            # Prepare landmarks array (all 6 landmarks: 8-13)
            all_landmarks = np.array([landmark_locations_world[lm_id] for lm_id in landmark_ids])
            # Extract landmarks 10-13 (indices 2-5) for plane computation
            landmarks_for_plane = all_landmarks[2:6]
            
            # Get mask points (threshold = 0 for resampled scans)
            mask_points = np.argwhere(img_tensor.cpu().numpy()[0, 0] > 0).astype(np.float32)
            
            # Align scan points
            aligned_points, rotation, orig_normal, center, z_rotation_angle = align_to_horizontal(
                mask_points, landmarks_for_plane
            )
            
            corrected_normal = np.array([0, 0, -1])
            angles = rotation.as_euler('xyz', degrees=True)
            
            print(f"  Original Frankfort Plane Normal: {orig_normal}")
            print(f"  Corrected Plane Normal: {corrected_normal}")
            print(f"  Plane Center: {center}")
            print(f"  Rotation applied (degrees): {angles}")
            print(f"  Z-axis rotation: {z_rotation_angle:.2f}°")
            
            # Check if visualization is needed (angle > 10 degrees)
            rotation_threshold = 10.0
            needs_visualization = abs(z_rotation_angle) > rotation_threshold
            
            if needs_visualization:
                print(f"  ⚠️  Large rotation detected ({abs(z_rotation_angle):.2f}°)! Creating visualization...")
                vis_path = os.path.join(scan_viz_dir, f"{scan_name}_large_rotation_vis.png")
                visualize_landmarks_with_scan(
                    img_tensor.cpu().numpy()[0, 0], 
                    all_landmarks, 
                    vis_path, 
                    threshold=-300
                )
                
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
            aligned_lm_path = os.path.join(aligned_landmarks_dir, f"{scan_name}_lm_aligned.npy")
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
            
            transform_data["transformations"].append({
                "step": 10,
                "operation": "frankfort_plane_alignment",
                "parameters": {
                    "alignment_method": "Frankfort plane (landmarks 10-13)",
                    "original_plane_normal": [float(x) for x in orig_normal],
                    "target_plane_normal": [float(x) for x in corrected_normal],
                    "rotation_center_mm": [float(x) for x in center],
                    "rotation_matrix": [[float(x) for x in row] for row in rotation_matrix],
                    "rotation_angles_xyz_degrees": [float(x) for x in angles],
                    "z_rotation_angle_degrees": float(z_rotation_angle),
                    "interpolation_order": 1,
                    "fill_value": -1000,
                    "flagged_for_review": bool(needs_visualization)
                },
                "aligned_landmarks": aligned_landmarks_dict,
                "output_files": {
                    "aligned_scan": aligned_scan_path,
                    "aligned_landmarks_npy": aligned_lm_path
                }
            })
            
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
    csv_file = os.path.join(output_dir, "all_landmark_predictions.csv")
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
        flagged_csv_path = os.path.join(output_dir, "flagged_large_rotations.csv")
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
    print(f"  - Landmarks: {landmarks_output_dir}")
    print(f"  - Aligned Scans: {aligned_scans_dir}")
    print(f"  - Aligned Landmarks: {aligned_landmarks_dir}")
    print(f"  - Visualizations (large rotations): {visualizations_output_dir}")
    print(f"  - Predictions CSV: {csv_file}")
    if all_aligned_landmarks:
        print(f"  - Aligned Landmarks CSV: {aligned_csv_file}")
    if flagged_scans:
        print(f"  - Flagged Scans CSV: {flagged_csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"LANDMARK DETECTION PIPELINE - P2 Final Preprocessing")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Input directory: {Processed_scans_dir}")
    print(f"Model path: {landmark_detection_model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Landmarks to detect: {landmark_ids}")
    print(f"{'='*60}\n")
    
    process_scans()
