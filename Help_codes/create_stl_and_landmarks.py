import nrrd
import nibabel as nib
import json
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh
import os
from pathlib import Path
import pyvista as pv
pv.OFF_SCREEN = True  # Enable off-screen rendering for headless environments

# Input directories and processing configuration
MASK_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Quality_Check/Canal_good_quality/masks"
JSON_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_Markups_Corrected"
OUTPUT_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Quality_Check/Canal_good_quality/stl files"

# Number of scans to process (set to None to process all available scans)
NUM_SCANS = None

# Output generation options: "both", "stl", "landmarks"
GENERATE_OUTPUT = "stl"  # Set to "stl" for STL only, "landmarks" for landmarks only, or "both" for both

# Visualization settings
VISUALIZE_NORMALS = False  # Set to True to visualize mesh normals
NORMAL_SAMPLE_RATE = 100  # Show normals for every Nth vertex (higher = fewer arrows)
NORMAL_LENGTH_SCALE = 5.0  # Scale factor for normal arrow length


def visualize_mesh_with_normals(mesh, output_path):
    """
    Save mesh visualization with surface normals as arrows to an image file using PyVista.
    
    Args:
        mesh: trimesh object
        output_path: Path to save visualization as PNG image
    """
    # Sample vertices to display normals (not all vertices to avoid clutter)
    num_vertices = len(mesh.vertices)
    sample_indices = np.arange(0, num_vertices, NORMAL_SAMPLE_RATE)
    
    # Get sampled vertices and normals
    sampled_vertices = mesh.vertices[sample_indices]
    sampled_normals = mesh.vertex_normals[sample_indices]
    
    # Calculate average edge length for scaling normal arrows
    if len(mesh.edges) > 0:
        edge_lengths = np.linalg.norm(
            mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], 
            axis=1
        )
        avg_edge_length = np.mean(edge_lengths)
        normal_length = avg_edge_length * NORMAL_LENGTH_SCALE
    else:
        # Fallback if no edges
        bbox = mesh.bounds
        bbox_size = np.linalg.norm(bbox[1] - bbox[0])
        normal_length = bbox_size * 0.05 * NORMAL_LENGTH_SCALE
    
    print(f"Creating visualization with {len(sample_indices)} normal vectors (sampling every {NORMAL_SAMPLE_RATE}th vertex)...")
    
    try:
        # Create PyVista plotter with off-screen rendering
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        
        # Convert trimesh to PyVista mesh
        pv_mesh = pv.wrap(mesh)
        
        # Add the mesh to the plotter
        plotter.add_mesh(
            pv_mesh,
            color='cyan',
            opacity=0.6,
            show_edges=False,
            lighting=True,
            smooth_shading=True
        )
        
        # Create arrows for normal vectors
        # PyVista expects arrows as points (origins) and vectors (directions)
        arrow_origins = sampled_vertices
        arrow_directions = sampled_normals * normal_length
        
        # Add arrows to the plotter
        plotter.add_arrows(
            arrow_origins,
            arrow_directions,
            mag=1.0,  # magnitude is already in the direction vectors
            color='red',
            opacity=0.8
        )
        
        # Set up camera and lighting
        plotter.add_light(pv.Light(position=(1, 1, 1), light_type='headlight'))
        
        # Set camera view: Z vertical, X horizontal, Y going into screen
        # This is equivalent to viewing from the front along the Y-axis
        plotter.view_xz()
        
        plotter.add_axes()
        plotter.add_text('Mesh with Surface Normals', position='upper_edge', font_size=14)
        
        # Save the image
        print(f"Saving visualization to: {output_path}")
        plotter.screenshot(output_path)
        plotter.close()
        
        print(f"✓ Visualization saved successfully")
        
    except Exception as e:
        print(f"Warning: Could not save visualization: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with processing...")


def create_stl_from_mask(mask_path, output_stl_path):
    """
    Create an STL file from a binary mask using marching cubes algorithm.
    Supports both .nrrd and .nii.gz formats.
    
    Args:
        mask_path: Path to the mask file (.nrrd or .nii.gz)
        output_stl_path: Path where the STL file will be saved
    """
    print(f"Reading mask from: {mask_path}")
    
    # Determine file format and read accordingly
    if mask_path.endswith('.nii.gz') or mask_path.endswith('.nii'):
        # Read NIfTI file
        nii_img = nib.load(mask_path)
        mask_data = nii_img.get_fdata()
        spacing = np.array(nii_img.header.get_zooms()[:3])
    else:
        # Read NRRD file
        mask_data, mask_header = nrrd.read(mask_path)
        
        # Get spacing information for correct physical dimensions
        spacing = mask_header.get('space directions', None)
        if spacing is not None:
            # Extract diagonal elements (spacing) from the space directions matrix
            spacing = np.array([spacing[0][0], spacing[1][1], spacing[2][2]])
        else:
            spacing = mask_header.get('spacings', np.array([1.0, 1.0, 1.0]))
    
    print(f"Mask shape: {mask_data.shape}")
    print(f"Spacing: {spacing}")
    
    # Ensure mask is binary
    mask_binary = mask_data > 0
    
    # Apply Gaussian smoothing to the mask for smoother surface
    print("Applying Gaussian smoothing to mask...")
    sigma = 1.5  # Smoothing parameter (adjust for more/less smoothing)
    mask_smoothed = gaussian_filter(mask_binary.astype(float), sigma=sigma)
    
    # Use marching cubes to extract surface mesh
    print("Extracting surface mesh using marching cubes...")
    verts, faces, normals, values = measure.marching_cubes(
        mask_smoothed, 
        level=0.5,
        spacing=spacing
    )
    
    print(f"Generated mesh with {len(verts)} vertices and {len(faces)} faces")
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Apply additional Laplacian smoothing to the mesh
    print("Applying Laplacian smoothing to mesh...")
    trimesh.smoothing.filter_laplacian(mesh, iterations=5)
    
    # Save as STL
    print(f"Saving STL to: {output_stl_path}")
    mesh.export(output_stl_path)
    
    return mesh


def create_simplified_landmarks_json(json_path, output_json_path):
    """
    Create a simplified JSON file with landmark id, label, and location.
    
    Args:
        json_path: Path to the original landmarks JSON file
        output_json_path: Path where the simplified JSON will be saved
    """
    print(f"Reading landmarks from: {json_path}")
    
    # Read the original JSON file
    with open(json_path, 'r') as f:
        landmarks_data = json.load(f)
    
    # Extract relevant information
    simplified_landmarks = []
    
    # Handle different JSON structures (Slicer markups format)
    if 'markups' in landmarks_data:
        markups = landmarks_data['markups']
        for markup in markups:
            if 'controlPoints' in markup:
                for idx, point in enumerate(markup['controlPoints']):
                    landmark = {
                        'id': point.get('id', f"point_{idx}"),
                        'label': point.get('label', f"Landmark_{idx}"),
                        'position': point.get('position', [0, 0, 0])
                    }
                    simplified_landmarks.append(landmark)
    # Handle simple list format
    elif isinstance(landmarks_data, list):
        for idx, point in enumerate(landmarks_data):
            landmark = {
                'id': point.get('id', f"point_{idx}"),
                'label': point.get('label', f"Landmark_{idx}"),
                'position': point.get('position', point.get('location', [0, 0, 0]))
            }
            simplified_landmarks.append(landmark)
    # Handle dictionary with landmarks key
    elif 'landmarks' in landmarks_data:
        for idx, point in enumerate(landmarks_data['landmarks']):
            landmark = {
                'id': point.get('id', f"point_{idx}"),
                'label': point.get('label', f"Landmark_{idx}"),
                'position': point.get('position', point.get('location', [0, 0, 0]))
            }
            simplified_landmarks.append(landmark)
    
    print(f"Found {len(simplified_landmarks)} landmarks")
    
    # Create simplified structure
    output_data = {
        'landmarks': simplified_landmarks,
        'count': len(simplified_landmarks)
    }
    
    # Save simplified JSON
    print(f"Saving simplified landmarks to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_data


def process_patient(patient_id, mask_dir, json_dir, output_dir, mask_filename=None):
    """
    Process a single patient: create STL and/or simplified landmarks JSON based on GENERATE_OUTPUT setting.
    
    Args:
        patient_id: Patient identifier (e.g., "CHUM-013_right_ear")
        mask_dir: Directory containing mask files
        json_dir: Directory containing JSON landmark files
        output_dir: Directory where outputs will be saved
        mask_filename: Optional full filename of the mask file (with extension)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Construct file paths
    if mask_filename is None:
        # Try to find the mask file with any supported extension
        for ext in ['.nrrd', '.nii.gz', '.nii']:
            test_path = os.path.join(mask_dir, f"{patient_id}{ext}")
            if os.path.exists(test_path):
                mask_path = test_path
                break
        else:
            mask_path = os.path.join(mask_dir, f"{patient_id}.nrrd")  # Default fallback
    else:
        mask_path = os.path.join(mask_dir, mask_filename)
    
    json_path = os.path.join(json_dir, f"{patient_id}.mrk.json")
    output_stl_path = os.path.join(output_dir, f"{patient_id}.stl")
    output_json_path = os.path.join(output_dir, f"{patient_id}_landmarks.json")
    
    # Determine what to generate
    generate_stl = GENERATE_OUTPUT in ["both", "stl"]
    generate_landmarks = GENERATE_OUTPUT in ["both", "landmarks"]
    
    # Check if required input files exist
    if generate_stl and not os.path.exists(mask_path):
        print(f"✗ Mask file not found: {mask_path}")
        return False
    
    if generate_landmarks and not os.path.exists(json_path):
        print(f"✗ JSON file not found: {json_path}")
        return False
    
    try:
        print("\n" + "="*60)
        print(f"PROCESSING: {patient_id}")
        print("="*60)
        
        mesh = None
        landmarks = None
        step_num = 1
        
        # Create STL from mask if requested
        if generate_stl:
            print(f"STEP {step_num}: Creating STL from mask")
            mesh = create_stl_from_mask(mask_path, output_stl_path)
            print(f"✓ STL file created successfully")
            step_num += 1
        
        # Create simplified landmarks JSON if requested
        if generate_landmarks:
            print(f"STEP {step_num}: Creating simplified landmarks JSON")
            landmarks = create_simplified_landmarks_json(json_path, output_json_path)
            print(f"✓ Landmarks JSON created successfully")
            step_num += 1
        
        # Visualize mesh with normals if enabled (only if STL was generated)
        if VISUALIZE_NORMALS and mesh is not None:
            print(f"STEP {step_num}: Visualizing mesh with normals")
            output_viz_path = os.path.join(output_dir, f"{patient_id}_visualization.png")
            visualize_mesh_with_normals(mesh, output_viz_path)
        
        # Summary
        print(f"\n✓ {patient_id} processed successfully")
        if generate_stl:
            print(f"  Output STL: {output_stl_path}")
        if generate_landmarks:
            print(f"  Output JSON: {output_json_path}")
        if VISUALIZE_NORMALS and mesh is not None:
            print(f"  Visualization: {output_viz_path}")
        if mesh is not None:
            print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, Watertight: {mesh.is_watertight}")
        if landmarks is not None:
            print(f"  Landmarks: {landmarks['count']} landmarks")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process all specified patients."""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Scan mask directory for available patients
    print("="*60)
    print("SCANNING MASK DIRECTORY FOR PATIENTS")
    print("="*60)
    print(f"Mask directory: {MASK_DIR}")
    
    # Get all .nrrd and .nii.gz files in mask directory
    mask_files = sorted([f for f in os.listdir(MASK_DIR) 
                         if f.endswith('.nrrd') or f.endswith('.nii.gz') or f.endswith('.nii')])
    
    if not mask_files:
        print(f"✗ No .nrrd or .nii.gz files found in {MASK_DIR}")
        return 1
    
    # Extract patient IDs (remove file extensions)
    all_patient_ids = []
    patient_file_map = {}  # Map patient_id to full filename
    for f in mask_files:
        if f.endswith('.nii.gz'):
            patient_id = f[:-7]  # Remove .nii.gz
        elif f.endswith('.nii'):
            patient_id = f[:-4]  # Remove .nii
        else:  # .nrrd
            patient_id = f[:-5]  # Remove .nrrd
        all_patient_ids.append(patient_id)
        patient_file_map[patient_id] = f
    
    # Determine which patients to process
    if NUM_SCANS is None:
        patient_ids_to_process = all_patient_ids
        print(f"Found {len(all_patient_ids)} patients - processing all")
    else:
        patient_ids_to_process = all_patient_ids[:NUM_SCANS]
        print(f"Found {len(all_patient_ids)} patients - processing first {len(patient_ids_to_process)}")
    
    print("="*60)
    print("STL AND LANDMARKS BATCH PROCESSING")
    print("="*60)
    print(f"Output mode: {GENERATE_OUTPUT.upper()}")
    print(f"Mask directory: {MASK_DIR}" if GENERATE_OUTPUT in ["both", "stl"] else "")
    print(f"JSON directory: {JSON_DIR}" if GENERATE_OUTPUT in ["both", "landmarks"] else "")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Patients to process: {len(patient_ids_to_process)}")
    print("="*60)
    
    # Process each patient
    successful = []
    failed = []
    
    for patient_id in patient_ids_to_process:
        # Get the actual filename with correct extension
        mask_filename = patient_file_map.get(patient_id, f"{patient_id}.nrrd")
        success = process_patient(patient_id, MASK_DIR, JSON_DIR, OUTPUT_DIR, mask_filename)
        if success:
            successful.append(patient_id)
        else:
            failed.append(patient_id)
    
    # Final summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully processed: {len(successful)}/{len(patient_ids_to_process)}")
    if successful:
        print("\nSuccessful:")
        for patient_id in successful:
            print(f"  ✓ {patient_id}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for patient_id in failed:
            print(f"  ✗ {patient_id}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
