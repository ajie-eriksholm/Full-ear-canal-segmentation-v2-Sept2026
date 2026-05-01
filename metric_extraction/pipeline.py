# -*- coding: utf-8 -*-
"""
Full Ear Processing Pipeline.

Combines all processing steps into a unified pipeline:
1. Load and invert mask
2. Create mesh from inverted mask
3. Load landmarks
4. Compute geodesic path (Top RS -> Bottom RS)
5. Extract centerline with plane-based trimming (VMTK)
6. Validate centerline
7. Fit plane and trim centerline
8. Compute metrics (length, tortuosity, cross-sections, volume)
9. Save outputs (VTK, STL, JSON)

Usage:
    python -m full_ear_processing.pipeline --input data_dir --output output_dir
    python -m full_ear_processing.pipeline --input data_dir --output output_dir --viz
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyvista as pv
import vtk

from utils import (
    # Volume computation
    compute_voxel_mesh_volume,
    # Mask processing
    calculate_isthmus_to_eardrum_length_and_tortuosity,
    load_mask,
    load_and_invert_mask,
    save_inverted_mask,
    # Mesh creation
    create_mesh_from_mask,
    simplify_mesh_for_vmtk,
    save_mesh_as_stl,
    # Landmark I/O
    load_landmarks_from_json,
    save_landmarks_json,
    create_complete_landmarks_json,
    compute_middle_rs,
    # VTK I/O
    save_vtk_polydata,
    extract_points_from_polydata,
    create_polydata_from_points,
    # Path computation
    compute_geodesic_path,
    compute_geodesic_raw,
    compute_centerline_vmtk,
    # Plane fitting
    fit_plane_to_points,
    find_plane_centerline_intersection,
    resample_centerline_spline,
    # Metrics
    compute_path_length,
    compute_tortuosity_index,
    compute_volume_between_planes,
    compute_clipped_mesh_between_planes,
    extract_isthmus_eardrum_segment,
    split_canal_at_cbj_plane,
    split_canal_at_cbj_plane_intelligent,
    # Validation
    validate_centerline,
    find_geodesic_furthest_point,
    # Centerline refinement
    refine_centerline_endpoint,
    # CBJ processing
    load_cbj_landmarks_from_json,
    fit_plane_to_cbj_points,
    find_closest_point_to_plane,
    extract_cbj_cross_section_features,
    compute_cbj_centerline_metrics,
    # Feature extraction
    extract_centerline_features,
    extract_landmark_cross_sections,
    # Visualization
    visualize_mesh_with_landmarks,
    visualize_results,
    visualize_debug_failure,
    visualize_landmark_cross_sections,
    visualize_full_results,
    visualize_volume_segment
)


def detect_geodesic_oscillation(
    geodesic_points: np.ndarray,
    window_size: int = 5,
    oscillation_threshold: float = 0.15,
    verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Detect oscillation in geodesic path, especially at endpoints.
    
    Oscillation is detected by measuring local curvature and path deviation.
    
    Args:
        geodesic_points: Array of 3D points along geodesic path
        window_size: Number of points to consider for local curvature
        oscillation_threshold: Threshold for curvature variation (0-1, normalized by path length)
        verbose: Print debug info
    
    Returns:
        (has_oscillation, debug_info_dict)
    """
    if len(geodesic_points) < window_size + 2:
        return False, {"reason": "Path too short to analyze"}
    
    debug_info = {}
    
    # Compute local curvatures along path
    curvatures = []
    for i in range(1, len(geodesic_points) - 1):
        # Get vectors to neighbors
        v1 = geodesic_points[i] - geodesic_points[i-1]
        v2 = geodesic_points[i+1] - geodesic_points[i]
        
        len_v1 = np.linalg.norm(v1) + 1e-10
        len_v2 = np.linalg.norm(v2) + 1e-10
        
        # Normalize
        v1_norm = v1 / len_v1
        v2_norm = v2 / len_v2
        
        # Curvature is inversely related to dot product (1 = straight, -1 = sharp turn)
        # Convert to 0-2 scale (0 = straight, 2 = U-turn)
        curvature = 1.0 - np.dot(v1_norm, v2_norm)
        curvatures.append(curvature)
    
    curvatures = np.array(curvatures)
    
    # Compute local curvature variation in end regions (first and last window_size points)
    start_region_curvatures = curvatures[:window_size] if len(curvatures) > window_size else curvatures
    end_region_curvatures = curvatures[-window_size:] if len(curvatures) > window_size else curvatures
    
    start_variation = np.std(start_region_curvatures) / (np.mean(curvatures) + 1e-10)
    end_variation = np.std(end_region_curvatures) / (np.mean(curvatures) + 1e-10)
    
    debug_info["start_region_variation"] = float(start_variation)
    debug_info["end_region_variation"] = float(end_variation)
    debug_info["mean_curvature"] = float(np.mean(curvatures))
    debug_info["max_curvature"] = float(np.max(curvatures))
    
    # Check if either end has high oscillation
    has_oscillation = start_variation > oscillation_threshold or end_variation > oscillation_threshold
    
    if verbose and has_oscillation:
        print(f"\n  [GEODESIC OSCILLATION DETECTED]")
        print(f"    Start region variation: {start_variation:.3f}")
        print(f"    End region variation: {end_variation:.3f}")
        print(f"    Threshold: {oscillation_threshold:.3f}")
        if start_variation > oscillation_threshold:
            print(f"    ⚠ High oscillation at START of path")
        if end_variation > oscillation_threshold:
            print(f"    ⚠ High oscillation at END of path")
    
    return has_oscillation, debug_info


def project_point_to_mesh_surface(
    point: np.ndarray,
    mesh: 'pv.PolyData',
    verbose: bool = False
) -> np.ndarray:
    """
    Project a 3D point to the nearest point on mesh surface.
    
    Args:
        point: 3D point to project
        mesh: PyVista mesh
        verbose: Print debug info
    
    Returns:
        Projected point on mesh surface
    """
    # Use VTK locator to find closest surface point
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    
    closest_point = np.zeros(3)
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    distance = vtk.mutable(0.0)
    
    locator.FindClosestPoint(point, closest_point, cell_id, sub_id, distance)
    distance_val = distance.get()
    
    if verbose and distance_val > 0.1:
        print(f"    Point projection distance: {distance_val:.3f} mm")
    
    return closest_point


def extract_largest_connected_component(mesh: 'pv.PolyData', verbose: bool = True) -> 'pv.PolyData':
    """
    Extract the largest connected component from a mesh.
    
    Args:
        mesh: Input mesh (vtkPolyData)
        verbose: Print information about components
    
    Returns:
        Cleaned mesh containing only the largest connected component
    """
    if mesh.GetNumberOfCells() == 0:
        if verbose:
            print(f"  ⚠ Empty mesh, skipping component extraction")
        return mesh
    
    # Extract all connected components
    connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
    connectivity_filter.SetInputData(mesh)
    connectivity_filter.SetExtractionModeToAllRegions()
    connectivity_filter.Update()
    
    num_components = connectivity_filter.GetNumberOfExtractedRegions()
    
    if num_components <= 1:
        if verbose:
            print(f"  ✓ Single connected component (no cleaning needed)")
        return mesh
    
    if verbose:
        print(f"  Found {num_components} connected component(s), extracting largest...")
    
    # Find the largest component
    largest_component_id = 0
    largest_component_size = 0
    
    for comp_id in range(num_components):
        extractor = vtk.vtkPolyDataConnectivityFilter()
        extractor.SetInputData(mesh)
        extractor.SetExtractionModeToSpecifiedRegions()
        extractor.AddSpecifiedRegion(comp_id)
        extractor.Update()
        
        output = extractor.GetOutput()
        component_size = output.GetNumberOfCells()
        
        if verbose:
            print(f"    Component {comp_id}: {component_size} cells")
        
        if component_size > largest_component_size:
            largest_component_size = component_size
            largest_component_id = comp_id
    
    # Extract the largest component
    largest_extractor = vtk.vtkPolyDataConnectivityFilter()
    largest_extractor.SetInputData(mesh)
    largest_extractor.SetExtractionModeToSpecifiedRegions()
    largest_extractor.AddSpecifiedRegion(largest_component_id)
    largest_extractor.Update()
    
    cleaned_mesh = largest_extractor.GetOutput()
    
    if verbose:
        print(f"  ✓ Selected component {largest_component_id} (size: {cleaned_mesh.GetNumberOfCells()} cells)")
    
    return cleaned_mesh


def validate_and_correct_landmarks(
    landmarks: Dict,
    mesh: 'pv.PolyData',
    verbose: bool = True
) -> Tuple[Dict, str]:
    """
    Validate landmarks are within mesh bounds. If not, try coordinate transformations.
    
    Prioritizes (-x, -y, z) when landmarks are completely outside bounds.
    
    Args:
        landmarks: Dictionary of landmark names → [x, y, z] coordinates
        mesh: PyVista mesh to check bounds against
        verbose: Print debug info
    
    Returns:
        (corrected_landmarks_dict, transformation_applied_str)
    """
    mesh_bounds = mesh.bounds  # (x_min, x_max, y_min, y_max, z_min, z_max)
    mesh_center = [(mesh_bounds[0] + mesh_bounds[1]) / 2,
                   (mesh_bounds[2] + mesh_bounds[3]) / 2,
                   (mesh_bounds[4] + mesh_bounds[5]) / 2]
    
    if verbose:
        print(f"\n  [LANDMARK VALIDATION]")
        print(f"    Mesh bounds: x=[{mesh_bounds[0]:.1f}, {mesh_bounds[1]:.1f}], "
              f"y=[{mesh_bounds[2]:.1f}, {mesh_bounds[3]:.1f}], "
              f"z=[{mesh_bounds[4]:.1f}, {mesh_bounds[5]:.1f}]")
        print(f"    Original landmarks:")
        for name, point in landmarks.items():
            if isinstance(point, (list, tuple)) and len(point) >= 3:
                print(f"      {name}: [{point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}]")
    
    def is_inside_mesh_bounds(point, bounds, tolerance=50.0):
        """Check if point is within mesh bounds (with tolerance)."""
        x, y, z = point
        return (bounds[0] - tolerance <= x <= bounds[1] + tolerance and
                bounds[2] - tolerance <= y <= bounds[3] + tolerance and
                bounds[4] - tolerance <= z <= bounds[5] + tolerance)
    
    def count_landmarks_inside(landmarks_dict, bounds, tolerance=50.0):
        """Count how many landmarks are inside bounds."""
        count = 0
        for name, point in landmarks_dict.items():
            # Handle both lists and numpy arrays
            if (isinstance(point, (list, tuple, np.ndarray)) and 
                hasattr(point, '__len__') and len(point) >= 3):
                if is_inside_mesh_bounds(point, bounds, tolerance):
                    count += 1
        return count
    
    def check_critical_landmarks(landmarks_dict, bounds, tolerance=50.0):
        """Check specifically if critical visualization landmarks are inside bounds."""
        critical = ['Top RS', 'Bottom RS', 'Eardrum']
        inside = []
        outside = []
        for name in critical:
            if name in landmarks_dict:
                point = landmarks_dict[name]
                # Handle both lists and numpy arrays
                if (isinstance(point, (list, tuple, np.ndarray)) and 
                    hasattr(point, '__len__') and len(point) >= 3):
                    if is_inside_mesh_bounds(point, bounds, tolerance):
                        inside.append(name)
                    else:
                        outside.append(name)
        return inside, outside
    
    # Check original coordinates first
    original_score = count_landmarks_inside(landmarks, mesh_bounds, tolerance=50.0)
    original_critical_inside, original_critical_outside = check_critical_landmarks(landmarks, mesh_bounds, tolerance=50.0)
    
    if verbose:
        print(f"    Original: {original_score}/{len(landmarks)} landmarks inside bounds")
        print(f"      Critical landmarks (Top RS, Bottom RS, Eardrum):")
        print(f"        Inside: {original_critical_inside if original_critical_inside else 'none'}")
        print(f"        Outside: {original_critical_outside if original_critical_outside else 'all inside'}")
    
    # CRITICAL FIX: If ALL landmarks are outside, apply (-x, -y, z) automatically
    # This is the most common coordinate system flip and should fix nearly all cases
    if original_score == 0:
        if verbose:
            print(f"    ⚠ All landmarks outside bounds - automatically applying (-x, -y, z) transformation")
        
        auto_transform_func = lambda p: np.array([-p[0], -p[1], p[2]])
        auto_transformed = {}
        for name, point in landmarks.items():
            # Handle both lists and numpy arrays
            if (isinstance(point, (list, tuple, np.ndarray)) and 
                hasattr(point, '__len__') and len(point) >= 3):
                try:
                    transformed_point = auto_transform_func(np.array(point[:3])).tolist()
                    auto_transformed[name] = transformed_point
                except Exception as e:
                    if verbose:
                        print(f"      DEBUG ERROR transforming {name}: {e}")
                    auto_transformed[name] = point
            else:
                auto_transformed[name] = point
        
        if verbose:
            print(f"    ✓ Applied (-x, -y, z) transformation")
            print(f"      Transformed landmarks (all {len(auto_transformed)} points):")
            for name, point in auto_transformed.items():
                if isinstance(point, (list, tuple)) and len(point) >= 3:
                    print(f"        {name}: [{point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}]")
                else:
                    print(f"        {name}: {point} (NOT A VALID POINT)")
        
        return auto_transformed, "(-x, -y, z)"
    
    # If original coordinates are mostly okay, try to find the best match
    if verbose:
        print(f"    Checking other transformations to find best match...")
    
    # Try transformations in priority order
    priority_transformations = [
        ("original", lambda p: p),
        ("(-x, -y, z)", lambda p: np.array([-p[0], -p[1], p[2]])),
        ("(-x, y, z)", lambda p: np.array([-p[0], p[1], p[2]])),
        ("(x, -y, z)", lambda p: np.array([p[0], -p[1], p[2]])),
        ("(-x, -y, -z)", lambda p: np.array([-p[0], -p[1], -p[2]])),
        ("(z, y, x)", lambda p: np.array([p[2], p[1], p[0]])),
        ("(-z, y, -x)", lambda p: np.array([-p[2], p[1], -p[0]])),
        ("(y, x, z)", lambda p: np.array([p[1], p[0], p[2]])),
        ("(-y, -x, -z)", lambda p: np.array([-p[1], -p[0], -p[2]])),
    ]
    
    best_transform_name = "original"
    best_score = original_score
    best_critical_inside = original_critical_inside
    best_landmarks = None
    
    for transform_name, transform_func in priority_transformations:
        try:
            transformed = {}
            for name, point in landmarks.items():
                # Handle both lists and numpy arrays
                if (isinstance(point, (list, tuple, np.ndarray)) and 
                    hasattr(point, '__len__') and len(point) >= 3):
                    transformed[name] = transform_func(np.array(point[:3])).tolist()
                else:
                    transformed[name] = point
            
            # Score this transformation
            score = count_landmarks_inside(transformed, mesh_bounds, tolerance=50.0)
            critical_inside, critical_outside = check_critical_landmarks(transformed, mesh_bounds, tolerance=50.0)
            
            # Prioritize transformations that put ALL critical landmarks inside
            all_critical_inside = len(critical_outside) == 0 and len(critical_inside) == 3
            
            if verbose and (score > 0 or all_critical_inside):
                print(f"    {transform_name}: {score}/{len(landmarks)} total, "
                      f"critical: {len(critical_inside)}/3 inside")
                if all_critical_inside and verbose:
                    print(f"      → Top RS: {transformed.get('Top RS')}")
                    print(f"      → Bottom RS: {transformed.get('Bottom RS')}")
                    print(f"      → Eardrum: {transformed.get('Eardrum')}")
            
            # Use this transformation if it puts all critical landmarks inside
            if all_critical_inside:
                best_score = score
                best_critical_inside = critical_inside
                best_transform_name = transform_name
                best_landmarks = transformed
                if verbose:
                    print(f"      ✓ ALL CRITICAL LANDMARKS INSIDE - using this transformation")
                break  # Found the best solution, stop searching
            
            # Otherwise, update if this is better than current best
            elif score > best_score:
                best_score = score
                best_critical_inside = critical_inside
                best_transform_name = transform_name
                best_landmarks = transformed
        
        except Exception as e:
            if verbose:
                print(f"    {transform_name}: Error - {str(e)}")
            continue
    
    # If no transformation improved the score, use the original
    if best_landmarks is None:
        best_landmarks = landmarks
        best_transform_name = "original"
    
    if verbose:
        if best_transform_name != "original":
            print(f"    ✓ Applied coordinate transformation: {best_transform_name}")
            print(f"      All landmarks: {best_score}/{len(landmarks)} inside bounds")
            print(f"      Critical landmarks (Top RS, Bottom RS, Eardrum): {len(best_critical_inside)}/3 inside")
            if best_critical_inside:
                print(f"        ✓ Inside: {', '.join(best_critical_inside)}")
            print(f"      Transformed landmarks:")
            for name, point in best_landmarks.items():
                if isinstance(point, (list, tuple)) and len(point) >= 3:
                    marker = "  ✓" if name in best_critical_inside or name not in ['Top RS', 'Bottom RS', 'Eardrum'] else "  ⚠"
                    print(f"{marker}        {name}: [{point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}]")
        else:
            print(f"    ✓ Landmarks already correct (no transformation needed)")
    
    return best_landmarks, best_transform_name


def adjust_isthmus_for_large_volume(
    centerline_points: np.ndarray,
    isthmus_idx: int,
    eardrum_pt: np.ndarray,
    eardrum_normal: np.ndarray,
    eardrum_curve: 'pv.PolyData',
    mesh: 'pv.PolyData',
    non_inverted_mesh: 'pv.PolyData',
    inverted_image,
    current_volume: float,
    sample_name: str,
    output_dir: str,
    max_volume_threshold: float = 1500.0,
    volume_drop_percentage: float = 20.0,
    verbose: bool = True
) -> Tuple[int, float, Optional[Tuple]]:
    """
    Adjust the isthmus point along the centerline until bounding box doesn't touch mask edges.
    
    Uses bounding box volume as a fast proxy instead of computing actual mesh volume.
    CRITICAL: Ensures bounding box does NOT touch the physical dimensions of the mask 3D image.
    
    Moves isthmus point FORWARD (toward eardrum) incrementally until:
    1. Bounding box doesn't touch the physical edges of the mask image
    
    Args:
        centerline_points: Nx3 array of centerline points
        isthmus_idx: Current index of isthmus in centerline
        eardrum_pt: Eardrum point position
        eardrum_normal: Eardrum plane normal
        eardrum_curve: Eardrum cross-section polydata
        mesh: Simplified mesh for cross-section extraction
        non_inverted_mesh: Non-inverted mesh for volume computation
        inverted_image: SimpleITK image - used to get physical bounds of the mask
        current_volume: Initial volume (for reference)
        sample_name: Sample identifier
        output_dir: Output directory
        max_volume_threshold: Volume threshold (not used with bbox proxy)
        volume_drop_percentage: Volume drop percentage (not used with bbox proxy)
        verbose: Print progress messages
    
    Returns:
        (best_isthmus_idx, best_bbox_volume, best_cross_section_data)
    """
    print(f"\n[VOLUME ADJUSTMENT] Optimizing isthmus using bounding box proxy (initial volume: {current_volume:.2f} mm³)")
    
    best_isthmus_idx = isthmus_idx
    best_bbox_volume = current_volume
    best_cross_section_data = None
    
    # Get physical bounds of the mask image (in mm)
    # This is the 3D extent of the image volume, regardless of mask values
    origin = np.array(inverted_image.GetOrigin())
    size = np.array(inverted_image.GetSize())
    spacing = np.array(inverted_image.GetSpacing())
    
    # Compute physical bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
    mask_physical_bounds = np.array([
        origin[0],                           # x_min
        origin[0] + (size[0] - 1) * spacing[0],  # x_max
        origin[1],                           # y_min
        origin[1] + (size[1] - 1) * spacing[1],  # y_max
        origin[2],                           # z_min
        origin[2] + (size[2] - 1) * spacing[2],  # z_max
    ])
    
    if verbose:
        print(f"  Original isthmus index: {isthmus_idx}")
        print(f"  Original volume: {current_volume:.2f} mm³")
        print(f"  Mask physical bounds (from image dimensions):")
        print(f"    x: [{mask_physical_bounds[0]:.1f}, {mask_physical_bounds[1]:.1f}] mm")
        print(f"    y: [{mask_physical_bounds[2]:.1f}, {mask_physical_bounds[3]:.1f}] mm")
        print(f"    z: [{mask_physical_bounds[4]:.1f}, {mask_physical_bounds[5]:.1f}] mm")
    
    # Apply 10% margin - the mesh bbox should not extend into the outer 10% of the image
    margin_fraction = 0.10
    inner_bounds = np.zeros(6)
    for i in range(3):
        extent = mask_physical_bounds[2*i + 1] - mask_physical_bounds[2*i]
        margin = extent * margin_fraction
        inner_bounds[2*i] = mask_physical_bounds[2*i] + margin      # min + margin
        inner_bounds[2*i + 1] = mask_physical_bounds[2*i + 1] - margin  # max - margin
    
    if verbose:
        print(f"  Inner bounds (10% margin):")
        print(f"    x: [{inner_bounds[0]:.1f}, {inner_bounds[1]:.1f}] mm")
        print(f"    y: [{inner_bounds[2]:.1f}, {inner_bounds[3]:.1f}] mm")
        print(f"    z: [{inner_bounds[4]:.1f}, {inner_bounds[5]:.1f}] mm")
    
    def check_bbox_touches_mask_edges(bbox):
        """Check if bbox extends beyond the inner bounds (10% margin from edges)."""
        touching_edges = []
        for i, dim_name in enumerate(['x', 'y', 'z']):
            # Check min edge - bbox min should be >= inner_bounds min
            if bbox[2*i] < inner_bounds[2*i]:
                touching_edges.append(f"{dim_name}_min (bbox: {bbox[2*i]:.1f}, limit: {inner_bounds[2*i]:.1f})")
            # Check max edge - bbox max should be <= inner_bounds max
            if bbox[2*i + 1] > inner_bounds[2*i + 1]:
                touching_edges.append(f"{dim_name}_max (bbox: {bbox[2*i + 1]:.1f}, limit: {inner_bounds[2*i + 1]:.1f})")
        return len(touching_edges) > 0, touching_edges
    
    def compute_bbox_volume(bbox):
        """Compute bounding box volume in mm³."""
        return (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) * (bbox[5] - bbox[4])
    
    # First, extract the initial segment and check if it touches edges
    from .utils import extract_isthmus_eardrum_segment, extract_cross_section_at_point
    
    # Get initial isthmus data
    initial_isthmus_pt = centerline_points[isthmus_idx]
    if isthmus_idx < len(centerline_points) - 1:
        initial_isthmus_normal = centerline_points[isthmus_idx + 1] - centerline_points[isthmus_idx]
        initial_isthmus_normal = initial_isthmus_normal / (np.linalg.norm(initial_isthmus_normal) + 1e-10)
    else:
        initial_isthmus_normal = eardrum_normal
    
    # Extract cross-section at initial isthmus position
    initial_isthmus_curve, _, _, _ = extract_cross_section_at_point(
        mesh=mesh,
        centerline_points=centerline_points,
        point_index=isthmus_idx,
        override_point=initial_isthmus_pt,
        override_tangent=initial_isthmus_normal
    )
    
    # Get initial segment mesh
    initial_clipped_mesh, _ = extract_isthmus_eardrum_segment(
        mesh=non_inverted_mesh,
        isthmus_point=np.array(initial_isthmus_pt),
        isthmus_normal=np.array(initial_isthmus_normal),
        eardrum_point=np.array(eardrum_pt),
        eardrum_normal=np.array(eardrum_normal),
        isthmus_curve=initial_isthmus_curve,
        eardrum_curve=eardrum_curve,
        centerline_points=centerline_points,
        isthmus_idx=isthmus_idx,
        eardrum_idx=None,
        second_bend_point=None,
        output_dir=output_dir,
        sample_name=sample_name,
        visualize=False
    )
    
    if initial_clipped_mesh is None or initial_clipped_mesh.GetNumberOfCells() == 0:
        if verbose:
            print(f"  ⚠ Initial segment mesh is empty, returning original values")
        return best_isthmus_idx, best_bbox_volume, best_cross_section_data
    
    # Check initial bounding box - handle both VTK and PyVista objects
    if hasattr(initial_clipped_mesh, 'bounds'):
        initial_bbox = initial_clipped_mesh.bounds
    else:
        initial_bbox = initial_clipped_mesh.GetBounds()
    initial_bbox_vol = compute_bbox_volume(initial_bbox)
    touches_edge, touching_edges = check_bbox_touches_mask_edges(initial_bbox)
    
    if verbose:
        print(f"\n  Initial segment bounding box:")
        print(f"    x: [{initial_bbox[0]:.1f}, {initial_bbox[1]:.1f}] mm")
        print(f"    y: [{initial_bbox[2]:.1f}, {initial_bbox[3]:.1f}] mm")
        print(f"    z: [{initial_bbox[4]:.1f}, {initial_bbox[5]:.1f}] mm")
        print(f"    Bbox volume: {initial_bbox_vol:.0f} mm³")
        if touches_edge:
            print(f"    ⚠ TOUCHES MASK EDGES: {', '.join(touching_edges)}")
        else:
            print(f"    ✓ Does not touch mask edges")
    
    if not touches_edge:
        # Already good, no adjustment needed
        if verbose:
            print(f"\n  ✓ No adjustment needed - bounding box does not touch mask edges")
        return isthmus_idx, initial_bbox_vol, None
    
    # Move FORWARD along centerline until bounding box doesn't touch edges
    if verbose:
        print(f"\n  Moving FORWARD along centerline to separate from mask edges...")
    
    max_steps = min(len(centerline_points) - isthmus_idx - 5, 50)  # Don't get within 5 points of end
    
    for step in range(1, max_steps + 1):
        forward_idx = isthmus_idx + step
        
        if forward_idx >= len(centerline_points) - 5:
            if verbose:
                print(f"  Step {step}: Reached safety limit (too close to centerline end)")
            break
        
        try:
            # Get new isthmus point from centerline
            new_isthmus_pt = centerline_points[forward_idx]
            
            # Compute normal at this point
            if forward_idx < len(centerline_points) - 1:
                new_isthmus_normal = centerline_points[forward_idx + 1] - centerline_points[forward_idx]
                new_isthmus_normal = new_isthmus_normal / (np.linalg.norm(new_isthmus_normal) + 1e-10)
            else:
                new_isthmus_normal = eardrum_normal
            
            # Extract cross-section at new position
            new_isthmus_curve, _, _, _ = extract_cross_section_at_point(
                mesh=mesh,
                centerline_points=centerline_points,
                point_index=forward_idx,
                override_point=new_isthmus_pt,
                override_tangent=new_isthmus_normal
            )
            
            if new_isthmus_curve is None or new_isthmus_curve.GetNumberOfCells() == 0:
                if verbose:
                    print(f"  Step {step}: Could not extract cross-section at idx {forward_idx}")
                continue
            
            # Extract segment between new isthmus and eardrum
            volume_clipped_mesh, _ = extract_isthmus_eardrum_segment(
                mesh=non_inverted_mesh,
                isthmus_point=np.array(new_isthmus_pt),
                isthmus_normal=np.array(new_isthmus_normal),
                eardrum_point=np.array(eardrum_pt),
                eardrum_normal=np.array(eardrum_normal),
                isthmus_curve=new_isthmus_curve,
                eardrum_curve=eardrum_curve,
                centerline_points=centerline_points,
                isthmus_idx=forward_idx,
                eardrum_idx=None,
                second_bend_point=None,
                output_dir=output_dir,
                sample_name=sample_name,
                visualize=False
            )
            
            if volume_clipped_mesh is None or volume_clipped_mesh.GetNumberOfCells() == 0:
                if verbose:
                    print(f"  Step {step}: Empty mesh segment at idx {forward_idx}")
                continue
            
            # Check bounding box against mask physical edges - handle both VTK and PyVista objects
            if hasattr(volume_clipped_mesh, 'bounds'):
                clipped_bbox = volume_clipped_mesh.bounds
            else:
                clipped_bbox = volume_clipped_mesh.GetBounds()
            bbox_vol = compute_bbox_volume(clipped_bbox)
            touches_edge_now, touching_edges_now = check_bbox_touches_mask_edges(clipped_bbox)
            
            if verbose:
                status = f"TOUCHES: {', '.join(touching_edges_now)}" if touches_edge_now else "✓ CLEAR"
                print(f"  Step {step}: idx={forward_idx}, bbox_vol={bbox_vol:.0f} mm³, {status}")
            
            # If bounding box no longer touches mask edges, we found our position
            if not touches_edge_now:
                best_isthmus_idx = forward_idx
                best_bbox_volume = bbox_vol
                best_cross_section_data = {
                    'point': new_isthmus_pt.tolist(),
                    'tangent': new_isthmus_normal.tolist(),
                    'cross_section': new_isthmus_curve,
                    'index': forward_idx
                }
                
                if verbose:
                    print(f"\n  ✓ Found valid position: idx={forward_idx}, bbox_vol={bbox_vol:.0f} mm³")
                    print(f"    Bounding box does not touch mask physical edges")
                break
            
        except Exception as e:
            if verbose:
                print(f"  Step {step}: Error at idx {forward_idx}: {str(e)}")
            continue
    
    if verbose:
        print(f"\n  ✓ Optimization complete: idx={best_isthmus_idx}, bbox_vol={best_bbox_volume:.0f} mm³")
        if best_isthmus_idx != isthmus_idx:
            print(f"    Moved {best_isthmus_idx - isthmus_idx} steps forward")
    
    return best_isthmus_idx, best_bbox_volume, best_cross_section_data


def should_rerun_due_to_volume(
    json_path: str,
    volume_hard_threshold_lower: float = 100.0,
    volume_hard_threshold_upper: float = 1500.0,
    volume_soft_threshold_lower: float = 100.0,
    volume_soft_threshold_upper: float = 1500.0,
    volume_tolerance_percent: float = 15.0
) -> Tuple[bool, str]:
    """
    Check if a file should be reprocessed based on volume values.
    
    Returns (should_rerun, reason_string)
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check volumes in metrics section
        metrics = data.get('metrics', {})
        volumes = data.get('volumes', {})
        
        # Get volume values (check both locations)
        isthmus_eardrum_vol = metrics.get('isthmus_eardrum_volume') or volumes.get('isthmus_eardrum_volume')
        volume_hard = metrics.get('volume_hard') or volumes.get('volume_hard')
        volume_soft = metrics.get('volume_soft') or volumes.get('volume_soft')
        
        reasons = []
        
        # Check 1: isthmus_eardrum > 1500
        if isthmus_eardrum_vol is not None:
            if isthmus_eardrum_vol > 1500:
                reasons.append(f"isthmus_eardrum_volume ({isthmus_eardrum_vol:.2f}) > 1500")
        
        # Check 2: volume_hard + volume_soft not approximately equal to isthmus_eardrum
        if volume_hard is not None and volume_soft is not None and isthmus_eardrum_vol is not None:
            combined_volume = volume_hard + volume_soft
            volume_diff_percent = abs(combined_volume - isthmus_eardrum_vol) / isthmus_eardrum_vol * 100 if isthmus_eardrum_vol > 0 else 0
            
            if volume_diff_percent > volume_tolerance_percent:
                reasons.append(
                    f"volume_hard ({volume_hard:.2f}) + volume_soft ({volume_soft:.2f}) = {combined_volume:.2f} "
                    f"differs from isthmus_eardrum ({isthmus_eardrum_vol:.2f}) by {volume_diff_percent:.1f}% (threshold: {volume_tolerance_percent}%)"
                )
        
        if reasons:
            return True, "\n  ".join(reasons)
        else:
            return False, "Volume values are within acceptable range"
    
    except Exception as e:
        return False, f"Could not parse JSON: {str(e)}"


def process_single_sample(
    sample_name: str,
    mask_path: str,
    landmarks_path: str,
    stl_path: str,
    output_dir: str,
    cbj_landmarks_path: str = None,
    enable_viz: bool = False,
    rerun_big_volumes: bool = False
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Process a single sample through the full pipeline.
    
    Args:
        sample_name: Sample identifier (e.g., "CHUM-001_left_ear")
        mask_path: Path to NIfTI mask file
        landmarks_path: Path to landmarks JSON file
        output_dir: Output directory for results
        cbj_landmarks_path: Optional path to CBJ landmarks
        enable_viz: Enable interactive visualizations
        rerun_big_volumes: If True, reprocess files with anomalous volume values
    
    Returns:
        dict: Computed metrics or None if processing failed
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {sample_name}")
    print(f"{'='*70}\n")

    
    # Define output paths
    output_stl = os.path.join(output_dir, "stl", f"{sample_name}_outside.stl")
    output_inverted_mask = os.path.join(output_dir, "masks", f"{sample_name}_inverted.nii.gz")
    output_centerline = os.path.join(output_dir, "vtk", f"{sample_name}_centerline.vtk")
    output_centerline_raw = os.path.join(output_dir, "vtk", f"{sample_name}_centerline_raw.vtk")
    output_centerline_refined = os.path.join(output_dir, "vtk", f"{sample_name}_centerline_refined.vtk")
    output_centerline_refined_raw = os.path.join(output_dir, "vtk", f"{sample_name}_centerline_refined_raw.vtk")
    output_centerline_features = os.path.join(output_dir, "vtk", f"{sample_name}_centerline_features.vtk")
    output_geodesic = os.path.join(output_dir, "vtk", f"{sample_name}_geodesic.vtk")
    output_geodesic_raw = os.path.join(output_dir, "vtk", f"{sample_name}_geodesic_raw.vtk")
    output_landmarks = os.path.join(output_dir, "markups", f"{sample_name}.json")
    
    # Check if this sample has already been processed
    if os.path.exists(output_landmarks):
        # If rerun_big_volumes is enabled, check if we should reprocess
        should_rerun = False
        rerun_reason = ""
        
        if rerun_big_volumes:
            should_rerun, rerun_reason = should_rerun_due_to_volume(output_landmarks)
            if should_rerun:
                print(f"[RERUN] Volume-based reprocessing triggered:")
                print(f"  {rerun_reason}")
                print(f"  Reprocessing {sample_name}...\n")
            else:
                print(f"[SKIP] Markup file already exists and volume values are acceptable: {output_landmarks}")
                print(f"[SKIP] Skipping processing for {sample_name}")
                return None, "Already processed"
        else:
            print(f"[SKIP] Markup file already exists: {output_landmarks}")
            print(f"[SKIP] Skipping processing for {sample_name}")
            return None, "Already processed"
    
    try:
        # =====================================================================
        # STEP 1: Load and invert mask
        # =====================================================================
        print(f"[STEP 1] Loading and inverting mask")
        
        # Load original mask (for dynamic plane offset computation)
        original_mask = load_mask(mask_path)
        
        # Load and invert mask for mesh creation
        inverted_image = load_and_invert_mask(mask_path)
        
        # Save inverted mask
        print(f"\n  Saving inverted mask...")
        save_inverted_mask(inverted_image, output_inverted_mask)
        
        # =====================================================================
        # STEP 2: Create mesh from inverted mask
        # =====================================================================
        print(f"\n[STEP 2] Creating 3D mesh from inverted mask")
        mesh = create_mesh_from_mask(inverted_image)
        
        # Extract largest connected component before saving
        print(f"\n  Extracting largest connected component...")
        mesh = pv.wrap(extract_largest_connected_component(mesh, verbose=True))
        
        # Save mesh as STL
        print(f"\n  Saving mesh as STL...")
        save_mesh_as_stl(mesh, output_stl)
        
        # =====================================================================
        # STEP 3: Load landmarks
        # =====================================================================
        print(f"\n[STEP 3] Loading landmarks from JSON")
        landmarks = load_landmarks_from_json(landmarks_path)
        
        # Validate and correct landmarks if they're not in the same coordinate system as mesh
        landmarks, transform_applied = validate_and_correct_landmarks(landmarks, mesh, verbose=True)
        
        # Extract required landmarks
        try:
            top_rs = landmarks['Top RS']
            bottom_rs = landmarks['Bottom RS']
            eardrum = landmarks['Eardrum']
        except KeyError as e:
            print(f"  ✗ Missing required landmark: {e}")
            print(f"  Available landmarks: {list(landmarks.keys())}")
            return None, f"Missing required landmark: {str(e)}"
        
        if transform_applied != "original":
            print(f"  ✓ Using transformed landmarks: {transform_applied}")
            print(f"    Top RS: {top_rs}")
            print(f"    Bottom RS: {bottom_rs}")
            print(f"    Eardrum: {eardrum}")
        
        # =====================================================================
        # STEP 3b: Load CBJ landmarks (optional, early loading)
        # =====================================================================
        print(f"\n[STEP 3b] Loading CBJ landmarks (optional)")
        cbj_points = None
        cbj_loaded = False
        if cbj_landmarks_path is not None:
            if os.path.exists(cbj_landmarks_path):
                cbj_points = load_cbj_landmarks_from_json(cbj_landmarks_path)
                if cbj_points is not None and len(cbj_points) > 0:
                    print(f"  ✓ Successfully loaded {len(cbj_points)} CBJ landmark points from:")
                    print(f"    {cbj_landmarks_path}")
                    
                    # CRITICAL: If main landmarks were transformed, apply the same transformation to CBJ points
                    # Since CBJ landmarks are now in the same unified JSON file, they have the same coordinate
                    # system issues as the main landmarks and need the same correction
                    if transform_applied != "original":
                        print(f"  → Applying same coordinate transformation to CBJ landmarks: {transform_applied}")
                        print(f"    Original CBJ points:")
                        for i, p in enumerate(cbj_points):
                            print(f"      CBJ{i+1}: [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]")
                        
                        # Apply the transformation that was used for main landmarks
                        if transform_applied == "(-x, -y, z)":
                            cbj_points = np.array([[-p[0], -p[1], p[2]] for p in cbj_points])
                        elif transform_applied == "(-x, y, z)":
                            cbj_points = np.array([[-p[0], p[1], p[2]] for p in cbj_points])
                        elif transform_applied == "(x, -y, z)":
                            cbj_points = np.array([[p[0], -p[1], p[2]] for p in cbj_points])
                        elif transform_applied == "(-x, -y, -z)":
                            cbj_points = np.array([[-p[0], -p[1], -p[2]] for p in cbj_points])
                        elif transform_applied == "(z, y, x)":
                            cbj_points = np.array([[p[2], p[1], p[0]] for p in cbj_points])
                        elif transform_applied == "(-z, y, -x)":
                            cbj_points = np.array([[-p[2], p[1], -p[0]] for p in cbj_points])
                        elif transform_applied == "(y, x, z)":
                            cbj_points = np.array([[p[1], p[0], p[2]] for p in cbj_points])
                        elif transform_applied == "(-y, -x, -z)":
                            cbj_points = np.array([[-p[1], -p[0], -p[2]] for p in cbj_points])
                        
                        print(f"    Transformed CBJ points:")
                        for i, p in enumerate(cbj_points):
                            print(f"      CBJ{i+1}: [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]")
                    
                    # Verify CBJ points are inside bounds
                    cbj_inside_bounds = sum(1 for p in cbj_points 
                                           if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 3 and
                                           mesh.bounds[0] <= p[0] <= mesh.bounds[1] and
                                           mesh.bounds[2] <= p[1] <= mesh.bounds[3] and
                                           mesh.bounds[4] <= p[2] <= mesh.bounds[5])
                    
                    if cbj_inside_bounds == len(cbj_points):
                        print(f"  ✓ CBJ points in correct coordinate system ({cbj_inside_bounds}/{len(cbj_points)} inside bounds)")
                        cbj_loaded = True
                    else:
                        print(f"  ⚠ {len(cbj_points) - cbj_inside_bounds}/{len(cbj_points)} CBJ points outside bounds")
                        if cbj_inside_bounds > 0:
                            # Some points inside, continue with warning
                            print(f"    Continuing with {cbj_inside_bounds} valid CBJ points")
                            cbj_loaded = True
                        else:
                            # All points outside, don't use them
                            print(f"    All CBJ points outside bounds - CBJ processing will be skipped")
                            cbj_points = None
                            cbj_loaded = False
                else:
                    print(f"  ⚠ CBJ landmarks file exists but contains no valid points")
                    print(f"    File: {cbj_landmarks_path}")
            else:
                print(f"  ✗ CBJ landmarks file not found:")
                print(f"    Expected: {cbj_landmarks_path}")
        else:
            print(f"  (No CBJ landmarks path provided - skipping CBJ loading)")
        
        # =====================================================================
        # STEP 4: Compute middle RS
        # =====================================================================
        print(f"\n[STEP 4] Computing middle RS point")
        middle_rs = compute_middle_rs(top_rs, bottom_rs)
        
        # =====================================================================
        # STEP 5: Visualize mesh with landmarks (optional)
        # =====================================================================
        if enable_viz:
            cbj_status = " [WITH CBJ]" if cbj_loaded else ""
            print(f"\n[STEP 5] Visualizing mesh with landmarks{cbj_status}")
            visualize_mesh_with_landmarks(
                mesh, middle_rs, eardrum, top_rs, bottom_rs,
                cbj_points=cbj_points,
                title=f"{sample_name} - Mesh with Landmarks{cbj_status}"
            )
        else:
            print(f"\n[STEP 5] Skipping visualization (enable_viz=False)")
        
        # =====================================================================
        # STEP 6: Simplify mesh for VMTK
        # =====================================================================
        print(f"\n[STEP 6] Simplifying mesh for VMTK")
        mesh_simplified = simplify_mesh_for_vmtk(mesh, target_reduction=0.95)
        
        # =====================================================================
        # STEP 7: Compute geodesic path (Top RS → Bottom RS)
        # =====================================================================
        print(f"\n[STEP 7] Computing geodesic path (Top RS → Bottom RS)")
        geodesic_path_raw, geodesic_points_raw = compute_geodesic_raw(
            mesh_simplified, top_rs, bottom_rs
        )
        
        # Save raw geodesic
        save_vtk_polydata(geodesic_path_raw, output_geodesic_raw)
        print(f"  ✓ Raw geodesic saved to: {output_geodesic_raw}")
        
        # Resample geodesic for final output
        geodesic_path, geodesic_points = compute_geodesic_path(
            mesh_simplified, top_rs, bottom_rs, resample=True, num_points=100
        )
        
        # Save resampled geodesic
        save_vtk_polydata(geodesic_path, output_geodesic)
        print(f"  ✓ Resampled geodesic saved to: {output_geodesic}")
        
        # Check for oscillation in geodesic path
        has_oscillation, osc_debug = detect_geodesic_oscillation(
            geodesic_points, window_size=5, oscillation_threshold=0.15, verbose=True
        )
        
        if has_oscillation:
            print(f"\n  [GEODESIC CORRECTION] Recomputing with mesh-projected landmarks...")
            
            # Project Top RS and Bottom RS onto mesh surface
            top_rs_projected = project_point_to_mesh_surface(top_rs, mesh_simplified, verbose=True)
            bottom_rs_projected = project_point_to_mesh_surface(bottom_rs, mesh_simplified, verbose=True)
            
            print(f"    Top RS: {top_rs} → {top_rs_projected}")
            print(f"    Bottom RS: {bottom_rs} → {bottom_rs_projected}")
            
            # Recompute geodesic with projected landmarks
            try:
                geodesic_path_corrected, geodesic_points_corrected = compute_geodesic_path(
                    mesh_simplified, top_rs_projected, bottom_rs_projected, resample=True, num_points=100
                )
                
                # Check if corrected path has less oscillation
                has_oscillation_corrected, osc_debug_corrected = detect_geodesic_oscillation(
                    geodesic_points_corrected, window_size=5, oscillation_threshold=0.15, verbose=False
                )
                
                if not has_oscillation_corrected or (osc_debug_corrected["max_curvature"] < osc_debug["max_curvature"]):
                    print(f"    ✓ Corrected path has smoother oscillation profile")
                    print(f"      Before: max_curvature={osc_debug['max_curvature']:.3f}")
                    print(f"      After:  max_curvature={osc_debug_corrected['max_curvature']:.3f}")
                    
                    # Use corrected geodesic
                    geodesic_path = geodesic_path_corrected
                    geodesic_points = geodesic_points_corrected
                    geodesic_path_raw = geodesic_path_corrected
                    geodesic_points_raw = geodesic_points_corrected
                    
                    # Save corrected geodesic
                    save_vtk_polydata(geodesic_path, output_geodesic)
                    save_vtk_polydata(geodesic_path_raw, output_geodesic_raw)
                    print(f"    ✓ Saved corrected geodesic")
                else:
                    print(f"    ⚠ Corrected path not better, keeping original")
            except Exception as e:
                print(f"    ⚠ Could not recompute corrected geodesic: {e}")
                print(f"    Keeping original geodesic")
        
        # =====================================================================
        # STEP 8: Extract centerline with plane-based trimming
        # =====================================================================
        print(f"\n[STEP 8] Extracting centerline (Middle RS → Eardrum)")
        
        # Track start point (may change if validation fails)
        current_start_point = middle_rs
        start_point_source = "Middle RS"
        centerline_valid = False
        retry_with_geodesic_furthest = False
        
        # Debug info for visualization on failure
        debug_info = {
            'raw_centerline_points': None,
            'extended_seed': None,
            'vmtk_seed_on_surface': None,
            'geodesic_furthest_point': None,
            'plane_normal': None,
            'plane_centroid': None,
            'failure_reason': None
        }
        
        # First attempt with Middle RS
        try:
            centerline_result = compute_centerline_vmtk(
                mesh_simplified, current_start_point, eardrum,
                geodesic_points=geodesic_points_raw,
                reference_point=middle_rs,
                original_mask=original_mask
            )
            
            if centerline_result is None:
                debug_info['failure_reason'] = "Centerline computation returned None (first attempt)"
                raise RuntimeError("Centerline computation returned None")
            
            # Store debug info
            debug_info['raw_centerline_points'] = centerline_result.get('raw_centerline_points')
            debug_info['extended_seed'] = centerline_result.get('extended_seed')
            debug_info['vmtk_seed_on_surface'] = centerline_result.get('vmtk_seed_on_surface')
            debug_info['plane_normal'] = centerline_result.get('plane_normal')
            debug_info['plane_centroid'] = centerline_result.get('plane_centroid')
            
            # Validate centerline
            raw_centerline_points = centerline_result['raw_centerline_points']
            
            print(f"\n  [VALIDATION] Checking centerline quality...")
            is_valid, validation_msg = validate_centerline(
                raw_centerline_points, current_start_point, eardrum
            )
            
            if is_valid:
                centerline_valid = True
                print(f"  ✓ Centerline validation PASSED: {validation_msg}")
            else:
                print(f"  ⚠ Centerline validation FAILED: {validation_msg}")
                debug_info['failure_reason'] = f"Validation failed (first attempt): {validation_msg}"
                retry_with_geodesic_furthest = True
                
        except Exception as e:
            print(f"  ⚠ Centerline computation failed: {e}")
            if debug_info['failure_reason'] is None:
                debug_info['failure_reason'] = f"Exception (first attempt): {e}"
            retry_with_geodesic_furthest = True
        
        # Retry with geodesic furthest point if needed
        if retry_with_geodesic_furthest:
            print(f"\n  [RETRY] Attempting with geodesic furthest point from eardrum...")
            
            try:
                geodesic_furthest_point, geodesic_dist = find_geodesic_furthest_point(
                    mesh_simplified, eardrum
                )
                
                debug_info['geodesic_furthest_point'] = geodesic_furthest_point
                
                current_start_point = geodesic_furthest_point
                start_point_source = "Geodesic Furthest Point"
                
                print(f"    Using new start point: {current_start_point}")
                print(f"    Geodesic distance from eardrum: {geodesic_dist:.2f} mm")
                
                centerline_result = compute_centerline_vmtk(
                    mesh_simplified, current_start_point, eardrum,
                    geodesic_points=geodesic_points_raw,
                    reference_point=middle_rs,
                    original_mask=original_mask
                )
                
                if centerline_result is None:
                    debug_info['failure_reason'] = "Centerline computation returned None (retry)"
                    raise RuntimeError("Centerline computation returned None on retry")
                
                # Update debug info with retry results
                debug_info['raw_centerline_points'] = centerline_result.get('raw_centerline_points')
                debug_info['extended_seed'] = centerline_result.get('extended_seed')
                debug_info['vmtk_seed_on_surface'] = centerline_result.get('vmtk_seed_on_surface')
                debug_info['plane_normal'] = centerline_result.get('plane_normal')
                debug_info['plane_centroid'] = centerline_result.get('plane_centroid')
                
                raw_centerline_points = centerline_result['raw_centerline_points']
                
                print(f"\n  [VALIDATION] Checking centerline quality (retry)...")
                is_valid, validation_msg = validate_centerline(
                    raw_centerline_points, current_start_point, eardrum
                )
                
                if is_valid:
                    centerline_valid = True
                    print(f"  ✓ Centerline validation PASSED: {validation_msg}")
                else:
                    print(f"  ⚠ Centerline validation FAILED on retry: {validation_msg}")
                    debug_info['failure_reason'] = f"Validation failed (retry): {validation_msg}"
                    
            except Exception as retry_error:
                print(f"  ⚠ Retry failed: {retry_error}")
                if debug_info['failure_reason'] is None or "retry" not in debug_info['failure_reason']:
                    debug_info['failure_reason'] = f"Exception (retry): {retry_error}"
        
        # Check if centerline is valid
        if not centerline_valid:
            warning_msg = f"Centerline for {sample_name} failed validation after retry"
            warnings.warn(warning_msg)
            print(f"\n  {'!'*60}")
            print(f"  {warning_msg}")
            print(f"  {'!'*60}\n")
            
            # Show debug visualization if viz is enabled
            if enable_viz:
                print(f"\n  [DEBUG] Opening detailed visualization to diagnose failure...")
                visualize_debug_failure(
                    mesh=mesh,
                    middle_rs=middle_rs,
                    eardrum=eardrum,
                    top_rs=top_rs,
                    bottom_rs=bottom_rs,
                    geodesic_path=geodesic_path_raw,
                    raw_centerline_points=debug_info['raw_centerline_points'],
                    extended_seed=debug_info['extended_seed'],
                    vmtk_seed_on_surface=debug_info['vmtk_seed_on_surface'],
                    geodesic_furthest_point=debug_info['geodesic_furthest_point'],
                    plane_normal=debug_info['plane_normal'],
                    plane_centroid=debug_info['plane_centroid'],
                    failure_reason=debug_info['failure_reason'] or "Unknown failure",
                    title=f"DEBUG: {sample_name}"
                )
            
            return None, debug_info.get('failure_reason', 'Centerline computation failed')
        
        # Extract centerline data
        centerline = centerline_result['centerline']
        original_start = centerline_result['original_start']
        trimmed_start = centerline_result['trimmed_start']
        end_point = centerline_result['end_point']
        beginning_eac = centerline_result.get('beginning_eac')
        beginning_centerline = centerline_result.get('beginning_centerline')
        plane_normal = centerline_result.get('plane_normal')
        plane_centroid = centerline_result.get('plane_centroid')
        offset_plane_centroid = centerline_result.get('offset_plane_centroid')
        mask_boundary_point = centerline_result.get('mask_boundary_point')
        
        # =====================================================================
        # STEP 8b: Refine centerline endpoint
        # =====================================================================
        print(f"\n[STEP 8b] Refining centerline endpoint using median normal")
        
        # Refine the endpoint using median of last 10 normals
        refined_endpoint, _ = refine_centerline_endpoint(
            mesh=mesh_simplified,
            raw_centerline_points=raw_centerline_points,
            original_endpoint=end_point,
            num_normals=10
        )
        
        # Store original endpoint (from landmark)
        original_eardrum_point = end_point.copy()
        
        max_refinement_distance = 5.0  # mm - maximum allowed refinement distance
        
        if refined_endpoint is not None:
            refinement_distance = np.linalg.norm(refined_endpoint - end_point)
            if refinement_distance > 0.1:
                if refinement_distance > max_refinement_distance:
                    print(f"  ⚠ Endpoint refined too far ({refinement_distance:.2f} mm > {max_refinement_distance:.2f} mm threshold)")
                    print(f"    Rejecting refined endpoint and using original eardrum point")
                    refined_endpoint = None
                    end_point = original_eardrum_point
                else:
                    print(f"  ✓ Endpoint refined (moved {refinement_distance:.2f} mm)")
                    end_point = refined_endpoint
            else:
                print(f"  ⚠ Refinement resulted in same endpoint, using original")
                refined_endpoint = None
        
        # Recompute centerline using the refined endpoint
        print(f"\n[STEP 8c] Recomputing centerline with refined endpoint")
        
        try:
            centerline_refined_result = compute_centerline_vmtk(
                mesh_simplified, current_start_point, end_point,
                geodesic_points=geodesic_points_raw,
                reference_point=middle_rs,
                original_mask=original_mask
            )
            
            if centerline_refined_result is None:
                print(f"  ⚠ Refined centerline computation failed, using original centerline")
                centerline_refined = centerline
                refined_centerline_points = raw_centerline_points
            else:
                # Extract refined centerline data
                centerline_refined = centerline_refined_result['centerline']
                refined_centerline_points = centerline_refined_result['raw_centerline_points']
                
                # Validate refined centerline
                is_valid_refined, validation_msg_refined = validate_centerline(
                    refined_centerline_points, current_start_point, end_point
                )
                
                # Compute centerline lengths to detect collapse
                original_length = np.sum([
                    np.linalg.norm(raw_centerline_points[i+1] - raw_centerline_points[i])
                    for i in range(len(raw_centerline_points) - 1)
                ])
                
                refined_length = np.sum([
                    np.linalg.norm(refined_centerline_points[i+1] - refined_centerline_points[i])
                    for i in range(len(refined_centerline_points) - 1)
                ])
                
                length_ratio = refined_length / original_length if original_length > 0 else 0.0
                
                print(f"    Original centerline: {len(raw_centerline_points)} points, {original_length:.2f} mm")
                print(f"    Refined centerline: {len(refined_centerline_points)} points, {refined_length:.2f} mm")
                print(f"    Length ratio (refined/original): {length_ratio:.2%}")
                
                # Check for centerline collapse (refined < 50% of original)
                if length_ratio < 0.5:
                    print(f"  ✗ REJECTING refined centerline: collapsed to {length_ratio:.2%} of original length")
                    print(f"    Using original centerline instead")
                    centerline_refined = centerline
                    refined_centerline_points = raw_centerline_points
                    is_valid_refined = True  # Use original validation result
                elif is_valid_refined:
                    print(f"  ✓ Refined centerline validation PASSED")
                else:
                    print(f"  ⚠ Refined centerline validation FAILED: {validation_msg_refined}")
                    print(f"    Keeping refined centerline anyway (proceeding with caution)")
                
                # Update other endpoints from refined result
                trimmed_start_refined = centerline_refined_result['trimmed_start']
                beginning_eac_refined = centerline_refined_result.get('beginning_eac')
                beginning_centerline_refined = centerline_refined_result.get('beginning_centerline')
                mask_boundary_point_refined = centerline_refined_result.get('mask_boundary_point')
                
                print(f"    Refined centerline endpoints:")
                print(f"      Refined trimmed start: {trimmed_start_refined}")
                print(f"      Refined end point: {end_point}")
                if beginning_eac_refined is not None:
                    print(f"      Refined beginning EAC: {beginning_eac_refined}")
                if beginning_centerline_refined is not None:
                    print(f"      Refined beginning centerline: {beginning_centerline_refined}")
                
                # Update landmarks to refined versions
                trimmed_start = trimmed_start_refined
                if beginning_eac_refined is not None:
                    beginning_eac = beginning_eac_refined
                if beginning_centerline_refined is not None:
                    beginning_centerline = beginning_centerline_refined
                if mask_boundary_point_refined is not None:
                    mask_boundary_point = mask_boundary_point_refined
            
            # Save refined centerline outputs
            refined_centerline_polydata = create_polydata_from_points(refined_centerline_points)
            save_vtk_polydata(refined_centerline_polydata, output_centerline_refined_raw)
            print(f"  ✓ Refined raw centerline saved to: {output_centerline_refined_raw}")
            
            save_vtk_polydata(centerline_refined, output_centerline_refined)
            print(f"  ✓ Refined resampled centerline saved to: {output_centerline_refined}")
            
            # From now on, use refined centerline for analysis
            centerline = centerline_refined
            raw_centerline_points = refined_centerline_points
            
        except Exception as e:
            print(f"  ⚠ Exception during refined centerline computation: {e}")
            print(f"    Continuing with original centerline")
            refined_endpoint = None
        
        print(f"\n  Centerline landmarks (using {start_point_source}):")
        print(f"    Original start: {original_start}")
        print(f"    Trimmed start (beginning_centerline): {trimmed_start}")
        if beginning_eac is not None:
            print(f"    Beginning EAC (geodesic plane): {beginning_eac}")
        if beginning_centerline is not None:
            print(f"    Beginning Centerline (offset plane): {beginning_centerline}")
        if mask_boundary_point is not None:
            print(f"    Mask boundary point (outermost): {mask_boundary_point}")
        print(f"    End point (Eardrum): {end_point}")
        if refined_endpoint is not None:
            print(f"    Refined end point: {refined_endpoint}")
            print(f"    Original eardrum point: {original_eardrum_point}")

        
        # Save raw centerline
        raw_centerline_polydata = create_polydata_from_points(raw_centerline_points)
        save_vtk_polydata(raw_centerline_polydata, output_centerline_raw)
        print(f"  ✓ Raw centerline saved to: {output_centerline_raw}")
        
        # Save resampled centerline
        save_vtk_polydata(centerline, output_centerline)
        print(f"  ✓ Resampled centerline saved to: {output_centerline}")
        
        # Save centerline landmarks
        landmark_json_path = output_centerline.replace('.vtk', '_landmarks.json')
        landmark_data = {
            'original_start_middle_rs': original_start.tolist(),
            'trimmed_start_at_plane': trimmed_start.tolist(),
            'end_point_eardrum': end_point.tolist()
        }
        # Add original eardrum point before refinement if it was refined
        if refined_endpoint is not None:
            landmark_data['original_eardrum_landmark'] = original_eardrum_point.tolist()
            landmark_data['new_eardrum'] = refined_endpoint.tolist()
            landmark_data['refined'] = True
        else:
            landmark_data['refined'] = False
        if beginning_eac is not None:
            landmark_data['beginning_eac'] = beginning_eac.tolist()
        if beginning_centerline is not None:
            landmark_data['beginning_centerline'] = beginning_centerline.tolist()
        if mask_boundary_point is not None:
            landmark_data['mask_boundary_point'] = mask_boundary_point.tolist()
        with open(landmark_json_path, 'w') as f:
            json.dump(landmark_data, f, indent=2)
        print(f"  ✓ Centerline landmarks saved to: {landmark_json_path}")
        
        # Initialize volume variable (will be computed in STEP 11a)
        isthmus_eardrum_volume = None
        
        # =====================================================================
        # STEP 9: Save complete landmarks JSON
        # =====================================================================
        print(f"\n[STEP 9] Saving complete landmarks JSON")
        complete_landmarks = create_complete_landmarks_json(
            landmarks, beginning_eac, beginning_centerline, mask_boundary_point
        )
        
        os.makedirs(os.path.dirname(output_landmarks), exist_ok=True)
        # NOTE: Complete landmarks JSON will be saved after landmark cross-sections extraction
        # This ensures isthmus is included if detected
        print(f"  (Complete landmarks JSON will be saved after cross-section extraction)")
        
        # =====================================================================
        # STEP 10: Load and clean non-inverted mesh for feature extraction
        # =====================================================================
        print(f"\n[STEP 10] Preparing mesh for feature extraction")
        
        # Extract centerline points early for coordinate system checking
        centerline_points = extract_points_from_polydata(centerline)
        print(f"  ✓ Extracted {len(centerline_points)} centerline points")
        
        # Define cleaned mesh path
        stl_folder = os.path.join(output_dir, "stl")
        os.makedirs(stl_folder, exist_ok=True)
        cleaned_stl_path = os.path.join(stl_folder, f"{sample_name}_clean_one_blob.stl")
        
        # Check if cleaned mesh already exists
        if os.path.exists(cleaned_stl_path):
            print(f"\n[STEP 10.1] Cleaned mesh already exists, loading it")
            non_inverted_mesh = pv.read(cleaned_stl_path)
            print(f"  ✓ Loaded cleaned mesh from: {cleaned_stl_path}")
            print(f"    Mesh: {non_inverted_mesh.GetNumberOfCells()} cells, {non_inverted_mesh.GetNumberOfPoints()} points")
            
            # ===================================================================
            # STEP 10.1a: Check for coordinate system mismatch
            # ===================================================================
            print(f"\n[STEP 10.1a] Checking for coordinate system mismatch between mesh and centerline")
            
            # Get a sample mesh point and centerline point to compare coordinate systems
            mesh_points = non_inverted_mesh.GetPoints()
            if mesh_points.GetNumberOfPoints() > 0:
                sample_mesh_pt = np.array(mesh_points.GetPoint(0))
                sample_centerline_pt = centerline_points[0]
                
                print(f"  Sample mesh point: {sample_mesh_pt}")
                print(f"  Sample centerline point: {sample_centerline_pt}")
                
                # Check if coordinates have opposite signs in x,y (indicating inversion)
                mesh_xy_positive = sample_mesh_pt[0] > 0 and sample_mesh_pt[1] > 0
                centerline_xy_positive = sample_centerline_pt[0] > 0 and sample_centerline_pt[1] > 0
                
                if mesh_xy_positive != centerline_xy_positive:
                    print(f"  ⚠ Coordinate system mismatch detected!")
                    print(f"    Mesh: x={sample_mesh_pt[0]:.1f}, y={sample_mesh_pt[1]:.1f}")
                    print(f"    Centerline: x={sample_centerline_pt[0]:.1f}, y={sample_centerline_pt[1]:.1f}")
                    print(f"  → Correcting mesh coordinates by flipping x and y...")
                    
                    # Transform all mesh points: negate x and y to match centerline system
                    points_array = np.array([mesh_points.GetPoint(i) for i in range(mesh_points.GetNumberOfPoints())])
                    points_array[:, 0] *= -1  # flip x
                    points_array[:, 1] *= -1  # flip y
                    
                    # Create new point set with transformed coordinates
                    new_points = vtk.vtkPoints()
                    for pt in points_array:
                        new_points.InsertNextPoint(pt)
                    
                    non_inverted_mesh.SetPoints(new_points)
                    print(f"  ✓ Mesh coordinates corrected (x and y flipped)")
                else:
                    print(f"  ✓ Coordinate systems match - no correction needed")
        else:
            # Load original mesh and clean it
            print(f"\n[STEP 10.1] Loading original mesh and cleaning")
            if os.path.exists(stl_path):
                mesh_original = pv.read(stl_path)
                print(f"  ✓ Loaded original mesh from: {stl_path}")
            else:
                print(f"  ✗ WARNING: STL not found in input_dir/stl, creating from mask")
                mesh_original = pv.wrap(create_mesh_from_mask(original_mask))
            
            # ===================================================================
            # Check for coordinate system mismatch before cleaning
            # ===================================================================
            print(f"\n  Checking for coordinate system mismatch...")
            
            mesh_points = mesh_original.GetPoints()
            if mesh_points.GetNumberOfPoints() > 0:
                sample_mesh_pt = np.array(mesh_points.GetPoint(0))
                sample_centerline_pt = centerline_points[0]
                
                # Check if coordinates have opposite signs in x,y (indicating inversion)
                mesh_xy_positive = sample_mesh_pt[0] > 0 and sample_mesh_pt[1] > 0
                centerline_xy_positive = sample_centerline_pt[0] > 0 and sample_centerline_pt[1] > 0
                
                if mesh_xy_positive != centerline_xy_positive:
                    print(f"  ⚠ Coordinate system mismatch detected!")
                    print(f"    Mesh: x={sample_mesh_pt[0]:.1f}, y={sample_mesh_pt[1]:.1f}")
                    print(f"    Centerline: x={sample_centerline_pt[0]:.1f}, y={sample_centerline_pt[1]:.1f}")
                    print(f"  → Correcting mesh coordinates by flipping x and y...")
                    
                    # Transform all mesh points: negate x and y to match centerline system
                    points_array = np.array([mesh_points.GetPoint(i) for i in range(mesh_points.GetNumberOfPoints())])
                    points_array[:, 0] *= -1  # flip x
                    points_array[:, 1] *= -1  # flip y
                    
                    # Create new point set with transformed coordinates
                    new_points = vtk.vtkPoints()
                    for pt in points_array:
                        new_points.InsertNextPoint(pt)
                    
                    mesh_original.SetPoints(new_points)
                    print(f"  ✓ Mesh coordinates corrected (x and y flipped)")
                else:
                    print(f"  ✓ Coordinate systems match - no correction needed")
            
            print(f"  Mesh before cleaning: {mesh_original.GetNumberOfCells()} cells, {mesh_original.GetNumberOfPoints()} points")
            
            # Extract all connected components
            connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
            connectivity_filter.SetInputData(mesh_original)
            connectivity_filter.SetExtractionModeToAllRegions()
            connectivity_filter.Update()
            
            num_components = connectivity_filter.GetNumberOfExtractedRegions()
            print(f"  Found {num_components} connected component(s)")
            
            if num_components > 1:
                # Find the largest component
                largest_component_id = 0
                largest_component_size = 0
                
                for comp_id in range(num_components):
                    extractor = vtk.vtkPolyDataConnectivityFilter()
                    extractor.SetInputData(mesh_original)
                    extractor.SetExtractionModeToSpecifiedRegions()
                    extractor.AddSpecifiedRegion(comp_id)
                    extractor.Update()
                    component = extractor.GetOutput()
                    
                    component_size = component.GetNumberOfCells()
                    print(f"    Component {comp_id}: {component_size} cells")
                    
                    if component_size > largest_component_size:
                        largest_component_size = component_size
                        largest_component_id = comp_id
                
                # Extract the largest component
                largest_extractor = vtk.vtkPolyDataConnectivityFilter()
                largest_extractor.SetInputData(mesh_original)
                largest_extractor.SetExtractionModeToSpecifiedRegions()
                largest_extractor.AddSpecifiedRegion(largest_component_id)
                largest_extractor.Update()
                non_inverted_mesh = largest_extractor.GetOutput()
                
                print(f"  ✓ Selected component {largest_component_id} (largest: {largest_component_size} cells)")
            else:
                print(f"  ✓ Only one component found, no cleaning needed")
                non_inverted_mesh = mesh_original
            
            print(f"  Mesh after cleaning: {non_inverted_mesh.GetNumberOfCells()} cells, {non_inverted_mesh.GetNumberOfPoints()} points")
            
            # Save the cleaned mesh
            pv_mesh = pv.wrap(non_inverted_mesh)
            pv_mesh.save(cleaned_stl_path)
            print(f"  ✓ Cleaned mesh saved to: {cleaned_stl_path}")
        
        # =====================================================================
        # STEP 10a: Extract cross-sectional features using non-inverted mesh
        # =====================================================================
        print(f"\n[STEP 10a] Extracting cross-sectional features from cleaned mesh")
        
        # Extract features (min/max radius, perimeter, area at each point) using non-inverted mesh
        centerline_features = extract_centerline_features(
            non_inverted_mesh, centerline_points, output_vtk_path=output_centerline_features
        )
        
        if centerline_features is None:
            print(f"    ⚠ Warning: Feature extraction failed, continuing without features")
        else:
            print(f"  ✓ Centerline features saved to: {output_centerline_features}")
        
        # =====================================================================
        # STEP 10b: Extract cross-sections at landmark locations
        # =====================================================================
        print(f"\n[STEP 10b] Extracting cross-sections at landmark locations")
        
        # Prepare landmarks dict for cross-section extraction
        landmark_positions = {
            '1st bend': landmarks.get('1st bend'),
            '2nd bend': landmarks.get('2nd bend'),
            'Eardrum': eardrum
        }
        # Filter out None values
        landmark_positions = {k: v for k, v in landmark_positions.items() if v is not None}
        
        # Non-inverted mesh is already loaded in STEP 10 for consistent feature extraction
        landmark_cross_sections = extract_landmark_cross_sections(
            mesh, centerline_points, landmark_positions,
            centerline_features=centerline_features,  # For isthmus detection
            isthmus_method="closed_section",
            non_inverted_mesh=non_inverted_mesh,
            inverted_mask=inverted_image 
            )
        
        # =====================================================================
        # STEP 10c: Save complete landmarks JSON (now includes isthmus)
        # =====================================================================
        print(f"\n[STEP 10c] Saving complete landmarks JSON with isthmus and cross-section data")
        
        # Add isthmus to landmarks if found in landmark_cross_sections
        if landmark_cross_sections is not None and 'isthmus' in landmark_cross_sections:
            isthmus_data = landmark_cross_sections['isthmus']
            if 'point' in isthmus_data:
                landmarks['isthmus'] = isthmus_data['point']
        
        # Add new_eardrum if available (refined endpoint)
        if refined_endpoint is not None:
            landmarks['new_eardrum'] = refined_endpoint
        
        # Add new_first_bend and new_second_bend (center points of cross-sections)
        if landmark_cross_sections is not None:
            if '1st bend' in landmark_cross_sections:
                bend_data = landmark_cross_sections['1st bend']
                if 'point' in bend_data:
                    landmarks['new_first_bend'] = bend_data['point']
            if '2nd bend' in landmark_cross_sections:
                bend_data = landmark_cross_sections['2nd bend']
                if 'point' in bend_data:
                    landmarks['new_second_bend'] = bend_data['point']
        
        # Add CBJ landmarks if available
        if cbj_points is not None and len(cbj_points) > 0:
            cbj_landmarks = []
            for i, cbj_pt in enumerate(cbj_points):
                cbj_landmarks.append({
                    'id': str(i + 1),
                    'label': f'CBJ_{i + 1}',
                    'position': cbj_pt.tolist() if hasattr(cbj_pt, 'tolist') else cbj_pt
                })
            landmarks['cbj'] = cbj_landmarks
        
        # Create complete landmarks including isthmus
        complete_landmarks = create_complete_landmarks_json(
            landmarks, beginning_eac, beginning_centerline, mask_boundary_point
        )
        
        # Add cross-sectional features for each landmark to the JSON
        if landmark_cross_sections is not None:
            # Add cross-sectional data section to landmarks JSON
            cross_section_data = {}
            for landmark_name, lm_data in landmark_cross_sections.items():
                if lm_data is not None and 'area' in lm_data and lm_data['area'] is not None:
                    # Calculate aspect ratio for this cross-section
                    min_r = lm_data.get('min_radius', 0)
                    max_r = lm_data.get('max_radius', 0)
                    aspect_ratio = max_r / min_r if min_r > 0 else None
                    
                    # Get point and normal (tangent) if available
                    point = lm_data.get('point')
                    tangent = lm_data.get('tangent')
                    
                    cross_section_data[landmark_name] = {
                        'area': float(lm_data['area']),
                        'perimeter': float(lm_data.get('perimeter', 0)),
                        'min_radius': float(min_r),
                        'max_radius': float(max_r),
                        'aspect_ratio': float(aspect_ratio) if aspect_ratio is not None else None,
                        'centerline_index': int(lm_data.get('index', -1)),
                        'point': point.tolist() if hasattr(point, 'tolist') else point,
                        'normal': tangent.tolist() if hasattr(tangent, 'tolist') else tangent
                    }
            
            if cross_section_data:
                complete_landmarks['cross_sections'] = cross_section_data
        
        # NOTE: JSON save moved to after STEP 11a (volume computation) to include volume in output
            
        # =====================================================================
        # STEP 11: Compute metrics
        # =====================================================================
        print(f"\n[STEP 11] Computing path metrics")

        
        centerline_length = compute_path_length(centerline_points)
        centerline_tortuosity = compute_tortuosity_index(centerline_points)
        geodesic_length = compute_path_length(geodesic_points)
        geodesic_tortuosity = compute_tortuosity_index(geodesic_points)


        
        
        print(f"    Centerline length: {centerline_length:.2f} mm")
        print(f"    Centerline tortuosity: {centerline_tortuosity:.3f}")
        print(f"    Geodesic arch length: {geodesic_length:.2f} mm")
        print(f"    Geodesic arch tortuosity: {geodesic_tortuosity:.3f}")
        isthmus_point = None
        if landmark_cross_sections is not None and 'isthmus' in landmark_cross_sections:
            isthmus_point = landmark_cross_sections['isthmus'].get('point')
        if isthmus_point is None and 'isthmus' in landmarks:
            isthmus_point = landmarks['isthmus']
        eardrum_point = eardrum


        if isthmus_point is not None and eardrum_point is not None:
            try:
                # Use the centerline PolyData for the calculation
                isthmus_length, isthmus_tortuosity = calculate_isthmus_to_eardrum_length_and_tortuosity(
                    centerline=centerline,
                    isthmus_point=np.array(isthmus_point),
                    eardrum_point=np.array(eardrum_point)
                )
                print(f"    Centerline isthmus-to-eardrum length: {isthmus_length:.2f} mm")
                print(f"    Centerline isthmus-to-eardrum tortuosity: {isthmus_tortuosity:.3f}")
            except Exception as err:
                print(f"    ⚠ Could not compute isthmus-to-eardrum metrics: {err}")
                isthmus_length = None
                isthmus_tortuosity = None
        else:
            print("    ⚠ Isthmus or eardrum point not found, skipping isthmus-to-eardrum metrics.")
            isthmus_length = None
            isthmus_tortuosity = None
        
        # NOTE: Mesh reconstruction (STEP 10.5) now happens inside extract_isthmus_eardrum_segment()
        # to keep the volume computation self-contained
        
        # =====================================================================
        # STEP 11a: Compute volume between isthmus and eardrum (optional)
        # =====================================================================
        isthmus_eardrum_volume = None
        volume_clipped_mesh = None
        
        if isthmus_point is not None and eardrum_point is not None and landmark_cross_sections is not None:
            print(f"\n[STEP 11a] Computing volume between isthmus and eardrum")
            try:
                # Get the cross-section data for isthmus and eardrum
                isthmus_data = landmark_cross_sections.get('isthmus') or landmark_cross_sections.get('Isthmus')
                eardrum_data = landmark_cross_sections.get('Eardrum') or landmark_cross_sections.get('eardrum')
                
                if isthmus_data is not None and eardrum_data is not None:
                    # Extract curve polydata, points, and normals from cross-sections
                    isthmus_curve = isthmus_data.get('cross_section')
                    eardrum_curve = eardrum_data.get('cross_section')
                    isthmus_pt = isthmus_data.get('point')
                    isthmus_normal = isthmus_data.get('tangent')
                    eardrum_pt = eardrum_data.get('point')
                    eardrum_normal = eardrum_data.get('tangent')
                    
                    if (isthmus_curve is not None and eardrum_curve is not None and
                        isthmus_pt is not None and isthmus_normal is not None and
                        eardrum_pt is not None and eardrum_normal is not None):
                        # Get indices for centerline-based component selection
                        isthmus_idx = isthmus_data.get('index')
                        eardrum_idx = eardrum_data.get('index')
                        
                        # Extract mesh segment by cutting at cross-section planes
                        second_bend_pt = landmarks.get('new_second_bend')
                        volume_clipped_mesh, _ = extract_isthmus_eardrum_segment(
                            mesh=non_inverted_mesh,
                            isthmus_point=np.array(isthmus_pt),
                            isthmus_normal=np.array(isthmus_normal),
                            eardrum_point=np.array(eardrum_pt),
                            eardrum_normal=np.array(eardrum_normal),
                            isthmus_curve=isthmus_curve,
                            eardrum_curve=eardrum_curve,
                            centerline_points=raw_centerline_points,
                            isthmus_idx=isthmus_idx,
                            eardrum_idx=eardrum_idx,
                            second_bend_point=second_bend_pt,
                            output_dir=output_dir,
                            sample_name=sample_name,
                            visualize=enable_viz
                        )
                        
                        # Check bounding box FIRST before expensive volume computation
                        if volume_clipped_mesh is not None and volume_clipped_mesh.GetNumberOfCells() > 0:
                            # Get physical bounds of mask image for bounding box check
                            origin = np.array(inverted_image.GetOrigin())
                            size = np.array(inverted_image.GetSize())
                            spacing = np.array(inverted_image.GetSpacing())
                            mask_physical_bounds = np.array([
                                origin[0], origin[0] + (size[0] - 1) * spacing[0],
                                origin[1], origin[1] + (size[1] - 1) * spacing[1],
                                origin[2], origin[2] + (size[2] - 1) * spacing[2],
                            ])
                            
                            # Apply 10% margin - the mesh bbox should not extend into the outer 10% of the image
                            margin_fraction = 0.10
                            inner_bounds = np.zeros(6)
                            for i in range(3):
                                extent = mask_physical_bounds[2*i + 1] - mask_physical_bounds[2*i]
                                margin = extent * margin_fraction
                                inner_bounds[2*i] = mask_physical_bounds[2*i] + margin      # min + margin
                                inner_bounds[2*i + 1] = mask_physical_bounds[2*i + 1] - margin  # max - margin
                            
                            # Get bounding box - handle both VTK and PyVista objects
                            if hasattr(volume_clipped_mesh, 'bounds'):
                                clipped_bbox = volume_clipped_mesh.bounds
                            else:
                                clipped_bbox = volume_clipped_mesh.GetBounds()
                            
                            # Check if mesh bbox extends beyond the inner bounds (10% margin)
                            bbox_touches_edge = False
                            touching_edges = []
                            for i, dim_name in enumerate(['x', 'y', 'z']):
                                if clipped_bbox[2*i] < inner_bounds[2*i]:
                                    bbox_touches_edge = True
                                    touching_edges.append(f"{dim_name}_min")
                                if clipped_bbox[2*i + 1] > inner_bounds[2*i + 1]:
                                    bbox_touches_edge = True
                                    touching_edges.append(f"{dim_name}_max")
                            
                            print(f"  [BOUNDING BOX CHECK (10% margin)]")
                            print(f"    Mesh bbox: x=[{clipped_bbox[0]:.1f}, {clipped_bbox[1]:.1f}], y=[{clipped_bbox[2]:.1f}, {clipped_bbox[3]:.1f}], z=[{clipped_bbox[4]:.1f}, {clipped_bbox[5]:.1f}]")
                            print(f"    Mask bounds: x=[{mask_physical_bounds[0]:.1f}, {mask_physical_bounds[1]:.1f}], y=[{mask_physical_bounds[2]:.1f}, {mask_physical_bounds[3]:.1f}], z=[{mask_physical_bounds[4]:.1f}, {mask_physical_bounds[5]:.1f}]")
                            print(f"    Inner bounds (10% margin): x=[{inner_bounds[0]:.1f}, {inner_bounds[1]:.1f}], y=[{inner_bounds[2]:.1f}, {inner_bounds[3]:.1f}], z=[{inner_bounds[4]:.1f}, {inner_bounds[5]:.1f}]")
                            
                            if bbox_touches_edge:
                                print(f"    ⚠ BBOX EXTENDS INTO MARGIN ZONE: {', '.join(touching_edges)}")
                                print(f"    → Will need isthmus adjustment after volume computation")
                            else:
                                print(f"    ✓ Bounding box within safe zone (10% margin from edges)")
                            
                            print(f"\n  → Computing volume using voxel-based method (voxel_spacing=0.5 mm)...")
                            
                            # Save the mesh to STL for processing
                            stl_dir = os.path.join(output_dir, "stl")
                            os.makedirs(stl_dir, exist_ok=True)
                            stl_path = os.path.join(stl_dir, f"{sample_name}_isthmus_eardrum_segment.stl")
                            
                            # Extract largest connected component before saving
                            cleaned_mesh = extract_largest_connected_component(volume_clipped_mesh, verbose=True)
                            pv.wrap(cleaned_mesh).save(stl_path)
                            print(f"  ✓ Saved segment to: {stl_path}")
                            
                            # Compute volume using voxel method (use 0.5mm spacing, same as tissue volumes)
                            print(f"  [DEBUG] About to compute voxel volume for {stl_path}")
                            isthmus_eardrum_volume, _ = compute_voxel_mesh_volume(
                                mesh_or_path=stl_path,
                                voxel_spacing=0.5,
                                save_cleaned_stl=None,
                                return_mesh=False,
                                verbose=True
                            )
                            print(f"  [DEBUG] compute_voxel_mesh_volume returned: {isthmus_eardrum_volume}")
                            
                            # BACKUP: Store original computed volume in case adjustment fails
                            original_computed_volume = isthmus_eardrum_volume
                            
                            # Print volume result (even if 0)
                            if isthmus_eardrum_volume is not None:
                                print(f"  ✓ Volume (voxel method): {isthmus_eardrum_volume:.2f} mm³")
                            else:
                                print(f"  ⚠ Volume computation returned None")
                            
                            # STEP 11a.0: Adjust isthmus if volume is too large (> 1500 mm³) OR bbox touches mask edges
                            # Check adjustment REGARDLESS of volume result, as bbox check is independent
                            needs_adjustment = ((isthmus_eardrum_volume is not None and isthmus_eardrum_volume > 1500) or 
                                               bbox_touches_edge)
                            if needs_adjustment and isthmus_idx is not None and centerline_points is not None:
                                    reason = []
                                    if isthmus_eardrum_volume is not None and isthmus_eardrum_volume > 1500:
                                        reason.append(f"volume {isthmus_eardrum_volume:.0f} > 1500 mm³")
                                    if bbox_touches_edge:
                                        reason.append(f"bbox touches mask edges ({', '.join(touching_edges)})")
                                    print(f"\n[STEP 11a.0] Adjusting isthmus point: {' OR '.join(reason)}")
                                    
                                    try:
                                        adjusted_idx, adjusted_volume, adjusted_isthmus_data = adjust_isthmus_for_large_volume(
                                            centerline_points=centerline_points,
                                            isthmus_idx=isthmus_idx,
                                            eardrum_pt=np.array(eardrum_pt),
                                            eardrum_normal=np.array(eardrum_normal),
                                            eardrum_curve=eardrum_curve,
                                            mesh=mesh_simplified,
                                            non_inverted_mesh=non_inverted_mesh,
                                            inverted_image=inverted_image,
                                            current_volume=isthmus_eardrum_volume,
                                            sample_name=sample_name,
                                            output_dir=output_dir,
                                            max_volume_threshold=1500.0,
                                            volume_drop_percentage=20.0,
                                            verbose=True
                                        )
                                        
                                        # Use adjusted values if better
                                        # If original volume was None/0, any adjustment is better
                                        should_use_adjusted = (isthmus_eardrum_volume is None or 
                                                             isthmus_eardrum_volume == 0 or
                                                             (adjusted_volume is not None and adjusted_volume < isthmus_eardrum_volume))
                                        if should_use_adjusted:
                                            if isthmus_eardrum_volume is not None:
                                                print(f"\n  ✓ Using adjusted isthmus: volume {isthmus_eardrum_volume:.2f} → {adjusted_volume:.2f} mm³")
                                            else:
                                                print(f"\n  ✓ Using adjusted isthmus: volume was problematic, now {adjusted_volume:.2f} mm³")
                                            isthmus_eardrum_volume = adjusted_volume
                                            isthmus_idx = adjusted_idx
                                            
                                            # Update isthmus_data with adjusted values
                                            if adjusted_isthmus_data is not None:
                                                isthmus_data.update(adjusted_isthmus_data)
                                                isthmus_pt = adjusted_isthmus_data['point']
                                                isthmus_normal = adjusted_isthmus_data['tangent']
                                                isthmus_curve = adjusted_isthmus_data['cross_section']
                                                
                                                # Recompute volume segment with adjusted isthmus
                                                volume_clipped_mesh, _ = extract_isthmus_eardrum_segment(
                                                    mesh=non_inverted_mesh,
                                                    isthmus_point=np.array(isthmus_pt),
                                                    isthmus_normal=np.array(isthmus_normal),
                                                    eardrum_point=np.array(eardrum_pt),
                                                    eardrum_normal=np.array(eardrum_normal),
                                                    isthmus_curve=isthmus_curve,
                                                    eardrum_curve=eardrum_curve,
                                                    centerline_points=centerline_points,
                                                    isthmus_idx=adjusted_idx,
                                                    eardrum_idx=eardrum_idx,
                                                    second_bend_point=landmarks.get('new_second_bend'),
                                                    output_dir=output_dir,
                                                    sample_name=sample_name,
                                                    visualize=False
                                                )
                                                
                                                # Re-save the adjusted segment STL
                                                if volume_clipped_mesh is not None:
                                                    adjusted_cleaned = extract_largest_connected_component(volume_clipped_mesh, verbose=False)
                                                    pv.wrap(adjusted_cleaned).save(stl_path)
                                                    print(f"  ✓ Re-saved adjusted segment to: {stl_path}")
                                                    
                                                    # CRITICAL: Recompute the actual voxel volume with adjusted isthmus
                                                    # (not the bbox proxy estimate from adjust_isthmus_for_large_volume)
                                                    print(f"  [RECOMPUTE] Computing actual voxel volume with adjusted isthmus...")
                                                    try:
                                                        adjusted_actual_volume, _ = compute_voxel_mesh_volume(
                                                            mesh_or_path=stl_path,
                                                            voxel_spacing=0.5,
                                                            save_cleaned_stl=None,
                                                            return_mesh=False,
                                                            verbose=False
                                                        )
                                                        if adjusted_actual_volume is not None and adjusted_actual_volume > 0:
                                                            print(f"  ✓ Recomputed volume with adjusted isthmus: {adjusted_actual_volume:.2f} mm³")
                                                            isthmus_eardrum_volume = adjusted_actual_volume
                                                            # Update backup as well
                                                            original_computed_volume = adjusted_actual_volume
                                                        else:
                                                            print(f"  ⚠ Recomputed volume invalid, keeping estimated: {adjusted_volume:.2f} mm³")
                                                            isthmus_eardrum_volume = adjusted_volume
                                                    except Exception as volrecomp_err:
                                                        print(f"  ⚠ Could not recompute volume: {volrecomp_err}, using estimate: {adjusted_volume:.2f} mm³")
                                                        isthmus_eardrum_volume = adjusted_volume
                                                
                                                # Update landmarks dict with adjusted isthmus position
                                                landmarks['isthmus'] = isthmus_pt
                                                isthmus_point = isthmus_pt  # Also update the variable used throughout
                                                print(f"  ✓ Updated landmarks['isthmus'] and isthmus_point with adjusted position")
                                                
                                                # Update landmark_cross_sections with adjusted data
                                                if landmark_cross_sections is not None:
                                                    landmark_cross_sections['isthmus'] = adjusted_isthmus_data
                                                    print(f"  ✓ Updated landmark_cross_sections['isthmus'] with adjusted data")
                                                
                                                # Update complete_landmarks['landmarks'] list entry for isthmus
                                                if 'landmarks' in complete_landmarks:
                                                    for lm_entry in complete_landmarks['landmarks']:
                                                        if lm_entry.get('label') == 'isthmus':
                                                            lm_entry['position'] = isthmus_pt.tolist() if hasattr(isthmus_pt, 'tolist') else isthmus_pt
                                                            print(f"  ✓ Updated complete_landmarks['landmarks'] isthmus entry")
                                                            break
                                                
                                                # Update complete_landmarks cross-section data for isthmus
                                                if 'cross_sections' in complete_landmarks and 'isthmus' in complete_landmarks['cross_sections']:
                                                    adjusted_min_r = adjusted_isthmus_data.get('min_radius', 0)
                                                    adjusted_max_r = adjusted_isthmus_data.get('max_radius', 0)
                                                    adjusted_aspect = adjusted_max_r / adjusted_min_r if adjusted_min_r > 0 else None
                                                    complete_landmarks['cross_sections']['isthmus'] = {
                                                        'area': float(adjusted_isthmus_data.get('area', 0)),
                                                        'perimeter': float(adjusted_isthmus_data.get('perimeter', 0)),
                                                        'min_radius': float(adjusted_min_r),
                                                        'max_radius': float(adjusted_max_r),
                                                        'aspect_ratio': float(adjusted_aspect) if adjusted_aspect else None,
                                                        'centerline_index': int(adjusted_idx),
                                                        'point': isthmus_pt.tolist() if hasattr(isthmus_pt, 'tolist') else isthmus_pt,
                                                        'normal': isthmus_normal.tolist() if hasattr(isthmus_normal, 'tolist') else isthmus_normal
                                                    }
                                                    print(f"  ✓ Updated complete_landmarks['cross_sections']['isthmus'] with adjusted data")
                                        else:
                                            print(f"\n  ⚠ Adjusted isthmus did not improve volume, keeping original")
                                    except Exception as err:
                                        print(f"\n  ⚠ Could not adjust isthmus: {err}")
                                        # Continue with original volume
                            else:
                                print(f"  ⚠ Voxel volume computation returned invalid result")
                                isthmus_eardrum_volume = None
                    else:
                        print(f"  ⚠ Missing cross-section data (curves, points, or normals)")
                else:
                    print(f"  ⚠ Isthmus or eardrum cross-section data not found")
            except Exception as err:
                print(f"  ⚠ Could not compute isthmus-to-eardrum volume: {err}")
                isthmus_eardrum_volume = None
                import traceback
                traceback.print_exc()
        
        # FAILSAFE: If volume was computed but then lost, try to restore from backup
        # This handles cases where the backup was created but volume became None during processing
        if 'original_computed_volume' in locals() and original_computed_volume is not None and isthmus_eardrum_volume is None:
            print(f"  [FAILSAFE] Restoring original computed volume: {original_computed_volume:.2f} mm³")
            isthmus_eardrum_volume = original_computed_volume
        
        # STEP 11a.1: Visualize volume segment with shaped caps
        if enable_viz and volume_clipped_mesh is not None and isthmus_eardrum_volume is not None:
            print(f"\n[STEP 11a.0] Visualizing volume segment with shaped caps...")
            isthmus_center = np.array(isthmus_data.get('point', [0, 0, 0]))
            eardrum_center = np.array(eardrum_data.get('point', [0, 0, 0]))
            visualize_volume_segment(
                mesh=volume_clipped_mesh,
                isthmus_point=isthmus_center,
                eardrum_point=eardrum_center,
                volume_mm3=isthmus_eardrum_volume,
                title=f"{sample_name} - Volume Segment (Isthmus-to-Eardrum)"
            )
            print(f"  ✓ Visualization window closed. Continuing pipeline...\n")
        
        # =====================================================================
        # STEP 11a.1: Save complete landmarks JSON (with volume and metrics)
        # =====================================================================
        print(f"\n[STEP 11a.1] Saving complete landmarks JSON with computed metrics")
        
        # Add volumes to a dedicated volumes section
        if 'volumes' not in complete_landmarks:
            complete_landmarks['volumes'] = {}
        
        if isthmus_eardrum_volume is not None:
            complete_landmarks['volumes']['isthmus_eardrum_volume'] = float(isthmus_eardrum_volume)
            print(f"  ✓ Isthmus-eardrum volume: {isthmus_eardrum_volume:.2f} mm³")
        else:
            print(f"  ⚠ Isthmus-eardrum volume is None - not saved to JSON")
            print(f"    Debugging info:")
            print(f"      isthmus_point: {isthmus_point}")
            print(f"      eardrum_point: {eardrum_point}")
            print(f"      landmark_cross_sections exists: {landmark_cross_sections is not None}")
            if landmark_cross_sections is not None:
                print(f"      landmark_cross_sections keys: {list(landmark_cross_sections.keys())}")
                isthmus_data_check = landmark_cross_sections.get('isthmus') or landmark_cross_sections.get('Isthmus')
                eardrum_data_check = landmark_cross_sections.get('Eardrum') or landmark_cross_sections.get('eardrum')
                print(f"      isthmus_data exists: {isthmus_data_check is not None}")
                print(f"      eardrum_data exists: {eardrum_data_check is not None}")
                if isthmus_data_check:
                    print(f"      isthmus_data keys: {list(isthmus_data_check.keys())}")
                if eardrum_data_check:
                    print(f"      eardrum_data keys: {list(eardrum_data_check.keys())}")
        
        os.makedirs(os.path.dirname(output_landmarks), exist_ok=True)
        
        # Debug: print what we're about to save
        print(f"  [DEBUG] complete_landmarks['volumes'] = {complete_landmarks.get('volumes', {})}")
        
        save_landmarks_json(complete_landmarks, output_landmarks)
        print(f"  ✓ Complete landmarks saved to: {output_landmarks}")
        if 'landmarks' in complete_landmarks:
            landmark_labels = [lm['label'] for lm in complete_landmarks['landmarks']]
            print(f"    Landmarks: {', '.join(landmark_labels)}")
        if 'new_first_bend' in (complete_landmarks.get('landmarks') or {}):
            print(f"    New first bend: Included")
        if 'new_second_bend' in (complete_landmarks.get('landmarks') or {}):
            print(f"    New second bend: Included")
        if 'new_eardrum' in (complete_landmarks.get('landmarks') or {}):
            print(f"    New eardrum: Included")
        if 'cbj' in (complete_landmarks.get('landmarks') or {}):
            cbj_count = len(complete_landmarks['landmarks']['cbj']) if isinstance(complete_landmarks['landmarks'].get('cbj'), list) else 0
            print(f"    CBJ landmarks: {cbj_count} points")
        if 'cross_sections' in complete_landmarks:
            print(f"    Cross-sections: {', '.join(complete_landmarks['cross_sections'].keys())}")
        if 'volumes' in complete_landmarks and 'isthmus_eardrum_volume' in complete_landmarks['volumes']:
            volume = complete_landmarks['volumes']['isthmus_eardrum_volume']
            print(f"    Volume (isthmus-eardrum): {volume:.2f} mm³")
        
        # =====================================================================
        # STEP 11b: Process CBJ (Cartilaginous-Bony Junction) landmarks (optional)
        # =====================================================================
        cbj_metrics = {}
        cbj_plane_normal = None
        cbj_plane_centroid = None
        cbj_closest_point = None
        cbj_cross_section_curve = None
        
        if cbj_points is not None and len(cbj_points) >= 3:
            print(f"\n[STEP 11b] Processing CBJ landmarks")
            print(f"  Using {len(cbj_points)} CBJ landmark points loaded in STEP 3b")
            print(f"  CBJ points:")
            for i, pt in enumerate(cbj_points):
                print(f"    CBJ{i+1}: [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}]")
            
            try:
                cbj_plane_normal, cbj_plane_centroid = fit_plane_to_cbj_points(cbj_points)
                print(f"  ✓ CBJ plane fitted")
                print(f"    CBJ plane normal: {cbj_plane_normal}")
                print(f"    CBJ plane centroid: {cbj_plane_centroid}")
                
                # Find closest centerline point to CBJ plane (using resampled centerline for consistency with feature extraction)
                cbj_closest_idx, cbj_closest_point, cbj_plane_distance = find_closest_point_to_plane(
                    centerline_points, cbj_plane_normal, cbj_plane_centroid
                )
                print(f"  ✓ CBJ intersection found at resampled centerline index {cbj_closest_idx}")
                print(f"    Closest point: {cbj_closest_point}")
                print(f"    Distance to plane: {cbj_plane_distance:.2f} mm")
                
                # Extract cross-section features at CBJ plane (using resampled centerline)
                cbj_cross_section_features, cbj_cross_section_curve = extract_cbj_cross_section_features(
                    mesh=non_inverted_mesh,
                    centerline_points=centerline_points,
                    cbj_plane_normal=cbj_plane_normal,
                    cbj_plane_centroid=cbj_plane_centroid,
                    cbj_closest_idx=cbj_closest_idx
                )
                
                if cbj_cross_section_features is not None:
                    print(f"  ✓ CBJ cross-section features extracted:")
                    print(f"    Area: {cbj_cross_section_features['area']:.2f} mm²")
                    print(f"    Perimeter: {cbj_cross_section_features['perimeter']:.2f} mm")
                    print(f"    Min radius: {cbj_cross_section_features['min_radius']:.2f} mm")
                    print(f"    Max radius: {cbj_cross_section_features['max_radius']:.2f} mm")
                    if cbj_cross_section_features['aspect_ratio'] is not None:
                        print(f"    Aspect ratio: {cbj_cross_section_features['aspect_ratio']:.3f}")
                    
                    # Add to metrics dict
                    cbj_metrics['cbj_area'] = float(cbj_cross_section_features['area'])
                    cbj_metrics['cbj_perimeter'] = float(cbj_cross_section_features['perimeter'])
                    cbj_metrics['cbj_min_radius'] = float(cbj_cross_section_features['min_radius'])
                    cbj_metrics['cbj_max_radius'] = float(cbj_cross_section_features['max_radius'])
                    cbj_metrics['cbj_aspect_ratio'] = float(cbj_cross_section_features['aspect_ratio']) if cbj_cross_section_features['aspect_ratio'] is not None else None
                else:
                    print(f"  ⚠ Warning: Could not extract CBJ cross-section features")
                    cbj_cross_section_curve = None
                
                # Compute centerline length metrics related to CBJ
                # First find isthmus index if available
                isthmus_idx = None
                if isthmus_point is not None:
                    # Find closest centerline point to isthmus (using resampled centerline)
                    isthmus_distances = np.linalg.norm(centerline_points - np.array(isthmus_point), axis=1)
                    isthmus_idx = np.argmin(isthmus_distances)
                
                cbj_length_metrics = compute_cbj_centerline_metrics(
                    centerline_points=centerline_points,
                    cbj_closest_idx=cbj_closest_idx,
                    isthmus_idx=isthmus_idx
                )
                
                print(f"  ✓ CBJ centerline metrics:")
                print(f"    Length from start to CBJ: {cbj_length_metrics['cbj_length_from_start']:.2f} mm")
                print(f"    Length from CBJ to end: {cbj_length_metrics['cbj_length_to_end']:.2f} mm")
                print(f"    CBJ proportion (0-1): {cbj_length_metrics['cbj_total_proportion']:.3f}")
                if 'cbj_isthmus_to_cbj_length' in cbj_length_metrics:
                    print(f"    Length from isthmus to CBJ: {cbj_length_metrics['cbj_isthmus_to_cbj_length']:.2f} mm")
                
                # Add length metrics to dict
                for key, value in cbj_length_metrics.items():
                    if isinstance(value, (int, float)):
                        cbj_metrics[key] = float(value) if not isinstance(value, bool) else value
                    else:
                        cbj_metrics[key] = value
                cbj_metrics['cbj_centerline_index'] = int(cbj_closest_idx)  # Ensure integer type
                cbj_metrics['cbj_plane_distance'] = float(cbj_plane_distance)
                
                # ===================================================================
                # STEP 11b.1: Compute tissue volumes
                # ===================================================================
                if volume_clipped_mesh is not None and isthmus_point is not None and eardrum_point is not None:
                    print(f"\n[STEP 11b.1] Computing soft and hard tissue volumes at CBJ plane")
                    try:
                        # Get capped tissue meshes (hard and soft) from CBJ split
                        hard_tissue_mesh, soft_tissue_mesh = split_canal_at_cbj_plane(
                            canal_mesh=volume_clipped_mesh,
                            cbj_plane_centroid=cbj_plane_centroid,
                            cbj_plane_normal=cbj_plane_normal,
                            isthmus_point=isthmus_point,
                            eardrum_point=eardrum_point,
                            output_dir=output_dir,
                            sample_name=sample_name
                        )
                        
                        if hard_tissue_mesh is not None and soft_tissue_mesh is not None:
                            # Compute voxel-based volumes directly from mesh objects
                            print(f"\n  → Computing tissue volumes using voxel-based method (voxel_spacing=0.05 mm)...")
                            
                            # Compute hard tissue volume
                            print(f"\n  ╔═ HARD TISSUE (voxel method) ═╗")
                            hard_tissue_vol, _ = compute_voxel_mesh_volume(
                                mesh_or_path=hard_tissue_mesh,
                                voxel_spacing=0.5,
                                save_cleaned_stl=None,
                                return_mesh=False,
                                verbose=True
                            )
                            if hard_tissue_vol is not None and hard_tissue_vol > 0:
                                print(f"  ✓ Hard tissue volume (voxel method): {hard_tissue_vol:.2f} mm³")
                            else:
                                hard_tissue_vol = None
                                print(f"  ⚠ Hard tissue voxel computation returned invalid result")
                            
                            # Compute soft tissue volume
                            print(f"\n  ╔═ SOFT TISSUE (voxel method) ═╗")
                            soft_tissue_vol, _ = compute_voxel_mesh_volume(
                                mesh_or_path=soft_tissue_mesh,
                                voxel_spacing=0.5,
                                save_cleaned_stl=None,
                                return_mesh=False,
                                verbose=True
                            )
                            if soft_tissue_vol is not None and soft_tissue_vol > 0:
                                print(f"  ✓ Soft tissue volume (voxel method): {soft_tissue_vol:.2f} mm³")
                            else:
                                soft_tissue_vol = None
                                print(f"  ⚠ Soft tissue voxel computation returned invalid result")
                            
                            # Add tissue volumes to cbj_metrics for JSON output
                            cbj_metrics['soft_tissue_portion_canal_volume'] = float(soft_tissue_vol) if soft_tissue_vol is not None else None
                            cbj_metrics['hard_tissue_portion_canal_volume'] = float(hard_tissue_vol) if hard_tissue_vol is not None else None
                            if soft_tissue_vol is not None and hard_tissue_vol is not None:
                                print(f"\n  ✓ Tissue volumes added to JSON output:")
                                print(f"    Soft tissue volume: {soft_tissue_vol:.2f} mm³")
                                print(f"    Hard tissue volume: {hard_tissue_vol:.2f} mm³")
                        else:
                            print(f"  ⚠ Could not obtain tissue meshes from CBJ split")
                    except Exception as err:
                        print(f"  ⚠ Error computing tissue volumes: {err}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"  ⚠ Error processing CBJ landmarks: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[STEP 11b] Skipping CBJ processing")
            print(f"  Reason: ", end="")
            if cbj_points is None:
                print(f"cbj_points is None (no CBJ landmarks loaded)")
            elif len(cbj_points) == 0:
                print(f"cbj_points is empty list")
            elif len(cbj_points) < 3:
                print(f"only {len(cbj_points)} CBJ landmark point(s) available, need at least 3")
                print(f"  Available CBJ points:")
                for i, pt in enumerate(cbj_points):
                    print(f"    CBJ{i+1}: [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}]")
            else:
                print(f"unknown reason (cbj_points={cbj_points})")
        
        # =====================================================================
        # STEP 11c: Add CBJ cross-section data to landmarks JSON (optional)
        # =====================================================================
        if cbj_metrics and len(cbj_metrics) > 0:
            print(f"\n[STEP 11c] Adding CBJ cross-section data to landmarks JSON")
            try:
                # Load the existing landmarks JSON
                if os.path.exists(output_landmarks):
                    with open(output_landmarks, 'r') as f:
                        landmarks_data = json.load(f)
                else:
                    landmarks_data = {}
                
                # Add CBJ cross-section metrics
                if 'cross_sections' not in landmarks_data:
                    landmarks_data['cross_sections'] = {}
                
                # Build CBJ cross-section data with proper type conversions
                cbj_data = {
                    'area': float(cbj_metrics.get('cbj_area', 0)) if cbj_metrics.get('cbj_area') is not None else None,
                    'perimeter': float(cbj_metrics.get('cbj_perimeter', 0)) if cbj_metrics.get('cbj_perimeter') is not None else None,
                    'min_radius': float(cbj_metrics.get('cbj_min_radius', 0)) if cbj_metrics.get('cbj_min_radius') is not None else None,
                    'max_radius': float(cbj_metrics.get('cbj_max_radius', 0)) if cbj_metrics.get('cbj_max_radius') is not None else None,
                    'aspect_ratio': float(cbj_metrics.get('cbj_aspect_ratio', 0)) if cbj_metrics.get('cbj_aspect_ratio') is not None else None,
                    'centerline_index': int(cbj_metrics.get('cbj_centerline_index', -1)) if cbj_metrics.get('cbj_centerline_index') is not None else -1,
                    'plane_distance': float(cbj_metrics.get('cbj_plane_distance', 0)) if cbj_metrics.get('cbj_plane_distance') is not None else None,
                    'length_from_start': float(cbj_metrics.get('cbj_length_from_start', 0)) if cbj_metrics.get('cbj_length_from_start') is not None else None,
                    'length_to_end': float(cbj_metrics.get('cbj_length_to_end', 0)) if cbj_metrics.get('cbj_length_to_end') is not None else None,
                    'total_proportion': float(cbj_metrics.get('cbj_total_proportion', 0)) if cbj_metrics.get('cbj_total_proportion') is not None else None,
                }
                
                # Include isthmus-to-CBJ length if available
                if 'cbj_isthmus_to_cbj_length' in cbj_metrics and cbj_metrics['cbj_isthmus_to_cbj_length'] is not None:
                    cbj_data['isthmus_to_cbj_length'] = float(cbj_metrics['cbj_isthmus_to_cbj_length'])
                
                landmarks_data['cross_sections']['cbj'] = cbj_data
                
                # Add tissue volumes to volumes section
                if 'volumes' not in landmarks_data:
                    landmarks_data['volumes'] = {}
                
                if 'soft_tissue_portion_canal_volume' in cbj_metrics and cbj_metrics['soft_tissue_portion_canal_volume'] is not None:
                    landmarks_data['volumes']['soft_tissue_portion_canal_volume'] = float(cbj_metrics['soft_tissue_portion_canal_volume'])
                if 'hard_tissue_portion_canal_volume' in cbj_metrics and cbj_metrics['hard_tissue_portion_canal_volume'] is not None:
                    landmarks_data['volumes']['hard_tissue_portion_canal_volume'] = float(cbj_metrics['hard_tissue_portion_canal_volume'])
                
                # Print volume summary for debugging
                if 'volumes' in landmarks_data:
                    if 'isthmus_eardrum_volume' in landmarks_data.get('volumes', {}):
                        iso_vol = landmarks_data['volumes'].get('isthmus_eardrum_volume')
                    else:
                        iso_vol = None
                    soft_vol = landmarks_data['volumes'].get('soft_tissue_portion_canal_volume')
                    hard_vol = landmarks_data['volumes'].get('hard_tissue_portion_canal_volume')
                    
                    if iso_vol and soft_vol and hard_vol:
                        total_tissue = soft_vol + hard_vol
                        difference = total_tissue - iso_vol
                        percent_diff = (difference / iso_vol * 100) if iso_vol > 0 else 0
                        print(f"\n  [Volume Validation]")
                        print(f"    Isthmus-eardrum: {iso_vol:.2f} mm³")
                        print(f"    Soft + Hard: {soft_vol:.2f} + {hard_vol:.2f} = {total_tissue:.2f} mm³")
                        print(f"    Difference: {difference:.2f} mm³ ({percent_diff:+.1f}%)")
                        if abs(percent_diff) > 10:
                            print(f"    ⚠ WARNING: Tissue volumes don't match isthmus-eardrum volume!")
                
                # Save updated landmarks JSON
                save_landmarks_json(landmarks_data, output_landmarks)
                print(f"  ✓ CBJ cross-section data added to landmarks JSON")
                print(f"    Area: {cbj_data.get('area', 'N/A')} mm²")
                print(f"    Perimeter: {cbj_data.get('perimeter', 'N/A')} mm")
                print(f"    Centerline index: {cbj_data.get('centerline_index', 'N/A')}")
                
            except Exception as e:
                print(f"  ⚠ Warning: Could not add CBJ data to landmarks JSON: {e}")
        
        # =====================================================================
        # STEP 12: Visualize final results (optional)
        # =====================================================================
        if enable_viz:
            cbj_status = " [WITH CBJ]" if cbj_loaded else ""
            print(f"\n[STEP 12] Visualizing final results{cbj_status}")
            if cbj_loaded:
                print(f"  CBJ plane: {cbj_plane_normal is not None}")
                print(f"  CBJ cross-section: {cbj_cross_section_curve is not None}")
            if volume_clipped_mesh is not None:
                print(f"  Volume mesh: {volume_clipped_mesh.GetNumberOfCells()} cells for visualization")
            visualize_results(
                mesh, centerline, geodesic_path,
                middle_rs, eardrum, top_rs, bottom_rs,
                trimmed_start=trimmed_start,
                plane_normal=plane_normal,
                plane_centroid=plane_centroid,
                offset_plane_centroid=offset_plane_centroid,
                mask_boundary_point=mask_boundary_point,
                cbj_points=cbj_points,
                cbj_plane_normal=cbj_plane_normal,
                cbj_plane_centroid=cbj_plane_centroid,
                cbj_closest_point=cbj_closest_point,
                cbj_cross_section=cbj_cross_section_curve,
                clipped_volume_mesh=volume_clipped_mesh,
                title=f"{sample_name} - Final Results{cbj_status}"
            )
            
            # Visualize landmark cross-sections
            cbj_status = " [WITH CBJ]" if cbj_loaded else ""
            print(f"\n[STEP 12b] Visualizing landmark cross-sections{cbj_status}")
            visualize_landmark_cross_sections(
                non_inverted_mesh, centerline, landmark_cross_sections,
                landmark_positions,
                cbj_points=cbj_points,
                cbj_closest_point=cbj_closest_point,
                cbj_cross_section=cbj_cross_section_curve,
                clipped_volume_mesh=volume_clipped_mesh,
                title=f"{sample_name} - Landmark Cross-Sections{cbj_status}"
            )

            # Combined visualization (calls both above)
            cbj_status = " [WITH CBJ]" if cbj_loaded else ""
            print(f"\n[STEP 12c] Visualizing full results (combined){cbj_status}")
            
            visualize_full_results(
                mesh=non_inverted_mesh,
                centerline=centerline,
                geodesic_path=geodesic_path,
                middle_rs=middle_rs,
                eardrum=eardrum,
                top_rs=top_rs,
                bottom_rs=bottom_rs,
                trimmed_start=trimmed_start,
                plane_normal=plane_normal,
                plane_centroid=plane_centroid,
                offset_plane_centroid=offset_plane_centroid,
                mask_boundary_point=mask_boundary_point,
                landmark_cross_sections=landmark_cross_sections,
                landmark_positions=landmark_positions,
                cbj_points=cbj_points,
                cbj_plane_normal=cbj_plane_normal,
                cbj_plane_centroid=cbj_plane_centroid,
                cbj_closest_point=cbj_closest_point,
                cbj_cross_section=cbj_cross_section_curve,
                clipped_volume_mesh=volume_clipped_mesh,
                title=f"{sample_name} - Full Results{cbj_status}"
            )
        else:
            print(f"\n[STEP 12] Skipping visualization (enable_viz=False)")
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE: {sample_name}")
        print(f"{'='*70}\n")
        
        # Build result dictionary with metrics
        result = {
        
            'sample_name': sample_name,
            'centerline_length': centerline_length,
            'centerline_tortuosity': centerline_tortuosity,
            'geodesic_length': geodesic_length,
            'geodesic_tortuosity': geodesic_tortuosity,
            'isthmus_eardrum_length': isthmus_length,
            'isthmus_eardrum_tortuosity': isthmus_tortuosity,
            'isthmus_eardrum_volume': isthmus_eardrum_volume,
            'start_point_source': start_point_source
        }
        
        # Add cross-sectional feature statistics if available
        if centerline_features is not None:
            result['min_radius_mean'] = float(np.mean(centerline_features['min_radius']))
            result['min_radius_std'] = float(np.std(centerline_features['min_radius']))
            result['max_radius_mean'] = float(np.mean(centerline_features['max_radius']))
            result['max_radius_std'] = float(np.std(centerline_features['max_radius']))
            result['area_mean'] = float(np.mean(centerline_features['area']))
            result['area_std'] = float(np.std(centerline_features['area']))
            result['perimeter_mean'] = float(np.mean(centerline_features['perimeter']))
            result['perimeter_std'] = float(np.std(centerline_features['perimeter']))
            
            # Calculate aspect ratio for all cross-sections along centerline
            min_radii = np.array(centerline_features['min_radius'])
            max_radii = np.array(centerline_features['max_radius'])
            # Avoid division by zero
            valid_ratios = max_radii[min_radii > 1e-6] / min_radii[min_radii > 1e-6]
            if len(valid_ratios) > 0:
                result['aspect_ratio_mean'] = float(np.mean(valid_ratios))
                result['aspect_ratio_std'] = float(np.std(valid_ratios))
            else:
                result['aspect_ratio_mean'] = None
                result['aspect_ratio_std'] = None
        
        # Add landmark-specific cross-section features with aspect ratios
        if landmark_cross_sections is not None:
            landmark_aspect_ratios = []
            for landmark_name, lm_data in landmark_cross_sections.items():
                prefix = landmark_name.replace(' ', '_').replace('-', '_')
                # Features are stored directly in lm_data (area, min_radius, etc.)
                if lm_data.get('area') is not None:
                    result[f'{prefix}_area'] = lm_data.get('area')
                    result[f'{prefix}_perimeter'] = lm_data.get('perimeter')
                    result[f'{prefix}_min_radius'] = lm_data.get('min_radius')
                    result[f'{prefix}_max_radius'] = lm_data.get('max_radius')
                    result[f'{prefix}_centerline_idx'] = lm_data.get('index')
                    
                    # Calculate and store aspect ratio for this landmark
                    min_r = lm_data.get('min_radius', 0)
                    max_r = lm_data.get('max_radius', 0)
                    if min_r > 1e-6:
                        aspect_ratio = max_r / min_r
                        result[f'{prefix}_aspect_ratio'] = float(aspect_ratio)
                        landmark_aspect_ratios.append(aspect_ratio)
                    else:
                        result[f'{prefix}_aspect_ratio'] = None
            
            # Calculate mean aspect ratio across all landmarks
            if landmark_aspect_ratios:
                result['landmark_aspect_ratio_mean'] = float(np.mean(landmark_aspect_ratios))
                result['landmark_aspect_ratio_std'] = float(np.std(landmark_aspect_ratios))
            else:
                result['landmark_aspect_ratio_mean'] = None
                result['landmark_aspect_ratio_std'] = None
        
        # Add CBJ metrics if available
        if cbj_metrics:
            result.update(cbj_metrics)
        
        return result, None
        
    except Exception as e:
        print(f"\n  ✗ ERROR processing {sample_name}: {e}")
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        return None, str(e)


def discover_samples(input_dir: str) -> List[str]:
    """
    Discover samples in input directory.
    
    Looks for matching pairs of .nii.gz masks and .json landmarks.
    Supports both our pipeline layout (masks/, markups/) and the legacy
    layout (all_masks/, all_markups/).
    
    Args:
        input_dir: Input directory containing masks/ and markups/ subdirectories
    
    Returns:
        list: Sample names (without extension)
    """
    # Try our pipeline layout first, fall back to legacy
    masks_dir = os.path.join(input_dir, "masks")
    markups_dir = os.path.join(input_dir, "markups")
    if not os.path.exists(masks_dir):
        masks_dir = os.path.join(input_dir, "all_masks")
    if not os.path.exists(markups_dir):
        markups_dir = os.path.join(input_dir, "all_markups")
    
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory not found (tried masks/ and all_masks/ in {input_dir})")
    if not os.path.exists(markups_dir):
        raise FileNotFoundError(f"Markups directory not found (tried markups/ and all_markups/ in {input_dir})")
    
    # Find mask files
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.nii.gz')]
    
    # Extract sample names and check for matching landmarks
    samples = []
    for mask_file in mask_files:
        sample_name = mask_file.replace('.nii.gz', '')
        landmarks_file = f"{sample_name}.json"
        
        if os.path.exists(os.path.join(markups_dir, landmarks_file)):
            samples.append(sample_name)
        else:
            print(f"  ⚠ Skipping {sample_name}: No matching landmarks file")
    
    return sorted(samples)


def run_pipeline(input_dir: str, output_dir: str, enable_viz: bool = False,
                 cbj_markups_dir: str = None, samples: List[str] = None,
                 rerun_big_volumes: bool = False) -> pd.DataFrame:
    """
    Run the full ear processing pipeline on all samples.
    
    Args:
        input_dir: Input directory containing Masks/ and Markups/
        output_dir: Output directory for results
        enable_viz: Enable interactive visualizations
        cbj_markups_dir: Optional directory containing CBJ markup files (or None to use input_dir/markups)
        samples: Optional list of specific sample names to process
        rerun_big_volumes: If True, reprocess files with anomalous volume values
    
    Returns:
        DataFrame: Results with metrics for all processed samples
    """
    print(f"\n{'#'*70}")
    print(f"# FULL EAR PROCESSING PIPELINE")
    print(f"{'#'*70}")
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if cbj_markups_dir:
        print(f"CBJ markups directory: {cbj_markups_dir}")
    print(f"Visualization: {'enabled' if enable_viz else 'disabled'}")
    print(f"Rerun big volumes: {'enabled' if rerun_big_volumes else 'disabled'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Discover samples
    if samples is None:
        print(f"\n[DISCOVERY] Finding samples in input directory...")
        samples = discover_samples(input_dir)
    
    print(f"\nFound {len(samples)} samples to process:")
    for s in samples:
        print(f"  - {s}")
    
    if len(samples) == 0:
        print("\nNo samples found. Exiting.")
        return pd.DataFrame()
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "stl"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "vtk"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "markups"), exist_ok=True)
    
    # Determine input subdirectories (our layout: masks/, markups/, stl/
    # legacy layout: all_masks/, all_markups/, all_stl/)
    _masks_subdir = "masks" if os.path.exists(os.path.join(input_dir, "masks")) else "all_masks"
    _markups_subdir = "markups" if os.path.exists(os.path.join(input_dir, "markups")) else "all_markups"
    _stl_subdir = "stl" if os.path.exists(os.path.join(input_dir, "stl")) else "all_stl"
    
    # Process each sample
    results = []
    successful = 0
    failed = 0
    failed_samples_with_errors = {}  # Dict to store error messages
    
    for i, sample_name in enumerate(samples, 1):
        print(f"\n{'*'*70}")
        print(f"SAMPLE [{i}/{len(samples)}]: {sample_name}")
        print(f"{'*'*70}")
        
        mask_path = os.path.join(input_dir, _masks_subdir, f"{sample_name}.nii.gz")
        landmarks_path = os.path.join(input_dir, _markups_subdir, f"{sample_name}.json")
        stl_path = os.path.join(input_dir, _stl_subdir, f"{sample_name}.stl")
        
        # CBJ landmarks are now inside the unified markup JSON (labels CBJ1-CBJ4).
        # Pass the same landmarks JSON as the CBJ source.
        cbj_landmarks_path = landmarks_path
        
        metrics, error_msg = process_single_sample(
            sample_name=sample_name,
            mask_path=mask_path,
            landmarks_path=landmarks_path,
            stl_path=stl_path,
            cbj_landmarks_path=cbj_landmarks_path,
            output_dir=output_dir,
            enable_viz=enable_viz,
            rerun_big_volumes=rerun_big_volumes
        )
        
        if metrics is not None:
            results.append(metrics)
            successful += 1
        else:
            failed += 1
            failed_samples_with_errors[sample_name] = error_msg if error_msg else "Unknown error"
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results CSV
    csv_path = os.path.join(output_dir, "processing_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Write failure report if there are failures
    if failed_samples_with_errors:
        failure_report_path = os.path.join(output_dir, "processing_failures.txt")
        with open(failure_report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PROCESSING FAILURE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total failed samples: {len(failed_samples_with_errors)}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("FAILED SAMPLES AND ERROR REASONS:\n")
            f.write("-" * 80 + "\n\n")
            
            for idx, (sample_name, error_msg) in enumerate(failed_samples_with_errors.items(), 1):
                f.write(f"{idx}. {sample_name}\n")
                f.write(f"   Error: {error_msg}\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"✓ Failure report saved to: {failure_report_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_samples_with_errors:
        print(f"\nFailed samples:")
        for name, error_msg in failed_samples_with_errors.items():
            print(f"  - {name}")
            print(f"    Reason: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}")
    
    if len(df_results) > 0:
        print(f"\nMetrics summary:")
        print(f"  Centerline length: {df_results['centerline_length'].mean():.2f} ± {df_results['centerline_length'].std():.2f} mm")
        print(f"  Centerline tortuosity: {df_results['centerline_tortuosity'].mean():.3f} ± {df_results['centerline_tortuosity'].std():.3f}")
        print(f"  Geodesic length: {df_results['geodesic_length'].mean():.2f} ± {df_results['geodesic_length'].std():.2f} mm")
        print(f"  Geodesic tortuosity: {df_results['geodesic_tortuosity'].mean():.3f} ± {df_results['geodesic_tortuosity'].std():.3f}")
        
        # Cross-sectional features (if available)
        if 'min_radius_mean' in df_results.columns:
            print(f"\n  Cross-sectional features:")
            print(f"    Min radius: {df_results['min_radius_mean'].mean():.2f} ± {df_results['min_radius_mean'].std():.2f} mm")
            print(f"    Max radius: {df_results['max_radius_mean'].mean():.2f} ± {df_results['max_radius_mean'].std():.2f} mm")
            print(f"    Area: {df_results['area_mean'].mean():.2f} ± {df_results['area_mean'].std():.2f} mm²")
        
        # Aspect ratio (if available)
        if 'aspect_ratio_mean' in df_results.columns:
            valid_ar = df_results['aspect_ratio_mean'].dropna()
            if len(valid_ar) > 0:
                print(f"\n  Aspect ratio (all cross-sections):")
                print(f"    Mean: {valid_ar.mean():.3f} ± {valid_ar.std():.3f}")
        
        if 'landmark_aspect_ratio_mean' in df_results.columns:
            valid_lar = df_results['landmark_aspect_ratio_mean'].dropna()
            if len(valid_lar) > 0:
                print(f"\n  Aspect ratio (landmarks):")
                print(f"    Mean: {valid_lar.mean():.3f} ± {valid_lar.std():.3f}")
    
    print(f"\n{'='*70}\n")
    
    return df_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Full Ear Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage — point --input to the Results/ directory from generate_results.py
    python run_pipeline.py --input /path/to/Results --output /path/to/metric_output
    
    # With visualization
    python run_pipeline.py --input /path/to/Results --output /path/to/metric_output --viz
    
    # With specific samples
    python run_pipeline.py --input /path/to/Results --output /path/to/metric_output --samples CHUM-001_left CHUM-001_right
    
    # With a single specific file
    python run_pipeline.py --input /path/to/Results --output /path/to/metric_output --file CHUM-001_left
    
Expected input structure (from generate_results.py output):
    Results/
    ├── masks/        # NIfTI tissue/air masks (.nii.gz)
    ├── markups/      # Unified landmark JSONs (canal 1-7, FH 8-10, CBJ 11-14)
    └── stl/          # Tissue/air STL meshes
    
Output structure:
    metric_output/
    ├── markups/              # Enriched landmarks + metrics JSON
    ├── vtk/                  # VTK centerline and geodesic files
    ├── stl/                  # Generated STL mesh files
    ├── masks/                # Inverted masks
    ├── processing_results.csv
    └── processing_failures.txt (if any failures)
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory containing masks/ and markups/ subdirectories (e.g., the Results/ folder from generate_results.py)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--viz', action='store_true',
                        help='Enable interactive visualizations')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='Specific sample name/filename to process (processes only this file)')
    parser.add_argument('--samples', nargs='+', type=str, default=None,
                        help='Specific sample names to process (optional)')
    parser.add_argument('--rerun-big-volumes', action='store_true',
                        help='Reprocess files with anomalous volume values (isthmus_eardrum > 1500 or volume_hard+volume_soft significantly different from isthmus_eardrum)')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    # Handle --file argument (takes priority over --samples)
    samples_to_process = None
    if args.file:
        samples_to_process = [args.file]
        print(f"Processing single file: {args.file}")
    elif args.samples:
        samples_to_process = args.samples
        print(f"Processing {len(samples_to_process)} specified sample(s)")
    
    # Run pipeline
    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        enable_viz=args.viz,
        samples=samples_to_process,
        rerun_big_volumes=args.rerun_big_volumes
    )


if __name__ == "__main__":
    main()
