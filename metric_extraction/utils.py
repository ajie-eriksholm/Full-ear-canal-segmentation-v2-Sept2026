# -*- coding: utf-8 -*-
"""
Utility functions for ear canal processing pipeline.

Contains shared functions for:
- Mask loading and inversion
- Mesh creation and manipulation
- Landmark I/O
- Centerline and geodesic computation (VMTK)
- Path metrics computation
- Cross-sectional feature extraction
- Volume computation (voxel-based)
- CBJ (Cartilaginous-Bony Junction) processing
- Visualization helpers
"""

import json
import os
import traceback
from itertools import groupby
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import SimpleITK as sitk
import trimesh
import vtk
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter
from skimage import measure
from vmtk import vmtkscripts
from vtk.util import numpy_support

from constants import MIN_CENTERLINE_LENGTH_MM, MIN_ENDPOINT_DISTANCE_MM

# =============================================================================
# Voxel-based Volume Computation
# =============================================================================

def ensure_mesh_is_single_solid(mesh: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    Ensure mesh is a single solid object with no disconnected components.
    Extracts largest connected component if needed.
    
    Args:
        mesh: vtkPolyData mesh that may have disconnected parts
    
    Returns:
        Single connected, closed mesh
    """
    print(f"  → Verifying single solid mesh...")
    
    # Use vtkConnectivityFilter to find connected components
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputConnection(connectivity.GetOutputPort())
    geo_filter.Update()
    
    result = geo_filter.GetOutput()
    if result.GetNumberOfCells() < mesh.GetNumberOfCells():
        removed = mesh.GetNumberOfCells() - result.GetNumberOfCells()
        print(f"  ⚠ Removed {removed} cells from disconnected components")
    else:
        print(f"  ✓ Single connected volume")
    
    return result


def compute_voxel_mesh_volume(
    mesh_or_path: object,
    voxel_spacing: float = 0.05,
    save_cleaned_stl: Optional[str] = None,
    return_mesh: bool = False,
    verbose: bool = True
) -> Tuple[float, Optional[vtk.vtkPolyData]]:
    """
    Compute mesh volume using voxel-based interior counting method.
    
    This method is robust for meshes with complex topology:
    1. Cleans and triangulates the input mesh
    2. Closes any boundary openings (iterative hole-filling + fallback capping)
    3. Orients normals outward
    4. Converts mesh surface to voxel stencil
    5. Counts interior voxels and calculates volume
    
    Args:
        mesh_or_path: Either a vtkPolyData mesh or path to STL file
        voxel_spacing: Voxel size in mm (default: 0.05 for fine detail)
        save_cleaned_stl: Optional path to save the cleaned/closed mesh as STL
        return_mesh: If True, return cleaned mesh along with volume
        verbose: Print progress information
    
    Returns:
        Tuple of (volume_mm3, cleaned_mesh or None)
    
    Raises:
        ValueError: If input is invalid or volume computation fails
        IOError: If STL file cannot be read
    """
    
    # Load mesh if path provided
    if isinstance(mesh_or_path, str):
        if not os.path.exists(mesh_or_path):
            raise IOError(f"STL file not found: {mesh_or_path}")
        if verbose:
            print(f"  → Loading mesh from: {mesh_or_path}")
        mesh = pv.read(mesh_or_path)
        mesh_working = mesh.extract_surface()
    else:
        mesh_working = mesh_or_path
    
    if not isinstance(mesh_working, (vtk.vtkPolyData, pv.PolyData)):
        raise ValueError("Input must be vtkPolyData, PyVista mesh, or path to STL file")
    
    # Convert PyVista to VTK if needed
    if isinstance(mesh_working, pv.PolyData):
        mesh_working = mesh_working.extract_surface()
    
    if verbose:
        print(f"  ✓ Input mesh: {mesh_working.GetNumberOfCells()} cells, {mesh_working.GetNumberOfPoints()} points")
    
    # STEP 1: Clean and triangulate
    if verbose:
        print(f"  → Cleaning mesh...")
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(mesh_working)
    cleaner.SetTolerance(0.0001)
    cleaner.Update()
    mesh_working = cleaner.GetOutput()
    
    if verbose:
        print(f"  ✓ After cleaning: {mesh_working.GetNumberOfCells()} cells")
    
    # Ensure triangles only
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(mesh_working)
    triangle_filter.Update()
    mesh_working = triangle_filter.GetOutput()
    
    # STEP 2: Check for and close boundaries
    if verbose:
        print(f"  → Checking for boundary edges...")
    
    boundary_check = vtk.vtkFeatureEdges()
    boundary_check.SetInputData(mesh_working)
    boundary_check.BoundaryEdgesOn()
    boundary_check.FeatureEdgesOff()
    boundary_check.NonManifoldEdgesOff()
    boundary_check.ManifoldEdgesOff()
    boundary_check.Update()
    
    num_boundaries = boundary_check.GetOutput().GetNumberOfCells()
    
    if num_boundaries > 0:
        if verbose:
            print(f"  ⚠ Found {num_boundaries} boundary edges - closing mesh...")
        
        # Iterative hole-filling
        for fill_iter in range(3):
            hole_filler = vtk.vtkFillHolesFilter()
            hole_filler.SetInputData(mesh_working)
            hole_filler.SetHoleSize(1000)
            hole_filler.Update()
            mesh_working = hole_filler.GetOutput()
            
            # Check boundaries again
            boundary_check = vtk.vtkFeatureEdges()
            boundary_check.SetInputData(mesh_working)
            boundary_check.BoundaryEdgesOn()
            boundary_check.FeatureEdgesOff()
            boundary_check.NonManifoldEdgesOff()
            boundary_check.ManifoldEdgesOff()
            boundary_check.Update()
            num_boundaries = boundary_check.GetOutput().GetNumberOfCells()
            
            if verbose:
                print(f"  → Hole-fill iteration {fill_iter + 1}: {num_boundaries} boundaries remain")
            
            if num_boundaries == 0:
                if verbose:
                    print(f"  ✓ Mesh closed after {fill_iter + 1} hole-fill iterations")
                break
        
        # Fallback: cap any remaining boundaries
        if num_boundaries > 0:
            if verbose:
                print(f"  → Applying fallback boundary capping...")
            
            for cap_iter in range(3):
                boundary_check = vtk.vtkFeatureEdges()
                boundary_check.SetInputData(mesh_working)
                boundary_check.BoundaryEdgesOn()
                boundary_check.FeatureEdgesOff()
                boundary_check.NonManifoldEdgesOff()
                boundary_check.ManifoldEdgesOff()
                boundary_check.Update()
                boundary_polydata = boundary_check.GetOutput()
                
                if boundary_polydata.GetNumberOfCells() == 0:
                    break
                
                boundary_points = boundary_polydata.GetPoints()
                num_points = boundary_points.GetNumberOfPoints()
                
                if num_points < 3:
                    break
                
                # Compute center of boundary
                center = [0, 0, 0]
                for i in range(num_points):
                    p = boundary_points.GetPoint(i)
                    center[0] += p[0]
                    center[1] += p[1]
                    center[2] += p[2]
                center[0] /= num_points
                center[1] /= num_points
                center[2] /= num_points
                
                # Create triangular fan from boundary center
                vertices = []
                faces = []
                vertices.append(center)
                for i in range(num_points):
                    p = boundary_points.GetPoint(i)
                    vertices.append([p[0], p[1], p[2]])
                for i in range(num_points):
                    faces.extend([3, 0, i + 1, (i + 1) % num_points + 1])
                
                cap_mesh = pv.PolyData()
                cap_mesh.points = np.array(vertices)
                cap_mesh.faces = np.array(faces)
                
                # Combine meshes
                mesh_pv = pv.wrap(mesh_working)
                combined = mesh_pv + cap_mesh
                mesh_working = combined.extract_surface()
                
                # Check boundaries again
                boundary_check = vtk.vtkFeatureEdges()
                boundary_check.SetInputData(mesh_working)
                boundary_check.BoundaryEdgesOn()
                boundary_check.FeatureEdgesOff()
                boundary_check.NonManifoldEdgesOff()
                boundary_check.ManifoldEdgesOff()
                boundary_check.Update()
                num_boundaries = boundary_check.GetOutput().GetNumberOfCells()
                
                if verbose:
                    print(f"  → Cap iteration {cap_iter + 1}: {num_boundaries} boundaries remain")
                
                if num_boundaries == 0:
                    if verbose:
                        print(f"  ✓ Mesh closed with fallback capping")
                    break
    else:
        if verbose:
            print(f"  ✓ Mesh is already closed")
    
    # STEP 3: Get mesh bounds and compute voxel spacing
    bounds = mesh_working.GetBounds()
    diagonal = np.sqrt((bounds[1] - bounds[0])**2 + 
                      (bounds[3] - bounds[2])**2 + 
                      (bounds[5] - bounds[4])**2)
    
    if verbose:
        print(f"  ✓ Mesh diagonal: {diagonal:.2f} mm")
        print(f"  ✓ Voxel spacing: {voxel_spacing:.4f} mm")
    
    # Create voxel grid bounds (with small padding)
    padding = voxel_spacing * 2
    voxel_bounds = [
        bounds[0] - padding, bounds[1] + padding,
        bounds[2] - padding, bounds[3] + padding,
        bounds[4] - padding, bounds[5] + padding
    ]
    
    # Calculate voxel grid dimensions
    nx = int((voxel_bounds[1] - voxel_bounds[0]) / voxel_spacing) + 1
    ny = int((voxel_bounds[3] - voxel_bounds[2]) / voxel_spacing) + 1
    nz = int((voxel_bounds[5] - voxel_bounds[4]) / voxel_spacing) + 1
    
    if verbose:
        print(f"  ✓ Grid dimensions: {nx} × {ny} × {nz} = {nx*ny*nz:,} voxels")
    
    # STEP 4: Prepare mesh for containment testing
    if verbose:
        print(f"  → Preparing mesh for containment testing...")
    
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(mesh_working)
    cleaner.SetTolerance(0.0001)
    cleaner.Update()
    mesh_for_test = cleaner.GetOutput()
    
    # Ensure single solid
    if verbose:
        print(f"  → Ensuring single solid mesh...")
    mesh_for_test = ensure_mesh_is_single_solid(mesh_for_test)
    
    # Orient normals outward (critical for containment testing)
    if verbose:
        print(f"  → Orienting normals outward...")
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh_for_test)
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    mesh_for_test = normals.GetOutput()
    
    # STEP 5: Voxel containment testing using stencil
    if verbose:
        print(f"  → Converting mesh to voxel stencil...")
    
    # Create image data (voxel grid)
    voxel_grid = vtk.vtkImageData()
    voxel_grid.SetDimensions(nx, ny, nz)
    voxel_grid.SetSpacing(voxel_spacing, voxel_spacing, voxel_spacing)
    voxel_grid.SetOrigin(voxel_bounds[0], voxel_bounds[2], voxel_bounds[4])
    
    total_points = nx * ny * nz
    
    # Convert mesh to stencil
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(mesh_for_test)
    poly_to_stencil.SetOutputOrigin(voxel_bounds[0], voxel_bounds[2], voxel_bounds[4])
    poly_to_stencil.SetOutputSpacing(voxel_spacing, voxel_spacing, voxel_spacing)
    poly_to_stencil.SetOutputWholeExtent(0, nx-1, 0, ny-1, 0, nz-1)
    poly_to_stencil.Update()
    
    stencil_data = poly_to_stencil.GetOutput()
    
    # Create output image data to mark interior points
    if verbose:
        print(f"  → Computing interior voxels...")
    
    interior_image = vtk.vtkImageData()
    interior_image.SetDimensions(nx, ny, nz)
    interior_image.SetSpacing(voxel_spacing, voxel_spacing, voxel_spacing)
    interior_image.SetOrigin(voxel_bounds[0], voxel_bounds[2], voxel_bounds[4])
    
    # Initialize with scalar data (all 255 = all white/interior)
    scalars_array = np.full(total_points, 255, dtype=np.uint8)
    scalars = numpy_support.numpy_to_vtk(scalars_array, deep=True)
    scalars.SetNumberOfComponents(1)
    interior_image.GetPointData().SetScalars(scalars)
    
    # Use stencil to mark exterior as 0, interior stays 255
    stencil_filter = vtk.vtkImageStencil()
    stencil_filter.SetInputData(interior_image)
    stencil_filter.SetStencilData(stencil_data)
    stencil_filter.SetBackgroundValue(0)  # Exterior = 0
    stencil_filter.Update()
    
    output_image = stencil_filter.GetOutput()
    
    # Get the scalars and count interior voxels (255 = interior)
    output_scalars = output_image.GetPointData().GetScalars()
    inside_count = 0
    if output_scalars:
        for i in range(total_points):
            if output_scalars.GetValue(i) > 128:  # > 128 means interior
                inside_count += 1
    
    if verbose:
        print(f"  ✓ Interior voxels: {inside_count:,} / {total_points:,}")
    
    # Calculate volume
    if inside_count == 0:
        if verbose:
            print(f"  ⚠ No interior voxels found - containment test may have failed")
            print(f"  → Falling back to trimesh/divergence theorem volume calculation...")
        
        # Fall back to traditional methods
        trimesh_volume = calculate_mesh_volume_trimesh(mesh_for_test, method="auto")
        divergence_volume = calculate_mesh_volume_divergence(mesh_for_test)
        
        if trimesh_volume is not None and trimesh_volume > 0:
            total_volume = trimesh_volume
            if verbose:
                print(f"  ✓ Using trimesh result: {total_volume:.2f} mm³")
        elif divergence_volume is not None and divergence_volume > 0:
            total_volume = divergence_volume
            if verbose:
                print(f"  ✓ Using divergence theorem result: {total_volume:.2f} mm³")
        else:
            if verbose:
                print(f"  ✗ All volume methods failed")
            total_volume = 0.0
    else:
        # Calculate volume from voxel count
        voxel_volume = voxel_spacing ** 3
        total_volume = inside_count * voxel_volume
        if verbose:
            print(f"  ✓ Volume per voxel: {voxel_volume:.6f} mm³")
    
    if verbose:
        print(f"  ✓ Total volume: {total_volume:.2f} mm³")
    
    # Save cleaned mesh if requested
    if save_cleaned_stl is not None:
        if verbose:
            print(f"  → Saving cleaned mesh to: {save_cleaned_stl}")
        os.makedirs(os.path.dirname(save_cleaned_stl), exist_ok=True)
        pv.wrap(mesh_for_test).save(save_cleaned_stl)
        if verbose:
            print(f"  ✓ Saved")
    
    # Return volume and optionally mesh
    if return_mesh:
        return total_volume, mesh_for_test
    else:
        return total_volume, None


def calculate_isthmus_to_eardrum_length_and_tortuosity(
    centerline: vtk.vtkPolyData,
    isthmus_point: np.ndarray,
    eardrum_point: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate the length and tortuosity of the centerline segment from isthmus to eardrum.

    Args:
        centerline: Centerline as VTK PolyData (ordered points).
        isthmus_point: 3D coordinates of the isthmus landmark.
        eardrum_point: 3D coordinates of the eardrum landmark.

    Returns:
        Tuple of (segment_length, tortuosity).
    Raises:
        ValueError: If centerline or points are invalid.
    """
    if centerline is None or centerline.GetNumberOfPoints() == 0:
        raise ValueError("Centerline is empty.")
    if isthmus_point is None or eardrum_point is None:
        raise ValueError("Landmark points are missing.")

    vtk_points = centerline.GetPoints()
    points = np.array([vtk_points.GetPoint(i) for i in range(centerline.GetNumberOfPoints())])

    # Find closest centerline indices to isthmus and eardrum
    isthmus_idx = np.argmin(np.linalg.norm(points - isthmus_point, axis=1))
    eardrum_idx = np.argmin(np.linalg.norm(points - eardrum_point, axis=1))

    # Ensure correct order (isthmus to eardrum)
    start_idx, end_idx = sorted([isthmus_idx, eardrum_idx])
    segment = points[start_idx:end_idx + 1]

    # Compute segment length
    segment_length = np.sum(
        np.linalg.norm(segment[1:] - segment[:-1], axis=1)
    )
    # Compute straight-line distance
    straight_distance = np.linalg.norm(segment[0] - segment[-1])

    # Avoid division by zero
    tortuosity = segment_length / straight_distance if straight_distance > 0 else np.nan

    return segment_length, tortuosity

def visualize_full_results(
    mesh: vtk.vtkPolyData,
    centerline: vtk.vtkPolyData,
    geodesic_path: vtk.vtkPolyData,
    middle_rs: np.ndarray,
    eardrum: np.ndarray,
    top_rs: np.ndarray,
    bottom_rs: np.ndarray,
    trimmed_start: np.ndarray = None,
    plane_normal: np.ndarray = None,
    plane_centroid: np.ndarray = None,
    offset_plane_centroid: np.ndarray = None,
    mask_boundary_point: np.ndarray = None,
    landmark_cross_sections: Dict[str, Dict] = None,
    landmark_positions: Dict[str, np.ndarray] = None,
    cbj_points: np.ndarray = None,
    cbj_plane_normal: np.ndarray = None,
    cbj_plane_centroid: np.ndarray = None,
    cbj_closest_point: np.ndarray = None,
    cbj_cross_section: vtk.vtkPolyData = None,
    clipped_volume_mesh: vtk.vtkPolyData = None,
    title: str = "Full Results Visualization"
) -> None:
    """
    Visualize both the main results and landmark cross-sections in sequence.
    Calls visualize_results and visualize_landmark_cross_sections.

    Args:
        mesh: surface mesh
        centerline: centerline polydata
        geodesic_path: geodesic path polydata
        middle_rs: middle RS position
        eardrum: eardrum position
        top_rs: top RS position
        bottom_rs: bottom RS position
        trimmed_start: trimmed start position (optional)
        plane_normal: normal vector of the trimming plane (optional)
        plane_centroid: centroid point of the geodesic plane / beginning_eac (optional)
        offset_plane_centroid: centroid of offset plane / beginning_centerline (optional)
        mask_boundary_point: outermost mask point along plane normal (optional)
        landmark_cross_sections: Output from extract_landmark_cross_sections
        landmark_positions: Dictionary of landmark positions
        cbj_points: Nx3 array of CBJ landmark positions (optional)
        cbj_plane_normal: Normal of best-fit CBJ plane (optional)
        cbj_plane_centroid: Centroid of CBJ plane (optional)
        cbj_closest_point: Closest centerline point to CBJ plane (optional)
        title: Window title
    """
    # Combined visualization: mesh, centerline, geodesic, landmarks, planes, cross-sections, CBJ
    print(f"\n  Opening combined visualization window: {title}")
    import vtk
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)

    # Add mesh (semi-transparent)
    mapper_mesh = vtk.vtkPolyDataMapper()
    mapper_mesh.SetInputData(mesh)
    actor_mesh = vtk.vtkActor()
    actor_mesh.SetMapper(mapper_mesh)
    actor_mesh.GetProperty().SetOpacity(0.3)
    actor_mesh.GetProperty().SetColor(0.8, 0.8, 0.8)
    renderer.AddActor(actor_mesh)

    # Add centerline (red tube)
    cl_mapper = vtk.vtkPolyDataMapper()
    cl_mapper.SetInputData(centerline)
    cl_actor = vtk.vtkActor()
    cl_actor.SetMapper(cl_mapper)
    cl_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    cl_actor.GetProperty().SetLineWidth(3)
    renderer.AddActor(cl_actor)

    # Add geodesic path (blue line)
    if geodesic_path is not None:
        geo_mapper = vtk.vtkPolyDataMapper()
        geo_mapper.SetInputData(geodesic_path)
        geo_actor = vtk.vtkActor()
        geo_actor.SetMapper(geo_mapper)
        geo_actor.GetProperty().SetColor(0.0, 0.0, 1.0)
        geo_actor.GetProperty().SetLineWidth(2)
        renderer.AddActor(geo_actor)

    # Add clipped volume (light cyan, semi-transparent) - visualization of computed volume
    if clipped_volume_mesh is not None and clipped_volume_mesh.GetNumberOfCells() > 0:
        volume_mapper = vtk.vtkPolyDataMapper()
        volume_mapper.SetInputData(clipped_volume_mesh)
        volume_actor = vtk.vtkActor()
        volume_actor.SetMapper(volume_mapper)
        volume_actor.GetProperty().SetOpacity(0.4)
        volume_actor.GetProperty().SetColor(0.0, 1.0, 1.0)  # Cyan
        renderer.AddActor(volume_actor)

    # Add landmark spheres (middle_rs, eardrum, top_rs, bottom_rs, trimmed_start, etc.)
    def add_sphere(pos, color, radius=1.5, opacity=1.0):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(pos)
        sphere.SetRadius(radius)
        sphere.Update()
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputData(sphere.GetOutput())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(*color)
        sphere_actor.GetProperty().SetOpacity(opacity)
        renderer.AddActor(sphere_actor)

    add_sphere(middle_rs, (0.0, 0.0, 0.0), 1.2, 1.0)  # Black
    add_sphere(eardrum, (1.0, 0.0, 1.0), 1.2, 1.0)     # Magenta
    add_sphere(top_rs, (0.0, 0.0, 1.0), 1.2, 1.0)      # Blue
    add_sphere(bottom_rs, (0.0, 1.0, 0.0), 1.2, 1.0)   # Green
    if trimmed_start is not None:
        add_sphere(trimmed_start, (1.0, 0.5, 0.0), 1.0, 1.0)  # Orange
    if mask_boundary_point is not None:
        add_sphere(mask_boundary_point, (0.5, 0.5, 0.5), 1.0, 1.0)

    # Add cross-sectional curves and landmark spheres from cross-section data
    if landmark_cross_sections is not None and landmark_positions is not None:
        colors = {
            '1st_bend': (1.0, 0.5, 0.0),      # Orange
            '2nd_bend': (0.0, 1.0, 0.5),      # Green
            'eardrum': (1.0, 0.0, 1.0),       # Magenta
            'isthmus': (1.0, 1.0, 0.0),       # Yellow
        }
        for name, data in landmark_cross_sections.items():
            color = colors.get(name, (1.0, 1.0, 0.0))
            point = data['point']
            # Add cross-section curve if available
            if data.get('cross_section') is not None:
                curve_mapper = vtk.vtkPolyDataMapper()
                curve_mapper.SetInputData(data['cross_section'])
                curve_actor = vtk.vtkActor()
                curve_actor.SetMapper(curve_mapper)
                curve_actor.GetProperty().SetColor(*color)
                curve_actor.GetProperty().SetLineWidth(4)
                renderer.AddActor(curve_actor)
            # Add sphere at centerline point
            add_sphere(point, color, 1.5, 1.0)
        # Add original landmark positions as small spheres
        landmark_colors = {
            '1st bend': (1.0, 0.5, 0.0),
            '2nd bend': (0.0, 1.0, 0.5),
            'beginning_eac': (0.0, 0.5, 1.0),
            'Eardrum': (1.0, 0.0, 1.0),
        }
        for name, pos in landmark_positions.items():
            if name in landmark_colors:
                add_sphere(pos, landmark_colors[name], 0.8, 0.7)

    # Add CBJ visualization if available
    if cbj_points is not None and len(cbj_points) > 0:
        # Add CBJ landmark points (purple spheres)
        for i, cbj_pt in enumerate(cbj_points):
            add_sphere(cbj_pt, (0.8, 0.2, 0.8), 1.0, 0.9)  # Purple
    
    if cbj_plane_normal is not None and cbj_plane_centroid is not None:
        # Draw CBJ plane as a disk
        try:
            plane_normal_norm = cbj_plane_normal / np.linalg.norm(cbj_plane_normal)
            
            # Create orthonormal basis for the plane
            if abs(plane_normal_norm[0]) < 0.9:
                v1 = np.array([1.0, 0.0, 0.0])
            else:
                v1 = np.array([0.0, 1.0, 0.0])
            
            v1 = v1 - np.dot(v1, plane_normal_norm) * plane_normal_norm
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(plane_normal_norm, v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Create disk geometry
            n_segments = 32
            radius = 5.0
            disk_points = vtk.vtkPoints()
            disk_cells = vtk.vtkCellArray()
            
            # Center point
            disk_points.InsertNextPoint(cbj_plane_centroid)
            
            # Ring of points around the plane
            for i in range(n_segments):
                angle = 2 * np.pi * i / n_segments
                point = (cbj_plane_centroid + 
                        radius * np.cos(angle) * v1 + 
                        radius * np.sin(angle) * v2)
                disk_points.InsertNextPoint(point)
            
            # Create triangles from center to ring
            for i in range(n_segments):
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, 0)
                triangle.GetPointIds().SetId(1, 1 + i)
                triangle.GetPointIds().SetId(2, 1 + (i + 1) % n_segments)
                disk_cells.InsertNextCell(triangle)
            
            disk_polydata = vtk.vtkPolyData()
            disk_polydata.SetPoints(disk_points)
            disk_polydata.SetPolys(disk_cells)
            
            disk_mapper = vtk.vtkPolyDataMapper()
            disk_mapper.SetInputData(disk_polydata)
            disk_actor = vtk.vtkActor()
            disk_actor.SetMapper(disk_mapper)
            disk_actor.GetProperty().SetColor(0.8, 0.2, 0.8)  # Purple
            disk_actor.GetProperty().SetOpacity(0.3)
            renderer.AddActor(disk_actor)
        except Exception as e:
            print(f"    Warning: Could not draw CBJ plane: {e}")
    
    # Add CBJ cross-section curve if available
    if cbj_cross_section is not None:
        try:
            cbj_curve_mapper = vtk.vtkPolyDataMapper()
            cbj_curve_mapper.SetInputData(cbj_cross_section)
            cbj_curve_actor = vtk.vtkActor()
            cbj_curve_actor.SetMapper(cbj_curve_mapper)
            cbj_curve_actor.GetProperty().SetColor(0.8, 0.2, 0.8)  # Purple
            cbj_curve_actor.GetProperty().SetLineWidth(5)
            renderer.AddActor(cbj_curve_actor)
        except Exception as e:
            print(f"    Warning: Could not draw CBJ cross-section: {e}")
    
    if cbj_closest_point is not None:
        add_sphere(cbj_closest_point, (0.8, 0.2, 0.8), 1.5, 1.0)  # Purple - CBJ intersection

    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName(title)
    render_window.SetSize(1200, 800)
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    renderer.ResetCamera()
    render_window.Render()

    print(f"\n  Legend:")
    print(f"    - Gray transparent: Mesh surface")
    print(f"    - Red line: Centerline")
    print(f"    - Blue line: Geodesic path")
    print(f"    - Orange/Green/Magenta/Yellow: Cross-sections and landmarks")
    if cbj_points is not None or cbj_plane_normal is not None:
        print(f"    - Purple disk: CBJ plane")
        print(f"    - Purple small spheres: CBJ landmark points (4 points)")
        print(f"    - Purple line: CBJ cross-section curve")
        print(f"    - Purple large sphere: CBJ-centerline intersection point")
    print(f"\n  Close window to continue...")

    interactor.Start()


# =============================================================================
# Mask Loading and Processing
# =============================================================================
def create_surface_mesh_from_mask(sitk_image: sitk.Image) -> vtk.vtkPolyData:
    """
    Create 3D surface mesh from SimpleITK binary mask using marching cubes.

    Extracts only the surface at 1/0 transitions, excluding the edges of the image.
    This prevents surfaces at the image boundary, so only internal object surfaces
    are extracted.

    Args:
        sitk_image: SimpleITK binary mask image with physical properties

    Returns:
        vtkPolyData: surface mesh in physical coordinates

    Raises:
        ValueError: if mask is empty or too small for surface extraction
    """
    print(f"  Creating 3D mesh from mask (surface only at 1/0 transitions, no edges)")
    print(f"    Image size: {sitk_image.GetSize()}")
    print(f"    Spacing: {sitk_image.GetSpacing()}")

    # Get numpy array from SimpleITK image
    np_array = sitk.GetArrayFromImage(sitk_image)
    spacing = np.array(sitk_image.GetSpacing())
    origin = np.array(sitk_image.GetOrigin())

    print(f"    Array shape (z, y, x): {np_array.shape}")

    # Ensure mask is binary
    mask_binary = np_array > 0

    # Remove edges: set border voxels to 0 so marching cubes doesn't create surface at image edges
    for axis in range(mask_binary.ndim):
        mask_binary = np.copy(mask_binary)
        mask_binary = np.swapaxes(mask_binary, 0, axis)
        mask_binary[0, ...] = 0
        mask_binary[-1, ...] = 0
        mask_binary = np.swapaxes(mask_binary, 0, axis)

    # Check if mask is empty after removing edges
    if not np.any(mask_binary):
        raise ValueError("Mask is empty after removing edges; cannot extract surface.")

    # Apply Gaussian smoothing for smoother surface
    print(f"    Applying Gaussian smoothing for smoother surface...")
    sigma = 1.5
    mask_smoothed = gaussian_filter(mask_binary.astype(float), sigma=sigma)

    # Use marching cubes to extract surface mesh
    print(f"    Running marching cubes algorithm...")
    verts, faces, normals, _ = measure.marching_cubes(
        mask_smoothed,
        level=0.5,
        spacing=(1.0, 1.0, 1.0)
    )

    print(f"    Generated mesh with {len(verts)} vertices and {len(faces)} faces")

    # Create trimesh object
    trimesh_obj = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Apply Laplacian smoothing
    print(f"    Applying Laplacian smoothing to mesh...")
    trimesh.smoothing.filter_laplacian(trimesh_obj, iterations=5)

    # Transform vertices to physical coordinates
    print(f"    Transforming vertices to physical coordinates...")

    # Reorder from (z,y,x) to (x,y,z)
    verts_xyz_indices = trimesh_obj.vertices[:, [2, 1, 0]]

    # Apply spacing and origin
    verts_final = verts_xyz_indices * spacing + origin

    # Convert to VTK PolyData
    print(f"    Converting to VTK format...")
    points = vtk.vtkPoints()
    for v in verts_final:
        points.InsertNextPoint(v)

    triangles = vtk.vtkCellArray()
    for face in trimesh_obj.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, face[0])
        triangle.GetPointIds().SetId(1, face[1])
        triangle.GetPointIds().SetId(2, face[2])
        triangles.InsertNextCell(triangle)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)

    # Clean the mesh
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(mesh)
    cleaner.Update()

    mesh_cleaned = cleaner.GetOutput()

    # Compute and orient normals
    print(f"    Computing and orienting normals for surface...")
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(mesh_cleaned)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.ConsistencyOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.Update()

    mesh_with_normals = normals_filter.GetOutput()

    print(f"  ✓ Mesh created: {mesh_with_normals.GetNumberOfPoints()} points, {mesh_with_normals.GetNumberOfCells()} cells")

    return mesh_with_normals
def load_mask(mask_path: str) -> sitk.Image:
    """
    Load binary mask and preserve physical properties.
    
    Uses SimpleITK to ensure spacing, origin, and direction are preserved.
    
    Args:
        mask_path: path to NIfTI mask file (.nii.gz)
    
    Returns:
        sitk.Image: mask with preserved physical properties
    
    Raises:
        FileNotFoundError: if mask file doesn't exist
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    print(f"  Loading mask: {os.path.basename(mask_path)}")
    
    # Load NIfTI file with SimpleITK (preserves all physical properties)
    image = sitk.ReadImage(mask_path)
    
    print(f"    Mask size: {image.GetSize()}")
    print(f"    Spacing: {image.GetSpacing()}")
    print(f"    Origin: {image.GetOrigin()}")
    
    # Get statistics
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    print(f"    Value range: [{stats.GetMinimum()}, {stats.GetMaximum()}]")
    
    return image


def load_and_invert_mask(mask_path: str) -> sitk.Image:
    """
    Load binary mask and invert it (1→0, 0→1) while preserving physical properties.
    
    Uses SimpleITK to ensure spacing, origin, and direction are preserved.
    
    Args:
        mask_path: path to NIfTI mask file (.nii.gz)
    
    Returns:
        sitk.Image: inverted mask with preserved physical properties
    
    Raises:
        FileNotFoundError: if mask file doesn't exist
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    print(f"  Loading mask: {os.path.basename(mask_path)}")
    
    # Load NIfTI file with SimpleITK (preserves all physical properties)
    image = sitk.ReadImage(mask_path)
    
    print(f"    Mask size: {image.GetSize()}")
    print(f"    Spacing: {image.GetSpacing()}")
    print(f"    Origin: {image.GetOrigin()}")
    
    # Get statistics before inversion
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    print(f"    Value range: [{stats.GetMinimum()}, {stats.GetMaximum()}]")
    
    # Invert the mask: 1→0, 0→1
    inverted_image = 1 - image
    
    # Ensure physical properties are preserved
    inverted_image.SetSpacing(image.GetSpacing())
    inverted_image.SetOrigin(image.GetOrigin())
    inverted_image.SetDirection(image.GetDirection())
    
    # Get statistics after inversion
    stats.Execute(inverted_image)
    print(f"  ✓ Mask inverted")
    print(f"    Inverted value range: [{stats.GetMinimum()}, {stats.GetMaximum()}]")
    
    return inverted_image


def save_inverted_mask(sitk_image: sitk.Image, output_path: str) -> None:
    """
    Save inverted mask as NIfTI file.
    
    Args:
        sitk_image: SimpleITK image to save
        output_path: path to output NIfTI file
    """
    print(f"  Saving inverted mask to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(sitk_image, output_path)
    print(f"  ✓ Inverted mask saved")


# =============================================================================
# Mesh Creation and Manipulation
# =============================================================================

def create_mesh_from_mask(sitk_image: sitk.Image) -> vtk.vtkPolyData:
    """
    Create 3D surface mesh from SimpleITK binary mask using marching cubes.
    
    Uses skimage marching cubes with Gaussian smoothing for better quality mesh,
    then converts to VTK format for VMTK compatibility.
    
    Args:
        sitk_image: SimpleITK binary mask image with physical properties
    
    Returns:
        vtkPolyData: surface mesh in physical coordinates
    """
    print(f"  Creating 3D mesh from inverted mask")
    print(f"    Image size: {sitk_image.GetSize()}")
    print(f"    Spacing: {sitk_image.GetSpacing()}")
    
    # Get numpy array from SimpleITK image
    np_array = sitk.GetArrayFromImage(sitk_image)
    spacing = np.array(sitk_image.GetSpacing())
    origin = np.array(sitk_image.GetOrigin())
    
    print(f"    Array shape (z, y, x): {np_array.shape}")
    
    # Ensure mask is binary
    mask_binary = np_array > 0
    
    # Pad the mask with zeros on all sides to ensure closed surface
    print(f"    Padding mask to ensure closed surface...")
    pad_width = 2
    mask_padded = np.pad(mask_binary, pad_width=pad_width, mode='constant', constant_values=0)
    
    # Apply Gaussian smoothing for smoother surface
    print(f"    Applying Gaussian smoothing for smoother surface...")
    sigma = 1.5
    mask_smoothed = gaussian_filter(mask_padded.astype(float), sigma=sigma)
    
    # Use marching cubes to extract surface mesh
    print(f"    Running marching cubes algorithm...")
    verts, faces, normals, values = measure.marching_cubes(
        mask_smoothed,
        level=0.5,
        spacing=(1.0, 1.0, 1.0)
    )
    
    print(f"    Generated mesh with {len(verts)} vertices and {len(faces)} faces")
    
    # Create trimesh object
    trimesh_obj = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Apply Laplacian smoothing
    print(f"    Applying Laplacian smoothing to mesh...")
    trimesh.smoothing.filter_laplacian(trimesh_obj, iterations=5)
    
    # Transform vertices to physical coordinates
    print(f"    Transforming vertices to physical coordinates...")
    
    # Account for padding offset
    verts_unpadded = trimesh_obj.vertices - pad_width
    
    # Reorder from (z,y,x) to (x,y,z)
    verts_xyz_indices = verts_unpadded[:, [2, 1, 0]]
    
    # Apply spacing and origin
    verts_final = verts_xyz_indices * spacing + origin
    
    # Convert to VTK PolyData
    print(f"    Converting to VTK format...")
    points = vtk.vtkPoints()
    for v in verts_final:
        points.InsertNextPoint(v)
    
    triangles = vtk.vtkCellArray()
    for face in trimesh_obj.faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, face[0])
        triangle.GetPointIds().SetId(1, face[1])
        triangle.GetPointIds().SetId(2, face[2])
        triangles.InsertNextCell(triangle)
    
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)
    
    # Clean the mesh
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(mesh)
    cleaner.Update()
    
    mesh_cleaned = cleaner.GetOutput()
    
    # Compute and orient normals
    print(f"    Computing and orienting normals for closed surface...")
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(mesh_cleaned)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.ConsistencyOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.Update()
    
    mesh_with_normals = normals_filter.GetOutput()
    
    # Fill any holes in the mesh to ensure it's watertight for VMTK
    print(f"    Filling holes to ensure watertight mesh...")
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(mesh_with_normals)
    fill_holes.SetHoleSize(1000.0)  # Large value to fill all holes
    fill_holes.Update()
    
    final_mesh = fill_holes.GetOutput()
    
    # Check if surface is closed
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(final_mesh)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()
    
    num_boundary_edges = feature_edges.GetOutput().GetNumberOfCells()
    
    if num_boundary_edges == 0:
        print(f"    ✓ Surface is closed (manifold)")
    else:
        print(f"    ⚠ Warning: Surface still has {num_boundary_edges} boundary edges after hole filling")
    
    print(f"  ✓ Mesh created: {final_mesh.GetNumberOfPoints()} points, {final_mesh.GetNumberOfCells()} cells")
    
    return final_mesh


def simplify_mesh_for_vmtk(mesh: vtk.vtkPolyData, target_reduction: float = 0.9) -> vtk.vtkPolyData:
    """
    Simplify mesh to reduce complexity for VMTK centerline computation.
    
    Args:
        mesh: input mesh
        target_reduction: fraction of triangles to remove (0.9 = 90% reduction)
    
    Returns:
        vtkPolyData: simplified mesh
    """
    print(f"  Simplifying mesh for VMTK (target reduction: {target_reduction*100:.0f}%)")
    print(f"    Original: {mesh.GetNumberOfPoints()} points, {mesh.GetNumberOfCells()} cells")
    
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(target_reduction)
    decimate.PreserveTopologyOn()
    decimate.Update()
    
    simplified = decimate.GetOutput()
    
    print(f"    Simplified: {simplified.GetNumberOfPoints()} points, {simplified.GetNumberOfCells()} cells")
    print(f"    Reduction: {(1 - simplified.GetNumberOfCells()/mesh.GetNumberOfCells())*100:.1f}%")
    
    return simplified


def save_mesh_as_stl(mesh: vtk.vtkPolyData, output_path: str) -> None:
    """
    Save mesh as STL file.
    
    Args:
        mesh: vtkPolyData mesh
        output_path: path to output STL file
    """
    print(f"  Saving mesh to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(mesh)
    writer.Write()
    
    print(f"  ✓ STL file saved")


# =============================================================================
# Landmark I/O
# =============================================================================

def load_landmarks_from_json(json_path: str) -> Dict[str, np.ndarray]:
    """
    Load landmarks from JSON file.
    
    Expects our pipeline's unified markup format where all landmarks
    (canal 1-7, FH 8-10, CBJ 11-14) are in a single JSON.
    Coordinates are used as-is (no flipping) since our pipeline
    already outputs them in the correct coordinate system.
    
    Args:
        json_path: path to JSON landmark file
    
    Returns:
        dict: mapping from label to position array
    
    Raises:
        FileNotFoundError: if JSON file doesn't exist
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Landmarks file not found: {json_path}")
    
    print(f"  Loading landmarks: {os.path.basename(json_path)}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    landmarks = {}
    for lm in data['landmarks']:
        label = lm['label']
        position_raw = lm['position']
        
        # Handle different position formats
        if position_raw is None:
            print(f"    {label}: None (skipped - null position)")
            continue
        
        position = np.array(position_raw, dtype=float)
        
        # Ensure it's a 3D coordinate
        if position.ndim == 0 or (position.ndim == 1 and len(position) != 3):
            print(f"    {label}: {position_raw} (skipped - invalid format, expected [x, y, z])")
            continue
        
        # Flatten if needed
        if position.ndim > 1:
            position = position.flatten()
        
        # Use coordinates as-is — our pipeline already outputs in the correct system
        landmarks[label] = position
        print(f"    {label}: {position} (x={position[0]:.1f}, y={position[1]:.1f}, z={position[2]:.1f})")
    
    print(f"  ✓ Loaded {len(landmarks)} landmarks")
    
    return landmarks


def save_landmarks_json(landmarks: dict, output_path: str) -> None:
    """
    Save landmarks to JSON file.
    
    Args:
        landmarks: dict mapping from label to position array or Slicer format
        output_path: path to JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(landmarks, f, indent=2)


def create_complete_landmarks_json(original_landmarks: Dict[str, np.ndarray],
                                   beginning_eac: np.ndarray = None,
                                   beginning_centerline: np.ndarray = None,
                                   mask_boundary_point: np.ndarray = None) -> dict:
    """
    Create complete landmarks JSON including original landmarks and computed landmarks.
    
    Args:
        original_landmarks: dict with original landmarks from input JSON
        beginning_eac: 3D coordinates of geodesic plane intersection point
        beginning_centerline: 3D coordinates of offset plane intersection
        mask_boundary_point: 3D coordinates of outermost mask point along plane normal
    
    Returns:
        dict: Slicer-compatible landmarks structure
    """
    landmarks_list = []
    landmark_id = 1
    
    # Add original landmarks
    for label, position in original_landmarks.items():
        landmarks_list.append({
            "id": str(landmark_id),
            "label": label,
            "position": position.tolist() if isinstance(position, np.ndarray) else position
        })
        landmark_id += 1
    
    # Add beginning_eac if it exists
    if beginning_eac is not None:
        landmarks_list.append({
            "id": str(landmark_id),
            "label": "beginning_eac",
            "position": beginning_eac.tolist() if isinstance(beginning_eac, np.ndarray) else beginning_eac
        })
        landmark_id += 1
    
    # Add beginning_centerline if it exists
    if beginning_centerline is not None:
        landmarks_list.append({
            "id": str(landmark_id),
            "label": "beginning_centerline",
            "position": beginning_centerline.tolist() if isinstance(beginning_centerline, np.ndarray) else beginning_centerline
        })
        landmark_id += 1
    
    # Add mask_boundary_point if it exists
    if mask_boundary_point is not None:
        landmarks_list.append({
            "id": str(landmark_id),
            "label": "mask_boundary_point",
            "position": mask_boundary_point.tolist() if isinstance(mask_boundary_point, np.ndarray) else mask_boundary_point
        })
    
    return {"landmarks": landmarks_list}


def compute_middle_rs(top_rs: np.ndarray, bottom_rs: np.ndarray) -> np.ndarray:
    """
    Compute midpoint between top RS and bottom RS.
    
    Args:
        top_rs: top RS position [x, y, z] (numpy array or list)
        bottom_rs: bottom RS position [x, y, z] (numpy array or list)
    
    Returns:
        numpy array: middle RS position
    """
    # Convert to numpy arrays if needed (handle both lists and arrays)
    top_rs = np.array(top_rs) if not isinstance(top_rs, np.ndarray) else top_rs
    bottom_rs = np.array(bottom_rs) if not isinstance(bottom_rs, np.ndarray) else bottom_rs
    
    middle = (top_rs + bottom_rs) / 2.0
    print(f"  Middle RS computed: {middle}")
    return middle


# =============================================================================
# VTK I/O
# =============================================================================

def load_vtk_polydata(vtk_path: str) -> vtk.vtkPolyData:
    """
    Load VTK polydata file.
    
    Args:
        vtk_path: path to VTK file
    
    Returns:
        vtkPolyData: loaded polydata
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    return reader.GetOutput()


def save_vtk_polydata(polydata: vtk.vtkPolyData, output_path: str) -> None:
    """
    Save VTK polydata to file.
    
    Args:
        polydata: VTK polydata to save
        output_path: path to output VTK file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()


def extract_points_from_polydata(polydata: vtk.vtkPolyData) -> np.ndarray:
    """
    Extract points from VTK polydata as numpy array.
    
    Args:
        polydata: VTK polydata
    
    Returns:
        numpy array: Nx3 array of points
    """
    points = []
    for i in range(polydata.GetNumberOfPoints()):
        points.append(polydata.GetPoint(i))
    return np.array(points)


def create_polydata_from_points(points: np.ndarray) -> vtk.vtkPolyData:
    """
    Create VTK polydata line from points array.
    
    Args:
        points: Nx3 array of points
    
    Returns:
        vtkPolyData: polydata with line cells
    """
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p)
    
    lines = vtk.vtkCellArray()
    for i in range(len(points) - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)
    
    return polydata


# =============================================================================
# Plane Fitting
# =============================================================================

def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to 3D points using SVD (least squares plane fitting).
    
    Args:
        points: Nx3 array of points
    
    Returns:
        tuple: (normal_vector, centroid)
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Center the points
    centered_points = points - centroid
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(centered_points)
    
    # Normal vector is the last row of Vt
    normal = Vt[-1, :]
    normal = normal / np.linalg.norm(normal)
    
    return normal, centroid


def find_non_intersecting_parallel_plane(
    plane_normal: np.ndarray,
    plane_centroid: np.ndarray,
    original_mask: 'sitk.Image',
    eardrum_point: np.ndarray,
    step_size: float = 0.5,
    max_offset: float = 30.0,
    plane_sample_radius: float = 15.0,
    num_samples: int = 50
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Find the closest parallel plane to the fitted plane that doesn't intersect the mask.
    
    Projects all mask voxels onto the plane normal direction and finds the boundary
    point (furthest from eardrum), then creates a parallel plane at that location.
    
    Args:
        plane_normal: Normal vector of the plane
        plane_centroid: Centroid of the original plane
        original_mask: SimpleITK image of the original (non-inverted) mask
        eardrum_point: Position of eardrum to determine direction
        step_size: Unused, kept for API compatibility
        max_offset: Unused, kept for API compatibility
        plane_sample_radius: Unused, kept for API compatibility
        num_samples: Unused, kept for API compatibility
    
    Returns:
        tuple: (offset_plane_centroid, offset_distance, boundary_point)
               boundary_point is the outermost mask voxel along the plane normal
    """

    
    print(f"    Finding mask boundary along plane normal...")
    
    # Convert mask to numpy arSray
    mask_array = sitk.GetArrayFromImage(original_mask)  # Shape: (z, y, x)
    
    # Get spacing and origin for coordinate transformation
    # NOTE: We intentionally ignore direction matrix to match mesh coordinate system
    # The mesh is created using: verts_xyz * spacing + origin (no direction matrix)
    spacing = np.array(original_mask.GetSpacing())  # (x, y, z)
    origin = np.array(original_mask.GetOrigin())    # (x, y, z)
    
    # Find all voxels with value = 1
    mask_indices = np.where(mask_array > 0.5)  # Returns (z_indices, y_indices, x_indices)
    
    if len(mask_indices[0]) == 0:
        print(f"      ⚠ No mask voxels found, using original plane")
        return plane_centroid, 0.0, None
    
    # Convert all mask voxel indices to physical coordinates
    z_indices, y_indices, x_indices = mask_indices
    num_voxels_original = len(z_indices)
    
    print(f"      Found {num_voxels_original} mask voxels")
    
    # Convert voxel indices to physical coordinates
    # Use same transformation as mesh: (x_idx, y_idx, z_idx) * spacing + origin
    # This ignores direction matrix intentionally to match mesh coordinate system
    physical_points_all = np.zeros((num_voxels_original, 3))
    physical_points_all[:, 0] = x_indices * spacing[0] + origin[0]  # X
    physical_points_all[:, 1] = y_indices * spacing[1] + origin[1]  # Y
    physical_points_all[:, 2] = z_indices * spacing[2] + origin[2]  # Z
    
    # Filter to keep only central 50% in physical Y and Z dimensions (discard 25% on each side)
    # This is direction-matrix agnostic since we filter on physical coordinates
    phys_y = physical_points_all[:, 1]  # Physical Y
    phys_z = physical_points_all[:, 2]  # Physical Z
    
    y_min_phys, y_max_phys = phys_y.min(), phys_y.max()
    z_min_phys, z_max_phys = phys_z.min(), phys_z.max()
    
    y_range_phys = y_max_phys - y_min_phys
    z_range_phys = z_max_phys - z_min_phys
    
    y_lower_phys = y_min_phys + 0.25 * y_range_phys
    y_upper_phys = y_max_phys - 0.25 * y_range_phys
    z_lower_phys = z_min_phys + 0.25 * z_range_phys
    z_upper_phys = z_max_phys - 0.25 * z_range_phys
    
    # Create mask for central region in physical coordinates (no filtering in X)
    central_mask = (
        (phys_y >= y_lower_phys) & (phys_y <= y_upper_phys) &
        (phys_z >= z_lower_phys) & (phys_z <= z_upper_phys)
    )
    
    # Apply filter
    physical_points = physical_points_all[central_mask]
    num_voxels = len(physical_points)
    
    print(f"      Filtered to central 50% in physical Y/Z: {num_voxels} voxels")
    
    if num_voxels == 0:
        print(f"      ⚠ No voxels after filtering, using all voxels")
        physical_points = physical_points_all
        num_voxels = num_voxels_original
    
    # Determine which direction is away from eardrum
    direction_to_eardrum = eardrum_point - plane_centroid
    if np.dot(plane_normal, direction_to_eardrum) > 0:
        # Normal points towards eardrum, use opposite direction
        outward_normal = -plane_normal
    else:
        # Normal points away from eardrum
        outward_normal = plane_normal
    
    # Project all mask points onto the outward normal direction
    # (relative to plane centroid)
    projections = np.dot(physical_points - plane_centroid, outward_normal)
    
    # Find the point with maximum projection (furthest in outward direction)
    max_idx = np.argmax(projections)
    max_projection = projections[max_idx]
    boundary_point = physical_points[max_idx]
    
    print(f"      Boundary point (max projection): {boundary_point}")
    print(f"      Projection distance: {max_projection:.2f}mm")
    
    # Create parallel plane: shift original centroid along normal by the max projection
    # This places the plane at the boundary of the mask
    offset_plane_centroid = plane_centroid + outward_normal * max_projection
    
    # The offset distance is the projection value
    offset_distance = abs(max_projection)
    
    print(f"      Offset plane centroid: {offset_plane_centroid}")
    print(f"      Offset distance from geodesic plane: {offset_distance:.1f}mm")
    
    return offset_plane_centroid, offset_distance, boundary_point


def find_plane_centerline_intersection(centerline_points: np.ndarray,
                                       plane_normal: np.ndarray,
                                       plane_point: np.ndarray,
                                       reference_point: np.ndarray = None,
                                       select_closest_to_eardrum: np.ndarray = None) -> Tuple[np.ndarray, int]:
    """
    Find intersection point between centerline and plane.
    
    Args:
        centerline_points: Nx3 array of centerline points
        plane_normal: normal vector to the plane
        plane_point: a point on the plane (e.g., centroid)
        reference_point: optional reference point to select among multiple intersections
        select_closest_to_eardrum: if provided, select intersection closest to this eardrum point
    
    Returns:
        tuple: (intersection_point, index)
    """
    # Compute signed distance from each point to the plane
    distances = np.dot(centerline_points - plane_point, plane_normal)
    
    # Find sign changes (plane crossings)
    sign_changes = np.where(np.diff(np.sign(distances)))[0]
    
    if len(sign_changes) > 0:
        print(f"    Found {len(sign_changes)} plane crossing(s) at indices: {sign_changes.tolist()}")
        
        # Compute all intersection points
        intersections = []
        for idx in sign_changes:
            p1 = centerline_points[idx]
            p2 = centerline_points[idx + 1]
            d1 = distances[idx]
            d2 = distances[idx + 1]
            
            # Linear interpolation
            t = abs(d1) / (abs(d1) + abs(d2))
            intersection = p1 + t * (p2 - p1)
            intersections.append((intersection, idx + 1))
        
        # Select intersection based on criteria
        if len(intersections) > 1:
            if select_closest_to_eardrum is not None:
                # Select the intersection closest to the eardrum point
                print(f"    Multiple intersections found, selecting closest to eardrum")
                
                best_intersection = None
                best_idx = None
                best_distance = float('inf')
                
                for i, (inter_point, inter_idx) in enumerate(intersections):
                    dist_to_eardrum = np.linalg.norm(inter_point - select_closest_to_eardrum)
                    print(f"      Intersection {i+1}: index={inter_idx}, distance to eardrum = {dist_to_eardrum:.2f}mm")
                    
                    if dist_to_eardrum < best_distance:
                        best_distance = dist_to_eardrum
                        best_intersection = inter_point
                        best_idx = inter_idx
                
                print(f"    ✓ Selected intersection closest to eardrum (distance: {best_distance:.2f}mm)")
                return best_intersection, best_idx
            elif reference_point is not None:
                print(f"    Multiple intersections found, selecting closest to reference point")
                
                best_intersection = None
                best_idx = None
                best_distance = float('inf')
                
                for i, (inter_point, inter_idx) in enumerate(intersections):
                    dist_to_ref = np.linalg.norm(inter_point - reference_point)
                    print(f"      Intersection {i+1}: index={inter_idx}, distance to reference = {dist_to_ref:.2f}mm")
                    
                    if dist_to_ref < best_distance:
                        best_distance = dist_to_ref
                        best_intersection = inter_point
                        best_idx = inter_idx
                
                print(f"    ✓ Selected intersection closest to reference (distance: {best_distance:.2f}mm)")
                return best_intersection, best_idx
            else:
                # No selection criteria, use first
                intersection, idx = intersections[0]
                print(f"    Using first intersection at index {idx}")
                return intersection, idx
        else:
            intersection, idx = intersections[0]
            print(f"    Single plane crossing at index {sign_changes[0]}")
            return intersection, idx
    else:
        # No crossing found, use closest point to plane
        abs_distances = np.abs(distances)
        idx = np.argmin(abs_distances)
        intersection = centerline_points[idx]
        
        print(f"    No plane crossing found, using closest point at index {idx}")
        print(f"    Distance to plane: {abs_distances[idx]:.2f}mm")
        
        return intersection, idx


# =============================================================================
# Spline Resampling
# =============================================================================

def resample_centerline_spline(points: np.ndarray, l_0: np.ndarray, 
                               l_1: np.ndarray, num_points: int = 100) -> np.ndarray:
    """
    Resample centerline to have a fixed number of points using spline interpolation.
    
    Args:
        points: Nx3 array of centerline points
        l_0: start point to prepend
        l_1: end point to append
        num_points: number of points in resampled line
    
    Returns:
        numpy array: resampled centerline (num_points x 3)
    """
    # Add start and end points
    points = np.vstack([l_0, points, l_1])
    
    # Remove duplicates while preserving order
    centerline, idx = np.unique(points, axis=0, return_index=True)
    centerline_clean = centerline[np.argsort(idx)]
    
    # Fit cubic spline
    tck, u = splprep(centerline_clean.T, k=3, s=0.2)
    
    # Resample uniformly
    u_fine = np.linspace(0, 1, num_points)
    centerline_resampled = np.array(splev(u_fine, tck)).T
    
    return centerline_resampled


# =============================================================================
# Path Metrics
# =============================================================================

def compute_path_length(points: np.ndarray) -> float:
    """
    Compute total arc length of a path defined by points.
    
    Args:
        points: Nx3 array of path coordinates
    
    Returns:
        float: total path length in mm
    """
    if len(points) < 2:
        return 0.0
    
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.sum(segment_lengths))


def compute_tortuosity_index(points: np.ndarray) -> float:
    """
    Compute tortuosity index as ratio of path length to straight-line distance.
    
    Args:
        points: Nx3 array of path coordinates
    
    Returns:
        float: tortuosity index (>= 1.0)
    """
    if len(points) < 2:
        return 1.0
    
    arc_length = compute_path_length(points)
    straight_distance = np.linalg.norm(points[-1] - points[0])
    
    if straight_distance == 0:
        return 1.0
    
    return arc_length / straight_distance


def compute_volume_between_planes(mesh: vtk.vtkPolyData,
                                 plane1_centroid: np.ndarray,
                                 plane1_normal: np.ndarray,
                                 plane2_centroid: np.ndarray,
                                 plane2_normal: np.ndarray) -> Optional[float]:
    """
    Compute volume of mesh between two planes.
    
    Args:
        mesh: VTK PolyData surface mesh (must be closed)
        plane1_centroid: Center point of first plane
        plane1_normal: Normal vector of first plane
        plane2_centroid: Center point of second plane
        plane2_normal: Normal vector of second plane

    Returns:
        float: Volume in mm³, or None if computation failed
    """
    try:
        if mesh.GetNumberOfCells() == 0:
            return None
        
        # Normalize plane normals
        plane1_normal = plane1_normal / np.linalg.norm(plane1_normal)
        plane2_normal = plane2_normal / np.linalg.norm(plane2_normal)
        
        # Create first clipping plane
        plane1 = vtk.vtkPlane()
        plane1.SetOrigin(plane1_centroid)
        plane1.SetNormal(plane1_normal)
        
        # Create second clipping plane
        plane2 = vtk.vtkPlane()
        plane2.SetOrigin(plane2_centroid)
        plane2.SetNormal(-plane2_normal)  # Invert normal to clip from the other side
        
        # Create implicit function combining both planes (intersection region)
        implicit_function = vtk.vtkImplicitBoolean()
        implicit_function.SetOperationTypeToIntersection()
        implicit_function.AddFunction(plane1)
        implicit_function.AddFunction(plane2)
        
        # Create clipped mesh using implicit function
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(mesh)
        clipper.SetClipFunction(implicit_function)
        clipper.Update()
        
        clipped_mesh = clipper.GetOutput()
        
        if clipped_mesh.GetNumberOfCells() == 0:
            return None
        
        # Compute volume using VTK's mass properties
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(clipped_mesh)
        mass_properties.Update()
        
        volume = mass_properties.GetVolume()
        
        return float(volume) if volume > 0 else None
    
    except Exception as e:
        print(f"    ⚠ Error computing volume: {e}")
        return None


def calculate_mesh_volume_trimesh(mesh: vtk.vtkPolyData, method: str = "auto") -> Optional[float]:
    """
    Calculate mesh volume using trimesh library (more robust than VTK).
    
    Trimesh uses signed volume calculation which is more accurate for complex geometries.
    Automatically handles mesh repair if needed.
    
    Args:
        mesh: VTK PolyData surface mesh (can be open or closed)
        method: Volume calculation method
            - "auto": Uses trimesh default (signed volume)
            - "convex": Uses convex hull approximation
            - "sample": Uses ray-casting/Monte Carlo method
    
    Returns:
        float: Volume in mm³, or None if calculation failed
    
    Raises:
        ImportError: If trimesh is not installed
    """
    try:
        # Convert VTK mesh to trimesh
        vertices = np.zeros((mesh.GetNumberOfPoints(), 3))
        for i in range(mesh.GetNumberOfPoints()):
            vertices[i] = mesh.GetPoint(i)
        
        # Extract face indices
        cell_array = mesh.GetPolys()
        faces = []
        cell_array.InitTraversal()
        while True:
            ids = vtk.vtkIdList()
            if cell_array.GetNextCell(ids) == 0:
                break
            # Only accept triangles
            if ids.GetNumberOfIds() == 3:
                faces.append([ids.GetId(j) for j in range(3)])
        
        if len(faces) == 0:
            print(f"    ⚠ No triangles found in mesh")
            return None
        
        faces = np.array(faces)
        
        # Create trimesh object
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        if method == "auto":
            # Use signed volume (most accurate for closed meshes)
            volume = tm.volume
            if volume < 0:
                # Mesh has inverted normals, flip and try again
                tm.invert()
                volume = tm.volume
            print(f"    ✓ Trimesh (signed volume): {abs(volume):.2f} mm³")
            return float(abs(volume))
        
        elif method == "convex":
            # Use convex hull approximation
            if tm.is_watertight:
                volume = tm.volume
            else:
                # Try to fill holes and recalculate
                tm.fill_holes()
                volume = tm.volume
            print(f"    ✓ Trimesh (with hole filling): {volume:.2f} mm³")
            return float(volume)
        
        elif method == "sample":
            # Ray-casting method (slower but works for complex shapes)
            volume = tm.volume
            # Verify with bounds check
            bounds = tm.bounds
            bounds_volume = np.prod(bounds[1] - bounds[0])
            if volume > bounds_volume:
                print(f"    ⚠ Volume sanity check failed ({volume:.2f} > bounds {bounds_volume:.2f})")
                volume = None
            else:
                print(f"    ✓ Trimesh (ray-casting): {volume:.2f} mm³")
            return float(volume) if volume else None
        
        else:
            print(f"    ⚠ Unknown method: {method}, using 'auto'")
            return calculate_mesh_volume_trimesh(mesh, method="auto")
    
    except ImportError:
        print(f"    ✗ trimesh not installed. Install with: pip install trimesh")
        return None
    except Exception as e:
        print(f"    ⚠ Error calculating volume with trimesh: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_mesh_volume_scipy(mesh: vtk.vtkPolyData) -> Optional[float]:
    """
    Calculate mesh volume using scipy's ConvexHull (convex approximation).
    
    Useful for getting a conservative volume estimate for non-convex meshes.
    
    Args:
        mesh: VTK PolyData surface mesh
    
    Returns:
        float: Volume in mm³ (convex approximation), or None if calculation failed
    
    Raises:
        ImportError: If scipy is not installed
    """
    try:
        from scipy.spatial import ConvexHull
        
        # Extract vertices
        vertices = np.zeros((mesh.GetNumberOfPoints(), 3))
        for i in range(mesh.GetNumberOfPoints()):
            vertices[i] = mesh.GetPoint(i)
        
        if len(vertices) < 4:
            print(f"    ⚠ Not enough vertices for ConvexHull ({len(vertices)} < 4)")
            return None
        
        # Compute convex hull
        hull = ConvexHull(vertices)
        volume = hull.volume
        
        print(f"    ✓ SciPy ConvexHull (convex approx): {volume:.2f} mm³")
        return float(volume)
    
    except ImportError:
        print(f"    ✗ scipy not installed. Install with: pip install scipy")
        return None
    except Exception as e:
        print(f"    ⚠ Error calculating volume with scipy: {e}")
        return None


def calculate_mesh_volume_divergence(mesh: vtk.vtkPolyData) -> Optional[float]:
    """
    Calculate mesh volume using divergence theorem (manual implementation).
    
    Volume = (1/6) * |Σ (p_i · (p_j × p_k))|
    for each triangle with vertices p_i, p_j, p_k
    
    This is the most mathematically direct method and doesn't require external libraries.
    Works best for closed, properly-oriented meshes.
    
    Args:
        mesh: VTK PolyData surface mesh (must be closed and consistently oriented)
    
    Returns:
        float: Volume in mm³, or None if calculation failed
    """
    try:
        vertices = np.zeros((mesh.GetNumberOfPoints(), 3))
        for i in range(mesh.GetNumberOfPoints()):
            vertices[i] = mesh.GetPoint(i)
        
        # Extract triangles
        cell_array = mesh.GetPolys()
        volume = 0.0
        triangle_count = 0
        
        cell_array.InitTraversal()
        while True:
            ids = vtk.vtkIdList()
            if cell_array.GetNextCell(ids) == 0:
                break
            
            if ids.GetNumberOfIds() == 3:
                p1 = vertices[ids.GetId(0)]
                p2 = vertices[ids.GetId(1)]
                p3 = vertices[ids.GetId(2)]
                
                # Scalar triple product: (p1 · (p2 × p3))
                v2_cross_v3 = np.cross(p2, p3)
                signed_volume_element = np.dot(p1, v2_cross_v3)
                volume += signed_volume_element
                triangle_count += 1
        
        volume = abs(volume) / 6.0
        
        print(f"    ✓ Divergence theorem ({triangle_count} triangles): {volume:.2f} mm³")
        return float(volume)
    
    except Exception as e:
        print(f"    ⚠ Error calculating volume with divergence theorem: {e}")
        return None


def calculate_mesh_volume_comparison(mesh: vtk.vtkPolyData) -> Dict[str, Optional[float]]:
    """
    Calculate mesh volume using multiple methods and compare results.
    
    Useful for debugging volume calculation issues and validating results.
    
    Args:
        mesh: VTK PolyData surface mesh
    
    Returns:
        dict: Keys are method names, values are volumes in mm³
    
    Example output:
        {
            'vtk_mass_properties': 1093.86,
            'trimesh_signed': 1093.85,
            'scipy_convex_hull': 1100.50,
            'divergence_theorem': 1093.86
        }
    """
    results = {}
    
    print(f"  Computing volume using multiple methods...")
    
    # Method 1: VTK MassProperties (original)
    try:
        mass_props = vtk.vtkMassProperties()
        mass_props.SetInputData(mesh)
        mass_props.Update()
        vtk_volume = mass_props.GetVolume()
        results['vtk_mass_properties'] = float(vtk_volume) if vtk_volume > 0 else None
        print(f"    ✓ VTK MassProperties: {results['vtk_mass_properties']:.2f} mm³" if results['vtk_mass_properties'] else "    ⚠ VTK MassProperties: Failed")
    except Exception as e:
        print(f"    ⚠ VTK MassProperties failed: {e}")
        results['vtk_mass_properties'] = None
    
    # Method 2: Trimesh
    trimesh_vol = calculate_mesh_volume_trimesh(mesh, method="auto")
    results['trimesh_signed'] = trimesh_vol
    
    # Method 3: SciPy ConvexHull
    scipy_vol = calculate_mesh_volume_scipy(mesh)
    results['scipy_convex_hull'] = scipy_vol
    
    # Method 4: Divergence Theorem
    div_vol = calculate_mesh_volume_divergence(mesh)
    results['divergence_theorem'] = div_vol
    
    # Summary
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        avg_volume = np.mean(list(valid_results.values()))
        print(f"\n  Summary:")
        print(f"    Methods succeeded: {len(valid_results)}/4")
        print(f"    Average volume: {avg_volume:.2f} mm³")
        if len(valid_results) > 1:
            max_vol = max(valid_results.values())
            min_vol = min(valid_results.values())
            variation = (max_vol - min_vol) / min_vol * 100 if min_vol > 0 else 0
            print(f"    Range: {min_vol:.2f} - {max_vol:.2f} mm³ ({variation:.1f}% variation)")
    
    return results


def compute_clipped_mesh_between_planes(mesh: vtk.vtkPolyData,
                                       plane1_centroid: np.ndarray,
                                       plane1_normal: np.ndarray,
                                       plane2_centroid: np.ndarray,
                                       plane2_normal: np.ndarray) -> Optional[vtk.vtkPolyData]:
    """
    Extract and return the mesh clipped between two planes (for visualization).
    
    Args:
        mesh: VTK PolyData surface mesh (must be closed)
        plane1_centroid: Center point of first plane
        plane1_normal: Normal vector of first plane
        plane2_centroid: Center point of second plane
        plane2_normal: Normal vector of second plane

    Returns:
        vtkPolyData: Clipped mesh region between planes, or None if computation failed
    """
    try:
        if mesh.GetNumberOfCells() == 0:
            return None
        
        # Normalize plane normals
        plane1_normal = plane1_normal / np.linalg.norm(plane1_normal)
        plane2_normal = plane2_normal / np.linalg.norm(plane2_normal)
        
        # Create first clipping plane
        plane1 = vtk.vtkPlane()
        plane1.SetOrigin(plane1_centroid)
        plane1.SetNormal(plane1_normal)
        
        # Create second clipping plane
        plane2 = vtk.vtkPlane()
        plane2.SetOrigin(plane2_centroid)
        plane2.SetNormal(-plane2_normal)  # Invert normal to clip from the other side
        
        # Create implicit function combining both planes (intersection region)
        implicit_function = vtk.vtkImplicitBoolean()
        implicit_function.SetOperationTypeToIntersection()
        implicit_function.AddFunction(plane1)
        implicit_function.AddFunction(plane2)
        
        # Create clipped mesh using implicit function
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(mesh)
        clipper.SetClipFunction(implicit_function)
        clipper.Update()
        
        clipped_mesh = clipper.GetOutput()
        
        if clipped_mesh.GetNumberOfCells() == 0:
            return None
        
        return clipped_mesh
        
    except Exception as e:
        print(f"    ⚠ Error computing clipped mesh: {e}")
        return None
        
    except Exception as e:
        print(f"    Warning: Could not compute volume between planes: {e}")
        return None


# =============================================================================
# Geodesic Path Computation
# =============================================================================

def compute_geodesic_path(mesh: vtk.vtkPolyData,
                          point1: np.ndarray,
                          point2: np.ndarray,
                          resample: bool = True,
                          num_points: int = 100) -> Tuple[vtk.vtkPolyData, np.ndarray]:
    """
    Compute geodesic shortest path on mesh surface between two points.
    
    Args:
        mesh: vtkPolyData mesh
        point1: start point [x, y, z] (Top RS)
        point2: end point [x, y, z] (Bottom RS)
        resample: whether to resample the path
        num_points: number of points if resampling
    
    Returns:
        tuple: (vtkPolyData, numpy array of points)
    """
    print(f"  Computing geodesic path on surface")
    print(f"    Start: {point1}")
    print(f"    End: {point2}")
    
    # Find nearest mesh points
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    
    start_id = locator.FindClosestPoint(point1)
    end_id = locator.FindClosestPoint(point2)
    
    start_actual = np.array(mesh.GetPoint(start_id))
    end_actual = np.array(mesh.GetPoint(end_id))
    
    print(f"    Closest mesh point to start: {start_actual}")
    print(f"    Closest mesh point to end: {end_actual}")
    
    # Use Dijkstra to find shortest path
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(mesh)
    dijkstra.SetStartVertex(start_id)
    dijkstra.SetEndVertex(end_id)
    dijkstra.Update()
    
    path = dijkstra.GetOutput()
    
    print(f"  ✓ Geodesic path computed: {path.GetNumberOfPoints()} points")
    
    # Extract points
    points = np.array([path.GetPoint(i) for i in range(path.GetNumberOfPoints())])
    
    if resample and len(points) >= 4:
        print(f"    Resampling geodesic path to {num_points} points...")
        points_resampled = resample_centerline_spline(points, point2, point1, num_points=num_points)
        
        # Create new polydata with resampled points
        path_resampled = create_polydata_from_points(points_resampled)
        print(f"  ✓ Geodesic path resampled: {len(points_resampled)} points")
        return path_resampled, points_resampled
    
    return path, points


def compute_geodesic_raw(mesh: vtk.vtkPolyData,
                         top_rs: np.ndarray,
                         bottom_rs: np.ndarray) -> Tuple[vtk.vtkPolyData, np.ndarray]:
    """
    Compute raw geodesic path without resampling.
    
    Args:
        mesh: VTK PolyData surface mesh
        top_rs: Start point [x, y, z]
        bottom_rs: End point [x, y, z]
    
    Returns:
        Tuple of (geodesic_polydata, raw_points)
    """
    return compute_geodesic_path(mesh, top_rs, bottom_rs, resample=False)


# =============================================================================
# Centerline Computation
# =============================================================================

def compute_centerline_vmtk(mesh: vtk.vtkPolyData,
                            initial_point: np.ndarray,
                            final_point: np.ndarray,
                            geodesic_points: np.ndarray = None,
                            reference_point: np.ndarray = None,
                            original_mask: 'sitk.Image' = None) -> Optional[dict]:
    """
    Extract centerline using VMTK.
    
    Args:
        mesh: vtkPolyData surface mesh
        initial_point: starting point [x, y, z] (Middle RS landmark) - numpy array or list
        final_point: ending point [x, y, z] (Eardrum landmark) - numpy array or list
        geodesic_points: optional Nx3 array of geodesic path points for plane-based trimming
        reference_point: optional reference point for plane intersection selection
        original_mask: optional SimpleITK image of original (non-inverted) mask for
                       dynamic offset plane computation
    
    Returns:
        dict with centerline data or None if computation fails
    """
    # Convert to numpy arrays if needed (handle both lists and arrays)
    initial_point = np.array(initial_point) if not isinstance(initial_point, np.ndarray) else initial_point
    final_point = np.array(final_point) if not isinstance(final_point, np.ndarray) else final_point
    
    if reference_point is None:
        reference_point = initial_point
    else:
        reference_point = np.array(reference_point) if not isinstance(reference_point, np.ndarray) else reference_point
    
    print(f"  Computing centerline with VMTK")
    print(f"    Initial point (middle RS): {initial_point}")
    print(f"    Final point (eardrum): {final_point}")
    
    # Extend initial point 2cm away from final point
    direction = initial_point - final_point
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm > 0:
        direction_unit = direction / direction_norm
        extended_initial = initial_point + direction_unit * 20.0
        print(f"    Extended initial point (+2cm away from final): {extended_initial}")
    else:
        extended_initial = initial_point
    
    # Project extended point onto mesh surface
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()
    
    closest_point = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)
    cell_locator.FindClosestPoint(extended_initial, closest_point, cell_id, sub_id, dist2)
    
    vmtk_seed_point = np.array(closest_point)
    distance_to_surface = np.sqrt(dist2.get())
    
    print(f"    Extended point projected onto surface: {vmtk_seed_point}")
    print(f"    Distance from extended point to surface: {distance_to_surface:.2f}mm")
    
    # Compute centerline using VMTK
    print(f"    Running VMTK centerline computation...")
    vtk.vtkObject.GlobalWarningDisplayOff()
    
    try:
        centerline_filter = vmtkscripts.vmtkCenterlines()
        centerline_filter.Surface = mesh
        centerline_filter.SeedSelectorName = "pointlist"
        centerline_filter.SourcePoints = list(vmtk_seed_point)
        centerline_filter.TargetPoints = list(final_point)
        centerline_filter.AppendEndPoints = 0
        centerline_filter.Execute()
        
        centerline = centerline_filter.Centerlines
        vtk.vtkObject.GlobalWarningDisplayOn()
        
        if centerline is None or centerline.GetNumberOfPoints() < 2:
            print(f"  ✗ ERROR: Centerline computation failed or too few points")
            return None
        
        print(f"  ✓ Centerline computed: {centerline.GetNumberOfPoints()} points")
        
        # Extract points
        points = extract_points_from_polydata(centerline)
        vmtk_original_points = points.copy()
        
        # Check direction and reverse if needed
        dist_start_to_initial = np.linalg.norm(points[0] - initial_point)
        dist_start_to_final = np.linalg.norm(points[0] - final_point)
        
        if dist_start_to_final < dist_start_to_initial:
            print(f"    ⚠ Reversing centerline (first point closer to eardrum)")
            points = points[::-1]
        else:
            print(f"    ✓ Centerline direction is correct")
        
        # Trim the centerline
        if geodesic_points is not None:
            # Use plane-based trimming
            print(f"  Using PLANE-BASED trimming (geodesic arch)...")
            
            plane_normal, plane_centroid = fit_plane_to_points(geodesic_points)
            print(f"    Plane normal: {plane_normal}")
            print(f"    Plane centroid: {plane_centroid}")
            
            # Find intersection with original geodesic plane (beginning_eac)
            # Use select_closest_to_eardrum to pick the intersection closest to the eardrum
            # when there are multiple intersections
            intersection_point, eac_trim_index = find_plane_centerline_intersection(
                points, plane_normal, plane_centroid, reference_point,
                select_closest_to_eardrum=final_point
            )
            beginning_eac = intersection_point.copy()
            print(f"    Beginning EAC (geodesic plane intersection): {beginning_eac}")
            
            # Create a parallel plane that doesn't intersect the ear canal
            mask_boundary_point = None
            if original_mask is not None:
                # Use dynamic computation based on mask
                offset_plane_centroid, offset_distance, mask_boundary_point = find_non_intersecting_parallel_plane(
                    plane_normal=plane_normal,
                    plane_centroid=plane_centroid,
                    original_mask=original_mask,
                    eardrum_point=final_point,
                    step_size=0.5,  # 0.5mm steps
                    max_offset=30.0,  # Max 3cm offset
                    plane_sample_radius=15.0,  # Sample 15mm radius disk
                    num_samples=50
                )
                print(f"    Offset plane centroid (dynamic, {offset_distance:.1f}mm from geodesic): {offset_plane_centroid}")
                if mask_boundary_point is not None:
                    print(f"    Mask boundary point (outermost): {mask_boundary_point}")
            else:
                # Fallback: use fixed 1cm offset
                offset_distance = 10.0  # 1cm in mm
                
                # Determine which direction is away from eardrum
                direction_to_eardrum = final_point - plane_centroid
                if np.dot(plane_normal, direction_to_eardrum) > 0:
                    # Plane normal points towards eardrum, flip the offset direction
                    offset_plane_centroid = plane_centroid - plane_normal * offset_distance
                else:
                    # Plane normal points away from eardrum, shift in normal direction
                    offset_plane_centroid = plane_centroid + plane_normal * offset_distance
                
                print(f"    Offset plane centroid (fixed +1cm from eardrum): {offset_plane_centroid}")
            
            # Find intersection with offset plane (beginning_centerline)
            # Use a plane with normal (-1, 0, 0) passing through mask_boundary_point
            if mask_boundary_point is not None:
                x_plane_normal = np.array([-1.0, 0.0, 0.0])
                beginning_centerline, trim_index = find_plane_centerline_intersection(
                    points, x_plane_normal, mask_boundary_point, reference_point,
                    select_closest_to_eardrum=final_point
                )
                print(f"    Beginning Centerline (X-plane at mask boundary): {beginning_centerline}")
            else:
                # Fallback to offset plane if no mask_boundary_point
                beginning_centerline, trim_index = find_plane_centerline_intersection(
                    points, plane_normal, offset_plane_centroid, reference_point,
                    select_closest_to_eardrum=final_point
                )
                print(f"    Beginning Centerline (offset plane intersection): {beginning_centerline}")
            
            points_trimmed = points[trim_index:]
            print(f"    Trimmed {trim_index} points. Centerline: {len(points)} → {len(points_trimmed)} points")
        else:
            # Use fixed 2cm trimming
            print(f"  Using FIXED 2cm trimming (no geodesic provided)...")
            
            cumulative_dist = 0.0
            trim_index = 0
            
            for i in range(1, len(points)):
                segment_dist = np.linalg.norm(points[i] - points[i-1])
                cumulative_dist += segment_dist
                
                if cumulative_dist >= 20.0:
                    trim_index = i
                    print(f"    Found point 2cm inside centerline at index {i}")
                    break
            
            if trim_index > 0:
                points_trimmed = points[trim_index:]
                print(f"    Trimmed {trim_index} points. Centerline: {len(points)} → {len(points_trimmed)} points")
            else:
                points_trimmed = points
                print(f"    Warning: Could not find 2cm point, using full centerline")
            
            beginning_eac = None
            beginning_centerline = None
        
        # Store landmarks
        original_start_point = initial_point.copy()
        
        # Use beginning_centerline (plane intersection) as start point if available,
        # otherwise fallback to first trimmed point
        if beginning_centerline is not None:
            trimmed_start_point = beginning_centerline.copy()
            print(f"    Using beginning_centerline as start point for resampling")
        else:
            trimmed_start_point = points_trimmed[0].copy()
            print(f"    Using first trimmed point as start point for resampling")
        
        # Compute eardrum plane intersection for end point
        # The eardrum plane is defined by the eardrum landmark and a normal
        # computed from the last few centerline segments (tangent direction)
        print(f"    Computing eardrum plane intersection for end point...")
        
        # Compute tangent at end of centerline (average of last 5 segments)
        n_tangent_points = min(5, len(points_trimmed) - 1)
        if n_tangent_points >= 1:
            end_tangent = points_trimmed[-1] - points_trimmed[-1 - n_tangent_points]
            end_tangent_norm = np.linalg.norm(end_tangent)
            if end_tangent_norm > 1e-6:
                end_tangent = end_tangent / end_tangent_norm
            else:
                end_tangent = np.array([0.0, 0.0, 1.0])  # fallback
        else:
            end_tangent = np.array([0.0, 0.0, 1.0])  # fallback
        
        print(f"    Eardrum plane normal (end tangent): {end_tangent}")
        print(f"    Eardrum plane point: {final_point}")
        
        # Find intersection of centerline with eardrum plane
        eardrum_plane_intersection, eardrum_intersect_idx = find_plane_centerline_intersection(
            points_trimmed, end_tangent, final_point, reference_point=final_point,
            select_closest_to_eardrum=final_point
        )
        
        # Use the intersection as end point
        end_point = eardrum_plane_intersection.copy()
        print(f"    End point (eardrum plane intersection): {end_point}")
        
        # Resample to 100 points
        print(f"    Resampling centerline to 100 points...")
        points_resampled = resample_centerline_spline(
            points_trimmed, trimmed_start_point, end_point, num_points=100
        )
        print(f"  ✓ Centerline resampled: {len(points_resampled)} points")
        
        # Create VTK polydata
        centerline_resampled = create_polydata_from_points(points_resampled)
        
    except Exception as e:
        vtk.vtkObject.GlobalWarningDisplayOn()
        print(f"  ✗ ERROR: VMTK centerline extraction failed: {e}")
        raise
    
    return {
        'centerline': centerline_resampled,
        'original_start': original_start_point,
        'trimmed_start': trimmed_start_point,
        'end_point': end_point,
        'raw_centerline_points': points,
        'vmtk_original_points': vmtk_original_points,
        'extended_seed': extended_initial,
        'vmtk_seed_on_surface': vmtk_seed_point,
        'beginning_eac': beginning_eac,
        'beginning_centerline': beginning_centerline,
        'plane_normal': plane_normal if geodesic_points is not None else None,
        'plane_centroid': plane_centroid if geodesic_points is not None else None,
        'offset_plane_centroid': offset_plane_centroid if geodesic_points is not None else None,
        'mask_boundary_point': mask_boundary_point if geodesic_points is not None else None
    }


# =============================================================================
# Centerline Validation
# =============================================================================

def validate_centerline(centerline_points: np.ndarray,
                        initial_point: np.ndarray,
                        final_point: np.ndarray) -> Tuple[bool, str]:
    """
    Validate centerline meets minimum quality criteria.
    
    Args:
        centerline_points: Nx3 array of centerline points
        initial_point: Start point of centerline
        final_point: End point of centerline
    
    Returns:
        Tuple of (is_valid, reason)
    """
    if centerline_points is None or len(centerline_points) < 2:
        return False, "Centerline has fewer than 2 points"
    
    # Compute centerline path length
    centerline_length = compute_path_length(centerline_points)
    
    # Distance between endpoints
    endpoint_distance = np.linalg.norm(final_point - initial_point)
    
    print(f"    Centerline validation:")
    print(f"      Path length: {centerline_length:.2f} mm (min: {MIN_CENTERLINE_LENGTH_MM} mm)")
    print(f"      Endpoint distance: {endpoint_distance:.2f} mm (min: {MIN_ENDPOINT_DISTANCE_MM} mm)")
    
    if centerline_length < MIN_CENTERLINE_LENGTH_MM:
        reason = f"Centerline too short: {centerline_length:.2f} mm < {MIN_CENTERLINE_LENGTH_MM} mm"
        return False, reason
    
    if endpoint_distance < MIN_ENDPOINT_DISTANCE_MM:
        reason = f"Endpoints too close: {endpoint_distance:.2f} mm < {MIN_ENDPOINT_DISTANCE_MM} mm"
        return False, reason
    
    return True, f"Valid (length: {centerline_length:.2f} mm, distance: {endpoint_distance:.2f} mm)"


def find_geodesic_furthest_point(mesh: vtk.vtkPolyData,
                                 reference_point: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find the point on the mesh that is geodesically furthest from a reference point.
    
    Args:
        mesh: VTK PolyData surface mesh
        reference_point: Reference point [x, y, z]
    
    Returns:
        Tuple of (furthest_point, geodesic_distance)
    """
    n_points = mesh.GetNumberOfPoints()
    if n_points == 0:
        raise ValueError("Mesh has no points")
    
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    
    ref_vertex_id = locator.FindClosestPoint(reference_point)
    if ref_vertex_id < 0:
        raise ValueError("Could not find closest vertex to reference point")
    
    print(f"    Finding geodesic furthest point from reference...")
    
    max_geodesic_distance = 0.0
    furthest_vertex_id = ref_vertex_id
    
    # Sample subset of vertices for efficiency
    sample_step = max(1, n_points // 1000)
    
    for candidate_id in range(0, n_points, sample_step):
        if candidate_id == ref_vertex_id:
            continue
        
        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(mesh)
        dijkstra.SetStartVertex(ref_vertex_id)
        dijkstra.SetEndVertex(candidate_id)
        dijkstra.Update()
        
        path = dijkstra.GetOutput()
        if path is None or path.GetNumberOfPoints() < 2:
            continue
        
        # Compute path length
        path_length = 0.0
        for i in range(1, path.GetNumberOfPoints()):
            p1 = np.array(path.GetPoint(i - 1))
            p2 = np.array(path.GetPoint(i))
            path_length += np.linalg.norm(p2 - p1)
        
        if path_length > max_geodesic_distance:
            max_geodesic_distance = path_length
            furthest_vertex_id = candidate_id
    
    furthest_point = np.array(mesh.GetPoint(furthest_vertex_id))
    
    print(f"    Geodesic furthest point: {furthest_point}")
    print(f"    Geodesic distance: {max_geodesic_distance:.2f} mm")
    
    return furthest_point, max_geodesic_distance


# =============================================================================
# Centerline Refinement
# =============================================================================

def compute_centerline_tangents(centerline_points: np.ndarray) -> np.ndarray:
    """
    Compute tangent (normal to cutting planes) at each centerline point.
    
    Args:
        centerline_points: Nx3 array of centerline points
    
    Returns:
        Nx3 array of normalized tangent vectors
    """
    n_points = len(centerline_points)
    tangents = np.zeros((n_points, 3))
    
    for i in range(n_points):
        # Compute tangent using forward difference at start,
        # backward difference at end, and central difference in middle
        if i == 0:
            p0 = centerline_points[i]
            p2 = centerline_points[min(i + 2, n_points - 1)]
        elif i == n_points - 1:
            p0 = centerline_points[max(i - 2, 0)]
            p2 = centerline_points[i]
        else:
            p0 = centerline_points[i - 1]
            p2 = centerline_points[i + 1]
        
        tangent = p2 - p0
        tangent_norm = np.linalg.norm(tangent)
        
        if tangent_norm > 1e-10:
            tangents[i] = tangent / tangent_norm
        else:
            tangents[i] = np.array([1.0, 0.0, 0.0])
    
    return tangents


def find_closest_surface_point(mesh: vtk.vtkPolyData, point: np.ndarray) -> np.ndarray:
    """
    Find the closest point on the mesh surface to a given point.
    
    Args:
        mesh: VTK PolyData surface mesh
        point: 3D point [x, y, z]
    
    Returns:
        Closest point on mesh surface [x, y, z]
    """
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    
    closest_point = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    subid = vtk.mutable(0)
    distance = vtk.mutable(0.0)
    
    locator.FindClosestPoint(point, closest_point, cell_id, subid, distance)
    
    return np.array(closest_point)


def validate_endpoint_on_surface(mesh: vtk.vtkPolyData, endpoint: np.ndarray, 
                                  max_distance: float = 2.0) -> Tuple[bool, np.ndarray]:
    """
    Validate that an endpoint is on the mesh surface.
    
    If the endpoint is too far from the surface, find and return the closest surface point instead.
    
    Args:
        mesh: VTK PolyData surface mesh
        endpoint: 3D point to validate [x, y, z]
        max_distance: Maximum allowed distance from surface (mm)
    
    Returns:
        Tuple of (is_valid, validated_endpoint)
        is_valid: True if endpoint is on surface (within max_distance)
        validated_endpoint: Either the original endpoint or closest surface point
    """
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    
    closest_point = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    subid = vtk.mutable(0)
    distance = vtk.mutable(0.0)
    
    locator.FindClosestPoint(endpoint, closest_point, cell_id, subid, distance)
    
    # Compute distance manually as fallback if mutable access fails
    closest_point_array = np.array(closest_point)
    dist = np.linalg.norm(endpoint - closest_point_array)
    
    is_valid = dist <= max_distance
    
    return is_valid, closest_point_array


def refine_centerline_endpoint(
    mesh: vtk.vtkPolyData,
    raw_centerline_points: np.ndarray,
    original_endpoint: np.ndarray,
    num_normals: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine the centerline endpoint using the median of the last N normals.
    
    This extracts a cross-section at the last centerline point using the median
    of the last N point normals, and uses the centroid of this cross-section as
    the refined endpoint. Validates the refined endpoint is on the mesh surface;
    if not, uses the closest surface point instead.
    
    Args:
        mesh: VTK PolyData surface mesh
        raw_centerline_points: Nx3 array of raw centerline points
        original_endpoint: The original endpoint [x, y, z]
        num_normals: Number of normals from the end to use for median (default: 10)
    
    Returns:
        Tuple of (refined_endpoint, refined_normal)
        refined_endpoint is the centroid of the cross-section at the last point
                         (or closest surface point if validation fails)
        refined_normal is the median normal direction
    """
    print(f"  Refining centerline endpoint...")
    
    if len(raw_centerline_points) < num_normals + 1:
        print(f"    ⚠ Warning: Centerline has fewer than {num_normals + 1} points")
        print(f"    Using original endpoint (no refinement)")
        return original_endpoint, None
    
    # Compute tangents along centerline
    tangents = compute_centerline_tangents(raw_centerline_points)
    
    # Get the last N tangents and compute their median
    last_tangents = tangents[-num_normals:]
    print(f"    Computed {len(last_tangents)} tangents from end of centerline")
    
    # Compute median tangent (each component separately)
    median_tangent = np.median(last_tangents, axis=0)
    median_tangent_norm = np.linalg.norm(median_tangent)
    
    if median_tangent_norm < 1e-10:
        print(f"    ⚠ Warning: Could not compute valid median tangent")
        return original_endpoint, None
    
    median_tangent = median_tangent / median_tangent_norm
    print(f"    Median tangent (normal): {median_tangent}")
    
    # Extract cross-section at the last centerline point using the median tangent
    last_point = raw_centerline_points[-1]
    print(f"    Last centerline point: {last_point}")
    
    refined_endpoint = None
    
    try:
        plane = vtk.vtkPlane()
        plane.SetOrigin(last_point)
        plane.SetNormal(median_tangent)
        
        # Cut mesh with plane
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(mesh)
        cutter.Update()
        
        intersected_points = cutter.GetOutput()
        
        # Find closed curve containing this centerline point
        closed_curve, perimeter, area = find_closed_curve_at_centerline_point(
            intersected_points, last_point
        )
        
        if closed_curve is None or closed_curve.GetNumberOfPoints() == 0:
            print(f"    ⚠ Warning: Could not extract valid cross-section")
            refined_endpoint = None
        else:
            # Compute centroid of the cross-section
            curve_points = np.array([
                closed_curve.GetPoint(j) for j in range(closed_curve.GetNumberOfPoints())
            ])
            
            if curve_points.size == 0:
                print(f"    ⚠ Warning: Cross-section has no points")
                refined_endpoint = None
            else:
                refined_endpoint = np.mean(curve_points, axis=0)
                print(f"    Refined endpoint (cross-section centroid): {refined_endpoint}")
                
                # Distance from last centerline point to refined endpoint
                distance = np.linalg.norm(refined_endpoint - last_point)
                print(f"    Distance from last point to refined endpoint: {distance:.2f} mm")
        
    except Exception as e:
        print(f"    ⚠ Warning: Exception during cross-section extraction: {e}")
        refined_endpoint = None
    
    # Validate the refined endpoint is on the mesh surface
    if refined_endpoint is not None:
        is_valid, closest_point = validate_endpoint_on_surface(mesh, refined_endpoint, max_distance=2.0)
        
        if is_valid:
            print(f"    ✓ Refined endpoint validated (on mesh surface)")
            return refined_endpoint, median_tangent
        else:
            dist_to_surface = np.linalg.norm(refined_endpoint - closest_point)
            print(f"    ⚠ Refined endpoint NOT on surface (distance: {dist_to_surface:.2f}mm)")
            print(f"    Using closest surface point instead")
            print(f"    Closest surface point: {closest_point}")
            return closest_point, median_tangent
    else:
        # If endpoint extraction failed, find closest surface point to original endpoint
        print(f"    ⚠ Endpoint refinement failed, finding closest surface point to original endpoint")
        closest_point = find_closest_surface_point(mesh, original_endpoint)
        dist_to_surface = np.linalg.norm(original_endpoint - closest_point)
        print(f"    Distance from original endpoint to surface: {dist_to_surface:.2f}mm")
        print(f"    Using closest surface point: {closest_point}")
        return closest_point, median_tangent


# =============================================================================
# CBJ (Cartilaginous-Bony Junction) Processing
# =============================================================================

def load_cbj_landmarks_from_json(json_path: str) -> Optional[np.ndarray]:
    """
    Load CBJ (Cartilaginous-Bony Junction) landmarks from JSON file.
    
    Supports two formats:
    1. Unified markup JSON (our pipeline) — CBJ points have labels "CBJ1"-"CBJ4"
       among other landmarks in the same file.
    2. Legacy separate CBJ JSON — all landmarks are CBJ points.
    
    Args:
        json_path: Path to landmarks JSON file (unified or CBJ-only)
    
    Returns:
        Nx3 array of CBJ landmark positions, or None if file not found/invalid
    """
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'landmarks' not in data or len(data['landmarks']) == 0:
            return None
        
        # Extract CBJ positions — look for labels starting with "CBJ"
        positions = []
        for landmark in data['landmarks']:
            label = landmark.get('label', '')
            if label.upper().startswith('CBJ') and 'position' in landmark and landmark['position'] is not None:
                pos = landmark['position']
                positions.append(pos)
                print(f"    {label}: {pos} (x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f})")
        
        if len(positions) == 0:
            return None
        
        return np.array(positions)
        
    except Exception as e:
        print(f"    ⚠ Warning: Could not load CBJ landmarks from {json_path}: {e}")
        return None


def fit_plane_to_cbj_points(cbj_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to CBJ landmark points and return the best-fit plane.
    
    Uses PCA to find the plane that best fits the CBJ points.
    
    Args:
        cbj_points: Nx3 array of CBJ landmark positions (N >= 3)
    
    Returns:
        Tuple of (plane_normal, plane_centroid)
        plane_normal: Unit normal vector of best-fit plane
        plane_centroid: Centroid of the CBJ points on the plane
    """
    if len(cbj_points) < 3:
        raise ValueError(f"Need at least 3 points to fit plane, got {len(cbj_points)}")
    
    # Compute centroid
    centroid = np.mean(cbj_points, axis=0)
    
    # Center the points
    centered_points = cbj_points - centroid
    
    # Compute covariance matrix
    cov = np.dot(centered_points.T, centered_points)
    
    # SVD to get principal components
    U, S, Vt = np.linalg.svd(cov)
    
    # Last singular vector (smallest eigenvalue) is the plane normal
    plane_normal = Vt[-1, :]
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    return plane_normal, centroid


def find_closest_point_to_plane(
    centerline_points: np.ndarray,
    plane_normal: np.ndarray,
    plane_centroid: np.ndarray
) -> Tuple[int, np.ndarray, float]:
    """
    Find the centerline point closest to a plane and its distance.
    
    Args:
        centerline_points: Nx3 array of centerline points
        plane_normal: Unit normal vector of the plane
        plane_centroid: Point on the plane
    
    Returns:
        Tuple of (closest_index, closest_point, distance_to_plane)
    """
    if len(centerline_points) == 0:
        raise ValueError("Empty centerline points array")
    
    # Normalize plane normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Compute distance from each point to plane
    # Distance = |dot(point - centroid, normal)|
    distances = np.abs(np.dot(centerline_points - plane_centroid, plane_normal))
    
    # Find closest point
    closest_idx = np.argmin(distances)
    closest_point = centerline_points[closest_idx]
    min_distance = distances[closest_idx]
    
    return closest_idx, closest_point, min_distance


def extract_cbj_cross_section_features(
    mesh: vtk.vtkPolyData,
    centerline_points: np.ndarray,
    cbj_plane_normal: np.ndarray,
    cbj_plane_centroid: np.ndarray,
    cbj_closest_idx: int
) -> Tuple[Optional[Dict], Optional[vtk.vtkPolyData]]:
    """
    Extract cross-sectional features at the CBJ plane intersection.
    
    Args:
        mesh: VTK PolyData surface mesh
        centerline_points: Nx3 array of centerline points
        cbj_plane_normal: Unit normal of the CBJ plane
        cbj_plane_centroid: Centroid of the CBJ plane
        cbj_closest_idx: Index of closest centerline point
    
    Returns:
        Tuple of (features_dict, closed_curve_polydata) where features_dict contains
        area, perimeter, min_radius, max_radius, aspect_ratio; and closed_curve_polydata
        is the VTK polydata curve for visualization (or None if extraction failed)
    """
    try:
        # Normalize plane normal
        cbj_plane_normal = cbj_plane_normal / np.linalg.norm(cbj_plane_normal)
        
        # Create cutting plane
        plane = vtk.vtkPlane()
        plane.SetOrigin(cbj_plane_centroid)
        plane.SetNormal(cbj_plane_normal)
        
        # Cut mesh with plane
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(mesh)
        cutter.Update()
        
        intersected_points = cutter.GetOutput()
        
        # Find closed curve at the intersection
        closest_point = centerline_points[cbj_closest_idx]
        closed_curve, perimeter, area = find_closed_curve_at_centerline_point(
            intersected_points, closest_point
        )
        
        if closed_curve is None or closed_curve.GetNumberOfPoints() == 0:
            print(f"    ⚠ Warning: No valid cross-section found at CBJ plane")
            return None, None
        
        # Extract curve points and compute distances from closest point
        curve_points = np.array([
            closed_curve.GetPoint(j) for j in range(closed_curve.GetNumberOfPoints())
        ])
        
        if curve_points.size == 0:
            return None, None
        
        distances = np.linalg.norm(curve_points - closest_point, axis=1)
        
        min_radius = float(np.min(distances))
        max_radius = float(np.max(distances))
        
        # Compute aspect ratio
        aspect_ratio = max_radius / min_radius if min_radius > 1e-6 else None
        
        features_dict = {
            'area': float(area) if area else 0.0,
            'perimeter': float(perimeter) if perimeter else 0.0,
            'min_radius': min_radius,
            'max_radius': max_radius,
            'aspect_ratio': float(aspect_ratio) if aspect_ratio else None,
            'centerline_index': cbj_closest_idx
        }
        
        return features_dict, closed_curve
        
    except Exception as e:
        print(f"    ⚠ Warning: Could not extract CBJ cross-section features: {e}")
        return None, None


def compute_cbj_centerline_metrics(
    centerline_points: np.ndarray,
    cbj_closest_idx: int,
    isthmus_idx: int = None
) -> Dict[str, float]:
    """
    Compute centerline length metrics related to CBJ intersection.
    
    Args:
        centerline_points: Nx3 array of centerline points (should be raw/unresampled)
        cbj_closest_idx: Index of CBJ intersection point on centerline
        isthmus_idx: Optional index of isthmus point (for isthmus-to-CBJ metric)
    
    Returns:
        Dict with metrics:
        - cbj_length_from_start: Length from start to CBJ
        - cbj_length_to_end: Length from CBJ to end
        - cbj_total_proportion: CBJ position as fraction of total length
        - cbj_isthmus_to_cbj_length: (optional) Length from isthmus to CBJ
    """
    if len(centerline_points) < 2:
        raise ValueError("Centerline must have at least 2 points")
    
    # Compute length from start to CBJ
    length_from_start = 0.0
    for i in range(1, cbj_closest_idx + 1):
        segment = np.linalg.norm(centerline_points[i] - centerline_points[i-1])
        length_from_start += segment
    
    # Compute length from CBJ to end
    length_to_end = 0.0
    for i in range(cbj_closest_idx + 1, len(centerline_points)):
        segment = np.linalg.norm(centerline_points[i] - centerline_points[i-1])
        length_to_end += segment
    
    # Total length
    total_length = length_from_start + length_to_end
    
    # Proportion
    cbj_proportion = length_from_start / total_length if total_length > 0 else 0.0
    
    metrics = {
        'cbj_length_from_start': float(length_from_start),
        'cbj_length_to_end': float(length_to_end),
        'cbj_total_proportion': float(cbj_proportion),  # 0-1 scale where 1 is at end
    }
    
    # Compute isthmus-to-CBJ length if isthmus index is provided
    if isthmus_idx is not None and isthmus_idx < cbj_closest_idx:
        isthmus_to_cbj_length = 0.0
        for i in range(isthmus_idx + 1, cbj_closest_idx + 1):
            segment = np.linalg.norm(centerline_points[i] - centerline_points[i-1])
            isthmus_to_cbj_length += segment
        metrics['cbj_isthmus_to_cbj_length'] = float(isthmus_to_cbj_length)
    
    return metrics


# =============================================================================
# Visualization Helpers
# =============================================================================

def create_sphere_actor(position: np.ndarray, color: Tuple[float, float, float],
                        radius: float = 2.0) -> vtk.vtkActor:
    """
    Create a sphere actor for visualization.
    
    Args:
        position: sphere center [x, y, z]
        color: RGB color tuple
        radius: sphere radius
    
    Returns:
        vtkActor: sphere actor
    """
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(position)
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(20)
    sphere.SetThetaResolution(20)
    sphere.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    
    return actor


def create_tube_actor(polydata: vtk.vtkPolyData, color: Tuple[float, float, float],
                      radius: float = 0.8, opacity: float = 1.0) -> vtk.vtkActor:
    """
    Create a tube actor for visualization.
    
    Args:
        polydata: vtkPolyData line
        color: RGB color tuple
        radius: tube radius
        opacity: tube opacity
    
    Returns:
        vtkActor: tube actor
    """
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(polydata)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(12)
    tube.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)
    
    return actor


def visualize_debug_failure(mesh: vtk.vtkPolyData,
                            middle_rs: np.ndarray,
                            eardrum: np.ndarray,
                            top_rs: np.ndarray,
                            bottom_rs: np.ndarray,
                            geodesic_path: vtk.vtkPolyData = None,
                            raw_centerline_points: np.ndarray = None,
                            extended_seed: np.ndarray = None,
                            vmtk_seed_on_surface: np.ndarray = None,
                            geodesic_furthest_point: np.ndarray = None,
                            plane_normal: np.ndarray = None,
                            plane_centroid: np.ndarray = None,
                            failure_reason: str = "Unknown",
                            title: str = "DEBUG: Processing Failure") -> None:
    """
    Detailed visualization for debugging failed samples.
    
    Shows all available intermediate data to help diagnose the issue.
    
    Args:
        mesh: surface mesh
        middle_rs: middle RS position
        eardrum: eardrum position
        top_rs: top RS position
        bottom_rs: bottom RS position
        geodesic_path: geodesic path polydata (optional)
        raw_centerline_points: raw centerline points before trimming (optional)
        extended_seed: extended seed point in free space (optional)
        vmtk_seed_on_surface: VMTK seed projected onto surface (optional)
        geodesic_furthest_point: geodesic furthest point from eardrum (optional)
        plane_normal: trimming plane normal (optional)
        plane_centroid: trimming plane centroid (optional)
        failure_reason: description of why processing failed
        title: window title
    """
    print(f"\n  {'='*60}")
    print(f"  DEBUG VISUALIZATION: {failure_reason}")
    print(f"  {'='*60}")
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background for debug/failure
    
    # Add mesh (semi-transparent)
    mapper_mesh = vtk.vtkPolyDataMapper()
    mapper_mesh.SetInputData(mesh)
    actor_mesh = vtk.vtkActor()
    actor_mesh.SetMapper(mapper_mesh)
    actor_mesh.GetProperty().SetOpacity(0.25)
    actor_mesh.GetProperty().SetColor(0.7, 0.7, 0.7)
    renderer.AddActor(actor_mesh)
    
    legend_items = []
    
    # Add geodesic path if available (yellow)
    if geodesic_path is not None:
        renderer.AddActor(create_tube_actor(geodesic_path, (1.0, 1.0, 0.0), radius=0.6))
        legend_items.append("Yellow tube: Geodesic path (Top RS → Bottom RS)")
    
    # Add raw centerline if available (red)
    if raw_centerline_points is not None and len(raw_centerline_points) > 1:
        raw_centerline_polydata = create_polydata_from_points(raw_centerline_points)
        renderer.AddActor(create_tube_actor(raw_centerline_polydata, (1.0, 0.0, 0.0), radius=0.8))
        legend_items.append(f"Red tube: Raw centerline ({len(raw_centerline_points)} points)")
        
        # Mark first and last points of raw centerline
        renderer.AddActor(create_sphere_actor(raw_centerline_points[0], (1.0, 0.5, 0.5), radius=3.0))  # Light red
        renderer.AddActor(create_sphere_actor(raw_centerline_points[-1], (0.5, 0.0, 0.0), radius=3.0))  # Dark red
        legend_items.append("Light red sphere: Centerline start")
        legend_items.append("Dark red sphere: Centerline end")
    
    # Add trimming plane if available (cyan, semi-transparent)
    if plane_normal is not None and plane_centroid is not None:
        plane_size = 80.0  # Size of plane visualization in mm
        plane_source = vtk.vtkPlaneSource()
        plane_source.SetOrigin(-plane_size/2, -plane_size/2, 0)
        plane_source.SetPoint1(plane_size/2, -plane_size/2, 0)
        plane_source.SetPoint2(-plane_size/2, plane_size/2, 0)
        plane_source.SetCenter(0, 0, 0)
        plane_source.Update()
        
        z_axis = np.array([0, 0, 1])
        plane_normal_unit = plane_normal / np.linalg.norm(plane_normal)
        
        rotation_axis = np.cross(z_axis, plane_normal_unit)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        transform = vtk.vtkTransform()
        transform.Translate(plane_centroid)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            cos_angle = np.dot(z_axis, plane_normal_unit)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            transform.RotateWXYZ(angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        elif np.dot(z_axis, plane_normal_unit) < 0:
            transform.RotateX(180)
        
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(plane_source.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        
        mapper_plane = vtk.vtkPolyDataMapper()
        mapper_plane.SetInputConnection(transform_filter.GetOutputPort())
        
        actor_plane = vtk.vtkActor()
        actor_plane.SetMapper(mapper_plane)
        actor_plane.GetProperty().SetColor(0.0, 0.8, 0.8)
        actor_plane.GetProperty().SetOpacity(0.4)
        renderer.AddActor(actor_plane)
        legend_items.append("Cyan plane: Trimming plane")
    
    # Add landmarks
    landmarks_info = [
        (middle_rs, (0.0, 1.0, 0.0), "Middle RS (green)"),
        (eardrum, (0.0, 0.0, 1.0), "Eardrum (blue)"),
        (top_rs, (1.0, 0.0, 1.0), "Top RS (magenta)"),
        (bottom_rs, (0.0, 1.0, 1.0), "Bottom RS (cyan)")
    ]
    
    for position, color, name in landmarks_info:
        renderer.AddActor(create_sphere_actor(position, color, radius=2.5))
        legend_items.append(f"{name}")
    
    # Add extended seed if available (orange)
    if extended_seed is not None:
        renderer.AddActor(create_sphere_actor(extended_seed, (1.0, 0.5, 0.0), radius=3.0))
        legend_items.append("Orange sphere: Extended seed (+2cm)")
    
    # Add VMTK seed on surface if available (dark orange)
    if vmtk_seed_on_surface is not None:
        renderer.AddActor(create_sphere_actor(vmtk_seed_on_surface, (0.8, 0.3, 0.0), radius=3.0))
        legend_items.append("Dark orange sphere: VMTK seed (on surface)")
        
        # Draw line from extended seed to vmtk seed
        if extended_seed is not None:
            line_points = vtk.vtkPoints()
            line_points.InsertNextPoint(extended_seed)
            line_points.InsertNextPoint(vmtk_seed_on_surface)
            
            line_cell = vtk.vtkLine()
            line_cell.GetPointIds().SetId(0, 0)
            line_cell.GetPointIds().SetId(1, 1)
            
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(line_cell)
            
            line_polydata = vtk.vtkPolyData()
            line_polydata.SetPoints(line_points)
            line_polydata.SetLines(lines)
            
            renderer.AddActor(create_tube_actor(line_polydata, (0.9, 0.4, 0.0), radius=0.3))
    
    # Add geodesic furthest point if available (white)
    if geodesic_furthest_point is not None:
        renderer.AddActor(create_sphere_actor(geodesic_furthest_point, (1.0, 1.0, 1.0), radius=3.5))
        legend_items.append("White sphere: Geodesic furthest point from eardrum")
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName(f"{title} - {failure_reason}")
    render_window.SetSize(1400, 900)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Add axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(20, 20, 20)
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.02)
    renderer.AddActor(axes)
    
    renderer.ResetCamera()
    render_window.Render()
    
    print(f"\n  FAILURE REASON: {failure_reason}")
    print(f"\n  Legend:")
    for item in legend_items:
        print(f"    - {item}")
    print(f"\n  Close window to continue...")
    
    interactor.Start()


def visualize_mesh_with_landmarks(mesh: vtk.vtkPolyData,
                                  middle_rs: np.ndarray,
                                  eardrum: np.ndarray,
                                  top_rs: np.ndarray,
                                  bottom_rs: np.ndarray,
                                  cbj_points: np.ndarray = None,
                                  title: str = "Mesh with Landmarks") -> None:
    """
    Visualize mesh with landmark points.
    
    Args:
        mesh: surface mesh
        middle_rs: middle RS position
        eardrum: eardrum position
        top_rs: top RS position
        bottom_rs: bottom RS position
        cbj_points: Optional CBJ landmark points (Nx3 array)
        title: window title
    """
    print(f"  Opening visualization: {title}")
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    
    # Add mesh (semi-transparent)
    mapper_mesh = vtk.vtkPolyDataMapper()
    mapper_mesh.SetInputData(mesh)
    actor_mesh = vtk.vtkActor()
    actor_mesh.SetMapper(mapper_mesh)
    actor_mesh.GetProperty().SetOpacity(0.3)
    actor_mesh.GetProperty().SetColor(0.8, 0.8, 0.8)
    renderer.AddActor(actor_mesh)
    
    # Add landmarks
    landmarks_info = [
        (middle_rs, (0.0, 1.0, 0.0), "Middle RS"),
        (eardrum, (0.0, 0.0, 1.0), "Eardrum"),
        (top_rs, (1.0, 0.0, 1.0), "Top RS"),
        (bottom_rs, (0.0, 1.0, 1.0), "Bottom RS")
    ]
    
    for position, color, name in landmarks_info:
        renderer.AddActor(create_sphere_actor(position, color))
    
    # Add CBJ landmarks if available
    if cbj_points is not None and len(cbj_points) > 0:
        print(f"    ✓ Adding {len(cbj_points)} CBJ landmark points (purple spheres)")
        for i, cbj_pt in enumerate(cbj_points):
            renderer.AddActor(create_sphere_actor(cbj_pt, (0.8, 0.2, 0.8)))  # Purple
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName(title)
    render_window.SetSize(1200, 800)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Add axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(20, 20, 20)
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.02)
    renderer.AddActor(axes)
    
    renderer.ResetCamera()
    render_window.Render()
    print(f"  ✓ Close window to continue...")
    interactor.Start()


def visualize_results(mesh: vtk.vtkPolyData,
                      centerline: vtk.vtkPolyData,
                      geodesic_path: vtk.vtkPolyData,
                      middle_rs: np.ndarray,
                      eardrum: np.ndarray,
                      top_rs: np.ndarray,
                      bottom_rs: np.ndarray,
                      trimmed_start: np.ndarray = None,
                      plane_normal: np.ndarray = None,
                      plane_centroid: np.ndarray = None,
                      offset_plane_centroid: np.ndarray = None,
                      mask_boundary_point: np.ndarray = None,
                      cbj_points: np.ndarray = None,
                      cbj_plane_normal: np.ndarray = None,
                      cbj_plane_centroid: np.ndarray = None,
                      cbj_closest_point: np.ndarray = None,
                      cbj_cross_section: vtk.vtkPolyData = None,
                      clipped_volume_mesh: vtk.vtkPolyData = None,
                      title: str = "Processing Results") -> None:
    """
    Visualize mesh with centerline, geodesic path, landmarks, and trimming planes.
    
    Args:
        mesh: surface mesh
        centerline: centerline polydata
        geodesic_path: geodesic path polydata
        middle_rs: middle RS position
        eardrum: eardrum position
        top_rs: top RS position
        bottom_rs: bottom RS position
        trimmed_start: trimmed start position (optional)
        plane_normal: normal vector of the trimming plane (optional)
        plane_centroid: centroid point of the geodesic plane / beginning_eac (optional)
        offset_plane_centroid: centroid of offset plane / beginning_centerline (optional)
        mask_boundary_point: outermost mask point along plane normal (optional)
        title: window title
    """
    print(f"\n  Opening visualization window")
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    
    # Add mesh (semi-transparent)
    mapper_mesh = vtk.vtkPolyDataMapper()
    mapper_mesh.SetInputData(mesh)
    actor_mesh = vtk.vtkActor()
    actor_mesh.SetMapper(mapper_mesh)
    actor_mesh.GetProperty().SetOpacity(0.3)
    actor_mesh.GetProperty().SetColor(0.8, 0.8, 0.8)
    renderer.AddActor(actor_mesh)
    
    # Add centerline (red tube)
    renderer.AddActor(create_tube_actor(centerline, (1.0, 0.0, 0.0), radius=0.8))
    
    # Add geodesic path (yellow tube)
    renderer.AddActor(create_tube_actor(geodesic_path, (1.0, 1.0, 0.0), radius=0.6))
    
    # Add clipped volume (light cyan, semi-transparent) - visualization of computed volume
    if clipped_volume_mesh is not None and clipped_volume_mesh.GetNumberOfCells() > 0:
        mapper_volume = vtk.vtkPolyDataMapper()
        mapper_volume.SetInputData(clipped_volume_mesh)
        actor_volume = vtk.vtkActor()
        actor_volume.SetMapper(mapper_volume)
        actor_volume.GetProperty().SetOpacity(0.4)
        actor_volume.GetProperty().SetColor(0.0, 1.0, 1.0)  # Cyan
        renderer.AddActor(actor_volume)
    
    # Add trimming plane (semi-transparent cyan)
    if plane_normal is not None and plane_centroid is not None:
        # Create a plane source centered at the plane centroid
        plane_size = 80.0  # Size of plane visualization in mm
        plane_source = vtk.vtkPlaneSource()
        plane_source.SetOrigin(-plane_size/2, -plane_size/2, 0)
        plane_source.SetPoint1(plane_size/2, -plane_size/2, 0)
        plane_source.SetPoint2(-plane_size/2, plane_size/2, 0)
        plane_source.SetCenter(0, 0, 0)
        plane_source.Update()
        
        # Transform plane to align with the actual plane normal and position
        # Compute rotation to align Z-axis with plane normal
        z_axis = np.array([0, 0, 1])
        plane_normal_unit = plane_normal / np.linalg.norm(plane_normal)
        
        # Compute rotation axis and angle for the plane
        rotation_axis = np.cross(z_axis, plane_normal_unit)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # Store the plane rotation parameters for reuse
        plane_angle_deg = 0.0
        plane_rotation_axis = rotation_axis.copy() if rotation_axis_norm > 1e-6 else np.array([1, 0, 0])
        
        transform = vtk.vtkTransform()
        transform.Translate(plane_centroid)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            plane_rotation_axis = rotation_axis.copy()
            cos_angle = np.dot(z_axis, plane_normal_unit)
            plane_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            transform.RotateWXYZ(plane_angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        elif np.dot(z_axis, plane_normal_unit) < 0:
            # Plane normal is opposite to z-axis, rotate 180 degrees
            transform.RotateX(180)
            plane_angle_deg = 180.0
            plane_rotation_axis = np.array([1, 0, 0])
        
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(plane_source.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        
        mapper_plane = vtk.vtkPolyDataMapper()
        mapper_plane.SetInputConnection(transform_filter.GetOutputPort())
        
        actor_plane = vtk.vtkActor()
        actor_plane.SetMapper(mapper_plane)
        actor_plane.GetProperty().SetColor(0.0, 0.8, 0.8)  # Cyan
        actor_plane.GetProperty().SetOpacity(0.5)
        renderer.AddActor(actor_plane)
        
        # Add plane normal arrow
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.3)
        arrow_source.SetTipRadius(0.1)
        arrow_source.SetShaftRadius(0.03)
        arrow_source.Update()
        
        # Transform arrow to start at plane centroid and point along normal
        arrow_transform = vtk.vtkTransform()
        arrow_transform.Translate(plane_centroid)
        
        # Align arrow with plane normal
        x_axis = np.array([1, 0, 0])
        arrow_rotation_axis = np.cross(x_axis, plane_normal_unit)
        arrow_rotation_axis_norm = np.linalg.norm(arrow_rotation_axis)
        
        if arrow_rotation_axis_norm > 1e-6:
            arrow_rotation_axis = arrow_rotation_axis / arrow_rotation_axis_norm
            cos_angle = np.dot(x_axis, plane_normal_unit)
            arrow_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            arrow_transform.RotateWXYZ(arrow_angle_deg, arrow_rotation_axis[0], arrow_rotation_axis[1], arrow_rotation_axis[2])
        elif np.dot(x_axis, plane_normal_unit) < 0:
            arrow_transform.RotateZ(180)
        
        arrow_transform.Scale(15, 15, 15)  # Scale arrow length
        
        arrow_transform_filter = vtk.vtkTransformPolyDataFilter()
        arrow_transform_filter.SetInputConnection(arrow_source.GetOutputPort())
        arrow_transform_filter.SetTransform(arrow_transform)
        arrow_transform_filter.Update()
        
        mapper_arrow = vtk.vtkPolyDataMapper()
        mapper_arrow.SetInputConnection(arrow_transform_filter.GetOutputPort())
        
        actor_arrow = vtk.vtkActor()
        actor_arrow.SetMapper(mapper_arrow)
        actor_arrow.GetProperty().SetColor(0.0, 0.8, 0.8)  # Cyan (same as plane)
        renderer.AddActor(actor_arrow)
        
        # Add second plane (offset plane at mask boundary) - Orange
        # Center the plane at mask_boundary_point with normal (-1, 0, 0)
        if mask_boundary_point is not None:
            plane_source2 = vtk.vtkPlaneSource()
            plane_source2.SetOrigin(-plane_size/2, -plane_size/2, 0)
            plane_source2.SetPoint1(plane_size/2, -plane_size/2, 0)
            plane_source2.SetPoint2(-plane_size/2, plane_size/2, 0)
            plane_source2.SetCenter(0, 0, 0)
            plane_source2.Update()
            
            transform2 = vtk.vtkTransform()
            # Center the plane at the mask_boundary_point
            transform2.Translate(mask_boundary_point)
            
            # Rotate plane to have normal (-1, 0, 0)
            # Original plane normal is (0, 0, 1), rotate 90 degrees around Y axis
            transform2.RotateY(90)
            
            transform_filter2 = vtk.vtkTransformPolyDataFilter()
            transform_filter2.SetInputConnection(plane_source2.GetOutputPort())
            transform_filter2.SetTransform(transform2)
            transform_filter2.Update()
            
            mapper_plane2 = vtk.vtkPolyDataMapper()
            mapper_plane2.SetInputConnection(transform_filter2.GetOutputPort())
            
            actor_plane2 = vtk.vtkActor()
            actor_plane2.SetMapper(mapper_plane2)
            actor_plane2.GetProperty().SetColor(1.0, 0.5, 0.0)  # Orange
            actor_plane2.GetProperty().SetOpacity(0.5)
            renderer.AddActor(actor_plane2)
    
    # Add landmarks
    landmarks_info = [
        (middle_rs, (0.0, 1.0, 0.0)),      # Green
        (eardrum, (0.0, 0.0, 1.0)),        # Blue
        (top_rs, (1.0, 0.0, 1.0)),         # Magenta
        (bottom_rs, (0.0, 1.0, 1.0))       # Cyan
    ]
    
    if trimmed_start is not None:
        landmarks_info.append((trimmed_start, (1.0, 0.5, 0.0)))  # Orange
    
    if mask_boundary_point is not None:
        landmarks_info.append((mask_boundary_point, (1.0, 0.0, 0.0)))  # Red
    
    for position, color in landmarks_info:
        renderer.AddActor(create_sphere_actor(position, color))
    
    # Add CBJ visualization if available
    cbj_rendered = False
    if cbj_points is not None and len(cbj_points) > 0:
        print(f"    ✓ Adding {len(cbj_points)} CBJ landmark points (purple spheres)")
        cbj_rendered = True
        for i, cbj_pt in enumerate(cbj_points):
            renderer.AddActor(create_sphere_actor(cbj_pt, (0.8, 0.2, 0.8)))  # Purple
    
    if cbj_plane_normal is not None and cbj_plane_centroid is not None:
        if not cbj_rendered:
            print(f"    ✓ Adding CBJ plane (purple disk)")
        # Draw CBJ plane as a disk
        try:
            plane_normal_norm = cbj_plane_normal / np.linalg.norm(cbj_plane_normal)
            
            # Create orthonormal basis for the plane
            if abs(plane_normal_norm[0]) < 0.9:
                v1 = np.array([1.0, 0.0, 0.0])
            else:
                v1 = np.array([0.0, 1.0, 0.0])
            
            v1 = v1 - np.dot(v1, plane_normal_norm) * plane_normal_norm
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(plane_normal_norm, v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Create disk geometry
            n_segments = 32
            radius = 5.0
            disk_points = vtk.vtkPoints()
            disk_cells = vtk.vtkCellArray()
            
            # Center point
            disk_points.InsertNextPoint(cbj_plane_centroid)
            
            # Ring of points around the plane
            for i in range(n_segments):
                angle = 2 * np.pi * i / n_segments
                point = (cbj_plane_centroid + 
                        radius * np.cos(angle) * v1 + 
                        radius * np.sin(angle) * v2)
                disk_points.InsertNextPoint(point)
            
            # Create triangles from center to ring
            for i in range(n_segments):
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, 0)
                triangle.GetPointIds().SetId(1, 1 + i)
                triangle.GetPointIds().SetId(2, 1 + (i + 1) % n_segments)
                disk_cells.InsertNextCell(triangle)
            
            disk_polydata = vtk.vtkPolyData()
            disk_polydata.SetPoints(disk_points)
            disk_polydata.SetPolys(disk_cells)
            
            disk_mapper = vtk.vtkPolyDataMapper()
            disk_mapper.SetInputData(disk_polydata)
            disk_actor = vtk.vtkActor()
            disk_actor.SetMapper(disk_mapper)
            disk_actor.GetProperty().SetColor(0.8, 0.2, 0.8)  # Purple
            disk_actor.GetProperty().SetOpacity(0.3)
            renderer.AddActor(disk_actor)
        except Exception as e:
            print(f"    Warning: Could not draw CBJ plane: {e}")
    
    # Add CBJ cross-section curve if available
    if cbj_cross_section is not None:
        try:
            print(f"    ✓ Adding CBJ cross-section curve (purple line)")
            cbj_curve_mapper = vtk.vtkPolyDataMapper()
            cbj_curve_mapper.SetInputData(cbj_cross_section)
            cbj_curve_actor = vtk.vtkActor()
            cbj_curve_actor.SetMapper(cbj_curve_mapper)
            cbj_curve_actor.GetProperty().SetColor(0.8, 0.2, 0.8)  # Purple
            cbj_curve_actor.GetProperty().SetLineWidth(4)
            renderer.AddActor(cbj_curve_actor)
        except Exception as e:
            print(f"    Warning: Could not draw CBJ cross-section: {e}")
    
    if cbj_closest_point is not None:
        print(f"    ✓ Adding CBJ intersection point on centerline (purple sphere)")
        renderer.AddActor(create_sphere_actor(cbj_closest_point, (0.8, 0.2, 0.8), radius=1.5))
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName(title)
    render_window.SetSize(1200, 800)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(20, 20, 20)
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.02)
    renderer.AddActor(axes)
    
    renderer.ResetCamera()
    render_window.Render()
    
    print(f"  ✓ Visualization window opened")
    print(f"\n  Legend:")
    print(f"    - Gray surface: Mesh")
    print(f"    - Red tube: Centerline")
    print(f"    - Yellow tube: Geodesic path")
    if plane_normal is not None and plane_centroid is not None:
        print(f"    - Cyan plane: Geodesic plane (beginning_eac)")
        print(f"    - Cyan arrow: Plane normal direction")
    if mask_boundary_point is not None:
        print(f"    - Orange plane: Plane at mask boundary (normal: -1,0,0)")
    print(f"    - Green sphere: Middle RS")
    print(f"    - Blue sphere: Eardrum")
    print(f"    - Magenta sphere: Top RS")
    print(f"    - Cyan sphere: Bottom RS")
    if trimmed_start is not None:
        print(f"    - Orange sphere: Trimmed start")
    if mask_boundary_point is not None:
        print(f"    - Red sphere: Mask boundary point (outermost)")
    print(f"\n  Close window to continue...")
    print(f"\n  Close window to continue...")
    
    interactor.Start()


# =============================================================================
# Cross-sectional Feature Extraction
# =============================================================================

def find_closed_curve_at_centerline_point(
    cutting_surfaces: vtk.vtkPolyData,
    centerline_point: np.ndarray
) -> Tuple[Optional[vtk.vtkPolyData], Optional[float], Optional[float]]:
    """
    Find the closed curve (cross-section) closest to a centerline point.
    
    When a plane cuts a tubular surface, it may produce multiple disconnected
    curves. This function finds the one closest to the centerline point and
    computes its perimeter length and enclosed area.
    
    Args:
        cutting_surfaces: VTK PolyData from plane cutting operation
        centerline_point: The centerline point [x, y, z] to find the curve for
    
    Returns:
        Tuple of (closed_curve_polydata, perimeter_length, area) or (None, None, None) if failed
    """
    # Use connectivity filter to find the region closest to the centerline point
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(cutting_surfaces)
    connectivity_filter.SetClosestPoint(centerline_point)
    connectivity_filter.SetExtractionModeToClosestPointRegion()
    connectivity_filter.Update()
    
    closest_points = connectivity_filter.GetOutput().GetPoints()
    if (not closest_points) or (closest_points.GetNumberOfPoints() == 0):
        return None, None, None
    
    closed_curve_polydata = connectivity_filter.GetOutput()
    
    # Calculate perimeter length
    perimeter_length = closed_curve_polydata.GetLength()
    
    # Calculate enclosed area using Delaunay triangulation
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(closed_curve_polydata)
    delaunay.Update()
    
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(delaunay.GetOutput())
    triangle_filter.Update()
    
    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(triangle_filter.GetOutput())
    area = mass_properties.GetSurfaceArea()
    
    return closed_curve_polydata, perimeter_length, area


def extract_centerline_features(
    mesh: vtk.vtkPolyData,
    centerline_points: np.ndarray,
    output_vtk_path: Optional[str] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract cross-sectional features along the centerline.
    
    For each point on the centerline:
    1. Compute the local tangent direction
    2. Create a plane perpendicular to the centerline
    3. Cut the mesh with that plane to get a cross-section
    4. Find the closed curve containing the centerline point
    5. Compute min/max radii (distances from centerline to curve)
    6. Compute cross-section perimeter and area
    
    Args:
        mesh: VTK PolyData surface mesh
        centerline_points: Nx3 array of centerline points (should be 100 points)
        output_vtk_path: Optional path to save VTK file with features as point data
    
    Returns:
        Dictionary with feature arrays:
        - 'min_radius': minimum radius at each point (mm)
        - 'max_radius': maximum radius at each point (mm)
        - 'perimeter': cross-section perimeter at each point (mm)
        - 'area': cross-section area at each point (mm²)
        Returns None if feature extraction fails
    """
    print(f"  Extracting cross-sectional features along centerline...")
    
    n_points = len(centerline_points)
    if n_points < 3:
        print(f"    ✗ ERROR: Centerline must have at least 3 points")
        return None
    
    min_radii = []
    max_radii = []
    perimeters = []
    areas = []
    
    for i in range(n_points):
        # Compute tangent direction using neighboring points
        if i == 0:
            p0 = centerline_points[i]
            p2 = centerline_points[min(i + 2, n_points - 1)]
        elif i == n_points - 1:
            p0 = centerline_points[max(i - 2, 0)]
            p2 = centerline_points[i]
        else:
            p0 = centerline_points[i - 1]
            p2 = centerline_points[i + 1]
        
        tangent = p2 - p0
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            print(f"    ⚠ Warning: Zero tangent at point {i}, using previous values")
            if min_radii:
                min_radii.append(min_radii[-1])
                max_radii.append(max_radii[-1])
                perimeters.append(perimeters[-1])
                areas.append(areas[-1])
            else:
                return None
            continue
        
        tangent = tangent / tangent_norm
        point = centerline_points[i]
        
        # Create cutting plane perpendicular to centerline
        plane = vtk.vtkPlane()
        plane.SetOrigin(point)
        plane.SetNormal(tangent)
        
        # Cut mesh with plane
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(mesh)
        cutter.Update()
        
        intersected_points = cutter.GetOutput()
        
        # Find closed curve containing this centerline point
        closed_curve, perimeter, area = find_closed_curve_at_centerline_point(
            intersected_points, point
        )
        
        if closed_curve is None:
            print(f"    ⚠ Warning: No closed curve at point {i}, using previous values")
            if min_radii:
                min_radii.append(min_radii[-1])
                max_radii.append(max_radii[-1])
                perimeters.append(perimeters[-1])
                areas.append(areas[-1])
            else:
                # First point failed - critical error
                print(f"    ✗ ERROR: Cannot compute cross-section at first point")
                return None
            continue
        
        # Extract curve points and compute distances to centerline point
        curve_points = np.array([
            closed_curve.GetPoint(j) for j in range(closed_curve.GetNumberOfPoints())
        ])
        
        if curve_points.size == 0:
            print(f"    ⚠ Warning: Empty curve at point {i}, using previous values")
            if min_radii:
                min_radii.append(min_radii[-1])
                max_radii.append(max_radii[-1])
                perimeters.append(perimeters[-1])
                areas.append(areas[-1])
            else:
                return None
            continue
        
        distances = np.linalg.norm(curve_points - point, axis=1)
        
        min_radii.append(np.min(distances))
        max_radii.append(np.max(distances))
        perimeters.append(perimeter if perimeter else 0.0)
        areas.append(area if area else 0.0)
        
        # Progress indicator
        if (i + 1) % 20 == 0 or i == n_points - 1:
            print(f"    Progress: {i + 1}/{n_points} points processed", end="\r")
    
    print()  # New line after progress
    
    # Check if we got features for all points
    if len(min_radii) != n_points:
        print(f"    ✗ ERROR: Only {len(min_radii)}/{n_points} points processed successfully")
        return None
    
    features = {
        'min_radius': np.array(min_radii),
        'max_radius': np.array(max_radii),
        'perimeter': np.array(perimeters),
        'area': np.array(areas)
    }
    
    # Print summary statistics
    print(f"    Feature extraction complete:")
    print(f"      Min radius: {np.mean(features['min_radius']):.2f} ± {np.std(features['min_radius']):.2f} mm")
    print(f"      Max radius: {np.mean(features['max_radius']):.2f} ± {np.std(features['max_radius']):.2f} mm")
    print(f"      Area: {np.mean(features['area']):.2f} ± {np.std(features['area']):.2f} mm²")
    
    # Save to VTK file with features as point data
    if output_vtk_path:
        save_centerline_with_features(centerline_points, features, output_vtk_path)
    
    return features


def save_centerline_with_features(
    centerline_points: np.ndarray,
    features: Dict[str, np.ndarray],
    output_path: str
) -> None:
    """
    Save centerline with cross-sectional features as VTK PolyData.
    
    The features are stored as point data arrays that can be visualized
    in ParaView or other VTK-compatible software.
    
    Args:
        centerline_points: Nx3 array of centerline points
        features: Dictionary of feature arrays (min_radius, max_radius, perimeter, area)
        output_path: Path to save VTK file
    """
    print(f"  Saving centerline with features to: {output_path}")
    
    # Create points
    vtk_points = vtk.vtkPoints()
    for point in centerline_points:
        vtk_points.InsertNextPoint(point)
    
    # Create lines connecting consecutive points
    lines = vtk.vtkCellArray()
    for i in range(len(centerline_points) - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)
    
    # Add feature arrays as point data
    for name, values in features.items():
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(len(values))
        for i, v in enumerate(values):
            array.SetValue(i, v)
        polydata.GetPointData().AddArray(array)
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"  ✓ Centerline with features saved")


def find_closest_centerline_index(
    centerline_points: np.ndarray,
    landmark: np.ndarray
) -> int:
    """
    Find the index of the centerline point closest to a landmark.
    
    Args:
        centerline_points: Nx3 array of centerline points
        landmark: 3D point [x, y, z]
    
    Returns:
        Index of closest centerline point
    """
    distances = np.linalg.norm(centerline_points - landmark, axis=1)
    return int(np.argmin(distances))


def extract_cross_section_at_point(
    mesh: vtk.vtkPolyData,
    centerline_points: np.ndarray,
    point_index: int,
    override_point: np.ndarray = None,
    override_tangent: np.ndarray = None
) -> Tuple[Optional[vtk.vtkPolyData], Optional[Dict[str, float]], np.ndarray, np.ndarray]:
    """
    Extract cross-sectional features at a specific centerline point.
    
    Args:
        mesh: VTK PolyData surface mesh
        centerline_points: Nx3 array of centerline points
        point_index: Index of the centerline point to extract features at
        override_point: Optional point to use instead of centerline point
        override_tangent: Optional tangent to use instead of computed tangent
    
    Returns:
        Tuple of (cross_section_polydata, features_dict, point, tangent)
        features_dict contains: min_radius, max_radius, perimeter, area
    """
    n_points = len(centerline_points)
    i = point_index
    
    # Use override point or centerline point
    if override_point is not None:
        point = np.array(override_point)
    else:
        point = centerline_points[i]
    
    # Use override tangent or compute from neighbors
    if override_tangent is not None:
        tangent = np.array(override_tangent)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            return None, None, point, np.array([1, 0, 0])
        tangent = tangent / tangent_norm
    else:
        # Compute tangent direction using neighboring points
        if i == 0:
            p0 = centerline_points[i]
            p2 = centerline_points[min(i + 2, n_points - 1)]
        elif i == n_points - 1:
            p0 = centerline_points[max(i - 2, 0)]
            p2 = centerline_points[i]
        else:
            p0 = centerline_points[i - 1]
            p2 = centerline_points[i + 1]
        
        tangent = p2 - p0
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            return None, None, point, np.array([1, 0, 0])
        
        tangent = tangent / tangent_norm
    
    # Create cutting plane perpendicular to centerline
    plane = vtk.vtkPlane()
    plane.SetOrigin(point)
    plane.SetNormal(tangent)
    
    # Cut mesh with plane
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(mesh)
    cutter.Update()
    
    intersected_points = cutter.GetOutput()
    
    # Find closed curve containing this centerline point
    closed_curve, perimeter, area = find_closed_curve_at_centerline_point(
        intersected_points, point
    )
    
    if closed_curve is None:
        return None, None, point, tangent
    
    # Extract curve points and compute distances
    curve_points = np.array([
        closed_curve.GetPoint(j) for j in range(closed_curve.GetNumberOfPoints())
    ])
    
    if curve_points.size == 0:
        return None, None, point, tangent
    
    distances = np.linalg.norm(curve_points - point, axis=1)
    
    features = {
        'min_radius': float(np.min(distances)),
        'max_radius': float(np.max(distances)),
        'perimeter': float(perimeter) if perimeter else 0.0,
        'area': float(area) if area else 0.0
    }
    
    return closed_curve, features, point, tangent


def detect_isthmus(
    areas: np.ndarray,
    min_idx: int = 10,
    max_idx: int = None,
    smoothing_window: int = 5
) -> Tuple[int, float]:
    """
    Detect the isthmus - the point of steepest relative area decrease.
    
    The isthmus is the anatomical narrowing between cartilaginous and bony
    portions of the ear canal. It's detected as the point where the relative
    area gradient is most negative (steepest proportional drop).
    
    Args:
        areas: Array of cross-sectional areas along centerline
        min_idx: Minimum index to search from (skip noisy start)
        max_idx: Maximum index to search to (default: 80% of length)
        smoothing_window: Window size for smoothing gradient
    
    Returns:
        Tuple of (isthmus_index, relative_gradient_at_isthmus)
    """
    n = len(areas)
    if max_idx is None:
        max_idx = int(n * 0.8)  # Don't look in the last 20%
    
    # Compute relative gradient: (area[i+1] - area[i]) / area[i]
    # This captures proportional change regardless of absolute area
    relative_gradients = np.zeros(n)
    for i in range(min_idx, min(max_idx, n - 1)):
        if areas[i] > 1e-6:  # Avoid division by zero
            relative_gradients[i] = (areas[i + 1] - areas[i]) / areas[i]
    
    # Apply smoothing to reduce noise
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(relative_gradients, kernel, mode='same')
    else:
        smoothed = relative_gradients
    
    # Find the point of maximum negative gradient (steepest drop)
    search_range = smoothed[min_idx:max_idx]
    if len(search_range) == 0:
        return min_idx, 0.0
    
    local_idx = np.argmin(search_range)
    isthmus_idx = min_idx + local_idx +1
    
    return isthmus_idx, float(smoothed[isthmus_idx+1])


def extract_landmark_cross_sections(
    mesh: vtk.vtkPolyData,
    centerline_points: np.ndarray,
    landmarks: Dict[str, np.ndarray],
    centerline_features: Dict[str, np.ndarray] = None,
    isthmus_method: str = "closed_selection",
    non_inverted_mesh: vtk.vtkPolyData = None,
    inverted_mask = None
    ) -> Dict[str, Dict]:
    """
    Extract cross-sectional features at specific landmark locations.
    
    Args:
        mesh: VTK PolyData surface mesh
        centerline_points: Nx3 array of centerline points
        landmarks: Dictionary mapping landmark names to 3D positions
            Expected: '1st bend', '2nd bend', 'Eardrum'
        centerline_features: Optional dict with 'area' array for isthmus detection
    
    Returns:
        Dictionary mapping landmark names to their cross-section data:
        {
            'landmark_name': {
                'index': centerline index,
                'point': centerline point,
                'tangent': tangent direction,
                'min_radius': float,
                'max_radius': float,
                'perimeter': float,
                'area': float,
                'cross_section': vtkPolyData (optional)
            }
        }
    """
    print(f"  Extracting cross-sections at landmark locations...")
    
    results = {}
    n_points = len(centerline_points)
    
    # Define landmarks to extract
    landmark_configs = [
        ('1st bend', '1st_bend'),
        ('2nd bend', '2nd_bend'),
    ]
    
    for landmark_name, result_key in landmark_configs:
        if landmark_name in landmarks:
            landmark_pos = landmarks[landmark_name]
            idx = find_closest_centerline_index(centerline_points, landmark_pos)
            
            cross_section, features, point, tangent = extract_cross_section_at_point(
                non_inverted_mesh, centerline_points, idx
            )
            
            result = {
                'index': idx,
                'point': point,
                'tangent': tangent,
                'cross_section': cross_section
            }
            
            if features:
                result.update(features)
                print(f"    {landmark_name}: idx={idx}, area={features['area']:.1f}mm², "
                      f"radii=[{features['min_radius']:.2f}, {features['max_radius']:.2f}]mm")
            else:
                print(f"    {landmark_name}: idx={idx}, features extraction failed")
            
            results[result_key] = result
    
    # Eardrum: use landmark position with averaged tangent from last 10 points
    # Average tangent from last 10 points (excluding the very last point for stability)
    # This gives a more stable plane normal for the eardrum cross-section
    eardrum_pos = landmarks.get('Eardrum')
    if eardrum_pos is not None:
        eardrum_idx = find_closest_centerline_index(centerline_points, eardrum_pos)
        
        # Compute averaged tangent from the last 10 points (indices n-11 to n-2)
        # excluding the last point (n-1) for stability
        start_idx = max(0, n_points - 11)
        end_idx = n_points - 1  # exclude last point
        
        tangents = []
        for j in range(start_idx, end_idx):
            if j == 0:
                t = centerline_points[j + 1] - centerline_points[j]
            elif j == n_points - 1:
                t = centerline_points[j] - centerline_points[j - 1]
            else:
                t = centerline_points[j + 1] - centerline_points[j - 1]
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-10:
                tangents.append(t / t_norm)
        
        if tangents:
            avg_tangent = np.mean(tangents, axis=0)
            avg_tangent = avg_tangent / np.linalg.norm(avg_tangent)
        else:
            avg_tangent = None
        
        cross_section, features, point, tangent = extract_cross_section_at_point(
            non_inverted_mesh, centerline_points, eardrum_idx,
            override_point=eardrum_pos,
            override_tangent=avg_tangent
        )
        
        result = {
            'index': eardrum_idx,
            'point': point,
            'tangent': tangent,
            'cross_section': cross_section
        }
        
        if features:
            result.update(features)
            print(f"    Eardrum (idx {eardrum_idx}): area={features['area']:.1f}mm², "
                  f"radii=[{features['min_radius']:.2f}, {features['max_radius']:.2f}]mm (avg tangent from last 10 pts)")
        else:
            print(f"    Eardrum (idx {eardrum_idx}): features extraction failed")
        
        results['eardrum'] = result
    else:
        print(f"    Eardrum: landmark not found in input")
    
    # Isthmus: alternative detection methods
    if centerline_features is not None and 'area' in centerline_features:
        if isthmus_method == "area_drop":
            areas = np.array(centerline_features['area'])
            isthmus_idx, gradient = detect_isthmus(areas)
            cross_section, features, point, tangent = extract_cross_section_at_point(
                non_inverted_mesh, centerline_points, isthmus_idx
            )
            result = {
                'index': isthmus_idx,
                'point': point,
                'tangent': tangent,
                'cross_section': cross_section,
                'relative_gradient': gradient
            }
            if features:
                result.update(features)
                print(f"    Isthmus (idx {isthmus_idx}): area={features['area']:.1f}mm², "
                      f"radii=[{features['min_radius']:.2f}, {features['max_radius']:.2f}]mm "
                      f"(gradient={gradient:.2%})")
            else:
                print(f"    Isthmus (idx {isthmus_idx}): features extraction failed")
            results['isthmus'] = result
        elif isthmus_method == "closed_section":
            # Generate mesh from inverted mask (surface only at 1/0 transition)
           
            if inverted_mask is not None:
                mesh_to_use = non_inverted_mesh#create_surface_mesh_from_mask(inverted_mask)
            else:
                mesh_to_use = non_inverted_mesh#mesh
            n = len(centerline_points)
            closed_flags = []
            cross_section_cache = []
            # Precompute closed status for all cross-sections
            for idx in range(n):
                cross_section, features, point, tangent = extract_cross_section_at_point(
                    mesh_to_use, centerline_points, idx
                )
                is_closed = False
                n_open_edges = -1
                n_regions = -1
                n_points = -1
                forms_loop = False
                if cross_section is not None:
                    # Use vtkStripper to convert lines to polylines
                    stripper = vtk.vtkStripper()
                    stripper.SetInputData(cross_section)
                    stripper.Update()
                    polydata = stripper.GetOutput()
                    n_cells = polydata.GetNumberOfCells()
                    for cell_id in range(n_cells):
                        cell = polydata.GetCell(cell_id)
                        pt_ids = cell.GetPointIds()
                        n_cell_pts = pt_ids.GetNumberOfIds()
                        if n_cell_pts > 2:
                            # Check if polyline is closed (first and last point IDs match)
                            if pt_ids.GetId(0) == pt_ids.GetId(n_cell_pts - 1):
                                is_closed = True
                                forms_loop = True
                                n_points = n_cell_pts
                                break
                    # For debug: count regions and open edges as before
                    feature_edges = vtk.vtkFeatureEdges()
                    feature_edges.SetInputData(cross_section)
                    feature_edges.BoundaryEdgesOn()
                    feature_edges.FeatureEdgesOff()
                    feature_edges.NonManifoldEdgesOff()
                    feature_edges.ManifoldEdgesOff()
                    feature_edges.Update()
                    n_open_edges = feature_edges.GetOutput().GetNumberOfCells()
                    n_regions = cross_section.GetNumberOfCells()
                #print(f"      idx={idx}: closed={is_closed}, open_edges={n_open_edges}, regions={n_regions}, points={n_points}, forms_loop={forms_loop}")
                closed_flags.append(is_closed)
                cross_section_cache.append((cross_section, features, point, tangent))
            # Find all contiguous closed regions and identify the longest one

            # Find indices of all closed cross-sections
            closed_indices = [i for i, flag in enumerate(closed_flags) if flag]

            # Group closed indices into contiguous regions
            closed_regions = []
            for k, g in groupby(enumerate(closed_indices), lambda ix: ix[0] - ix[1]):
                group = list(map(lambda x: x[1], g))
                closed_regions.append(group)

            # Find the longest contiguous closed region
            longest_region = max(closed_regions, key=len) if closed_regions else []

            # Use the initial point of the longest contiguous closed
            # region as the isthmus
            if longest_region:
                isthmus_idx = longest_region[0]
                cross_section, features, point, tangent = extract_cross_section_at_point(non_inverted_mesh, centerline_points, isthmus_idx)
                result = {
                    'index': isthmus_idx,
                    'point': point,
                    'tangent': tangent,
                    'cross_section': cross_section,
                    'relative_gradient': None
                }
                result.update(features)
                print(f"    Isthmus (idx {isthmus_idx}): area={features['area']:.1f}mm², "
                      f"radii=[{features['min_radius']:.2f}, {features['max_radius']:.2f}]mm (closed section method)")
                results['isthmus'] = result
            else:
                isthmus_idx = None
                print("    Falling back to area method")
                areas = np.array(centerline_features['area'])
                isthmus_idx, gradient = detect_isthmus(areas)
                cross_section, features, point, tangent = extract_cross_section_at_point(
                    non_inverted_mesh, centerline_points, isthmus_idx
                )
                result = {
                    'index': isthmus_idx,
                    'point': point,
                    'tangent': tangent,
                    'cross_section': cross_section,
                    'relative_gradient': gradient
                }
                if features:
                    result.update(features)
                    print(f"    Isthmus (idx {isthmus_idx}): area={features['area']:.1f}mm², "
                        f"radii=[{features['min_radius']:.2f}, {features['max_radius']:.2f}]mm "
                        f"(gradient={gradient:.2%})")
                else:
                    print(f"    Isthmus (idx {isthmus_idx}): features extraction failed")
                results['isthmus'] = result
            
        else:
            print(f"    Isthmus: unknown method '{isthmus_method}', skipping")
    else:
        print(f"    Isthmus: skipped (no centerline_features provided)")
    
    return results


def visualize_landmark_cross_sections(
    mesh: vtk.vtkPolyData,
    centerline: vtk.vtkPolyData,
    landmark_cross_sections: Dict[str, Dict],
    landmarks: Dict[str, np.ndarray],
    cbj_points: np.ndarray = None,
    cbj_closest_point: np.ndarray = None,
    cbj_cross_section: vtk.vtkPolyData = None,
    clipped_volume_mesh: vtk.vtkPolyData = None,
    title: str = "Landmark Cross-Sections"
) -> None:
    """
    Visualize mesh with cross-sectional planes at landmark locations.
    
    Args:
        mesh: VTK PolyData surface mesh
        centerline: VTK PolyData centerline
        landmark_cross_sections: Output from extract_landmark_cross_sections
        landmarks: Dictionary of landmark positions
        cbj_points: Optional array of CBJ landmark points (Nx3)
        cbj_closest_point: Optional CBJ centerline intersection point
        cbj_cross_section: Optional CBJ cross-section curve polydata
        title: Window title
    """
    print(f"  Opening visualization: {title}")
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background
    
    # Add mesh (transparent)
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(mesh)
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    mesh_actor.GetProperty().SetOpacity(0.3)
    renderer.AddActor(mesh_actor)
    
    # Add centerline
    cl_mapper = vtk.vtkPolyDataMapper()
    cl_mapper.SetInputData(centerline)
    cl_actor = vtk.vtkActor()
    cl_actor.SetMapper(cl_mapper)
    cl_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    cl_actor.GetProperty().SetLineWidth(3)
    renderer.AddActor(cl_actor)
    
    # Add clipped volume (light cyan, semi-transparent) - visualization of computed volume
    if clipped_volume_mesh is not None and clipped_volume_mesh.GetNumberOfCells() > 0:
        volume_mapper = vtk.vtkPolyDataMapper()
        volume_mapper.SetInputData(clipped_volume_mesh)
        volume_actor = vtk.vtkActor()
        volume_actor.SetMapper(volume_mapper)
        volume_actor.GetProperty().SetOpacity(0.4)
        volume_actor.GetProperty().SetColor(0.0, 1.0, 1.0)  # Cyan
        renderer.AddActor(volume_actor)
    
    # Colors for different landmarks
    colors = {
        '1st_bend': (1.0, 0.5, 0.0),      # Orange
        '2nd_bend': (0.0, 1.0, 0.5),      # Green
        'eardrum': (1.0, 0.0, 1.0),       # Magenta
        'isthmus': (1.0, 1.0, 0.0),       # Yellow
    }
    
    # Add cross-sectional curves, disks, and landmark spheres
    for name, data in landmark_cross_sections.items():
        color = colors.get(name, (1.0, 1.0, 0.0))
        point = data['point']
        tangent = data.get('tangent')
        
        # Add cross-section curve if available
        if data.get('cross_section') is not None:
            curve_mapper = vtk.vtkPolyDataMapper()
            curve_mapper.SetInputData(data['cross_section'])
            curve_actor = vtk.vtkActor()
            curve_actor.SetMapper(curve_mapper)
            curve_actor.GetProperty().SetColor(*color)
            curve_actor.GetProperty().SetLineWidth(4)
            renderer.AddActor(curve_actor)
        
        # Add semi-transparent disk/circle for cross-sectional area
        if data.get('area') is not None and tangent is not None:
            area = data['area']
            # Calculate radius from area: A = π*r² => r = √(A/π)
            disk_radius = np.sqrt(area / np.pi)
            
            # Create disk source (circle in XY plane)
            disk = vtk.vtkDiskSource()
            disk.SetInnerRadius(0.0)
            disk.SetOuterRadius(disk_radius)
            disk.SetCircumferentialResolution(32)
            disk.SetRadialResolution(2)
            disk.Update()
            
            # Transform disk to align with tangent direction and position
            # Tangent vector defines the plane normal
            tangent_unit = tangent / (np.linalg.norm(tangent) + 1e-10)
            
            # Compute rotation to align Z-axis (disk normal) with tangent
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, tangent_unit)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            transform = vtk.vtkTransform()
            transform.Translate(point)
            
            if rotation_axis_norm > 1e-6:
                # Compute rotation angle
                angle_rad = np.arccos(np.clip(np.dot(z_axis, tangent_unit), -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)
                rotation_axis_normalized = rotation_axis / rotation_axis_norm
                transform.RotateWXYZ(angle_deg, *rotation_axis_normalized)
            elif np.dot(z_axis, tangent_unit) < 0:
                # Z-axis and tangent point in opposite directions (180 degrees)
                transform.RotateWXYZ(180, 1, 0, 0)
            
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(disk.GetOutputPort())
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            
            disk_mapper = vtk.vtkPolyDataMapper()
            disk_mapper.SetInputConnection(transform_filter.GetOutputPort())
            
            disk_actor = vtk.vtkActor()
            disk_actor.SetMapper(disk_mapper)
            disk_actor.GetProperty().SetColor(*color)
            disk_actor.GetProperty().SetOpacity(0.35)  # Semi-transparent
            renderer.AddActor(disk_actor)
        
        # Add sphere at centerline point
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(point)
        sphere.SetRadius(1.5)
        sphere.Update()
        
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputData(sphere.GetOutput())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(*color)
        renderer.AddActor(sphere_actor)
    
    # Add original landmark positions as small spheres
    landmark_colors = {
        '1st bend': (1.0, 0.5, 0.0),
        '2nd bend': (0.0, 1.0, 0.5),
        'beginning_eac': (0.0, 0.5, 1.0),
        'Eardrum': (1.0, 0.0, 1.0),
    }
    
    for name, pos in landmarks.items():
        if name in landmark_colors:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(pos)
            sphere.SetRadius(0.8)
            sphere.Update()
            
            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputData(sphere.GetOutput())
            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.GetProperty().SetColor(*landmark_colors[name])
            sphere_actor.GetProperty().SetOpacity(0.7)
            renderer.AddActor(sphere_actor)
    
    # Add CBJ landmarks if available
    if cbj_points is not None and len(cbj_points) > 0:
        print(f"    ✓ Adding {len(cbj_points)} CBJ landmark points (purple spheres)")
        for i, cbj_pt in enumerate(cbj_points):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(cbj_pt)
            sphere.SetRadius(1.0)
            sphere.Update()
            
            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputData(sphere.GetOutput())
            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.GetProperty().SetColor(0.8, 0.2, 0.8)  # Purple
            renderer.AddActor(sphere_actor)
    
    # Add CBJ cross-section curve if available
    if cbj_cross_section is not None:
        try:
            print(f"    ✓ Adding CBJ cross-section curve (purple line)")
            cbj_curve_mapper = vtk.vtkPolyDataMapper()
            cbj_curve_mapper.SetInputData(cbj_cross_section)
            cbj_curve_actor = vtk.vtkActor()
            cbj_curve_actor.SetMapper(cbj_curve_mapper)
            cbj_curve_actor.GetProperty().SetColor(0.8, 0.2, 0.8)  # Purple
            cbj_curve_actor.GetProperty().SetLineWidth(4)
            renderer.AddActor(cbj_curve_actor)
        except Exception as e:
            print(f"    Warning: Could not draw CBJ cross-section: {e}")
    
    # Add CBJ centerline intersection point if available
    if cbj_closest_point is not None:
        print(f"    ✓ Adding CBJ centerline intersection point (purple sphere)")
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(cbj_closest_point)
        sphere.SetRadius(1.5)
        sphere.Update()
        
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputData(sphere.GetOutput())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(0.8, 0.2, 0.8)  # Purple
        renderer.AddActor(sphere_actor)
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName(title)
    render_window.SetSize(1200, 800)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # # Add axes
    # axes = vtk.vtkAxesActor()
    # axes.SetTotalLength(20, 20, 20)
    # renderer.AddActor(axes)
    
    renderer.ResetCamera()
    render_window.Render()
    
    print(f"\n  Legend:")
    print(f"    - Gray transparent: Mesh surface")
    print(f"    - Red line: Centerline")
    print(f"    - Orange disk/curve: 1st bend cross-section (disk shows area)")
    print(f"    - Green disk/curve: 2nd bend cross-section (disk shows area)")
    print(f"    - Magenta disk/curve: Eardrum cross-section (disk shows area)")
    print(f"    - Yellow disk/curve: Isthmus cross-section (disk shows area)")
    if cbj_points is not None or cbj_cross_section is not None:
        print(f"    - Purple small spheres: CBJ landmark points")
        print(f"    - Purple line: CBJ cross-section curve")
        print(f"    - Purple large sphere: CBJ-centerline intersection point")
    print(f"\n  Close window to continue...")
    
    interactor.Start()


def cut_mesh_at_plane(mesh: vtk.vtkPolyData, 
                      plane_centroid: np.ndarray,
                      plane_normal: np.ndarray) -> Optional[vtk.vtkPolyData]:
    """
    Cut mesh at a plane and return the clipped portion.
    
    Args:
        mesh: VTK PolyData surface mesh
        plane_centroid: Center point of cutting plane
        plane_normal: Normal vector of cutting plane
    
    Returns:
        vtkPolyData: Clipped mesh on positive side of plane, or None if failed
    """
    try:
        # Normalize normal
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # Create cutting plane
        vtk_plane = vtk.vtkPlane()
        vtk_plane.SetOrigin(plane_centroid)
        vtk_plane.SetNormal(plane_normal)
        
        # Clip mesh
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(mesh)
        clipper.SetClipFunction(vtk_plane)
        clipper.Update()
        
        return clipper.GetOutput()
    except Exception as e:
        print(f"    ⚠ Error cutting mesh: {e}")
        return None


def create_cap_from_curve(curve_points: np.ndarray,
                         curve_normal: np.ndarray,
                         curve_center: np.ndarray) -> Optional[vtk.vtkPolyData]:
    """
    Create a closed cap (surface) from a closed curve.
    
    Args:
        curve_points: Nx3 array of ordered points forming a closed curve
        curve_normal: Normal vector of the plane containing the curve
        curve_center: Center point of the curve
    
    Returns:
        vtkPolyData: Surface cap, or None if failed
    """
    try:
        # Create polydata from curve points
        pts = vtk.vtkPoints()
        for pt in curve_points:
            pts.InsertNextPoint(pt)
        
        # Create polygon lines connecting all points in a circle
        lines = vtk.vtkCellArray()
        for i in range(len(curve_points)):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, (i + 1) % len(curve_points))
            lines.InsertNextCell(line)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(pts)
        polydata.SetLines(lines)
        
        # Use vtkFillHolesFilter to create the cap (2D surface)
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputData(polydata)
        filler.SetHoleSize(1000.0)  # Large hole size to fill entire boundary
        filler.Update()
        
        cap = filler.GetOutput()
        return cap if cap.GetNumberOfCells() > 0 else None
        
    except Exception as e:
        print(f"    ⚠ Error creating cap: {e}")
        return None


def append_meshes(mesh1: vtk.vtkPolyData, 
                 mesh2: vtk.vtkPolyData) -> Optional[vtk.vtkPolyData]:
    """
    Append two meshes together.
    
    Args:
        mesh1: First mesh
        mesh2: Second mesh
    
    Returns:
        vtkPolyData: Combined mesh
    """
    try:
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(mesh1)
        appender.AddInputData(mesh2)
        appender.Update()
        return appender.GetOutput()
    except Exception as e:
        print(f"    ⚠ Error appending meshes: {e}")
        return None


def ensure_triangles_only(mesh: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    Convert all polygons in a mesh to triangles.
    
    Args:
        mesh: VTK PolyData mesh that may contain polygons
        
    Returns:
        New vtkPolyData with only triangles
    """
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(mesh)
    triangulator.Update()
    triangulated_mesh = triangulator.GetOutput()
    
    n_tris = triangulated_mesh.GetNumberOfCells()
    n_points = triangulated_mesh.GetNumberOfPoints()
    
    print(f"      ✓ Triangulated: {n_tris} cells, {n_points} points")
    
    return triangulated_mesh


def mesh_is_closed(mesh: vtk.vtkPolyData) -> bool:
    """
    Check if a mesh is closed (no boundary edges).
    
    Args:
        mesh: VTK PolyData surface mesh
    
    Returns:
        bool: True if mesh is closed, False otherwise
    """
    edges = vtk.vtkExtractEdges()
    edges.SetInputData(mesh)
    edges.Update()
    
    edge_data = edges.GetOutput()
    return edge_data.GetNumberOfLines() == 0


def close_mesh_iteratively(mesh: vtk.vtkPolyData, max_iterations: int = 5) -> vtk.vtkPolyData:
    """
    Iteratively close a mesh by repeatedly filling holes until no boundary edges remain.
    Triangulates after each fill to ensure only triangles.
    
    Args:
        mesh: VTK PolyData mesh to close
        max_iterations: Maximum number of fill/check iterations
    
    Returns:
        Closed vtkPolyData mesh with only triangles
    """
    current_mesh = mesh
    iteration = 0
    
    while iteration < max_iterations:
        is_closed = mesh_is_closed(current_mesh)
        n_cells = current_mesh.GetNumberOfCells()
        n_points = current_mesh.GetNumberOfPoints()
        
        if iteration == 0:
            print(f"      Starting mesh: {n_cells} cells, {n_points} points, closed={is_closed}")
        else:
            print(f"      Iteration {iteration}: {n_cells} cells, {n_points} points, closed={is_closed}")
        
        if is_closed:
            # Triangulate one final time before returning
            print(f"    ✓ Mesh is now fully closed after {iteration} fill operations")
            print(f"      Final triangulation...")
            current_mesh = ensure_triangles_only(current_mesh)
            return current_mesh
        
        # Fill holes with increasing tolerance
        hole_size = 10000.0 * (iteration + 1)  # Increase hole size each iteration
        print(f"      Filling holes with size threshold {hole_size:.1f} mm²...")
        
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputData(current_mesh)
        filler.SetHoleSize(hole_size)
        filler.Update()
        
        current_mesh = filler.GetOutput()
        n_cells_after = current_mesh.GetNumberOfCells()
        print(f"      After fill: {n_cells_after} cells")
        
        # Triangulate immediately after filling to convert any polygons to triangles
        current_mesh = ensure_triangles_only(current_mesh)
        
        # Stop if no new cells were added
        if n_cells_after == n_cells:
            print(f"    ⚠ No new cells added, mesh may have unfillable holes")
            break
        
        iteration += 1
    
    # Final check and triangulate
    is_closed_final = mesh_is_closed(current_mesh)
    n_cells_final = current_mesh.GetNumberOfCells()
    print(f"    Final result after {iteration} iterations: {n_cells_final} cells, closed={is_closed_final}")
    print(f"      Final triangulation...")
    current_mesh = ensure_triangles_only(current_mesh)
    
    return current_mesh


def extract_isthmus_eardrum_segment(mesh: vtk.vtkPolyData,
                                    isthmus_point: np.ndarray,
                                    isthmus_normal: np.ndarray,
                                    eardrum_point: np.ndarray,
                                    eardrum_normal: np.ndarray,
                                    isthmus_curve: vtk.vtkPolyData,
                                    eardrum_curve: vtk.vtkPolyData,
                                    centerline_points: np.ndarray = None,
                                    isthmus_idx: int = None,
                                    eardrum_idx: int = None,
                                    second_bend_point: Optional[np.ndarray] = None,
                                    output_dir: str = None,
                                    sample_name: str = None,
                                    visualize: bool = False) -> Tuple[Optional[vtk.vtkPolyData], Optional[float]]:
    """
    Extract mesh segment between isthmus and eardrum by cutting at isthmus plane,
    create shaped caps from the curves, and compute volume.
    
    Args:
        mesh: Full ear canal mesh
        isthmus_point: Center point of isthmus cross-section
        isthmus_normal: Normal vector at isthmus (pointing toward eardrum)
        eardrum_point: Center point of eardrum cross-section
        eardrum_normal: Normal vector at eardrum (pointing toward eardrum)
        isthmus_curve: VTK PolyData containing isthmus cross-section curve
        eardrum_curve: VTK PolyData containing eardrum cross-section curve
        centerline_points: Centerline points (optional, for identifying correct segment)
        isthmus_idx: Centerline index at isthmus (optional)
        eardrum_idx: Centerline index at eardrum (optional)
        visualize: Whether to visualize intermediate results
    
    Returns:
        Tuple of (closed_mesh, volume_mm3)
    """
    print(f"\n  Extracting mesh segment between isthmus and eardrum...")
    
    try:
        # Normalize normals
        isthmus_normal = np.array(isthmus_normal)
        eardrum_normal = np.array(eardrum_normal)
        
        norm = np.linalg.norm(isthmus_normal)
        if norm > 0:
            isthmus_normal = isthmus_normal / norm
        
        norm = np.linalg.norm(eardrum_normal)
        if norm > 0:
            eardrum_normal = eardrum_normal / norm
        
        print(f"    Isthmus plane at {isthmus_point} with normal {isthmus_normal}")
        print(f"    Eardrum plane at {eardrum_point} with normal {eardrum_normal}")
        
        # STEP 1: Crop at eardrum plane with 3cm x 3cm plane - creates an open surface
        print(f"    STEP 1: Cropping mesh at eardrum plane (3cm x 3cm plane)...")
        
        # Create a box (30mm x 30mm) as the cutting surface at eardrum
        eardrum_box = vtk.vtkPlaneSource()
        eardrum_box.SetOrigin(eardrum_point - np.array([15, 15, 0]) @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        eardrum_box.SetPoint1(eardrum_point - np.array([15, -15, 0]) @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        eardrum_box.SetPoint2(eardrum_point - np.array([-15, 15, 0]) @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        eardrum_box.Update()
        
        eardrum_plane = vtk.vtkPlane()
        eardrum_plane.SetOrigin(eardrum_point)
        eardrum_plane.SetNormal(eardrum_normal)
        
        clipper_eardrum = vtk.vtkClipPolyData()
        clipper_eardrum.SetInputData(mesh)
        clipper_eardrum.SetClipFunction(eardrum_plane)
        clipper_eardrum.InsideOutOn()  # Keep part on normal side (toward isthmus)
        clipper_eardrum.Update()
        
        cropped_mesh = clipper_eardrum.GetOutput()
        n_cells = cropped_mesh.GetNumberOfCells()
        n_points = cropped_mesh.GetNumberOfPoints()
        print(f"      ✓ After eardrum crop: {n_cells} cells, {n_points} points")
        
        if n_cells == 0:
            print(f"      ⚠ Eardrum cropping produced no cells, trying with InsideOutOff...")
            clipper_eardrum_alt = vtk.vtkClipPolyData()
            clipper_eardrum_alt.SetInputData(mesh)
            clipper_eardrum_alt.SetClipFunction(eardrum_plane)
            clipper_eardrum_alt.InsideOutOff()
            clipper_eardrum_alt.Update()
            cropped_mesh = clipper_eardrum_alt.GetOutput()
            print(f"      ✓ With InsideOutOff: {cropped_mesh.GetNumberOfCells()} cells")
        
        if cropped_mesh.GetNumberOfCells() == 0:
            print(f"      ⚠ Both eardrum clip methods produced 0 cells, returning None")
            return None, None
        
        # Create a capping surface using the eardrum cross-sectional curve
        print(f"    STEP 1.2: Creating cap surface from eardrum curve...")
        
        # eardrum_curve is the cross-section at the eardrum location
        # Convert to vtkPolyData if needed
        if not isinstance(eardrum_curve, vtk.vtkPolyData):
            eardrum_curve_poly = pv.wrap(eardrum_curve).extract_geometry()
        else:
            eardrum_curve_poly = eardrum_curve
        
        # Create a planar surface capped by the eardrum curve boundary
        curve_points = eardrum_curve_poly.GetPoints()
        n_curve_points = curve_points.GetNumberOfPoints()
        
        if n_curve_points > 2:
            # Use vtkContourTriangulator to fill the area bounded by the curve
            print(f"      Creating planar surface bounded by curve ({n_curve_points} points)...")
            contour_triangulator = vtk.vtkContourTriangulator()
            contour_triangulator.SetInputData(eardrum_curve_poly)
            contour_triangulator.Update()
            cap_surface = contour_triangulator.GetOutput()
            
            print(f"      + Generated cap surface: {cap_surface.GetNumberOfCells()} triangles, {cap_surface.GetNumberOfPoints()} points")
            
            # Save the eardrum cap as a separate STL file
            if output_dir and sample_name:
                stl_folder = os.path.join(output_dir, "stl")
                os.makedirs(stl_folder, exist_ok=True)
                eardrum_cap_stl = os.path.join(stl_folder, f"{sample_name}_eardrum.stl")
                writer = vtk.vtkSTLWriter()
                writer.SetFileName(eardrum_cap_stl)
                writer.SetInputData(cap_surface)
                writer.Write()
                print(f"      + Saved eardrum cap to: {eardrum_cap_stl}")
            
            # Append cap to the cropped mesh
            appender = vtk.vtkAppendPolyData()
            appender.AddInputData(cropped_mesh)
            appender.AddInputData(cap_surface)
            appender.Update()
            cropped_mesh = appender.GetOutput()
            print(f"      + Added eardrum surface cap: {cropped_mesh.GetNumberOfCells()} cells")
        else:
            print(f"      - Eardrum curve has insufficient points, skipping cap")
        
        # Save the eardrum-cut mesh with cap as STL (BEFORE ensure_single_solid to preserve cap)
        if output_dir and sample_name:
            stl_folder = os.path.join(output_dir, "stl")
            os.makedirs(stl_folder, exist_ok=True)
            stl_path = os.path.join(stl_folder, f"{sample_name}_open_surface_cut_eardrum.stl")
            # Save directly without ensure_mesh_is_single_solid to preserve the appended cap
            pv_mesh = pv.wrap(cropped_mesh)
            pv_mesh.save(stl_path)
            print(f"      + Saved eardrum-cut mesh to: {stl_path}")
        
        # Visualize the eardrum-cut mesh
        if visualize:
            print(f"    STEP 1.5: Visualizing mesh with eardrum cut...")
            visualize_volume_segment(cropped_mesh, eardrum_point, eardrum_point, 0.0, 
                                     title="Mesh Cut at Eardrum Plane (3cm x 3cm)")
        
        # STEP 2: Crop at isthmus plane - keep the part toward eardrum
        print(f"    STEP 2: Cropping mesh at isthmus plane (keeping eardrum direction)...")
        
        isthmus_plane = vtk.vtkPlane()
        isthmus_plane.SetOrigin(isthmus_point)
        isthmus_plane.SetNormal(-isthmus_normal)  # Invert normal to keep correct side
        
        clipper1 = vtk.vtkClipPolyData()
        clipper1.SetInputData(cropped_mesh)
        clipper1.SetClipFunction(isthmus_plane)
        clipper1.InsideOutOn()  # Keep part on normal side (toward eardrum)
        clipper1.Update()
        
        cropped_mesh = clipper1.GetOutput()
        n_cells = cropped_mesh.GetNumberOfCells()
        n_points = cropped_mesh.GetNumberOfPoints()
        print(f"      ✓ After isthmus crop: {n_cells} cells, {n_points} points")
        
        if n_cells == 0:
            print(f"      ⚠ Isthmus cropping produced no cells, trying with InsideOutOff...")
            clipper1_alt = vtk.vtkClipPolyData()
            clipper1_alt.SetInputData(cropped_mesh)
            clipper1_alt.SetClipFunction(isthmus_plane)
            clipper1_alt.InsideOutOff()
            clipper1_alt.Update()
            cropped_mesh = clipper1_alt.GetOutput()
            print(f"      ✓ With InsideOutOff: {cropped_mesh.GetNumberOfCells()} cells")
        
        if cropped_mesh.GetNumberOfCells() == 0:
            print(f"      ⚠ Both isthmus clip methods produced 0 cells, returning None")
            return None, None
        
        # STEP 3: Clean
        print(f"    STEP 3: Cleaning mesh...")
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(cropped_mesh)
        cleaner.Update()
        closed_mesh = cleaner.GetOutput()
        print(f"      ✓ After cleaning: {closed_mesh.GetNumberOfCells()} cells")
        
        # STEP 4: Create cap at isthmus cut location using isthmus curve
        print(f"    STEP 4: Creating cap surface from isthmus curve...")
        
        # isthmus_curve is the cross-section at the isthmus location
        if not isinstance(isthmus_curve, vtk.vtkPolyData):
            isthmus_curve_poly = pv.wrap(isthmus_curve).extract_geometry()
        else:
            isthmus_curve_poly = isthmus_curve
        
        # Create a planar surface capped by the isthmus curve boundary
        icurve_points = isthmus_curve_poly.GetPoints()
        n_icurve_points = icurve_points.GetNumberOfPoints()
        
        if n_icurve_points > 2:
            # Use vtkContourTriangulator to fill the area bounded by the curve
            print(f"      Creating planar surface bounded by curve ({n_icurve_points} points)...")
            contour_triangulator = vtk.vtkContourTriangulator()
            contour_triangulator.SetInputData(isthmus_curve_poly)
            contour_triangulator.Update()
            isthmus_cap_surface = contour_triangulator.GetOutput()
            
            print(f"      + Generated cap surface: {isthmus_cap_surface.GetNumberOfCells()} triangles, {isthmus_cap_surface.GetNumberOfPoints()} points")
            
            # Save the isthmus cap as a separate STL file
            if output_dir and sample_name:
                stl_folder = os.path.join(output_dir, "stl")
                os.makedirs(stl_folder, exist_ok=True)
                isthmus_cap_stl = os.path.join(stl_folder, f"{sample_name}_isthmus.stl")
                writer = vtk.vtkSTLWriter()
                writer.SetFileName(isthmus_cap_stl)
                writer.SetInputData(isthmus_cap_surface)
                writer.Write()
                print(f"      + Saved isthmus cap to: {isthmus_cap_stl}")
            
            # Append isthmus cap to the mesh
            appender2 = vtk.vtkAppendPolyData()
            appender2.AddInputData(closed_mesh)
            appender2.AddInputData(isthmus_cap_surface)
            appender2.Update()
            closed_mesh = appender2.GetOutput()
            print(f"      + Added isthmus surface cap: {closed_mesh.GetNumberOfCells()} cells")
            
            # CRITICAL: Triangulate immediately after appending cap
            closed_mesh = ensure_triangles_only(closed_mesh)
        else:
            print(f"      - Isthmus curve has insufficient points, skipping cap")
        
        # STEP 5: Select connected component containing new_second_bend landmark
        print(f"    STEP 5: Selecting component containing new_second_bend landmark...")
        
        # Get all connected components
        all_components = vtk.vtkPolyDataConnectivityFilter()
        all_components.SetInputData(closed_mesh)
        all_components.SetExtractionModeToAllRegions()
        all_components.Update()
        
        n_components = all_components.GetNumberOfExtractedRegions()
        print(f"      Found {n_components} connected component(s)")
        
        if n_components == 0:
            print(f"      ⚠ No connected components found")
            return None, None
        
        # Debug: print component sizes
        print(f"      Component Breakdown:")
        for comp_id in range(n_components):
            extractor = vtk.vtkPolyDataConnectivityFilter()
            extractor.SetInputData(closed_mesh)
            extractor.SetExtractionModeToSpecifiedRegions()
            extractor.AddSpecifiedRegion(comp_id)
            extractor.Update()
            component = extractor.GetOutput()
            print(f"        Component {comp_id}: {component.GetNumberOfCells()} cells, {component.GetNumberOfPoints()} points")
        
        # Visualize all components with different colors
        if visualize:
            print(f"      Visualizing all {n_components} component(s)...")
            plotter = pv.Plotter(title="Connected Components")
            
            # Color map for components
            colors = [
                'red', 'blue', 'green', 'yellow', 'cyan', 'magenta',
                'orange', 'purple', 'brown', 'pink', 'gray', 'olive'
            ]
            
            for comp_id in range(n_components):
                # Extract this component
                extractor = vtk.vtkPolyDataConnectivityFilter()
                extractor.SetInputData(closed_mesh)
                extractor.SetExtractionModeToSpecifiedRegions()
                extractor.AddSpecifiedRegion(comp_id)
                extractor.Update()
                component = extractor.GetOutput()
                
                color = colors[comp_id % len(colors)]
                n_cells = component.GetNumberOfCells()
                plotter.add_mesh(component, color=color, label=f"Component {comp_id} ({n_cells} cells)")
            
            # Add isthmus and eardrum landmarks
            isthmus_pt = np.array(isthmus_point)
            eardrum_pt = np.array(eardrum_point)
            plotter.add_points(isthmus_pt.reshape(1, -1), color='white', point_size=10, label='isthmus')
            plotter.add_points(eardrum_pt.reshape(1, -1), color='yellow', point_size=10, label='eardrum')
            
            plotter.add_legend()
            plotter.show()
            print(f"      ✓ Visualization closed. Continuing...")
        
        # Jitter the landmark points slightly (move them inward from surface)
        # This avoids the issue where landmarks exactly on mesh vertices show 0.00mm on all components
        print(f"    Jittering landmark points to disambiguate components...")
        
        # Move landmarks slightly inward/offset from their exact surface position
        # Direction: toward centerline (use average direction toward center of mesh)
        from scipy.spatial import cKDTree
        
        # Get component 0 as reference for jitter direction (largest component)
        ref_extractor = vtk.vtkPolyDataConnectivityFilter()
        ref_extractor.SetInputData(closed_mesh)
        ref_extractor.SetExtractionModeToSpecifiedRegions()
        ref_extractor.AddSpecifiedRegion(0)
        ref_extractor.Update()
        ref_component = ref_extractor.GetOutput()
        
        # Compute centroid of largest component
        mesh_center = np.zeros(3)
        n_ref_points = ref_component.GetNumberOfPoints()
        for i in range(n_ref_points):
            mesh_center += np.array(ref_component.GetPoint(i))
        mesh_center = mesh_center / n_ref_points if n_ref_points > 0 else mesh_center
        
        # Jitter isthmus point 0.3mm toward the centerline
        isthmus_array = np.array(isthmus_point)
        isthmus_to_center = mesh_center - isthmus_array
        isthmus_to_center_norm = np.linalg.norm(isthmus_to_center)
        if isthmus_to_center_norm > 1e-6:
            isthmus_jitter_direction = isthmus_to_center / isthmus_to_center_norm
        else:
            isthmus_jitter_direction = np.array([0, 0, 1])
        isthmus_jittered = isthmus_array + 0.3 * isthmus_jitter_direction
        
        # Jitter eardrum point 0.3mm toward the centerline
        eardrum_array = np.array(eardrum_point)
        eardrum_to_center = mesh_center - eardrum_array
        eardrum_to_center_norm = np.linalg.norm(eardrum_to_center)
        if eardrum_to_center_norm > 1e-6:
            eardrum_jitter_direction = eardrum_to_center / eardrum_to_center_norm
        else:
            eardrum_jitter_direction = np.array([0, 0, 1])
        eardrum_jittered = eardrum_array + 0.3 * eardrum_jitter_direction
        
        print(f"      Isthmus: {isthmus_array} → {isthmus_jittered} (moved {np.linalg.norm(isthmus_jittered - isthmus_array):.3f}mm)")
        print(f"      Eardrum: {eardrum_array} → {eardrum_jittered} (moved {np.linalg.norm(eardrum_jittered - eardrum_array):.3f}mm)")
        
        # Find component containing both isthmus and eardrum landmarks
        print(f"    Finding component containing both isthmus and eardrum landmarks...")
        
        # IMPORTANT: First extract each component as a truly separate mesh
        print(f"    Separating all components into independent meshes...")
        separated_components = []
        for comp_id in range(n_components):
            # Extract this component
            extractor = vtk.vtkPolyDataConnectivityFilter()
            extractor.SetInputData(closed_mesh)
            extractor.SetExtractionModeToSpecifiedRegions()
            extractor.AddSpecifiedRegion(comp_id)
            extractor.Update()
            component_mesh = extractor.GetOutput()
            
            # Clean to ensure it's a standalone mesh with no shared references
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(component_mesh)
            cleaner.Update()
            cleaned_component = cleaner.GetOutput()
            
            n_cells = cleaned_component.GetNumberOfCells()
            n_points = cleaned_component.GetNumberOfPoints()
            
            separated_components.append({
                'id': comp_id,
                'mesh': cleaned_component,
                'n_cells': n_cells,
                'n_points': n_points
            })
            print(f"      Separated Component {comp_id}: {n_cells} cells, {n_points} points")
        
        # Now compute distances from landmarks to each separated component
        print(f"\n    Computing landmark distances to separated components...")
        component_landmark_scores = []
        
        for comp_info in separated_components:
            comp_id = comp_info['id']
            component_mesh = comp_info['mesh']
            n_cells = comp_info['n_cells']
            
            if n_cells == 0:
                continue
            
            # Build spatial index for this separated component
            from scipy.spatial import cKDTree
            component_points_array = np.array([
                component_mesh.GetPoint(i) for i in range(component_mesh.GetNumberOfPoints())
            ])
            tree = cKDTree(component_points_array)
            
            # Find closest distance from isthmus jittered point to this component
            isthmus_dist, _ = tree.query(isthmus_jittered)
            
            # Find closest distance from eardrum jittered point to this component
            eardrum_dist, _ = tree.query(eardrum_jittered)
            
            # Score: use maximum distance (both landmarks must be on surface)
            landmark_score = max(isthmus_dist, eardrum_dist)
            
            component_landmark_scores.append({
                'id': comp_id,
                'n_cells': n_cells,
                'isthmus_dist': isthmus_dist,
                'eardrum_dist': eardrum_dist,
                'score': landmark_score
            })
            
            print(f"        Component {comp_id}: {n_cells} cells | Isthmus dist: {isthmus_dist:.3f}mm | Eardrum dist: {eardrum_dist:.3f}mm | Max: {landmark_score:.3f}mm")
        
        # Sort by landmark score (lowest = both landmarks closest), then by cell count (descending)
        component_landmark_scores.sort(key=lambda x: (x['score'], -x['n_cells']))
        
        print(f"\n      Sorted by landmark proximity (lowest score = best):")
        for comp in component_landmark_scores:
            print(f"        Component {comp['id']}: score={comp['score']:.3f}mm (isthmus={comp['isthmus_dist']:.3f}mm, eardrum={comp['eardrum_dist']:.3f}mm)")
        
        if len(component_landmark_scores) == 0:
            print(f"      ⚠ No valid components found")
            return None, None
        
        # Select component with lowest score (both landmarks closest to surface)
        selected_component_id = component_landmark_scores[0]['id']
        selected_info = component_landmark_scores[0]
        print(f"      ✓ Selected component: ID {selected_component_id} (score={selected_info['score']:.3f}mm: isthmus={selected_info['isthmus_dist']:.3f}mm, eardrum={selected_info['eardrum_dist']:.3f}mm)")
        
        # Use the separated mesh for the selected component
        closed_mesh = separated_components[selected_component_id]['mesh']
        
        print(f"      ✓ Selected component {selected_component_id}")
        print(f"      ✓ Component: {closed_mesh.GetNumberOfCells()} cells, {closed_mesh.GetNumberOfPoints()} points")
        
        if closed_mesh.GetNumberOfCells() == 0:
            print(f"      ⚠ No cells in selected component")
            return None, None
        
        # STEP 6: Close any remaining holes before volume computation
        print(f"    STEP 6: Closing mesh for volume calculation (iterative)...")
        
        # Use iterative hole filling until mesh is fully closed
        closed_mesh = close_mesh_iteratively(closed_mesh, max_iterations=5)
        print(f"      ✓ Mesh closure complete: {closed_mesh.GetNumberOfCells()} cells")
        
        # STEP 7: Compute volume using voxel-based method
        print(f"    STEP 7: Computing volume using voxel-based method (voxel_spacing=0.05 mm)...")
        volume, _ = compute_voxel_mesh_volume(
            mesh_or_path=closed_mesh,
            voxel_spacing=0.05,
            save_cleaned_stl=None,
            return_mesh=False,
            verbose=True
        )
        
        if volume is None or volume <= 0:
            print(f"      ⚠ Voxel volume computation failed")
            volume = 0.0
        
        # STEP 8: Save canal mesh as STL if output_dir and sample_name provided
        # DISABLED: Not saving _canal_only.stl for now
        # if output_dir is not None and sample_name is not None:
        #     stl_dir = os.path.join(output_dir, "stl")
        #     os.makedirs(stl_dir, exist_ok=True)
        #     stl_path = os.path.join(stl_dir, f"{sample_name}_canal_only.stl")
        #     print(f"    STEP 8: Saving canal mesh to {stl_path}")
        #     writer = vtk.vtkSTLWriter()
        #     writer.SetFileName(stl_path)
        #     writer.SetInputData(closed_mesh)
        #     writer.Write()
        #     print(f"      ✓ Canal mesh saved")
        
        return closed_mesh, volume
        
    except Exception as e:
        print(f"  ⚠ Error extracting segment: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def split_canal_at_cbj_plane(canal_mesh: vtk.vtkPolyData,
                             cbj_plane_centroid: np.ndarray,
                             cbj_plane_normal: np.ndarray,
                             isthmus_point: np.ndarray,
                             eardrum_point: np.ndarray,
                             output_dir: Optional[str] = None,
                             sample_name: Optional[str] = None,
                             inverted_mesh: Optional[vtk.vtkPolyData] = None) -> Tuple[Optional[vtk.vtkPolyData], Optional[vtk.vtkPolyData]]:
    """
    Split canal mesh at CBJ plane and return capped mesh segments.
    
    Creates hard and soft tissue mesh segments by clipping at CBJ plane and capping
    the open boundaries with the cutting plane. Returns mesh objects for downstream
    volume computation using voxel-based methods.
    
    Args:
        canal_mesh: Closed mesh segment (e.g., isthmus-to-eardrum)
        cbj_plane_centroid: CBJ plane center point
        cbj_plane_normal: CBJ plane normal vector (unit length)
        isthmus_point: Isthmus landmark (for reference)
        eardrum_point: Eardrum landmark (for reference)
        output_dir: Optional output directory for STL files (saves both tissues)
        sample_name: Optional sample name for file naming
        inverted_mesh: Optional inverted mesh (currently used for direct clipping path)
    
    Returns:
        Tuple of (hard_tissue_mesh, soft_tissue_mesh) as vtkPolyData objects
        Both meshes are capped at CBJ plane and ready for volume computation
    """
    print(f"\n  Splitting closed canal mesh at CBJ plane (with boundary capping)...")
    
    try:
        if canal_mesh.GetNumberOfCells() == 0:
            print(f"    ⚠ Canal mesh is empty")
            return None, None
        
        # Normalize plane normal
        cbj_normal_unit = np.array(cbj_plane_normal)
        cbj_normal_unit = cbj_normal_unit / (np.linalg.norm(cbj_normal_unit) + 1e-10)
        
        # Create CBJ plane for cutting
        cbj_plane = vtk.vtkPlane()
        cbj_plane.SetOrigin(cbj_plane_centroid)
        cbj_plane.SetNormal(cbj_normal_unit)
        
        # ==================== KEY INSIGHT: CAPS REQUIRED ====================
        # vtkMassProperties uses the divergence theorem, which REQUIRES a closed surface.
        # When you clip a mesh, the result is OPEN at the cut plane.
        # To compute volume correctly on open meshes, we must:
        #   1. Cap the open boundary with the cutting plane
        #   2. Ensure cap normals are consistent with the mesh
        #   3. Compute volume on the CLOSED capped mesh
        # 
        # The partition property ensures: volume(hard) + volume(soft) = volume(original)
        
        print(f"  STATUS: Using capped mesh approach with hole filling...")
        
        # def create_and_append_cap(clipped_mesh: vtk.vtkPolyData, plane: vtk.vtkPlane) -> vtk.vtkPolyData:
        #     """
        #     Create a planar cap from boundary edges and append to clipped mesh.
            
        #     Args:
        #         clipped_mesh: vtkPolyData that is open at cut plane
        #         plane: vtkPlane defining the cap orientation
            
        #     Returns:
        #         vtkPolyData with cap appended (may still have holes)
        #     """
        #     # Extract boundary edges
        #     boundary_extractor = vtk.vtkFeatureEdges()
        #     boundary_extractor.SetInputData(clipped_mesh)
        #     boundary_extractor.BoundaryEdgesOn()
        #     boundary_extractor.FeatureEdgesOff()
        #     boundary_extractor.NonManifoldEdgesOff()
        #     boundary_extractor.ManifoldEdgesOff()
        #     boundary_extractor.Update()
        #     boundary_edges = boundary_extractor.GetOutput()
            
        #     # Triangulate boundary into cap
        #     triangulator = vtk.vtkContourTriangulator()
        #     triangulator.SetInputData(boundary_edges)
        #     triangulator.Update()
        #     cap = triangulator.GetOutput()
            
        #     # Append cap to clipped mesh
        #     appender = vtk.vtkAppendPolyData()
        #     appender.AddInputData(clipped_mesh)
        #     appender.AddInputData(cap)
        #     appender.Update()
        #     result = appender.GetOutput()
            
        #     # Clean to merge duplicate points
        #     cleaner = vtk.vtkCleanPolyData()
        #     cleaner.SetInputData(result)
        #     cleaner.SetTolerance(0.01)
        #     cleaner.Update()
            
        #     return cleaner.GetOutput()
        
        # ==================== CHOOSE CALCULATION METHOD ====================
        if inverted_mesh is not None:
            print(f"  Using INVERTED MESH method for volume calculation")
            print(f"  (More robust: uses mesh complement for validation)")
            
            # Normalize plane normal
            cbj_normal_unit = np.array(cbj_plane_normal, dtype=float)
            cbj_normal_unit = cbj_normal_unit / (np.linalg.norm(cbj_normal_unit) + 1e-10)
            
            # Create CBJ plane for cutting
            cbj_plane = vtk.vtkPlane()
            cbj_plane.SetOrigin(cbj_plane_centroid)
            cbj_plane.SetNormal(cbj_normal_unit)
            
            # Extract boundary from canal mesh for capping
            cutter = vtk.vtkCutter()
            cutter.SetInputData(canal_mesh)
            cutter.SetCutFunction(cbj_plane)
            cutter.Update()
            cbj_boundary = cutter.GetOutput()
            print(f"  ✓ CBJ boundary: {cbj_boundary.GetNumberOfCells()} cells")
            
            # Create shared cap
            triangulator = vtk.vtkContourTriangulator()
            triangulator.SetInputData(cbj_boundary)
            triangulator.Update()
            cbj_cap = triangulator.GetOutput()
            print(f"  ✓ CBJ cap: {cbj_cap.GetNumberOfCells()} cells")
            
            # ===== HARD TISSUE using INVERTED mesh =====
            print(f"\n  ╔═ HARD TISSUE (from inverted mesh) ═╗")
            
            # Clip inverted mesh below CBJ (hard tissue side)
            clipper_inv_hard = vtk.vtkClipPolyData()
            clipper_inv_hard.SetInputData(inverted_mesh)
            clipper_inv_hard.SetClipFunction(cbj_plane)
            clipper_inv_hard.InsideOutOff()
            clipper_inv_hard.Update()
            hard_inv_clipped = clipper_inv_hard.GetOutput()
            print(f"  ✓ Inverted hard clipped: {hard_inv_clipped.GetNumberOfCells()} cells")
            
            # Append cap to inverted hard tissue
            hard_inv_appender = vtk.vtkAppendPolyData()
            hard_inv_appender.AddInputData(hard_inv_clipped)
            hard_inv_appender.AddInputData(cbj_cap)
            hard_inv_appender.Update()
            hard_tissue_inv = ensure_triangles_only(hard_inv_appender.GetOutput())
            print(f"  ✓ Inverted hard with cap: {hard_tissue_inv.GetNumberOfCells()} cells")
            
            print(f"  ✓ Hard tissue mesh (capped): {hard_tissue_inv.GetNumberOfCells()} cells")
            
            # ===== SOFT TISSUE using INVERTED mesh =====
            print(f"\n  ╔═ SOFT TISSUE (from inverted mesh) ═╗")
            
            # Create inverted plane for soft tissue side
            cbj_plane_inv = vtk.vtkPlane()
            cbj_plane_inv.SetOrigin(cbj_plane_centroid)
            cbj_plane_inv.SetNormal(-cbj_normal_unit)
            
            # Clip inverted mesh above CBJ (soft tissue side)
            clipper_inv_soft = vtk.vtkClipPolyData()
            clipper_inv_soft.SetInputData(inverted_mesh)
            clipper_inv_soft.SetClipFunction(cbj_plane_inv)
            clipper_inv_soft.InsideOutOff()
            clipper_inv_soft.Update()
            soft_inv_clipped = clipper_inv_soft.GetOutput()
            print(f"  ✓ Inverted soft clipped: {soft_inv_clipped.GetNumberOfCells()} cells")
            
            # Append same cap to inverted soft tissue
            soft_inv_appender = vtk.vtkAppendPolyData()
            soft_inv_appender.AddInputData(soft_inv_clipped)
            soft_inv_appender.AddInputData(cbj_cap)
            soft_inv_appender.Update()
            soft_tissue_inv = ensure_triangles_only(soft_inv_appender.GetOutput())
            print(f"  ✓ Soft tissue mesh (capped): {soft_tissue_inv.GetNumberOfCells()} cells")
            
            # Save STL files
            if output_dir is not None and sample_name is not None:
                stl_dir = os.path.join(output_dir, "stl")
                os.makedirs(stl_dir, exist_ok=True)
                hard_stl_path = os.path.join(stl_dir, f"{sample_name}_hard_tissue_from_inverted.stl")
                pv.wrap(hard_tissue_inv).save(hard_stl_path)
                print(f"  ✓ Saved hard tissue to {hard_stl_path}")
                
                soft_stl_path = os.path.join(stl_dir, f"{sample_name}_soft_tissue_from_inverted.stl")
                pv.wrap(soft_tissue_inv).save(soft_stl_path)
                print(f"  ✓ Saved soft tissue to {soft_stl_path}")
            
            # Return capped mesh objects for downstream voxel-based volume computation
            hard_tissue = hard_tissue_inv
            soft_tissue = soft_tissue_inv
        
        else:
            print(f"  Using direct CANAL MESH method for volume calculation")
            print(f"  (Direct clipping of isthmus-eardrum segment)")
            
            # Normalize plane normal
            cbj_normal_unit = np.array(cbj_plane_normal, dtype=float)
            cbj_normal_unit = cbj_normal_unit / (np.linalg.norm(cbj_normal_unit) + 1e-10)
            
            # Create CBJ plane for cutting
            cbj_plane = vtk.vtkPlane()
            cbj_plane.SetOrigin(cbj_plane_centroid)
            cbj_plane.SetNormal(cbj_normal_unit)
            
            # Extract boundary for capping
            cutter = vtk.vtkCutter()
            cutter.SetInputData(canal_mesh)
            cutter.SetCutFunction(cbj_plane)
            cutter.Update()
            cbj_boundary = cutter.GetOutput()
            print(f"  ✓ CBJ boundary: {cbj_boundary.GetNumberOfCells()} cells")
            
            # Create shared cap
            triangulator = vtk.vtkContourTriangulator()
            triangulator.SetInputData(cbj_boundary)
            triangulator.Update()
            cbj_cap = triangulator.GetOutput()
            print(f"  ✓ CBJ cap: {cbj_cap.GetNumberOfCells()} cells")
            
            # ===== HARD TISSUE (direct clipping) =====
            print(f"\n  ╔═ HARD TISSUE (direct clipping) ═╗")
            
            clipper_hard = vtk.vtkClipPolyData()
            clipper_hard.SetInputData(canal_mesh)
            clipper_hard.SetClipFunction(cbj_plane)
            clipper_hard.InsideOutOff()
            clipper_hard.Update()
            hard_tissue_clipped = clipper_hard.GetOutput()
            print(f"  ✓ Hard clipped: {hard_tissue_clipped.GetNumberOfCells()} cells")
            
            # Append cap to hard tissue
            hard_appender = vtk.vtkAppendPolyData()
            hard_appender.AddInputData(hard_tissue_clipped)
            hard_appender.AddInputData(cbj_cap)
            hard_appender.Update()
            hard_tissue = ensure_triangles_only(hard_appender.GetOutput())
            print(f"  ✓ Hard tissue mesh (capped): {hard_tissue.GetNumberOfCells()} cells")
            
            # Save hard tissue STL
            if output_dir is not None and sample_name is not None:
                stl_dir = os.path.join(output_dir, "stl")
                os.makedirs(stl_dir, exist_ok=True)
                hard_stl_path = os.path.join(stl_dir, f"{sample_name}_hard_tissue.stl")
                pv.wrap(hard_tissue).save(hard_stl_path)
                print(f"  ✓ Saved hard tissue to {hard_stl_path}")
            
            # ===== SOFT TISSUE (direct clipping with inverted plane) =====
            print(f"\n  ╔═ SOFT TISSUE (direct clipping) ═╗")
            
            cbj_plane_inv = vtk.vtkPlane()
            cbj_plane_inv.SetOrigin(cbj_plane_centroid)
            cbj_plane_inv.SetNormal(-cbj_normal_unit)
            
            clipper_soft = vtk.vtkClipPolyData()
            clipper_soft.SetInputData(canal_mesh)
            clipper_soft.SetClipFunction(cbj_plane_inv)
            clipper_soft.InsideOutOff()
            clipper_soft.Update()
            soft_tissue_clipped = clipper_soft.GetOutput()
            print(f"  ✓ Soft clipped: {soft_tissue_clipped.GetNumberOfCells()} cells")
            
            # Append cap to soft tissue
            soft_appender = vtk.vtkAppendPolyData()
            soft_appender.AddInputData(soft_tissue_clipped)
            soft_appender.AddInputData(cbj_cap)
            soft_appender.Update()
            soft_tissue = ensure_triangles_only(soft_appender.GetOutput())
            print(f"  ✓ Soft tissue mesh (capped): {soft_tissue.GetNumberOfCells()} cells")
            
            # Save soft tissue STL
            if output_dir is not None and sample_name is not None:
                stl_dir = os.path.join(output_dir, "stl")
                os.makedirs(stl_dir, exist_ok=True)
                soft_stl_path = os.path.join(stl_dir, f"{sample_name}_soft_tissue.stl")
                pv.wrap(soft_tissue).save(soft_stl_path)
                print(f"  ✓ Saved soft tissue to {soft_stl_path}")
        
        # ==================== RETURN CAPPED MESHES ====================
        print(f"\n  ╔═══════════════════════════════════════════════════════╗")
        print(f"  ║  CBJ TISSUE SPLIT MESHES (ready for voxel volume)    ║")
        print(f"  ╠═══════════════════════════════════════════════════════╣")
        print(f"  ║ Hard tissue mesh:  {hard_tissue.GetNumberOfCells():>9} cells       ║")
        print(f"  ║ Soft tissue mesh:  {soft_tissue.GetNumberOfCells():>9} cells       ║")
        print(f"  ║ (Both capped at CBJ plane, ready for voxelization)    ║")
        print(f"  ╚═══════════════════════════════════════════════════════╝")
        
        return hard_tissue, soft_tissue
        
    except Exception as e:
        print(f"    ⚠ Error splitting canal at CBJ plane: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def split_canal_at_cbj_plane_intelligent(canal_mesh: vtk.vtkPolyData,
                                         cbj_plane_centroid: np.ndarray,
                                         cbj_plane_normal: np.ndarray,
                                         isthmus_point: np.ndarray,
                                         eardrum_point: np.ndarray,
                                         output_dir: Optional[str] = None,
                                         sample_name: Optional[str] = None,
                                         visualize: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """
    Intelligently split canal mesh at CBJ plane by extracting connected components
    and selecting the ones closest to anatomical landmarks.
    
    Process:
    1. Cut mesh at CBJ plane into two halves
    2. Extract connected components from each half
    3. Hard tissue: Select component closest to EARDRUM
    4. Soft tissue: Select component closest to ISTHMUS
    5. Close both with CBJ plane caps (same method as isthmus-eardrum closing)
    6. Compute volumes (hard + soft = total)
    7. Visualize and save STL files
    
    Args:
        canal_mesh: Closed mesh segment (e.g., isthmus-to-eardrum)
        cbj_plane_centroid: CBJ plane center point
        cbj_plane_normal: CBJ plane normal vector  
        isthmus_point: Isthmus landmark (for proximity check)
        eardrum_point: Eardrum landmark (for proximity check)
        output_dir: Optional output directory for STL files
        sample_name: Optional sample name for file naming
        visualize: Whether to generate visualizations
    
    Returns:
        Tuple of (soft_tissue_volume, hard_tissue_volume) in mm³
    """
    print(f"\n  Splitting canal mesh at CBJ plane with intelligent component selection...")
    
    try:
        from scipy.spatial import cKDTree
        
        if canal_mesh.GetNumberOfCells() == 0:
            print(f"    ⚠ Canal mesh is empty")
            return None, None
        
        # Normalize plane normal
        cbj_normal_unit = np.array(cbj_plane_normal, dtype=float)
        cbj_normal_unit = cbj_normal_unit / (np.linalg.norm(cbj_normal_unit) + 1e-10)
        
        print(f"  CBJ Plane center: {cbj_plane_centroid}")
        print(f"  CBJ Plane normal: {cbj_normal_unit}")
        
        # Create CBJ plane for cutting
        cbj_plane = vtk.vtkPlane()
        cbj_plane.SetOrigin(cbj_plane_centroid)
        cbj_plane.SetNormal(cbj_normal_unit)
        
        # ==================== HARD TISSUE (Eardrum Side) ====================
        print(f"\n  ╔═══ HARD TISSUE (Bony, toward Eardrum) ═══╗")
        print(f"  Step 1a: Cutting at CBJ plane (eardrum side)...")
        
        # Clip hard tissue (eardrum side - negative normal direction)
        clipper_hard = vtk.vtkClipPolyData()
        clipper_hard.SetInputData(canal_mesh)
        clipper_hard.SetClipFunction(cbj_plane)
        clipper_hard.InsideOutOff()  # Keep on negative side of plane
        clipper_hard.Update()
        hard_clipped = clipper_hard.GetOutput()
        
        hard_cells_clipped = hard_clipped.GetNumberOfCells()
        print(f"    ✓ Hard tissue clipped: {hard_cells_clipped} cells")
        
        # Extract boundary edges from hard tissue clipped mesh
        print(f"  Step 1b: Extracting boundary curve from hard tissue clip...")
        hard_boundary_edges = vtk.vtkExtractEdges()
        hard_boundary_edges.SetInputData(hard_clipped)
        hard_boundary_edges.Update()
        hard_boundary = hard_boundary_edges.GetOutput()
        print(f"    ✓ Hard tissue boundary: {hard_boundary.GetNumberOfCells()} edges")
        
        # Create cap for hard tissue from its boundary
        print(f"  Step 1c: Creating cap for hard tissue...")
        if hard_boundary.GetNumberOfPoints() > 2:
            hard_triangulator = vtk.vtkContourTriangulator()
            hard_triangulator.SetInputData(hard_boundary)
            hard_triangulator.Update()
            hard_cap = hard_triangulator.GetOutput()
            
            # Reverse normals for hard tissue cap
            hard_reverser = vtk.vtkReverseSense()
            hard_reverser.SetInputData(hard_cap)
            hard_reverser.ReverseNormalsOn()
            hard_reverser.Update()
            hard_cap = hard_reverser.GetOutput()
            print(f"    ✓ Hard tissue cap created: {hard_cap.GetNumberOfCells()} triangles (normals reversed)")
        else:
            print(f"    ⚠ Hard tissue boundary has insufficient points")
            return None, None
        
        # ==================== SOFT TISSUE (Isthmus Side) ====================
        print(f"\n  ╔═══ SOFT TISSUE (Cartilaginous, toward Isthmus) ═══╗")
        print(f"  Step 2a: Cutting at CBJ plane (isthmus side)...")
        
        # Create inverted CBJ plane for soft tissue
        cbj_plane_inv = vtk.vtkPlane()
        cbj_plane_inv.SetOrigin(cbj_plane_centroid)
        cbj_plane_inv.SetNormal(-cbj_normal_unit)
        
        # Clip soft tissue (isthmus side - positive normal direction)
        clipper_soft = vtk.vtkClipPolyData()
        clipper_soft.SetInputData(canal_mesh)
        clipper_soft.SetClipFunction(cbj_plane_inv)
        clipper_soft.InsideOutOff()  # Keep on positive side of inverted plane
        clipper_soft.Update()
        soft_clipped = clipper_soft.GetOutput()
        
        soft_cells_clipped = soft_clipped.GetNumberOfCells()
        print(f"    ✓ Soft tissue clipped: {soft_cells_clipped} cells")
        
        # Extract boundary edges from soft tissue clipped mesh
        print(f"  Step 2b: Extracting boundary curve from soft tissue clip...")
        soft_boundary_edges = vtk.vtkExtractEdges()
        soft_boundary_edges.SetInputData(soft_clipped)
        soft_boundary_edges.Update()
        soft_boundary = soft_boundary_edges.GetOutput()
        print(f"    ✓ Soft tissue boundary: {soft_boundary.GetNumberOfCells()} edges")
        
        # Create cap for soft tissue from its boundary
        print(f"  Step 2c: Creating cap for soft tissue...")
        if soft_boundary.GetNumberOfPoints() > 2:
            soft_triangulator = vtk.vtkContourTriangulator()
            soft_triangulator.SetInputData(soft_boundary)
            soft_triangulator.Update()
            soft_cap = soft_triangulator.GetOutput()
            print(f"    ✓ Soft tissue cap created: {soft_cap.GetNumberOfCells()} triangles")
        else:
            print(f"    ⚠ Soft tissue boundary has insufficient points")
            return None, None
        
        # Extract connected components from hard tissue
        print(f"  Step 1d: Extracting connected components from hard tissue...")
        hard_connectivity = vtk.vtkPolyDataConnectivityFilter()
        hard_connectivity.SetInputData(hard_clipped)
        hard_connectivity.SetExtractionModeToAllRegions()
        hard_connectivity.Update()
        n_hard_components = hard_connectivity.GetNumberOfExtractedRegions()
        print(f"    ✓ Found {n_hard_components} component(s) in hard tissue")
        
        # Analyze all hard tissue components
        print(f"  Step 1e: Analyzing hard tissue components...")
        hard_components = []
        for comp_id in range(n_hard_components):
            extractor = vtk.vtkPolyDataConnectivityFilter()
            extractor.SetInputData(hard_clipped)
            extractor.SetExtractionModeToSpecifiedRegions()
            extractor.AddSpecifiedRegion(comp_id)
            extractor.Update()
            component = extractor.GetOutput()
            
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(component)
            cleaner.Update()
            cleaned = cleaner.GetOutput()
            
            # Get nearest distance to eardrum
            comp_points = np.array([cleaned.GetPoint(i) for i in range(cleaned.GetNumberOfPoints())])
            tree = cKDTree(comp_points)
            eardrum_dist, _ = tree.query(np.array(eardrum_point))
            
            n_cells = cleaned.GetNumberOfCells()
            hard_components.append({
                'id': comp_id,
                'mesh': cleaned,
                'eardrum_dist': eardrum_dist,
                'n_cells': n_cells
            })
            print(f"      Component {comp_id}: {n_cells} cells | Distance to eardrum: {eardrum_dist:.3f}mm")
        
        # Select hard tissue component closest to eardrum
        if len(hard_components) == 0:
            print(f"    ⚠ No hard tissue components found")
            return None, None
        
        hard_components.sort(key=lambda x: x['eardrum_dist'])
        selected_hard = hard_components[0]
        print(f"  Step 1f: Selected hard tissue component {selected_hard['id']} (eardrum_dist={selected_hard['eardrum_dist']:.3f}mm)")
        
        # Close hard tissue with inverted cap
        print(f"  Step 1g: Closing hard tissue with inverted CBJ cap...")
        
        # Clean hard tissue clipped mesh before appending cap (merge duplicate points)
        hard_cleaner = vtk.vtkCleanPolyData()
        hard_cleaner.SetInputData(selected_hard['mesh'])
        hard_cleaner.Update()
        hard_tissue_clean = hard_cleaner.GetOutput()
        print(f"    After cleaning: {hard_tissue_clean.GetNumberOfCells()} cells")
        
        hard_appender = vtk.vtkAppendPolyData()
        hard_appender.AddInputData(hard_tissue_clean)
        hard_appender.AddInputData(hard_cap)
        hard_appender.Update()
        hard_tissue_closed = hard_appender.GetOutput()
        hard_tissue_closed = ensure_triangles_only(hard_tissue_closed)
        print(f"    ✓ Hard tissue closed: {hard_tissue_closed.GetNumberOfCells()} cells")
        
        # Compute hard tissue volume
        print(f"  Step 2f: Computing hard tissue volume...")
        
        # Final cleanup of appended mesh to ensure seamless joining
        print(f"    Final cleanup of hard tissue mesh...")
        final_hard_cleaner = vtk.vtkCleanPolyData()
        final_hard_cleaner.SetInputData(hard_tissue_closed)
        final_hard_cleaner.Update()
        hard_tissue_closed = final_hard_cleaner.GetOutput()
        
        # Ensure mesh is closed before computing volume
        print(f"    Checking mesh closure...")
        is_hard_closed = mesh_is_closed(hard_tissue_closed)
        print(f"    Mesh is closed: {is_hard_closed}")
        
        if not is_hard_closed:
            print(f"    ⚠ Hard tissue mesh has boundary edges, attempting to close...")
            hard_tissue_closed = close_mesh_iteratively(hard_tissue_closed, max_iterations=3)
            is_hard_closed = mesh_is_closed(hard_tissue_closed)
            print(f"    After closure attempt - closed: {is_hard_closed}")
        
        hard_props = vtk.vtkMassProperties()
        hard_props.SetInputData(hard_tissue_closed)
        hard_props.Update()
        hard_volume = hard_props.GetVolume()
        print(f"    ✓ Hard tissue volume: {hard_volume:.2f} mm³")
        
        # Save hard tissue STL
        if output_dir and sample_name:
            stl_dir = os.path.join(output_dir, "stl")
            os.makedirs(stl_dir, exist_ok=True)
            hard_stl = os.path.join(stl_dir, f"{sample_name}_hard_tissue_cbj_intelligent.stl")
            pv.wrap(hard_tissue_closed).save(hard_stl)
            print(f"    ✓ Saved hard tissue to {hard_stl}")
        
        # Visualization for hard tissue components
        if visualize and n_hard_components > 1:
            print(f"  Step 2g: Visualizing hard tissue components...")
            plotter = pv.Plotter(title=f"Hard Tissue Components ({n_hard_components})")
            
            colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
            for i, comp in enumerate(hard_components):
                color = colors[i % len(colors)]
                is_selected = "✓ SELECTED" if comp['id'] == selected_hard['id'] else ""
                plotter.add_mesh(comp['mesh'], color=color, opacity=0.7,
                                label=f"Component {comp['id']} {is_selected}")
            
            plotter.add_points(np.array(eardrum_point).reshape(1, -1), color='white', 
                              point_size=10, label='Eardrum')
            plotter.add_legend()
            plotter.show()
            print(f"    ✓ Visualization closed")
        
        # ==================== SOFT TISSUE (Isthmus Side) ====================
        # (Already created hard_cap and soft_cap above)
        
        # Extract connected components from soft tissue
        print(f"  Step 2d: Extracting connected components from soft tissue...")
        soft_connectivity = vtk.vtkPolyDataConnectivityFilter()
        soft_connectivity.SetInputData(soft_clipped)
        soft_connectivity.SetExtractionModeToAllRegions()
        soft_connectivity.Update()
        n_soft_components = soft_connectivity.GetNumberOfExtractedRegions()
        print(f"    ✓ Found {n_soft_components} component(s) in soft tissue")
        
        # Analyze all soft tissue components
        print(f"  Step 2e: Analyzing soft tissue components...")
        soft_components = []
        for comp_id in range(n_soft_components):
            extractor = vtk.vtkPolyDataConnectivityFilter()
            extractor.SetInputData(soft_clipped)
            extractor.SetExtractionModeToSpecifiedRegions()
            extractor.AddSpecifiedRegion(comp_id)
            extractor.Update()
            component = extractor.GetOutput()
            
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(component)
            cleaner.Update()
            cleaned = cleaner.GetOutput()
            
            # Get nearest distance to isthmus
            comp_points = np.array([cleaned.GetPoint(i) for i in range(cleaned.GetNumberOfPoints())])
            tree = cKDTree(comp_points)
            isthmus_dist, _ = tree.query(np.array(isthmus_point))
            
            n_cells = cleaned.GetNumberOfCells()
            soft_components.append({
                'id': comp_id,
                'mesh': cleaned,
                'isthmus_dist': isthmus_dist,
                'n_cells': n_cells
            })
            print(f"      Component {comp_id}: {n_cells} cells | Distance to isthmus: {isthmus_dist:.3f}mm")
        
        # Select soft tissue component closest to isthmus
        if len(soft_components) == 0:
            print(f"    ⚠ No soft tissue components found")
            return None, None
        
        soft_components.sort(key=lambda x: x['isthmus_dist'])
        selected_soft = soft_components[0]
        print(f"  Step 2f: Selected soft tissue component {selected_soft['id']} (isthmus_dist={selected_soft['isthmus_dist']:.3f}mm)")
        
        # Close soft tissue with base cap
        print(f"  Step 2g: Closing soft tissue with CBJ cap (standard orientation)...")
        
        # Clean soft tissue clipped mesh before appending cap (merge duplicate points)
        soft_cleaner = vtk.vtkCleanPolyData()
        soft_cleaner.SetInputData(selected_soft['mesh'])
        soft_cleaner.Update()
        soft_tissue_clean = soft_cleaner.GetOutput()
        print(f"    After cleaning: {soft_tissue_clean.GetNumberOfCells()} cells")
        
        soft_appender = vtk.vtkAppendPolyData()
        soft_appender.AddInputData(soft_tissue_clean)
        soft_appender.AddInputData(soft_cap)
        soft_appender.Update()
        soft_tissue_closed = soft_appender.GetOutput()
        soft_tissue_closed = ensure_triangles_only(soft_tissue_closed)
        print(f"    ✓ Soft tissue closed: {soft_tissue_closed.GetNumberOfCells()} cells")
        
        # Compute soft tissue volume
        print(f"  Step 2h: Computing soft tissue volume...")
        
        # Final cleanup of appended mesh to ensure seamless joining
        print(f"    Final cleanup of soft tissue mesh...")
        final_soft_cleaner = vtk.vtkCleanPolyData()
        final_soft_cleaner.SetInputData(soft_tissue_closed)
        final_soft_cleaner.Update()
        soft_tissue_closed = final_soft_cleaner.GetOutput()
        
        # Ensure mesh is closed before computing volume
        print(f"    Checking mesh closure...")
        is_soft_closed = mesh_is_closed(soft_tissue_closed)
        print(f"    Mesh is closed: {is_soft_closed}")
        
        if not is_soft_closed:
            print(f"    ⚠ Soft tissue mesh has boundary edges, attempting to close...")
            soft_tissue_closed = close_mesh_iteratively(soft_tissue_closed, max_iterations=3)
            is_soft_closed = mesh_is_closed(soft_tissue_closed)
            print(f"    After closure attempt - closed: {is_soft_closed}")
        
        soft_props = vtk.vtkMassProperties()
        soft_props.SetInputData(soft_tissue_closed)
        soft_props.Update()
        soft_volume = soft_props.GetVolume()
        print(f"    ✓ Soft tissue volume: {soft_volume:.2f} mm³")
        
        # Save soft tissue STL
        if output_dir and sample_name:
            stl_dir = os.path.join(output_dir, "stl")
            os.makedirs(stl_dir, exist_ok=True)
            soft_stl = os.path.join(stl_dir, f"{sample_name}_soft_tissue_cbj_intelligent.stl")
            pv.wrap(soft_tissue_closed).save(soft_stl)
            print(f"    ✓ Saved soft tissue to {soft_stl}")
        
        # Visualization for soft tissue components
        if visualize and n_soft_components > 1:
            print(f"  Step 2i: Visualizing soft tissue components...")
            plotter = pv.Plotter(title=f"Soft Tissue Components ({n_soft_components})")
            
            colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
            for i, comp in enumerate(soft_components):
                color = colors[i % len(colors)]
                is_selected = "✓ SELECTED" if comp['id'] == selected_soft['id'] else ""
                plotter.add_mesh(comp['mesh'], color=color, opacity=0.7,
                                label=f"Component {comp['id']} {is_selected}")
            
            plotter.add_points(np.array(isthmus_point).reshape(1, -1), color='white',
                              point_size=10, label='Isthmus')
            plotter.add_legend()
            plotter.show()
            print(f"    ✓ Visualization closed")
        
        # ==================== SUMMARY ====================
        print(f"\n  Step 4: Final Summary")
        print(f"  ╔═══════════════════════════════════════════════════════╗")
        print(f"  ║  CBJ TISSUE SPLIT - INTELLIGENT COMPONENT SELECTION  ║")
        print(f"  ╠═══════════════════════════════════════════════════════╣")
        print(f"  ║ Hard Tissue (Bony):     {hard_volume:>7.2f} mm³           ║")
        print(f"  ║ Soft Tissue (Cartilag): {soft_volume:>7.2f} mm³           ║")
        print(f"  ║ Total Combined:         {hard_volume + soft_volume:>7.2f} mm³           ║")
        print(f"  ║                                                     ║")
        print(f"  ║ Hard: Component {selected_hard['id']} (eardrum_dist={selected_hard['eardrum_dist']:.2f}mm)        ║")
        print(f"  ║ Soft: Component {selected_soft['id']} (isthmus_dist={selected_soft['isthmus_dist']:.2f}mm)      ║")
        
        # Compute original mesh volume for validation
        orig_props = vtk.vtkMassProperties()
        orig_props.SetInputData(canal_mesh)
        orig_props.Update()
        orig_volume = orig_props.GetVolume()
        
        combined_volume = hard_volume + soft_volume
        volume_diff = combined_volume - orig_volume
        volume_ratio = (combined_volume / orig_volume) if orig_volume > 0 else 0
        
        print(f"  ║                                                     ║")
        print(f"  ║ [VOLUME VALIDATION]                                 ║")
        print(f"  ║ Original mesh volume: {orig_volume:>7.2f} mm³           ║")
        print(f"  ║ Hard + Soft:          {combined_volume:>7.2f} mm³           ║")
        print(f"  ║ Difference:           {volume_diff:>7.2f} mm³ ({volume_ratio*100:>5.1f}%)  ║")
        if abs(volume_diff) < orig_volume * 0.05:  # Within 5%
            print(f"  ║ ✓ VOLUMES MATCH (within 5%)                        ║")
        else:
            print(f"  ║ ⚠ WARNING: Volume mismatch detected!                ║")
        print(f"  ╚═══════════════════════════════════════════════════════╝")
        
        return float(soft_volume), float(hard_volume)
        
    except Exception as e:
        print(f"    ⚠ Error in intelligent CBJ plane split: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_surface_from_curve(curve: vtk.vtkPolyData) -> Optional[vtk.vtkPolyData]:
    """
    Create an open triangulated surface (face) from a closed boundary curve.
    Uses fan triangulation to create a single open surface, not a closed mesh.
    
    Args:
        curve: VTK PolyData containing curve points (closed boundary)
    
    Returns:
        vtkPolyData: Open surface with triangles (not closed), or None if failed
    """
    try:
        if curve.GetNumberOfPoints() < 3:
            print(f"        ⚠ Curve has insufficient points: {curve.GetNumberOfPoints()}")
            return None
        
        n_pts = curve.GetNumberOfPoints()
        print(f"        Creating open surface from curve with {n_pts} points...")
        
        # Extract points from the curve
        pts = vtk.vtkPoints()
        for i in range(n_pts):
            pt = curve.GetPoint(i)
            pts.InsertNextPoint(pt)
        
        # Calculate center point of the curve
        center = [0.0, 0.0, 0.0]
        for i in range(n_pts):
            pt = curve.GetPoint(i)
            center[0] += pt[0]
            center[1] += pt[1]
            center[2] += pt[2]
        center[0] /= n_pts
        center[1] /= n_pts
        center[2] /= n_pts
        
        # Insert center point first
        pts_fan = vtk.vtkPoints()
        pts_fan.InsertNextPoint(center)
        for i in range(n_pts):
            pt = curve.GetPoint(i)
            pts_fan.InsertNextPoint(pt)
        
        # Create triangles from center to curve edges (fan pattern)
        # This creates an OPEN surface with no opposite face
        cells = vtk.vtkCellArray()
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            cells.InsertNextCell(3)
            cells.InsertCellPoint(0)  # Center point
            cells.InsertCellPoint(i + 1)  # Current curve point
            cells.InsertCellPoint(next_i + 1)  # Next curve point
        
        surface = vtk.vtkPolyData()
        surface.SetPoints(pts_fan)
        surface.SetPolys(cells)
        
        print(f"        ✓ Fan surface created {surface.GetNumberOfCells()} triangles (open surface)")
        return surface
        
    except Exception as e:
        print(f"        ⚠ Error creating surface from curve: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_volume_segment(mesh: vtk.vtkPolyData,
                             isthmus_point: np.ndarray,
                             eardrum_point: np.ndarray,
                             volume_mm3: float,
                             title: str = "Isthmus-Eardrum Volume Segment") -> None:
    """
    Visualize the extracted volume segment between isthmus and eardrum (closed mesh with caps).
    
    Args:
        mesh: Closed mesh segment with caps
        isthmus_point: Center of isthmus cross-section
        eardrum_point: Center of eardrum cross-section
        volume_mm3: Computed volume
        title: Window title
    """
    print(f"\n  Opening volume mesh visualization: {title}")
    print(f"    Volume: {volume_mm3:.2f} mm³")
    print(f"    Mesh: {mesh.GetNumberOfCells()} cells, {mesh.GetNumberOfPoints()} points")
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background
    
    # Add mesh (main visualization)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.2, 0.8, 1.0)  # Light blue
    actor.GetProperty().SetOpacity(0.7)
    actor.GetProperty().EdgeVisibilityOn()  # Show edges
    actor.GetProperty().SetEdgeColor(0.0, 0.5, 1.0)
    renderer.AddActor(actor)
    
    # Add isthmus marker (green sphere)
    sphere_isthmus = vtk.vtkSphereSource()
    sphere_isthmus.SetCenter(isthmus_point)
    sphere_isthmus.SetRadius(2.0)
    sphere_isthmus.Update()
    
    mapper_isthmus = vtk.vtkPolyDataMapper()
    mapper_isthmus.SetInputData(sphere_isthmus.GetOutput())
    actor_isthmus = vtk.vtkActor()
    actor_isthmus.SetMapper(mapper_isthmus)
    actor_isthmus.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
    renderer.AddActor(actor_isthmus)
    
    # Add eardrum marker (red sphere)
    sphere_eardrum = vtk.vtkSphereSource()
    sphere_eardrum.SetCenter(eardrum_point)
    sphere_eardrum.SetRadius(2.0)
    sphere_eardrum.Update()
    
    mapper_eardrum = vtk.vtkPolyDataMapper()
    mapper_eardrum.SetInputData(sphere_eardrum.GetOutput())
    actor_eardrum = vtk.vtkActor()
    actor_eardrum.SetMapper(mapper_eardrum)
    actor_eardrum.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
    renderer.AddActor(actor_eardrum)
    
    # Create window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 600)
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(title)
    
    # Add interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Reset camera to fit all objects
    renderer.ResetCamera()
    
    # Print legend
    print(f"\n  Legend:")
    print(f"    - Light blue mesh: Volume segment (closed)")
    print(f"    - Blue edges: Mesh boundaries")
    print(f"    - Green sphere: Isthmus cross-section center")
    print(f"    - Red sphere: Eardrum cross-section center")
    print(f"    - Title: Volume = {volume_mm3:.2f} mm³")
    print(f"\n  Close window to continue...")
    
    interactor.Start()


def _get_cell_type_name(cell_type: int) -> str:
    """Get human-readable cell type name from VTK cell type integer."""
    cell_type_names = {
        0: "Empty",
        1: "Vertex",
        2: "Poly Vertex",
        3: "Line",
        4: "Poly Line",
        5: "Triangle",
        6: "Triangle Strip",
        7: "Polygon",
        8: "Pixel",
        9: "Quad",
        10: "Tetra",
        11: "Voxel",
        12: "Hexahedron",
        13: "Wedge",
        14: "Pyramid",
        15: "Pentagonal Prism",
        16: "Hexagonal Prism",
        42: "Lagrange Triangle",
        43: "Lagrange Quad",
    }
    return cell_type_names.get(cell_type, f"Unknown({cell_type})")


def visualize_tissue_mesh(tissue_mesh: vtk.vtkPolyData, 
                          tissue_name: str,
                          color: tuple,
                          mesh_quality: dict) -> None:
    """
    Visualize a single tissue mesh with quality metrics display.
    
    Args:
        tissue_mesh: VTK mesh to visualize
        tissue_name: Name of tissue (e.g., "Soft Tissue" or "Hard Tissue")
        color: RGB color tuple (0-1 range)
        mesh_quality: Dictionary with mesh quality metrics
    """
    print(f"\n  Opening {tissue_name} mesh visualization...")
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background
    
    # Add tissue mesh
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(tissue_mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color[0], color[1], color[2])
    actor.GetProperty().SetOpacity(0.9)
    actor.GetProperty().EdgeVisibilityOn()  # Show mesh edges
    renderer.AddActor(actor)
    
    # Add axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(20, 20, 20)
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.5)
    renderer.AddActor(axes)
    
    # Create window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1000, 800)
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(f"{tissue_name} Mesh Quality - Individual Visualization")
    
    # Add interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Reset camera to fit mesh
    renderer.ResetCamera()
    
    # Print detailed mesh metrics
    print(f"\n  ╔════════════════════════════════════════════════════╗")
    print(f"  ║ {tissue_name.upper()} MESH QUALITY ANALYSIS")
    print(f"  ╚════════════════════════════════════════════════════╝")
    print(f"\n  Topology:")
    print(f"    - Number of cells: {mesh_quality.get('cells', 0)}")
    print(f"    - Number of points: {mesh_quality.get('points', 0)}")
    print(f"    - Cell types: {mesh_quality.get('cell_types', {})}")
    print(f"\n  Geometry:")
    print(f"    - Surface area: {mesh_quality.get('surface_area', 0):.2f} mm²")
    print(f"    - Volume: {mesh_quality.get('volume', 0):.2f} mm³")
    print(f"    - Bounds: X=[{mesh_quality.get('bounds', [0,0,0,0,0,0])[0]:.2f}, {mesh_quality.get('bounds', [0,0,0,0,0,0])[1]:.2f}]")
    if len(mesh_quality.get('bounds', [])) >= 6:
        print(f"             Y=[{mesh_quality.get('bounds', [0,0,0,0,0,0])[2]:.2f}, {mesh_quality.get('bounds', [0,0,0,0,0,0])[3]:.2f}]")
        print(f"             Z=[{mesh_quality.get('bounds', [0,0,0,0,0,0])[4]:.2f}, {mesh_quality.get('bounds', [0,0,0,0,0,0])[5]:.2f}]")
    
    print(f"\n  Legend:")
    print(f"    - Mesh edges are visible (black lines)")
    print(f"    - Color indicates tissue type")
    print(f"    - You can rotate/zoom/pan the view")
    print(f"\n  Close window to continue...")
    
    render_window.Render()
    interactor.Start()


def visualize_soft_tissue(soft_tissue: vtk.vtkPolyData) -> None:
    """Visualize soft tissue (cartilaginous) mesh individually."""
    # Compute quality metrics
    props = vtk.vtkMassProperties()
    props.SetInputData(soft_tissue)
    props.Update()
    
    bounds = soft_tissue.GetBounds()
    
    # Count cell types
    cell_type_counter = {}
    for i in range(soft_tissue.GetNumberOfCells()):
        cell_type = soft_tissue.GetCell(i).GetCellType()
        cell_type_name = _get_cell_type_name(cell_type)
        cell_type_counter[cell_type_name] = cell_type_counter.get(cell_type_name, 0) + 1
    
    quality = {
        'cells': soft_tissue.GetNumberOfCells(),
        'points': soft_tissue.GetNumberOfPoints(),
        'cell_types': cell_type_counter,
        'surface_area': props.GetSurfaceArea(),
        'volume': props.GetVolume(),
        'bounds': bounds
    }
    
    visualize_tissue_mesh(soft_tissue, "Soft Tissue (Cartilaginous)", (0.2, 0.6, 1.0), quality)


def visualize_hard_tissue(hard_tissue: vtk.vtkPolyData) -> None:
    """Visualize hard tissue (bony) mesh individually."""
    # Compute quality metrics
    props = vtk.vtkMassProperties()
    props.SetInputData(hard_tissue)
    props.Update()
    
    bounds = hard_tissue.GetBounds()
    
    # Count cell types
    cell_type_counter = {}
    for i in range(hard_tissue.GetNumberOfCells()):
        cell_type = hard_tissue.GetCell(i).GetCellType()
        cell_type_name = _get_cell_type_name(cell_type)
        cell_type_counter[cell_type_name] = cell_type_counter.get(cell_type_name, 0) + 1
    
    quality = {
        'cells': hard_tissue.GetNumberOfCells(),
        'points': hard_tissue.GetNumberOfPoints(),
        'cell_types': cell_type_counter,
        'surface_area': props.GetSurfaceArea(),
        'volume': props.GetVolume(),
        'bounds': bounds
    }
    
    visualize_tissue_mesh(hard_tissue, "Hard Tissue (Bony)", (1.0, 0.7, 0.2), quality)


def visualize_tissue_segments(soft_tissue: vtk.vtkPolyData,
                              hard_tissue: vtk.vtkPolyData,
                              cbj_plane_centroid: np.ndarray,
                              cbj_plane_normal: np.ndarray,
                              isthmus_point: np.ndarray,
                              eardrum_point: np.ndarray) -> None:
    """
    Visualize soft and hard tissue segments split at CBJ plane.
    
    Displays:
    - Soft tissue (cartilaginous) in light blue
    - Hard tissue (bony) in light orange
    - CBJ plane in cyan (semi-transparent)
    - Isthmus reference point (green sphere)
    - Eardrum reference point (red sphere)
    
    Args:
        soft_tissue: Cartilaginous component mesh
        hard_tissue: Bony component mesh
        cbj_plane_centroid: Center of the CBJ plane
        cbj_plane_normal: Normal vector of the CBJ plane
        isthmus_point: Isthmus reference point
        eardrum_point: Eardrum reference point
    """
    print(f"\n  Opening tissue segment visualization...")
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background
    
    # Add soft tissue (light blue)
    soft_mapper = vtk.vtkPolyDataMapper()
    soft_mapper.SetInputData(soft_tissue)
    soft_actor = vtk.vtkActor()
    soft_actor.SetMapper(soft_mapper)
    soft_actor.GetProperty().SetColor(0.2, 0.6, 1.0)  # Light blue
    soft_actor.GetProperty().SetOpacity(0.8)
    renderer.AddActor(soft_actor)
    
    # Add hard tissue (light orange)
    hard_mapper = vtk.vtkPolyDataMapper()
    hard_mapper.SetInputData(hard_tissue)
    hard_actor = vtk.vtkActor()
    hard_actor.SetMapper(hard_mapper)
    hard_actor.GetProperty().SetColor(1.0, 0.7, 0.2)  # Light orange
    hard_actor.GetProperty().SetOpacity(0.8)
    renderer.AddActor(hard_actor)
    
    # Add CBJ plane (cyan, semi-transparent)
    plane_size = 50.0
    plane_source = vtk.vtkPlaneSource()
    plane_source.SetOrigin(-plane_size/2, -plane_size/2, 0)
    plane_source.SetPoint1(plane_size/2, -plane_size/2, 0)
    plane_source.SetPoint2(-plane_size/2, plane_size/2, 0)
    plane_source.Update()
    
    # Transform plane to align with CBJ plane normal and position
    z_axis = np.array([0, 0, 1])
    normal_unit = cbj_plane_normal / (np.linalg.norm(cbj_plane_normal) + 1e-10)
    
    rotation_axis = np.cross(z_axis, normal_unit)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    transform = vtk.vtkTransform()
    transform.Translate(cbj_plane_centroid)
    
    if rotation_axis_norm > 1e-6:
        rotation_axis_unit = rotation_axis / rotation_axis_norm
        angle_rad = np.arccos(np.clip(np.dot(z_axis, normal_unit), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        transform.RotateWXYZ(angle_deg, rotation_axis_unit[0], rotation_axis_unit[1], rotation_axis_unit[2])
    elif np.dot(z_axis, normal_unit) < 0:
        transform.RotateWXYZ(180, 1, 0, 0)
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(plane_source.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    
    plane_mapper = vtk.vtkPolyDataMapper()
    plane_mapper.SetInputConnection(transform_filter.GetOutputPort())
    plane_actor = vtk.vtkActor()
    plane_actor.SetMapper(plane_mapper)
    plane_actor.GetProperty().SetColor(0.0, 0.8, 0.8)  # Cyan
    plane_actor.GetProperty().SetOpacity(0.5)
    renderer.AddActor(plane_actor)
    
    # Add isthmus marker (green sphere)
    sphere_isthmus = vtk.vtkSphereSource()
    sphere_isthmus.SetCenter(isthmus_point)
    sphere_isthmus.SetRadius(2.5)
    sphere_isthmus.Update()
    
    mapper_isthmus = vtk.vtkPolyDataMapper()
    mapper_isthmus.SetInputData(sphere_isthmus.GetOutput())
    actor_isthmus = vtk.vtkActor()
    actor_isthmus.SetMapper(mapper_isthmus)
    actor_isthmus.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
    renderer.AddActor(actor_isthmus)
    
    # Add eardrum marker (red sphere)
    sphere_eardrum = vtk.vtkSphereSource()
    sphere_eardrum.SetCenter(eardrum_point)
    sphere_eardrum.SetRadius(2.5)
    sphere_eardrum.Update()
    
    mapper_eardrum = vtk.vtkPolyDataMapper()
    mapper_eardrum.SetInputData(sphere_eardrum.GetOutput())
    actor_eardrum = vtk.vtkActor()
    actor_eardrum.SetMapper(mapper_eardrum)
    actor_eardrum.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
    renderer.AddActor(actor_eardrum)
    
    # Add axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(20, 20, 20)
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.5)
    renderer.AddActor(axes)
    
    # Create window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1000, 700)
    render_window.AddRenderer(renderer)
    render_window.SetWindowName("CBJ Plane Split: Soft vs Hard Tissue")
    
    # Add interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Reset camera to fit all objects
    renderer.ResetCamera()
    
    # Print legend
    print(f"\n  Legend:")
    print(f"    - Light blue mesh: Soft tissue (cartilaginous, toward isthmus)")
    print(f"    - Light orange mesh: Hard tissue (bony, toward eardrum)")
    print(f"    - Cyan plane: CBJ (Cartilaginous-Bony Junction) plane")
    print(f"    - Green sphere: Isthmus reference point")
    print(f"    - Red sphere: Eardrum reference point")
    print(f"    - Soft tissue cells: {soft_tissue.GetNumberOfCells()}")
    print(f"    - Hard tissue cells: {hard_tissue.GetNumberOfCells()}")
    print(f"\n  Close window to continue to volume computation...")
    
    render_window.Render()
    interactor.Start()