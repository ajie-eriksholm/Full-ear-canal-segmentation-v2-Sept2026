"""
All-in-one ear canal feature extraction pipeline.

Given:
  - input_dir   : directory containing vtk/, markups/, stl/ subdirs
                  and a processing_results.csv (centerline features)
  - metadata_csv: CSV with patient demographics (age, sex)
  - output_dir  : where to write the output CSV

Produces a CSV with exactly the same columns as combined_ear_canal_features.csv:
  patient_id, EarID, age, sex, side,
  ear_size, tragus_size,
  isthmus_eardrum_length, isthmus_eardrum_tortuosity,
  area_mean,
  1st_bend_area, 2nd_bend_area, eardrum_area, isthmus_area,
  first_to_second_bend_distance, isthmus_to_first_bend_distance,
  second_bend_to_eardrum_distance,
  area_cv,
  1st_bend_max_min_r_ratio, 2nd_bend_max_min_r_ratio,
  eardrum_max_min_r_ratio, isthmus_max_min_r_ratio,
  isthmus_eardrum_cm3, 2nd_bend_eardrum_cm3, 1st_2nd_bend_cm3,
  isthmus_1st_bend_cm3, cbj_eardrum_cm3, cbj_plane_angle_deg_vol,
  ratio_mean_full, ratio_median_full,
  ratio_max_isthmus,
  cbj_pos_pct_isthmus_eardrum,
  angle_total_1st_bend_deg, angle_total_2nd_bend_deg,
  sagittal_arc_angle_deg, angle_entrance_to_eardrum_deg
"""

import json
import os
import re
import shutil
import traceback

import numpy as np
import pandas as pd
import pyvista as pv
import vtk

# ============================================================================
# CONFIGURATION  — edit these paths before running
# ============================================================================

input_dir    = "/projects/oticon/erhdata/Processed-Data/SBEO/HighRes_noeyes/Output/Metrics"

metadata_csv = [
    "/projects/oticon/erhdata/Processed-Data/SBEO/HighRes_noeyes/highres_metadata.csv",
]

output_dir   = "/projects/oticon/erhdata/Processed-Data/SBEO/HighRes_noeyes/Webpage_features"

#create output_dir if it doesn't exist, but be careful not to accidentally overwrite something important!
os.makedirs(output_dir, exist_ok=True)

# Pseudonymization mapping directory
pseudonym_dir = "/projects/oticon/erhdata/Output/SBEO/pseudonymization" #make sure its the centralized location for all pseudonymization mappings across all projects
pseudonym_mapping_csv = os.path.join(pseudonym_dir, "patient_id_mapping.csv")

# Sub-directories inside input_dir
vtk_dir     = os.path.join(input_dir, "vtk")
markups_dir = os.path.join(input_dir, "markups")
stl_dir     = os.path.join(input_dir, "stl")
inverted_mask_dir = os.path.join(input_dir, "masks") 

# Centerline features CSV (produced by the processing pipeline)
centerline_features_csv = os.path.join(input_dir, "processing_results.csv")

# Output file
output_csv        = os.path.join(output_dir, "all_ear_canal_features.csv")
output_stl_folder = os.path.join(output_dir, "stl_webpage")
output_block_stl_folder = os.path.join(output_dir, "stl_webpage_block")
output_vtk_folder = os.path.join(output_dir, "vtk_webpage")
output_markup_folder = os.path.join(output_dir, "markup_webpage")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_stl_folder, exist_ok=True)
os.makedirs(output_block_stl_folder, exist_ok=True)
os.makedirs(output_vtk_folder, exist_ok=True)
os.makedirs(output_markup_folder, exist_ok=True)
os.makedirs(pseudonym_dir, exist_ok=True)

# ── Metadata CSV column aliases ───────────────────────────────────────────
# Set to None to use auto-detection, or override with your exact column name.
META_PATIENT_ID_COL = None   # e.g. "Patient_ID" or "patientid"
META_AGE_COL        = None   # e.g. "Patient_Age" or "age"
META_SEX_COL        = None   # e.g. "Patient_Sex" or "gender"

# ============================================================================
# OUTPUT COLUMN ORDER (mirrors combined_ear_canal_features.csv exactly)
# ============================================================================
OUTPUT_COLUMNS = [
    "patient_id", "EarID", "age", "sex", "side",
    "ear_size", "tragus_size",
    "isthmus_eardrum_length", "isthmus_eardrum_tortuosity",
    "area_mean",
    "1st_bend_area", "2nd_bend_area", "eardrum_area", "isthmus_area",
    "first_to_second_bend_distance",
    "isthmus_to_first_bend_distance",
    "second_bend_to_eardrum_distance",
    "area_cv",
    "1st_bend_max_min_r_ratio", "2nd_bend_max_min_r_ratio",
    "eardrum_max_min_r_ratio", "isthmus_max_min_r_ratio",
    "isthmus_eardrum_cm3", "2nd_bend_eardrum_cm3", "1st_2nd_bend_cm3",
    "isthmus_1st_bend_cm3", "cbj_eardrum_cm3", "cbj_plane_angle_deg_vol",
    "ratio_mean_full", "ratio_median_full",
    "ratio_max_isthmus",
    "cbj_pos_pct_isthmus_eardrum",
    "angle_total_1st_bend_deg", "angle_total_2nd_bend_deg",
    "sagittal_arc_angle_deg", "angle_entrance_to_eardrum_deg",
]

# Columns allowed to be NaN in the final output (all others must be complete)
OPTIONAL_COLS = {"age", "sex"}

pv.set_plot_theme("document")

# ============================================================================
# PSEUDONYMIZATION HELPERS
# ============================================================================

def load_pseudonym_mapping(mapping_csv):
    """Load existing patient ID → pseudonym mapping from CSV.
    
    Returns:
        dict: {original_patient_id: pseudonym}
    """
    if not os.path.exists(mapping_csv):
        return {}
    
    df = pd.read_csv(mapping_csv)
    if df.empty or 'original_patient_id' not in df.columns or 'pseudonym' not in df.columns:
        return {}
    
    return dict(zip(df['original_patient_id'], df['pseudonym']))


def save_pseudonym_mapping(mapping_dict, mapping_csv):
    """Save patient ID → pseudonym mapping to CSV.
    
    Args:
        mapping_dict: {original_patient_id: pseudonym}
        mapping_csv: path to save CSV
    """
    df = pd.DataFrame([
        {'original_patient_id': orig, 'pseudonym': pseudo}
        for orig, pseudo in sorted(mapping_dict.items())
    ])
    df.to_csv(mapping_csv, index=False)
    print(f"\nPseudonym mapping saved: {mapping_csv}")
    print(f"  Total mappings: {len(mapping_dict)}")


def generate_pseudonym(counter, ear_side):
    """Generate a full pseudonym in format patient_00001_left.
    
    Args:
        counter: integer counter for the pseudonym
        ear_side: 'left' or 'right'
    
    Returns:
        str: full pseudonym like 'patient_00001_left'
    """
    return f"patient_{counter:05d}_{ear_side}"


def patient_base_id(patient_id):
    """Return side-agnostic patient base ID.

    Example:
      CHUM-001_left_ear  -> CHUM-001
      CHUM-001_right_ear -> CHUM-001
      0522c0014_left     -> 0522c0014
      0522c0014_right    -> 0522c0014
    """
    return re.sub(r"_(left|right)(_ear)?$", "", str(patient_id).strip(), flags=re.IGNORECASE)


def patient_ear_side(patient_id):
    """Return normalized ear side: 'left' or 'right' (fallback: 'unknown')."""
    pid = str(patient_id).lower()
    # Check for explicit patterns first
    m = re.search(r"_(left|right)(_ear)?$", pid)
    if m:
        return m.group(1)
    # Fallback: search anywhere in the string
    if "left" in pid:
        return "left"
    if "right" in pid:
        return "right"
    return "unknown"


def compose_ear_pseudonym(base_pseudonym, ear_side):
    """Compose final ear pseudonym as patient_{00000}_earside."""
    m = re.match(r"^(patient)_(\d+)$", str(base_pseudonym))
    if m:
        return f"{m.group(1)}_{m.group(2)}_{ear_side}"

    # Backward-compatible fallback for legacy values (e.g., ear_00001)
    m = re.match(r"^(.+?)_(\d+)$", str(base_pseudonym))
    if m:
        return f"{m.group(1)}_{m.group(2)}_{ear_side}"
    return f"{base_pseudonym}_{ear_side}"


def update_pseudonym_mapping(patient_ids, existing_mapping):
    """Update pseudonym mapping with new patients at ear level (includes left/right).
    
    Args:
        patient_ids: list of original ear-level IDs to process
        existing_mapping: dict of {patient_id: pseudonym}
    
    Returns:
        dict: updated mapping {patient_id: pseudonym}
    """
    mapping = existing_mapping.copy()
    
    # Find the next available counter by looking at existing pseudonyms
    if mapping:
        existing_pseudonyms = set(mapping.values())
        existing_numbers = []
        for pseudo in existing_pseudonyms:
            # Extract number from pattern like patient_00001_left
            m = re.search(r"patient_(\d+)_", str(pseudo))
            if m:
                try:
                    existing_numbers.append(int(m.group(1)))
                except ValueError:
                    pass
        next_counter = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        next_counter = 1
    
    # Group patient IDs by base ID to assign same counter to left/right pairs
    base_to_ears = {}
    for pid in patient_ids:
        base_pid = patient_base_id(pid)
        ear_side = patient_ear_side(pid)
        if base_pid not in base_to_ears:
            base_to_ears[base_pid] = []
        base_to_ears[base_pid].append((pid, ear_side))
    
    # Add new patient IDs (with same counter for both ears of same patient)
    new_count = 0
    for base_pid in sorted(base_to_ears.keys()):
        # Check if any ear from this patient already has a mapping
        existing_counter = None
        for pid, ear_side in base_to_ears[base_pid]:
            if pid in mapping:
                # Extract counter from existing pseudonym
                m = re.search(r"patient_(\d+)_", mapping[pid])
                if m:
                    existing_counter = int(m.group(1))
                    break
        
        # Use existing counter or assign new one
        if existing_counter is None:
            counter = next_counter
            next_counter += 1
        else:
            counter = existing_counter
        
        # Assign pseudonyms to both ears
        for pid, ear_side in base_to_ears[base_pid]:
            if pid not in mapping:
                mapping[pid] = generate_pseudonym(counter, ear_side)
                new_count += 1
    
    if new_count > 0:
        print(f"  Generated {new_count} new pseudonyms")
    
    return mapping


# ============================================================================
# METADATA LOADING HELPERS
# ============================================================================

def _detect_col(df, candidates):
    """Return the first matching column name (case-insensitive) or None."""
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _parse_age(value):
    """Parse age values: float, '048Y' (TCIA) → float, else NaN."""
    if pd.isna(value):
        return float("nan")
    m = re.match(r"^(\d+)[Yy]?$", str(value).strip())
    if m:
        return float(m.group(1))
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _parse_sex(value):
    """Normalise sex → 1 (male) / 0 (female) / NaN."""
    if pd.isna(value):
        return float("nan")
    v = str(value).strip().upper()
    if v in ("1", "M", "MALE"):
        return 1
    if v in ("0", "F", "FEMALE"):
        return 0
    return float("nan")


def _load_single_metadata(path):
    """Load one metadata CSV/TSV and return a tidy DataFrame with columns
    ['_pid_base', 'age', 'sex'].  Returns None if the file cannot be read."""
    if not os.path.exists(path):
        print(f"  WARNING: metadata file not found, skipping: {path}")
        return None
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        print(f"  WARNING: could not read metadata file ({e}): {path}")
        return None

    # Drop unnamed / all-NaN columns produced by duplicate headers in TSV exports
    df = df.loc[:, df.columns.notna()]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    cols_lower = {c.lower(): c for c in df.columns}

    # ── patient id ──────────────────────────────────────────────────────
    if META_PATIENT_ID_COL and META_PATIENT_ID_COL in df.columns:
        id_col = META_PATIENT_ID_COL
    elif "patient_id" in cols_lower:
        id_col = cols_lower["patient_id"]
    elif "patientid" in cols_lower:
        id_col = cols_lower["patientid"]
    elif "patient id" in cols_lower:
        id_col = cols_lower["patient id"]
    else:
        id_col = df.columns[0]

    df["_pid_base"] = (df[id_col].astype(str).str.strip()
                       .str.replace(r"_(left|right)_ear$", "", regex=True))

    # ── age ──────────────────────────────────────────────────────────────
    if META_AGE_COL and META_AGE_COL in df.columns:
        age_col = META_AGE_COL
    else:
        age_col = _detect_col(df, ["Patient Age", "Patient_Age", "age", "Age"])
    df["age"] = df[age_col].apply(_parse_age) if age_col else float("nan")

    # ── sex ──────────────────────────────────────────────────────────────
    if META_SEX_COL and META_SEX_COL in df.columns:
        sex_col = META_SEX_COL
    else:
        sex_col = _detect_col(df, ["Patient Sex", "Patient_Sex", "sex", "Sex",
                                    "gender", "Gender"])
    df["sex"] = df[sex_col].apply(_parse_sex) if sex_col else float("nan")

    return df[["_pid_base", "age", "sex"]]


def load_metadata(paths):
    """Load one or more metadata CSVs and return (DataFrame indexed by
    _pid_base, norm_index dict).  Accepts a single path string or a list.
    When multiple files contain the same patient, the first file wins."""
    if isinstance(paths, str):
        paths = [paths]

    frames = []
    for p in paths:
        part = _load_single_metadata(p)
        if part is not None:
            frames.append(part)

    if not frames:
        out = pd.DataFrame(columns=["age", "sex"])
        out.index.name = "_pid_base"
        return out, {}

    combined = pd.concat(frames, ignore_index=True)
    # De-duplicate: one row per base patient id (first file wins)
    combined = combined.drop_duplicates(subset="_pid_base", keep="first")
    out = combined.set_index("_pid_base")[["age", "sex"]]

    # Normalised index: remove dashes/underscores/spaces + lowercase
    norm_index = {
        re.sub(r"[-_\s]", "", k).lower(): k
        for k in out.index
    }
    return out, norm_index


# ============================================================================
# GEOMETRY / VTK HELPERS
# ============================================================================

AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def load_markups(json_path):
    """Return (landmarks dict  {label.lower(): np.array},
               cross_sections dict) from a markup JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    landmarks = {}
    for pt in data.get("landmarks", []):
        label = pt.get("label", "")
        pos   = pt.get("position", [])
        if label and isinstance(pos, list) and len(pos) == 3:
            if not isinstance(pos[0], dict):          # skip multi-point CBJ here
                landmarks[label.lower()] = np.array(pos, dtype=float)
    return landmarks, data.get("cross_sections", {})


def load_cbj_points(json_path):
    """Return Nx3 array of the 4 CBJ sub-point positions."""
    with open(json_path) as f:
        data = json.load(f)
    for entry in data.get("landmarks", []):
        if entry.get("label", "").lower() == "cbj":
            pos = entry.get("position", [])
            if isinstance(pos, list) and len(pos) > 0 and isinstance(pos[0], dict):
                return np.array([p["position"] for p in pos], dtype=float)
    raise ValueError("CBJ 4-point landmark not found in JSON.")


def load_centerline_vtk(vtk_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    return reader.GetOutput()


def cl_pts_array(vtk_polydata):
    n = vtk_polydata.GetNumberOfPoints()
    return np.array([vtk_polydata.GetPoint(i) for i in range(n)])


def cl_point_and_tangent(cl_pts, n_pts, index):
    idx = max(0, min(int(index), n_pts - 1))
    if idx == 0:
        p1, p2 = cl_pts[0], cl_pts[1]
    elif idx == n_pts - 1:
        p1, p2 = cl_pts[-2], cl_pts[-1]
    else:
        p1, p2 = cl_pts[idx - 1], cl_pts[idx + 1]
    d = p2 - p1
    n = np.linalg.norm(d)
    return cl_pts[idx], d / n if n > 0 else d


def cl_arc_length(cl_pts, start_idx, end_idx):
    """Arc-length distance along the centerline between two point indices."""
    s = max(0, int(min(start_idx, end_idx)))
    e = min(len(cl_pts) - 1, int(max(start_idx, end_idx)))
    if s >= e:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(cl_pts[s:e+1], axis=0), axis=1)))


def angle_vec_to_plane(vector, plane_normal):
    """Angle between a vector and a plane (complement of angle with normal)."""
    v = vector / np.linalg.norm(vector)
    n = plane_normal / np.linalg.norm(plane_normal)
    return 90.0 - float(np.degrees(np.arccos(np.clip(np.abs(np.dot(v, n)), 0.0, 1.0))))


def fit_plane_svd(points):
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    normal   = Vt[-1]
    return centroid, normal / np.linalg.norm(normal)


# ── Volume helpers ────────────────────────────────────────────────────────

def _get_plane(cross_sections, key, vtk_cl, n_pts):
    """Return (pos, unit_normal) for a cross-section key."""
    cs = cross_sections[key]
    if "point" in cs and "normal" in cs:
        pos = np.array(cs["point"],  dtype=float)
        nor = np.array(cs["normal"], dtype=float)
        return pos, nor / np.linalg.norm(nor)
    idx    = max(0, min(int(cs["centerline_index"]), n_pts - 1))
    origin = np.array(vtk_cl.GetPoint(idx))
    if idx == 0:
        p1, p2 = np.array(vtk_cl.GetPoint(0)), np.array(vtk_cl.GetPoint(1))
    elif idx == n_pts - 1:
        p1, p2 = np.array(vtk_cl.GetPoint(n_pts-2)), np.array(vtk_cl.GetPoint(n_pts-1))
    else:
        p1, p2 = np.array(vtk_cl.GetPoint(idx-1)), np.array(vtk_cl.GetPoint(idx+1))
    d = p2 - p1
    return origin, d / np.linalg.norm(d)


def clip_mesh_with_plane(mesh, origin, normal, keep_positive_side=True):
    plane  = vtk.vtkPlane()
    nor    = normal if keep_positive_side else -np.array(normal)
    plane.SetOrigin(origin)
    plane.SetNormal(nor)
    planes = vtk.vtkPlaneCollection()
    planes.AddItem(plane)
    clipper = vtk.vtkClipClosedSurface()
    clipper.SetInputData(mesh)
    clipper.SetClippingPlanes(planes)
    clipper.SetGenerateFaces(True)
    clipper.Update()
    return clipper.GetOutput()


def _closest_fragment(mesh, pt1, pt2=None):
    """Select the mesh fragment whose surface is closest to pt1 (and pt2)."""
    pv_mesh = pv.wrap(mesh)
    pt1     = np.array(pt1, dtype=float)
    labeled = pv_mesh.connectivity(largest=False)
    rids    = np.unique(labeled.point_data["RegionId"]).astype(int)
    if len(rids) <= 1:
        return labeled
    best_score, best_comp = float("inf"), None
    for rid in rids:
        comp = labeled.threshold([rid-0.5, rid+0.5], scalars="RegionId", preference="point")
        if comp.n_cells == 0:
            continue
        pts   = comp.points
        score = float(np.min(np.linalg.norm(pts - pt1, axis=1)))
        if pt2 is not None:
            score += float(np.min(np.linalg.norm(pts - np.array(pt2, dtype=float), axis=1)))
        if score < best_score:
            best_score, best_comp = score, comp
    return best_comp.extract_surface()


def close_and_volume(fragment):
    closed = pv.wrap(fragment)
    for _ in range(10):
        closed = closed.fill_holes(hole_size=1000).clean()
        if closed.n_open_edges == 0:
            break
    closed = closed.compute_normals(consistent_normals=True, auto_orient_normals=True)
    mp = vtk.vtkMassProperties()
    mp.SetInputData(closed)
    mp.Update()
    return mp.GetVolume() / 1000.0   # mm³ → cm³


def vol_single_clip(stl_vtk, origin, normal, far_pt, seed1, seed2):
    """Volume on the side of `origin`-plane that contains `far_pt`.

    Mirrors new_volumes_calculation_stl.py:
      direction = far_pt - origin  (vector pointing toward the kept side)
      keep      = dot(direction, normal) > 0
    """
    keep = np.dot(far_pt - origin, normal) > 0
    c    = clip_mesh_with_plane(stl_vtk, origin, normal, keep_positive_side=keep)
    return close_and_volume(_closest_fragment(c, seed1, seed2))


def vol_between_clips(stl_vtk, o1, n1, far1, o2, n2, far2, seed1, seed2):
    """Volume between two planes.

    Mirrors new_volumes_calculation_stl.py two-step clipping:
      Step 1 — keep the `far1` side of plane 1  (far1 = o2, the other plane's origin)
      Step 2 — keep the `far2` side of plane 2  (far2 = o1, pointing back)
    """
    k1 = np.dot(far1 - o1, n1) > 0
    c  = clip_mesh_with_plane(stl_vtk, o1, n1, keep_positive_side=k1)
    k2 = np.dot(far2 - o2, n2) > 0
    c  = clip_mesh_with_plane(c, o2, n2, keep_positive_side=k2)
    return close_and_volume(_closest_fragment(c, seed1, seed2))


# ── Block STL from inverted NIfTI mask ───────────────────────────────────

def nifti_mask_to_stl(mask_path, output_stl_path):
    """Convert a binary NIfTI mask into a solid block STL.

    Steps:
      1. Load the mask with nibabel (preserves full sform/qform affine).
      2. Invert the binary values (0 → 1, 1 → 0) so the solid region
         becomes the new foreground.
      3. Build a pyvista ImageData (voxel grid) from the inverted array.
      4. Threshold to keep only foreground voxels → solid block geometry.
      5. Extract the outer surface of those voxels (blocky, no marching
         cubes smoothing) and apply the full affine transform so the mesh
         is in world coordinates.
      6. Save as STL.
    """
    try:
        import nibabel as nib
    except ImportError as err:
        raise ImportError(
            "nibabel is required for block STL generation. "
            "Install it with: pip install nibabel"
        ) from err

    # ── 1. Load ──────────────────────────────────────────────────────────
    img    = nib.load(mask_path)
    data   = img.get_fdata()
    affine = np.array(img.affine, dtype=float)   # 4×4

    # ── 2. Invert (0 → 1, 1 → 0) ─────────────────────────────────────
    data_inv = (data < 0.5).astype(np.float32)

    # ── 3. Build pyvista voxel grid ──────────────────────────────────────
    # spacing is the column norms of the 3×3 part of the affine.
    # origin is taken directly from the affine translation column.
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(float)

    grid = pv.ImageData()
    grid.dimensions = np.array(data_inv.shape) + 1  # cell-based: N+1 points per axis
    grid.spacing    = spacing
    grid.origin     = affine[:3, 3]
    grid.cell_data["mask"] = data_inv.flatten(order="F").astype(float)

    # ── 4. Threshold → solid block cells ─────────────────────────────────
    block = grid.threshold(0.5, scalars="mask")
    if block.n_cells == 0:
        raise ValueError("Inverted mask is empty — no block voxels found.")

    # ── 5. Extract outer surface of the voxel block ───────────────────────
    surface = block.extract_surface()

    # ── 6. Apply Gaussian smoothing (sigma=1.5) ──────────────────────────
    # Use VTK's smoothing to reduce blocky appearance - more aggressive settings
    import vtk as _vtk
    
    # First pass: Windowed Sinc smoothing (high quality)
    sinc_smooth = _vtk.vtkWindowedSincPolyDataFilter()
    sinc_smooth.SetInputData(surface)
    sinc_smooth.SetNumberOfIterations(50)  # increased for stronger smoothing
    sinc_smooth.BoundarySmoothingOff()
    sinc_smooth.FeatureEdgeSmoothingOff()
    sinc_smooth.SetPassBand(0.01)  # much lower for aggressive smoothing (sigma ~1.5)
    sinc_smooth.NonManifoldSmoothingOn()
    sinc_smooth.NormalizeCoordinatesOn()
    sinc_smooth.Update()
    
    # Second pass: Laplacian smoothing for additional refinement
    laplacian = _vtk.vtkSmoothPolyDataFilter()
    laplacian.SetInputData(sinc_smooth.GetOutput())
    laplacian.SetNumberOfIterations(30)
    laplacian.SetRelaxationFactor(0.5)
    laplacian.FeatureEdgeSmoothingOff()
    laplacian.BoundarySmoothingOn()
    laplacian.Update()
    
    surface = pv.wrap(laplacian.GetOutput())

    # ── 7. Scale mm → m (ANSYS expects metres) ──────────────────────────
    surface.points *= 0.001

    # ── 8. Save ──────────────────────────────────────────────────────────
    surface.save(output_stl_path)


# ============================================================================
# LOAD CENTERLINE FEATURES CSV
# ============================================================================

def load_centerline_features(csv_path):
    """Return (features_dict, excluded_dict) from processing_results.csv.

    features_dict : {patient_id: {col: value}} — rows with NO missing values
    excluded_dict : {patient_id: [missing_col_names]} — rows with any NaN/empty
    """
    if not os.path.exists(csv_path):
        print(f"  WARNING: centerline features CSV not found: {csv_path}")
        return {}, {}
    df     = pd.read_csv(csv_path)
    id_col = df.columns[0]               # first column is always sample_name / patient_id
    df     = df.rename(columns={id_col: "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    data_cols     = [c for c in df.columns if c != "patient_id"]
    complete_mask = df[data_cols].notna().all(axis=1)

    excluded = {}
    for _, row in df[~complete_mask].iterrows():
        missing = [c for c in data_cols if pd.isna(row[c])]
        excluded[row["patient_id"]] = missing

    features_dict = (df[complete_mask]
                     .set_index("patient_id")
                     .to_dict(orient="index"))
    return features_dict, excluded


# ============================================================================
# PER-PARTICIPANT FEATURE EXTRACTION
# ============================================================================

def extract_features(patient_id, cl_features_row, markups_path,
                     cl_refined_path, cl_features_vtk_path, stl_path,
                     vol_diagnostics=None):
    """
    Extract all features for one participant.
    Returns a flat dict with keys matching OUTPUT_COLUMNS (NaN for missing).
    vol_diagnostics: if a dict is passed, volume-specific skip reasons are recorded into it.
    """
    nan = float("nan")
    feat = {c: nan for c in OUTPUT_COLUMNS}
    feat["patient_id"] = patient_id
    feat["EarID"]      = patient_id.replace("_ear", "")   # e.g. CHUM-001_left
    feat["side"]       = "left" if "left" in patient_id.lower() else "right"

    # ── A. Features from centerline features CSV ──────────────────────────
    r = cl_features_row or {}
    eps = 1e-10

    feat["bte_geodesic_length"]        = r.get("geodesic_length",          nan)
    feat["bte_geodesic_tortuosity"]    = r.get("geodesic_tortuosity",      nan)
    feat["isthmus_eardrum_length"]     = r.get("isthmus_eardrum_length",   nan)
    feat["isthmus_eardrum_tortuosity"] = r.get("isthmus_eardrum_tortuosity", nan)
    feat["area_mean"]                  = r.get("area_mean",                nan)
    feat["1st_bend_area"]              = r.get("1st_bend_area",            nan)
    feat["2nd_bend_area"]              = r.get("2nd_bend_area",            nan)
    feat["eardrum_area"]               = r.get("eardrum_area",             nan)
    feat["isthmus_area"]               = r.get("isthmus_area",             nan)

    # Distances between landmarks — computed from the refined centerline VTK
    # (populated later in section C; these CSV fallbacks are left as nan)
    feat["first_to_second_bend_distance"]   = nan
    feat["isthmus_to_first_bend_distance"]  = nan
    feat["second_bend_to_eardrum_distance"] = nan

    # Area coefficient of variation
    a_mean = r.get("area_mean", 0.0) or 0.0
    a_std  = r.get("area_std",  nan)
    feat["area_cv"] = a_std / a_mean if (a_mean > 0 and not np.isnan(a_std)) else nan

    # Max/min radius ratios at landmark cross-sections
    for lm in ["1st_bend", "2nd_bend", "eardrum", "isthmus"]:
        mn = r.get(f"{lm}_min_radius", nan)
        mx = r.get(f"{lm}_max_radius", nan)
        feat[f"{lm}_max_min_r_ratio"] = (mx / (mn + eps)
                                         if not (np.isnan(mn) or np.isnan(mx)) else nan)

    if not os.path.exists(markups_path):
        return feat   # nothing more possible without markups

    # ── B. Ear size & tragus size from markup landmarks ───────────────────
    try:
        landmarks, cross_sections = load_markups(markups_path)

        if "top rs" in landmarks and "bottom rs" in landmarks:
            feat["ear_size"] = float(np.linalg.norm(
                landmarks["top rs"] - landmarks["bottom rs"]))

        if "top tragus" in landmarks and "bottom tragus" in landmarks:
            feat["tragus_size"] = float(np.linalg.norm(
                landmarks["top tragus"] - landmarks["bottom tragus"]))

    except Exception as e:
        print(f"\n  [markup geometry] {patient_id}: {e}")
        landmarks, cross_sections = {}, {}

    # ── C. Ratio & angle features (centerline VTK + markups) ─────────────
    if os.path.exists(cl_refined_path):
        try:
            vtk_cl = load_centerline_vtk(cl_refined_path)
            n_pts  = vtk_cl.GetNumberOfPoints()
            cl_pts = cl_pts_array(vtk_cl)

            cs_idx = {k: int(v["centerline_index"])
                      for k, v in cross_sections.items()
                      if "centerline_index" in v}

            # ── C1. Radius ratios from features VTK ───────────────────────
            if os.path.exists(cl_features_vtk_path):
                try:
                    fm = pv.read(cl_features_vtk_path)
                    if "min_radius" in fm.point_data and "max_radius" in fm.point_data:
                        min_r = np.asarray(fm.point_data["min_radius"])
                        max_r = np.asarray(fm.point_data["max_radius"])
                        ratio = max_r / (min_r + eps)
                        feat["ratio_mean_full"]   = float(np.mean(ratio))
                        feat["ratio_median_full"] = float(np.median(ratio))

                        if "isthmus" in cs_idx:
                            ith_idx  = cs_idx["isthmus"]
                            drum_idx = cs_idx.get("eardrum", len(max_r) - 1)
                            r_ith    = (max_r[ith_idx:drum_idx+1]
                                        / (min_r[ith_idx:drum_idx+1] + eps))
                            feat["ratio_max_isthmus"] = float(r_ith.max())
                            seg_len = drum_idx - ith_idx
                            worst   = int(np.argmax(r_ith))
                            feat["ratio_worst_pos_pct_isthmus"] = round(
                                (worst / seg_len * 100.0) if seg_len > 0 else 0.0, 2)
                except Exception as e:
                    print(f"\n  [features VTK ratio] {patient_id}: {e}")

            # ── C1b. Arc-length distances between landmark cross-sections ──────
            if "1st_bend" in cs_idx and "2nd_bend" in cs_idx:
                feat["first_to_second_bend_distance"] = cl_arc_length(
                    cl_pts, cs_idx["1st_bend"], cs_idx["2nd_bend"])

            if "isthmus" in cs_idx and "1st_bend" in cs_idx:
                feat["isthmus_to_first_bend_distance"] = cl_arc_length(
                    cl_pts, cs_idx["isthmus"], cs_idx["1st_bend"])

            if "2nd_bend" in cs_idx and "eardrum" in cs_idx:
                feat["second_bend_to_eardrum_distance"] = cl_arc_length(
                    cl_pts, cs_idx["2nd_bend"], cs_idx["eardrum"])

            # ── C2. Sagittal / entrance reference ─────────────────────────
            entrance_key = next((k for k in landmarks
                                  if "top tragus" in k or k == "tragus"), None)
            if entrance_key:
                entrance_pt  = landmarks[entrance_key]
                sag_dists    = np.abs(np.dot(cl_pts - entrance_pt, [1.0, 0.0, 0.0]))
                sagittal_idx = int(np.argmin(sag_dists))
            else:
                entrance_pt  = cl_pts[0]
                sagittal_idx = 0
            sagittal_pt = cl_pts[sagittal_idx]

            # ── C3. Bend angles ───────────────────────────────────────────
            def tangent_at(idx):
                idx = int(idx)
                if   idx == 0:          t = cl_pts[1]     - cl_pts[0]
                elif idx == n_pts - 1:  t = cl_pts[-1]    - cl_pts[-2]
                else:                   t = cl_pts[idx+1] - cl_pts[idx-1]
                n = np.linalg.norm(t)
                return t / n if n > 0 else t

            if "1st_bend" in cs_idx and "2nd_bend" in cs_idx:
                b1_idx = cs_idx["1st_bend"]
                b2_idx = cs_idx["2nd_bend"]
                t1 = tangent_at(b1_idx)
                vec_in_1  = cl_pts[0]    - cl_pts[b1_idx]
                vec_out_1 = cl_pts[b2_idx] - cl_pts[b1_idx]
                feat["angle_total_1st_bend_deg"] = float(
                    angle_vec_to_plane(vec_in_1,  t1) +
                    angle_vec_to_plane(vec_out_1, t1))

            if "1st_bend" in cs_idx and "2nd_bend" in cs_idx and "eardrum" in cs_idx:
                b2_idx = cs_idx["2nd_bend"]
                t2 = tangent_at(b2_idx)
                vec_in_2  = cl_pts[cs_idx["1st_bend"]] - cl_pts[b2_idx]
                vec_out_2 = cl_pts[cs_idx["eardrum"]]  - cl_pts[b2_idx]
                feat["angle_total_2nd_bend_deg"] = float(
                    angle_vec_to_plane(vec_in_2,  t2) +
                    angle_vec_to_plane(vec_out_2, t2))

            # ── C4. Entrance → Eardrum direction angle ────────────────────
            if "eardrum" in cs_idx:
                vec_ee = cl_pts[cs_idx["eardrum"]] - sagittal_pt
                feat["angle_entrance_to_eardrum_deg"] = angle_vec_to_plane(
                    vec_ee, AXES["x"])

            # ── C5. Sagittal arc angle ────────────────────────────────────
            # Prefer stored JSON "point" for 1st bend (mirrors cs_center() in reference)
            if "1st_bend" in cs_idx:
                _cs1 = cross_sections.get("1st_bend", {})
                _b1_pt = (np.array(_cs1["point"], dtype=float)
                          if "point" in _cs1 else cl_pts[cs_idx["1st_bend"]])
                feat["sagittal_arc_angle_deg"] = angle_vec_to_plane(
                    _b1_pt - sagittal_pt, AXES["x"])

            # ── C6. CBJ position % & plane angle ─────────────────────────
            try:
                cbj_pts        = load_cbj_points(markups_path)
                cbj_pos, cbj_n = fit_plane_svd(cbj_pts)
                cbj_cl_idx     = int(np.argmin(
                    np.linalg.norm(cl_pts - cbj_pos, axis=1)))
                _, cbj_tangent = cl_point_and_tangent(cl_pts, n_pts, cbj_cl_idx)
                cos_a = float(np.clip(np.abs(np.dot(cbj_n, cbj_tangent)), 0.0, 1.0))
                feat["cbj_plane_angle_deg_vol"] = float(np.degrees(np.arccos(cos_a)))

                if "isthmus" in cs_idx and "eardrum" in cs_idx:
                    _ith  = cs_idx["isthmus"]
                    _drum = cs_idx["eardrum"]
                    _seg  = _drum - _ith
                    feat["cbj_pos_pct_isthmus_eardrum"] = round(
                        ((cbj_cl_idx - _ith) / _seg * 100.0) if _seg > 0 else 0.0, 2)
            except Exception:
                pass    # CBJ absent for some participants — stays NaN

        except Exception as e:
            print(f"\n  [angles/ratios] {patient_id}: {e}")

    # ── D. Volume features from STL ───────────────────────────────────────
    _diag = vol_diagnostics if vol_diagnostics is not None else {}

    if not os.path.exists(stl_path):
        _diag["all_volumes"] = f"STL file not found: {os.path.basename(stl_path)}"
    elif not os.path.exists(cl_refined_path):
        _diag["all_volumes"] = f"Centerline VTK not found: {os.path.basename(cl_refined_path)}"
    else:
        try:
            _, cross_sections_vol = load_markups(markups_path)
            vtk_cl = load_centerline_vtk(cl_refined_path)
            n_pts  = vtk_cl.GetNumberOfPoints()

            required = ["eardrum", "isthmus", "2nd_bend", "1st_bend"]
            missing_cs = [k for k in required
                          if k not in cross_sections_vol
                          or "centerline_index" not in cross_sections_vol[k]]
            if missing_cs:
                raise ValueError(f"Missing cross-section keys: {missing_cs}")

            ed_pos,  ed_n  = _get_plane(cross_sections_vol, "eardrum",   vtk_cl, n_pts)
            ith_pos, ith_n = _get_plane(cross_sections_vol, "isthmus",   vtk_cl, n_pts)
            sb_pos,  sb_n  = _get_plane(cross_sections_vol, "2nd_bend",  vtk_cl, n_pts)
            fb_pos,  fb_n  = _get_plane(cross_sections_vol, "1st_bend",  vtk_cl, n_pts)

            stl_pv  = pv.read(stl_path)
            stl_vtk = stl_pv.extract_surface()

            # 1. Isthmus → Eardrum
            # keep the eardrum side of the isthmus plane (direction: isthmus → eardrum)
            try:
                feat["isthmus_eardrum_cm3"] = vol_single_clip(
                    stl_vtk, ith_pos, ith_n,
                    far_pt=ed_pos,          # keep side where eardrum is
                    seed1=ith_pos, seed2=ed_pos)
            except Exception as e:
                _diag["isthmus_eardrum_cm3"] = f"Computation error: {e}"

            # 2. 2nd Bend → Eardrum
            # keep the eardrum side of the 2nd bend plane
            try:
                feat["2nd_bend_eardrum_cm3"] = vol_single_clip(
                    stl_vtk, sb_pos, sb_n,
                    far_pt=ed_pos,          # keep side where eardrum is
                    seed1=sb_pos, seed2=ed_pos)
            except Exception as e:
                _diag["2nd_bend_eardrum_cm3"] = f"Computation error: {e}"

            # 3. 1st Bend → 2nd Bend
            # Step 1: keep 2nd-bend side of 1st-bend plane  (far1 = sb_pos)
            # Step 2: keep 1st-bend side of 2nd-bend plane  (far2 = fb_pos)
            try:
                feat["1st_2nd_bend_cm3"] = vol_between_clips(
                    stl_vtk,
                    fb_pos, fb_n, far1=sb_pos,
                    o2=sb_pos, n2=sb_n, far2=fb_pos,
                    seed1=fb_pos, seed2=sb_pos)
            except Exception as e:
                _diag["1st_2nd_bend_cm3"] = f"Computation error: {e}"

            # 4. Isthmus → 1st Bend  (skip if same cross-section index)
            ith_idx = int(cross_sections_vol["isthmus"]["centerline_index"])
            fb_idx  = int(cross_sections_vol["1st_bend"]["centerline_index"])
            if ith_idx == fb_idx:
                _diag["isthmus_1st_bend_cm3"] = (
                    f"Isthmus and 1st_bend share the same centerline index "
                    f"({ith_idx}); segment has zero length — skipped")
            else:
                try:
                    # Step 1: keep 1st-bend side of isthmus plane  (far1 = fb_pos)
                    # Step 2: keep isthmus side of 1st-bend plane  (far2 = ith_pos)
                    feat["isthmus_1st_bend_cm3"] = vol_between_clips(
                        stl_vtk,
                        ith_pos, ith_n, far1=fb_pos,
                        o2=fb_pos, n2=fb_n, far2=ith_pos,
                        seed1=ith_pos, seed2=fb_pos)
                except Exception as e:
                    _diag["isthmus_1st_bend_cm3"] = f"Computation error: {e}"

            # 5. CBJ → Eardrum
            if ("cbj" not in cross_sections_vol
                    or "centerline_index" not in cross_sections_vol.get("cbj", {})):
                _diag["cbj_eardrum_cm3"] = "CBJ cross-section not present in markup JSON"
            else:
                try:
                    # CBJ plane: fit via SVD from the 4 ring points (matches
                    # new_volumes_calculation_stl.py which calls fit_plane_svd,
                    # NOT _get_plane which would use the stored centerline tangent)
                    cbj_pts_vol = load_cbj_points(markups_path)
                    cbj_p, cbj_n_vol = fit_plane_svd(cbj_pts_vol)
                    # keep the eardrum side of the CBJ plane (direction: CBJ → eardrum)
                    feat["cbj_eardrum_cm3"] = vol_single_clip(
                        stl_vtk, cbj_p, cbj_n_vol,
                        far_pt=ed_pos,          # keep side where eardrum is
                        seed1=cbj_p, seed2=ed_pos)
                except Exception as e:
                    _diag["cbj_eardrum_cm3"] = f"Computation error: {e}"

        except Exception as e:
            _diag["all_volumes"] = str(e)
            print(f"\n  [volumes] {patient_id}: {e}")
            if os.environ.get("DEBUG_VOLUMES"):
                traceback.print_exc()

    return feat


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ALL EAR CANAL FEATURES EXTRACTION")
    print("=" * 70)
    print(f"  input_dir   : {input_dir}")
    print(f"  metadata    : {metadata_csv if isinstance(metadata_csv, str) else str(len(metadata_csv)) + ' files'}")
    print(f"  output_csv  : {output_csv}")
    print()

    # ── Load metadata ─────────────────────────────────────────────────────
    print("Loading metadata...")
    try:
        meta_df, meta_norm = load_metadata(metadata_csv)
        print(f"  {len(meta_df)} unique subjects in metadata")
    except Exception as e:
        print(f"  WARNING: Could not load metadata ({e}). Age/sex will be NaN.")
        meta_df  = pd.DataFrame(columns=["age", "sex"])
        meta_df.index.name = "_pid_base"
        meta_norm = {}

    def _meta_lookup(pid_base):
        """Return (age, sex) from metadata; try exact then normalised prefix match."""
        if pid_base in meta_df.index:
            return meta_df.loc[pid_base, "age"], meta_df.loc[pid_base, "sex"]
        # Normalise the file-based key and try prefix match against metadata keys
        pid_norm = re.sub(r"[-_\s]", "", pid_base).lower()
        for norm_key, orig_key in meta_norm.items():
            if pid_norm.startswith(norm_key) or norm_key.startswith(pid_norm):
                return meta_df.loc[orig_key, "age"], meta_df.loc[orig_key, "sex"]
        return float("nan"), float("nan")

    # ── Load centerline features CSV ──────────────────────────────────────
    print("Loading centerline features CSV...")
    cl_features, cl_excluded = load_centerline_features(centerline_features_csv)
    print(f"  {len(cl_features)} participants with complete centerline features")
    if cl_excluded:
        print(f"  {len(cl_excluded)} participants will be excluded (missing values in CSV)")

    # ── Discover participants from markup files ───────────────────────────
    if not os.path.isdir(markups_dir):
        print(f"ERROR: markups directory not found: {markups_dir}")
        return

    json_files  = sorted(f for f in os.listdir(markups_dir) if f.endswith(".json"))
    patient_ids = [f[:-5] for f in json_files]
    print(f"\nDiscovered {len(patient_ids)} participants from markups directory")
    if cl_excluded:
        print(f"  (skipping {len([p for p in patient_ids if p in cl_excluded])} "
              f"due to missing centerline CSV values)")
    print()

    # ── Load/update pseudonym mapping ─────────────────────────────────────
    print("Loading pseudonym mapping...")
    # Load existing mapping at ear level (includes left/right)
    existing_mapping = load_pseudonym_mapping(pseudonym_mapping_csv)
    print(f"  Existing mappings: {len(existing_mapping)}")
    
    # Update mapping with all discovered patient IDs (even those we might skip)
    all_mapping = update_pseudonym_mapping(patient_ids, existing_mapping)
    
    # Save updated mapping immediately (so it's available even if processing fails)
    save_pseudonym_mapping(all_mapping, pseudonym_mapping_csv)
    print()

    results        = []
    errors         = []
    vol_issues     = {}   # {patient_id: {metric: reason}}
    output_excluded = {}  # {patient_id: [missing non-optional columns]}

    for i, pid in enumerate(patient_ids):
        # ── Skip participants with missing values in centerline features CSV ──
        if pid in cl_excluded:
            print(f"[{i+1}/{len(patient_ids)}] {pid} ... SKIPPED "
                  f"(missing in centerline CSV: {cl_excluded[pid]})")
            continue

        print(f"[{i+1}/{len(patient_ids)}] {pid}", end=" ... ", flush=True)
        try:
            diag = {}
            row = extract_features(
                patient_id           = pid,
                cl_features_row      = cl_features.get(pid, {}),
                markups_path         = os.path.join(markups_dir, f"{pid}.json"),
                cl_refined_path      = os.path.join(vtk_dir, f"{pid}_centerline_refined.vtk"),
                cl_features_vtk_path = os.path.join(vtk_dir, f"{pid}_centerline_features.vtk"),
                stl_path             = os.path.join(stl_dir,
                                           f"{pid}_open_surface_cut_eardrum.stl"),
                vol_diagnostics      = diag,
            )
            if diag:
                vol_issues[pid] = diag

            # Attach demographics
            pid_base = re.sub(r"_(left|right)_ear$", "", pid)
            row["age"], row["sex"] = _meta_lookup(pid_base)
            if np.isnan(row["age"]) and np.isnan(row["sex"]):
                print(f"\n    NOTE: '{pid_base}' not in metadata – age/sex set to NaN")

            # Check all required columns are non-null; exclude if any are missing
            missing_required = [
                c for c in OUTPUT_COLUMNS
                if c not in OPTIONAL_COLS
                and (c not in row or (isinstance(row.get(c), float) and np.isnan(row[c])))
            ]
            if missing_required:
                output_excluded[pid] = missing_required
                print(f"EXCLUDED  (missing required columns: {missing_required})")
                continue

            results.append(row)

            # Get full pseudonym (already includes ear side)
            pseudonym = all_mapping.get(pid, pid)
            pseudonym_base = pseudonym
            
            # Copy STL to output folder, renaming with pseudonym
            stl_src = os.path.join(stl_dir, f"{pid}_open_surface_cut_eardrum.stl")
            if os.path.exists(stl_src):
                stl_name = pseudonym_base + ".stl"
                try:
                    shutil.copy2(stl_src, os.path.join(output_stl_folder, stl_name))
                except Exception as _ce:
                    print(f"\n    WARNING: could not copy STL for {pid}: {_ce}")

            # Generate block STL from inverted NIfTI mask with pseudonym
            mask_src = os.path.join(inverted_mask_dir, f"{pid}_inverted.nii.gz")
            if os.path.exists(mask_src):
                block_stl_name = pseudonym_base + "_block.stl"
                block_stl_path = os.path.join(output_block_stl_folder, block_stl_name)
                try:
                    nifti_mask_to_stl(mask_src, block_stl_path)
                except Exception as _be:
                    print(f"\n    WARNING: could not generate block STL for {pid}: {_be}")
            else:
                print(f"\n    NOTE: inverted mask not found for {pid}: {os.path.basename(mask_src)}")

            # Copy VTK files to output folder with pseudonym
            vtk_files_to_copy = [
                (f"{pid}_centerline_refined.vtk", f"{pseudonym_base}_centerline_refined.vtk"),
                (f"{pid}_centerline_features.vtk", f"{pseudonym_base}_centerline_features.vtk"),
            ]
            for vtk_src_name, vtk_dst_name in vtk_files_to_copy:
                vtk_src = os.path.join(vtk_dir, vtk_src_name)
                if os.path.exists(vtk_src):
                    try:
                        shutil.copy2(vtk_src, os.path.join(output_vtk_folder, vtk_dst_name))
                    except Exception as _ve:
                        print(f"\n    WARNING: could not copy VTK {vtk_src_name} for {pid}: {_ve}")

            # Copy markup JSON to output folder with pseudonym
            markup_src = os.path.join(markups_dir, f"{pid}.json")
            if os.path.exists(markup_src):
                markup_dst_name = pseudonym_base + ".json"
                try:
                    shutil.copy2(markup_src, os.path.join(output_markup_folder, markup_dst_name))
                except Exception as _me:
                    print(f"\n    WARNING: could not copy markup for {pid}: {_me}")

            # Quick summary
            def _s(v, fmt):
                try:
                    return format(float(v), fmt) if not np.isnan(float(v)) else "?"
                except Exception:
                    return "?"
            print(f"OK  age={_s(row['age'],'.0f')}  "
                  f"ith→ed={_s(row['isthmus_eardrum_cm3'],'.2f')}cm³  "
                  f"1stBend={_s(row['angle_total_1st_bend_deg'],'.1f')}°")

        except Exception as e:
            errors.append({"patient_id": pid, "error": str(e)})
            print(f"ERROR: {e}")

    # ── Save CSV with pseudonymization ────────────────────────────────────
    if not results:
        print("\nNo results to save.")
        return

    df = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[OUTPUT_COLUMNS]
    
    # Apply pseudonymization to patient_id and EarID columns (already includes ear side)
    df['patient_id'] = df['patient_id'].apply(
        lambda x: all_mapping.get(x, x)
    )
    df['EarID'] = df['EarID'].apply(
        lambda x: all_mapping.get(x + '_ear', x) if not x.endswith('_ear') else all_mapping.get(x, x)
    )
    
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*70}")
    print(f"Saved {len(df)} rows  →  {output_csv}")
    print(f"Failed: {len(errors)}")
    for e in errors:
        print(f"  {e['patient_id']}: {e['error']}")

    # ── Unified exclusions txt file ───────────────────────────────────────
    # Covers: (1) missing values in centerline CSV, (2) missing required output columns
    total_excluded = len(cl_excluded) + len(output_excluded)
    excl_path = os.path.join(output_dir, "excluded_participants.txt")
    with open(excl_path, "w") as f:
        f.write("EXCLUDED PARTICIPANTS\n")
        f.write(f"Output CSV : {output_csv}\n")
        f.write(f"Total excluded: {total_excluded}  "
                f"(centerline CSV: {len(cl_excluded)}  |  incomplete output: {len(output_excluded)})\n")
        f.write("Note: only 'age' and 'sex' are allowed to be empty in the output.\n")
        f.write("=" * 70 + "\n")

        if not cl_excluded and not output_excluded:
            f.write("\nNo participants were excluded.\n")
        else:
            all_pids = sorted(set(cl_excluded) | set(output_excluded))
            max_pid  = max(len(p) for p in all_pids)
            if cl_excluded:
                f.write(f"\n[SECTION 1] Missing values in centerline features CSV "
                        f"({len(cl_excluded)} participants)\n")
                f.write("-" * 70 + "\n")
                for pid in sorted(cl_excluded):
                    missing_str = ", ".join(cl_excluded[pid])
                    f.write(f"  {pid:<{max_pid}}  missing: {missing_str}\n")
            if output_excluded:
                f.write(f"\n[SECTION 2] Missing required output columns after feature extraction "
                        f"({len(output_excluded)} participants)\n")
                f.write("-" * 70 + "\n")
                for pid in sorted(output_excluded):
                    missing_str = ", ".join(output_excluded[pid])
                    f.write(f"  {pid:<{max_pid}}  missing: {missing_str}\n")
    print(f"\nExcluded participants  →  {excl_path}")
    print(f"  Centerline CSV exclusions : {len(cl_excluded)}")
    print(f"  Incomplete output excluded: {len(output_excluded)}")
    print(f"  Total excluded            : {total_excluded}")

    # ── Volume diagnostics txt file ──────────────────────────────────────
    VOLUME_COLS = [
        "isthmus_eardrum_cm3", "2nd_bend_eardrum_cm3", "1st_2nd_bend_cm3",
        "isthmus_1st_bend_cm3", "cbj_eardrum_cm3",
    ]
    # vol_issues captures errors that occurred during extraction;
    # participants with unresolved NaN volumes are already in output_excluded.
    diag_path = os.path.join(output_dir, "volume_diagnostics.txt")
    with open(diag_path, "w") as f:
        f.write("VOLUME METRIC DIAGNOSTICS\n")
        f.write(f"Generated from: {output_csv}\n")
        f.write(f"Total participants: {len(df)}  |  "
                f"With missing volume metrics: {len(vol_issues)}\n")
        f.write("=" * 70 + "\n\n")

        if not vol_issues:
            f.write("All volume metrics computed successfully for all participants.\n")
        else:
            # Group by reason pattern for a summary first
            from collections import Counter
            reason_counter = Counter()
            for pid, reasons in vol_issues.items():
                for metric, reason in reasons.items():
                    reason_counter[reason.split(":")[0].strip()] += 1

            f.write("SUMMARY OF MISSING REASON CATEGORIES\n")
            f.write("-" * 40 + "\n")
            for reason, count in reason_counter.most_common():
                f.write(f"  [{count:3d}x]  {reason}\n")
            f.write("\n")

            f.write("PER-PARTICIPANT DETAILS\n")
            f.write("-" * 40 + "\n")
            for pid in sorted(vol_issues):
                f.write(f"\n{pid}\n")
                reasons = vol_issues[pid]
                if "all_volumes" in reasons:
                    f.write(f"  ALL VOLUMES  : {reasons['all_volumes']}\n")
                else:
                    for col in VOLUME_COLS:
                        if col in reasons:
                            f.write(f"  {col:<28s}: {reasons[col]}\n")

    print(f"\nVolume diagnostics  \u2192  {diag_path}")
    print(f"  Participants with missing volume metrics: "
          f"{len(vol_issues)}/{len(df)}")
    # ── Non-null summary ──────────────────────────────────────────────────
    print(f"\n--- Non-null counts ({len(df)} total rows) ---")
    check_cols = [c for c in OUTPUT_COLUMNS if c not in ("patient_id", "EarID", "side")]
    for c in check_cols:
        nn = df[c].notna().sum()
        if nn > 0:
            print(f"  {c:<42s}: {nn}/{len(df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()