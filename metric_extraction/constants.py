# -*- coding: utf-8 -*-
"""Pipeline-wide constants and thresholds."""

# Centerline validation thresholds
MIN_CENTERLINE_LENGTH_MM: float = 10.0  # 1 cm minimum centerline length
MIN_ENDPOINT_DISTANCE_MM: float = 20.0  # 2 cm minimum distance between endpoints

# Mesh simplification
DEFAULT_TARGET_REDUCTION: float = 0.95

# Geodesic resampling
DEFAULT_NUM_RESAMPLE_POINTS: int = 100

# Oscillation detection
DEFAULT_OSCILLATION_WINDOW: int = 5
DEFAULT_OSCILLATION_THRESHOLD: float = 0.15

# Endpoint refinement
MAX_REFINEMENT_DISTANCE_MM: float = 5.0
DEFAULT_NUM_NORMALS: int = 10

# Volume adjustment
MAX_VOLUME_THRESHOLD_MM3: float = 1500.0
BBOX_MARGIN_FRACTION: float = 0.10

# Voxel spacing for volume computation
DEFAULT_VOXEL_SPACING_MM: float = 0.5
FINE_VOXEL_SPACING_MM: float = 0.05
