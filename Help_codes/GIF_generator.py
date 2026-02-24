import numpy as np
import pyvista as pv
import json
import os

# ==================== CONFIGURATION ====================
patient_id = "MDA-265_left_ear"
include_landmarks = True  # Set False if you want STL only
final_visualization = True  # Set True to open interactive 3D viewer after GIF generation

stl_base_dir = r"\\kbnnfsserver\erhdata\Processed-Data\SBEO\Final_pipeline\Output\Quality_Check\Canal_good_quality\stl files"
landmarks_base_dir = r"\\kbnnfsserver\erhdata\Processed-Data\SBEO\Final_pipeline\Output\Quality_Check\All Markups (FH MODEL)"
output_dir = r"C:\Users\sbeo\OneDrive - Demant\Desktop\GIFs"

stl_path = os.path.join(stl_base_dir, f"{patient_id}.stl")
landmarks_path = os.path.join(landmarks_base_dir, f"{patient_id}.json")

print(f"Patient ID: {patient_id}")
print(f"Include landmarks: {include_landmarks}")

# ==================== LOAD LANDMARKS ====================
landmark_points = []
landmark_labels = []

if include_landmarks:
    print("\nLoading landmarks...")

    with open(landmarks_path, 'r') as f:
        landmarks_data = json.load(f)

    # Handle 3D Slicer markups format
    if 'markups' in landmarks_data:
        print("Using 3D Slicer markups format")
        for markup in landmarks_data['markups']:
            if 'controlPoints' in markup:
                control_points = markup['controlPoints']
                for idx, point in enumerate(control_points):
                    # Skip first 2 and last landmark
                    if idx < 2 or idx == len(control_points) - 1:
                        continue
                    position = point.get('position', [0, 0, 0])
                    label = point.get('label', 'Unknown')
                    landmark_points.append(position)
                    landmark_labels.append(label)

    # Handle simple landmarks dictionary format
    elif 'landmarks' in landmarks_data:
        print("Using landmarks dictionary format")
        total_landmarks = len(landmarks_data['landmarks'])
        for idx, point in enumerate(landmarks_data['landmarks']):
            if idx < 2 or idx == total_landmarks - 1:
                continue
            position = point.get('position', point.get('location', [0, 0, 0]))
            label = point.get('label', f"Landmark_{idx}")
            landmark_points.append(position)
            landmark_labels.append(label)

    landmark_points = np.array(landmark_points)
    print(f"Loaded {len(landmark_points)} landmarks")

# ==================== VISUALIZATION ====================
print("\nCreating visualization...")

surface = pv.read(stl_path)
print(f"STL loaded: {surface.n_points} points, {surface.n_cells} cells")

plotter = pv.Plotter(off_screen=True)
plotter.set_background(None)  # Transparent background

# ---- Add STL surface ----
plotter.add_mesh(
    surface,
    color='lightblue',
    opacity=1
)

# ---- Add landmarks (NO LIGHTING ARTIFACTS) ----
if include_landmarks and len(landmark_points) > 0:

    plotter.add_points(
        landmark_points,
        color='red',
        point_size=18,
        render_points_as_spheres=True,
        lighting=False
    )

# Remove axes for cleaner GIF
# plotter.add_axes()

plotter.camera_position = 'yz'

# ==================== CREATE ROTATING GIF ====================
print("Generating GIF...")

os.makedirs(output_dir, exist_ok=True)
gif_path = os.path.join(output_dir, f"{patient_id}.gif")

plotter.open_gif(gif_path, fps=15, transparent_background=True)

n_frames = 120
angle_step = 360 / n_frames

for _ in range(n_frames):
    plotter.camera.Azimuth(angle_step)  # capital A, incremental rotation
    plotter.render()
    plotter.write_frame()


plotter.close()

print(f"GIF saved to: {gif_path}")

# ==================== INTERACTIVE VISUALIZATION ====================
if final_visualization:
    print("\n" + "=" * 80)
    print("Opening interactive 3D viewer...")
    print("=" * 80)
    
    # Create interactive plotter
    interactive_plotter = pv.Plotter()
    interactive_plotter.set_background('white')
    
    # Add STL surface
    interactive_plotter.add_mesh(
        surface,
        color='lightblue',
        opacity=1,
        label='Ear Canal'
    )
    
    # Add landmarks if enabled
    if include_landmarks and len(landmark_points) > 0:
        interactive_plotter.add_points(
            landmark_points,
            color='red',
            point_size=18,
            render_points_as_spheres=True,
            lighting=False
        )
        
        # Add labels
        interactive_plotter.add_point_labels(
            landmark_points,
            landmark_labels,
            font_size=12,
            text_color='black',
            font_family='arial',
            bold=True,
            shape=None,
            always_visible=True
        )
    
    # Add axes for reference
    interactive_plotter.add_axes()
    
    # Set initial camera position
    interactive_plotter.camera_position = 'yz'
    
    # Show interactive window
    print("Close the window to exit...")
    interactive_plotter.show()

