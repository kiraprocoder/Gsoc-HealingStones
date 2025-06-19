import bpy
import os
import numpy as np
import sys

sys.path.append('/home/kira/Desktop/gsoc/Gsoc-HealingStones')
from align import get_rotation_angles


root_path = 'Ground_Truth'
blender_folder = os.path.join(root_path, 'blender', 'todo')
os.makedirs(blender_folder, exist_ok=True)

done_folder = os.path.join(root_path, 'blender', 'done')
os.makedirs(done_folder, exist_ok=True)

grid_size = 250


pair_folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f)) and f.startswith("pair_")]
pair_folders.sort()

if not pair_folders:
    print(f"No pair folders found in '{root_path}'!")
    sys.exit(1)

print("\nAvailable pair folders:")
for idx, name in enumerate(pair_folders):
    print(f"  [{idx + 1}] {name}")

try:
    selected_idx = int(input("\nEnter number of the pair to load: ")) - 1
    if not (0 <= selected_idx < len(pair_folders)):
        raise ValueError
except ValueError:
    print("Invalid selection. Exiting.")
    sys.exit(1)

selected_folder = pair_folders[selected_idx]
processed_folder = os.path.join(root_path, selected_folder)
print(f"Selected pair: {selected_folder}")


blend_file = os.path.join(blender_folder, f"{selected_folder}_todo.blend")

if os.path.exists(blend_file):
    print("Loading existing scene...")
    bpy.ops.wm.open_mainfile(filepath=blend_file)
else:
    print("Creating new empty scene...")
    bpy.ops.wm.read_factory_settings(use_empty=True)

# Scene Units and Packing Setup
bpy.data.use_autopack = True
bpy.context.scene.unit_settings.scale_length = 0.001
bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'


files = os.listdir(processed_folder)
ply_meshes_list = [file for file in files if file.lower().endswith('.ply')]

if not ply_meshes_list:
    print("No .ply files found in selected folder!")
    sys.exit(1)

print(f"\nFound {len(ply_meshes_list)} PLY files:", ply_meshes_list)

num_pieces = len(ply_meshes_list)
on_axis = np.ceil(np.sqrt(num_pieces)).astype(int)
x_ = np.linspace(-grid_size, grid_size, on_axis)
y_ = np.linspace(-grid_size, grid_size, on_axis)

counter = 0

for ply_mesh in ply_meshes_list:
    filepath_ply = os.path.join(processed_folder, ply_mesh)
    a_x, a_y, a_z = get_rotation_angles(filepath_ply)

    bpy.ops.import_mesh.ply(filepath=filepath_ply)
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')

    bpy.context.object.location = (x_[counter // on_axis], y_[counter % on_axis], 0)
    bpy.context.object.rotation_euler = (a_x, a_y, a_z)
    counter += 1


save_path = blend_file
bpy.ops.wm.save_mainfile(filepath=save_path)
print(f"\nSaved aligned scene to: {save_path}")
