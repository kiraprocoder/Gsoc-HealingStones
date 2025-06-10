import bpy
import os
import numpy as np
import sys

sys.path.append('/home/kira/Desktop/gsoc/Gsoc-HealingStones')
from methods import *

root_path = '/home/kira/Desktop/test_set'
target_gt_folder = os.path.join(root_path, 'ground_truth')
os.makedirs(target_gt_folder, exist_ok=True)
filepath = os.path.join(target_gt_folder, 'setup.blend')

print("start")

grid_size = 250

# 👉 YOU DON'T NEED groups anymore
processed_folder = root_path  # ← Look directly in test_set/

# Open setup.blend or start empty
if os.path.exists(filepath):
    bpy.ops.wm.open_mainfile(filepath=filepath)
else:
    print("⚠️ setup.blend not found. Using empty scene.")
    bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.data.use_autopack = True
bpy.context.scene.unit_settings.scale_length = 0.001
bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

# ✅ Check for .ply files directly in root_path
files = os.listdir(processed_folder)
ply_meshes_list = [file for file in files if file.endswith('.ply')]
print(ply_meshes_list)

num_pieces = len(ply_meshes_list)
if num_pieces == 0:
    print("❌ No .ply files found!")
else:
    on_axis = np.ceil(np.sqrt(num_pieces)).astype(int)
    x_ = np.linspace(-grid_size, grid_size, on_axis)
    y_ = np.linspace(-grid_size, grid_size, on_axis)
    counter = 0

    for ply_mesh in ply_meshes_list:
        a_x, a_y, a_z = get_rotation_angles(os.path.join(processed_folder, ply_mesh))

        filepath_ply = os.path.join(processed_folder, ply_mesh)
        bpy.ops.import_mesh.ply(filepath=filepath_ply)

        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
        bpy.context.object.location = (x_[counter // on_axis], y_[counter % on_axis], 0)
        bpy.context.object.rotation_euler = (a_x, a_y, a_z)
        counter += 1

    save_path = os.path.join(target_gt_folder, f"all_ply_files_TODO.blend")
    bpy.ops.wm.save_mainfile(filepath=save_path)
    print("✅ Saved:", save_path)

print("✅ Finished")
