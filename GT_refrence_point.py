import bpy
import os, json

#make a path in ground truth as "DONE" and save the blend
root_path = '/home/kira/Desktop/test_set/ground_truth'


solved_puzzles = os.path.join(root_path, 'DONE')
solved_puzzles_gt = os.path.join(root_path, 'gt_json')
os.makedirs(solved_puzzles_gt, exist_ok=True)

list_of_solved_puzzles = [sp for sp in os.listdir(solved_puzzles) if sp.endswith('blend')]
print("Found:", list_of_solved_puzzles)

for solved_puzzle in list_of_solved_puzzles:

    gt_dict = {}
    bpy.ops.wm.open_mainfile(filepath=os.path.join(solved_puzzles, solved_puzzle))
    bpy.data.use_autopack = True
    bpy.context.scene.unit_settings.scale_length = 0.001
    bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

    for obj in bpy.data.objects:
        if obj.type == 'MESH':  
            loc = obj.location
            rot_euler = obj.rotation_euler
            rot_quat = rot_euler.to_quaternion()
            gt_piece = {
                'location': [loc.x, loc.y, loc.z],
                'rotation_euler': [rot_euler.x, rot_euler.y, rot_euler.z],
                'rotation_quaternion': [rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z]
            }
            gt_dict[obj.name] = gt_piece

    target_gt_path = os.path.join(solved_puzzles_gt, f"{os.path.splitext(solved_puzzle)[0]}.json")
    with open(target_gt_path, 'w') as jtp:
        json.dump(gt_dict, jtp, indent=3)

    print(f"Saved JSON: {target_gt_path}")
