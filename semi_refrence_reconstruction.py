import open3d as o3d
import json, os
import numpy as np

# Add the proper paths 
ROOT_FOLDER = '/home/kira/Desktop/test_set' # root path where the fragments are saved 
JSON_PATH = '/home/kira/Desktop/test_set/ground_truth/gt_json/all_ply_files_DONE.json' # refrence json path
OUTPUT_FOLDER = '/home/kira/Desktop/test_set/ground_truth/reconstructed' 

def main():
    with open(JSON_PATH, 'r') as jp:
        gt = json.load(jp)

    meshes = []
    meshes_names = []
    all_pts = np.array([])

    for gtk in gt.keys():
        ply_path = os.path.join(ROOT_FOLDER, f"{gtk}.ply")
        meshes_names.append(f"{gtk}.ply")

        if not os.path.exists(ply_path):
            print(f"no available files: {ply_path}")
            continue

        mesh = o3d.io.read_triangle_mesh(ply_path, enable_post_processing=True)

        if len(np.asarray(mesh.vertices)) == 0:
            print(f"empty mesh {ply_path}")
            continue

        # place them in the origin
        mesh.translate(-mesh.get_center())

        # rotation in blender style
        rot_angles = gt[gtk]['rotation_euler']
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([rot_angles[0], 0, 0]), center=mesh.get_center())
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, rot_angles[1], 0]), center=mesh.get_center())
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, 0, rot_angles[2]]), center=mesh.get_center())

        # now translation
        mesh.translate(gt[gtk]['location'])
        print(f"{gtk} : {mesh.get_center()}")

        meshes.append(mesh)

        if all_pts.size == 0:
            all_pts = np.asarray(mesh.vertices)
        else:
            all_pts = np.concatenate((all_pts, np.asarray(mesh.vertices)))

    if all_pts.size == 0:
        print("empty mesh")
        return

    cframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    o3d.visualization.draw_geometries(meshes + [cframe])

    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(all_pts))
    translation = -pcd.get_center()
    print("np:", np.mean(all_pts, axis=0))
    print("o3d:", pcd.get_center())

    pcd.translate(translation)
    o3d.visualization.draw_geometries([pcd, cframe])

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for mesh, mesh_name in zip(meshes, meshes_names):
        mesh.translate(translation)
        o3d.io.write_triangle_mesh(filename=os.path.join(OUTPUT_FOLDER, mesh_name), mesh=mesh, write_ascii=True)

if __name__ == '__main__':
    main()
