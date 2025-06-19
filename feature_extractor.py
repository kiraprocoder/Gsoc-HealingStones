import numpy as np
import json
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import open3d as o3d
import os

class FragmentFeatureExtractor:
    def __init__(self, save_dir="features", target_points=30000):
        self.features = {}
        self.save_dir = save_dir
        self.target_points = target_points  # Default target, can be changed per fragment
        os.makedirs(self.save_dir, exist_ok=True)
    
    def load_ply_file(self, filepath, target_points=None):
        """Load PLY file and auto-downsample to target number of points"""
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            original_count = len(pcd.points)
            
            # Use instance target_points if not specified
            if target_points is None:
                target_points = self.target_points
            
            print(f"Original point count: {original_count}")
            
            # Only downsample if we have more points than target
            if original_count > target_points:
                # Compute bounding box diagonal for voxel size estimation
                max_bound = pcd.get_max_bound()
                min_bound = pcd.get_min_bound()
                bbox_diag = np.linalg.norm(max_bound - min_bound)
                
                if bbox_diag == 0 or np.isnan(bbox_diag):
                    raise ValueError(f"Invalid bounding box diagonal for file: {filepath}")
                
                # Iteratively find the right voxel size
                voxel_size = self._find_optimal_voxel_size(pcd, target_points, bbox_diag)
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                
                print(f"Downsampled to {len(pcd.points)} points (target: {target_points})")
            else:
                print(f"No downsampling needed - already below target of {target_points} points")

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals) if pcd.has_normals() else None

            return points, normals
            
        except Exception as e:
            print(f"Failed to load with open3d: {e}")
            raise e
    
    def _find_optimal_voxel_size(self, pcd, target_points, bbox_diag, max_iterations=10):
        """Iteratively find voxel size that gives approximately target_points"""
        # Initial estimate based on point density
        volume_estimate = bbox_diag ** 3  # Rough volume estimate
        target_density = target_points / volume_estimate
        voxel_size = (1.0 / target_density) ** (1/3)
        
        # Binary search approach
        min_voxel = bbox_diag / 1000  # Very fine
        max_voxel = bbox_diag / 10    # Very coarse
        
        for iteration in range(max_iterations):
            test_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            current_points = len(test_pcd.points)
            
            # Accept if within 10% of target
            if abs(current_points - target_points) / target_points < 0.1:
                break
                
            # Adjust voxel size based on result
            if current_points > target_points:
                # Too many points, increase voxel size
                min_voxel = voxel_size
                voxel_size = (voxel_size + max_voxel) / 2
            else:
                # Too few points, decrease voxel size
                max_voxel = voxel_size
                voxel_size = (min_voxel + voxel_size) / 2
            
            # Prevent infinite loop with very small adjustments
            if max_voxel - min_voxel < bbox_diag / 10000:
                break
        
        print(f"Found voxel size: {voxel_size:.6f} (iterations: {iteration + 1})")
        return voxel_size
    
    def compute_geometric_features(self, points, k=10):
        """Compute geometric features for each point"""
        features = []
        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(points)), algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        for i, point in enumerate(points):
            neighbor_indices = indices[i][1:]  # Exclude self
            neighbors = points[neighbor_indices]
            
            local_features = []
            local_density = np.mean(distances[i][1:])
            local_features.append(local_density)
            
            centered_neighbors = neighbors - point
            cov_matrix = np.cov(centered_neighbors.T)
            eigenvals = np.linalg.eigvalsh(cov_matrix)
            eigenvals = np.sort(eigenvals)
            
            linearity = (eigenvals[2] - eigenvals[1]) / eigenvals[2] if eigenvals[2] > 0 else 0
            planarity = (eigenvals[1] - eigenvals[0]) / eigenvals[2] if eigenvals[2] > 0 else 0
            sphericity = eigenvals[0] / eigenvals[2] if eigenvals[2] > 0 else 0
            
            height_var = np.var(neighbors[:, 2])
            centroid = np.mean(neighbors, axis=0)
            dist_to_centroid = np.linalg.norm(point - centroid)
            
            local_features.extend([linearity, planarity, sphericity, height_var, dist_to_centroid])
            features.append(local_features)
        
        return np.array(features)
    
    def compute_global_features(self, points):
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        bbox_size = max_coords - min_coords
        centroid = np.mean(points, axis=0)
        
        pca = PCA(n_components=3)
        pca.fit(points)
        principal_axes = pca.components_
        explained_variance = pca.explained_variance_ratio_
        
        global_features = {
            'bbox_size': bbox_size.tolist(),
            'centroid': centroid.tolist(),
            'principal_axes': principal_axes.tolist(),
            'explained_variance': explained_variance.tolist(),
            'num_points': int(len(points))
        }
        return global_features
    
    def extract_fragment_features(self, ply_filepath, fragment_id, target_points=None, save_to_disk=True):
        """Main function to extract all features from a fragment"""
        print(f"Processing fragment {fragment_id}...")
        
        points, normals = self.load_ply_file(ply_filepath, target_points)
        geometric_features = self.compute_geometric_features(points)
        global_features = self.compute_global_features(points)
        
        fragment_data = {
            'points': points.tolist(),  # For JSON compatibility
            'normals': normals.tolist() if normals is not None else None,
            'geometric_features': geometric_features.tolist(),
            'global_features': global_features,
            'fragment_id': fragment_id
        }
        
        self.features[fragment_id] = fragment_data
        
        if save_to_disk:
            save_path = os.path.join(self.save_dir, f"{fragment_id}.json")
            with open(save_path, 'w') as f:
                json.dump(fragment_data, f)
            print(f"Saved features to {save_path}")
        
        return fragment_data
    
    def set_target_points(self, target_points):
        """Change the default target point count"""
        self.target_points = target_points
        print(f"Target points set to: {target_points}")
    
    def load_ground_truth(self, json_filepath):
        with open(json_filepath, 'r') as f:
            return json.load(f)

# Example usage
if __name__ == "__main__":
    # Initialize with default 30k points
    extractor = FragmentFeatureExtractor(save_dir="features", target_points=30000)
    
    # Process fragments with auto-downsampling to 30k points
    fragment1_data = extractor.extract_fragment_features("data/NAR_ST_43B_FR_02_shrinkwrap_02mm.ply", "frag1")
    fragment2_data = extractor.extract_fragment_features("data/NAR_ST_43B_FR_10_shrinkwrap_02mm.ply", "frag2")
    
    # You can also override target points for specific fragments:
    # fragment3_data = extractor.extract_fragment_features("path/to/fragment3.ply", "frag3", target_points=50000)
    
    # Or change the default for all subsequent processing:
    # extractor.set_target_points(50000)
    
    ground_truth = extractor.load_ground_truth("data/ground_truth/gt_json/all_ply_files_DONE.json")
    
    print(f"Fragment 1 features shape: {np.array(fragment1_data['geometric_features']).shape}")
    print(f"Fragment 2 features shape: {np.array(fragment2_data['geometric_features']).shape}")