import os
import shutil
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Dict
import time

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available. Hard negative mining will use geometric heuristics instead of feature-based similarity.")

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def get_fragments(folder):
    return [f for f in os.listdir(folder) if f.endswith('.ply')]

def load_point_cloud(file_path):
    """Load point cloud and extract basic geometric features."""
    if OPEN3D_AVAILABLE:
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                return None
            
            # Extract geometric features
            points = np.asarray(pcd.points)
            centroid = np.mean(points, axis=0)
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            
            # Compute FPFH features for similarity comparison
            pcd.estimate_normals()
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
            
            return {
                'points': points,
                'centroid': centroid,
                'bbox_size': bbox_size,
                'fpfh': np.asarray(fpfh.data),
                'num_points': len(points)
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        # Fallback: basic geometric analysis without open3d
        try:
            # Simple PLY parser for coordinates
            points = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                vertex_count = 0
                reading_vertices = False
                
                for line in lines:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith('end_header'):
                        reading_vertices = True
                        continue
                    elif reading_vertices and vertex_count > 0:
                        coords = line.strip().split()[:3]
                        if len(coords) == 3:
                            try:
                                points.append([float(x) for x in coords])
                                vertex_count -= 1
                                if vertex_count == 0:
                                    break
                            except ValueError:
                                continue
            
            if not points:
                return None
                
            points = np.array(points)
            centroid = np.mean(points, axis=0)
            bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
            
            return {
                'points': points,
                'centroid': centroid,
                'bbox_size': bbox_size,
                'fpfh': None,
                'num_points': len(points)
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

def compute_similarity_score(features1, features2):
    """Compute similarity score between two fragments."""
    if features1 is None or features2 is None:
        return 0.0
    
    # Geometric similarity
    centroid_dist = np.linalg.norm(features1['centroid'] - features2['centroid'])
    bbox_similarity = 1.0 / (1.0 + np.linalg.norm(features1['bbox_size'] - features2['bbox_size']))
    size_similarity = min(features1['num_points'], features2['num_points']) / max(features1['num_points'], features2['num_points'])
    
    # Feature similarity (if available)
    feature_similarity = 0.5
    if OPEN3D_AVAILABLE and features1['fpfh'] is not None and features2['fpfh'] is not None:
        try:
            # Compute average feature similarity
            fpfh1_mean = np.mean(features1['fpfh'], axis=1)
            fpfh2_mean = np.mean(features2['fpfh'], axis=1)
            feature_similarity = np.dot(fpfh1_mean, fpfh2_mean) / (np.linalg.norm(fpfh1_mean) * np.linalg.norm(fpfh2_mean))
            feature_similarity = max(0, feature_similarity)  # Ensure non-negative
        except:
            feature_similarity = 0.5
    
    # Combine similarities
    total_similarity = (bbox_similarity * 0.3 + size_similarity * 0.3 + feature_similarity * 0.4) / (1.0 + centroid_dist * 0.1)
    return total_similarity

def load_fragment_features(data_path: str, fragments: List[str]) -> Dict[str, dict]:
    """Load features for all fragments with progress tracking."""
    print(f"Loading features for {len(fragments)} fragments...")
    
    def load_single_feature(frag):
        return frag, load_point_cloud(os.path.join(data_path, frag))
    
    features = {}
    with ThreadPoolExecutor(max_workers=min(8, len(fragments))) as executor:
        results = list(executor.map(load_single_feature, fragments))
    
    for frag, feat in results:
        features[frag] = feat
    
    valid_features = sum(1 for f in features.values() if f is not None)
    print(f"Successfully loaded features for {valid_features}/{len(fragments)} fragments")
    return features

def generate_negative_pair(all_frags: List[str], positive_pair: List[str], 
                          fragment_features: Dict[str, dict], hard_negative_ratio: float = 0.5):
    """Generate negative pair with option for hard negatives."""
    
    # Decide if this should be a hard negative
    use_hard_negative = random.random() < hard_negative_ratio
    
    if use_hard_negative and fragment_features:
        # Generate hard negative: find fragments similar to positive pair
        pos_features = [fragment_features.get(frag) for frag in positive_pair if fragment_features.get(frag) is not None]
        
        if len(pos_features) >= 1:
            # Find fragments most similar to the positive fragments
            similarities = []
            for frag in all_frags:
                if frag in positive_pair:
                    continue
                    
                frag_features = fragment_features.get(frag)
                if frag_features is None:
                    continue
                
                # Compute max similarity to any positive fragment
                max_sim = max(compute_similarity_score(pos_feat, frag_features) for pos_feat in pos_features)
                similarities.append((frag, max_sim))
            
            if similarities:
                # Sort by similarity and pick from top candidates
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_candidates = similarities[:max(1, len(similarities) // 4)]  # Top 25%
                
                # Select two different fragments from top candidates
                frag_a = random.choice(top_candidates)[0]
                remaining_candidates = [f for f, _ in top_candidates if f != frag_a]
                
                if remaining_candidates:
                    frag_b = random.choice(remaining_candidates)
                else:
                    # Fallback to any different fragment
                    frag_b = random.choice([f for f in all_frags if f != frag_a and f not in positive_pair])
                
                return frag_a, frag_b, True  # True indicates hard negative
    
    # Generate easy negative (random selection)
    frag_a = random.choice(all_frags)
    frag_b = random.choice(all_frags)
    while (frag_a == frag_b) or (set([frag_a, frag_b]) == set(positive_pair)):
        frag_a = random.choice(all_frags)
        frag_b = random.choice(all_frags)
    
    return frag_a, frag_b, False  # False indicates easy negative

def process_single_pair(args):
    """Process a single positive pair and generate its negative pairs."""
    (pair_folder, testdata_path, data_path, pairs_output, 
     num_neg_per_pos, all_fragments, fragment_features, pair_idx) = args
    
    pair_folder_path = os.path.join(testdata_path, pair_folder)
    pos_frags = get_fragments(pair_folder_path)
    
    if len(pos_frags) != 2:
        return f"Skipped {pair_folder}: does not contain exactly 2 .ply fragments.", 0, 0
    
    # Save positive pair
    pos_pair_folder = os.path.join(pairs_output, f'pos_pair_{pair_idx}')
    create_dir(pos_pair_folder)
    
    for frag in pos_frags:
        shutil.copy(os.path.join(pair_folder_path, frag), os.path.join(pos_pair_folder, frag))
    
    # Generate negative pairs
    neg_pairs_created = 0
    hard_neg_count = 0
    base_neg_counter = pair_idx * num_neg_per_pos
    
    for i in range(num_neg_per_pos):
        frag_a, frag_b, is_hard = generate_negative_pair(all_fragments, pos_frags, fragment_features)
        neg_counter = base_neg_counter + i + 1
        
        neg_pair_folder = os.path.join(pairs_output, f'neg_pair_{neg_counter}')
        create_dir(neg_pair_folder)
        
        shutil.copy(os.path.join(data_path, frag_a), os.path.join(neg_pair_folder, frag_a))
        shutil.copy(os.path.join(data_path, frag_b), os.path.join(neg_pair_folder, frag_b))
        
        neg_pairs_created += 1
        if is_hard:
            hard_neg_count += 1
    
    return f"Processed {pair_folder}: 1 pos, {neg_pairs_created} neg ({hard_neg_count} hard)", 1, neg_pairs_created

def main(dataset_root, num_neg_per_pos=3, hard_negative_ratio=0.5, max_workers=None, clear_output=True):
    """
    Main function with enhanced negative pair generation and parallelization.
    
    Args:
        dataset_root: Root directory of the dataset
        num_neg_per_pos: Number of negative pairs per positive pair (default: 3)
        hard_negative_ratio: Ratio of hard negatives (0.0 to 1.0, default: 0.5)
        max_workers: Maximum number of parallel workers (default: CPU count)
        clear_output: Whether to clear the output directory before generation (default: True)
    """
    start_time = time.time()
    
    testdata_path = os.path.join(dataset_root, 'testdata')
    data_path = os.path.join(dataset_root, 'data')
    pairs_output = os.path.join(dataset_root, 'pairs')
    
    # Handle output directory
    if clear_output and os.path.exists(pairs_output):
        print(f"Clearing existing output directory: {pairs_output}")
        shutil.rmtree(pairs_output)
        print("✓ Output directory cleared")
    
    create_dir(pairs_output)
    
    # Collect all .ply fragments from data/
    print("Collecting fragments...")
    all_fragments = get_fragments(data_path)
    print(f"Found {len(all_fragments)} fragments in data directory")
    
    # Load fragment features for hard negative mining
    fragment_features = {}
    if hard_negative_ratio > 0:
        fragment_features = load_fragment_features(data_path, all_fragments)
    
    # Get all pair folders
    pair_folders = [f for f in os.listdir(testdata_path) 
                   if f.startswith('pair_') and os.path.isdir(os.path.join(testdata_path, f))]
    
    print(f"Found {len(pair_folders)} pair folders to process")
    
    # Prepare arguments for parallel processing
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(pair_folders))
    
    args_list = []
    for i, pair_folder in enumerate(pair_folders, 1):
        args_list.append((
            pair_folder, testdata_path, data_path, pairs_output,
            num_neg_per_pos, all_fragments, fragment_features, i
        ))
    
    # Process pairs in parallel
    print(f"Processing pairs with {max_workers} workers...")
    total_pos_pairs = 0
    total_neg_pairs = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_pair, args_list))
    
    # Collect results
    for result_msg, pos_count, neg_count in results:
        print(result_msg)
        total_pos_pairs += pos_count
        total_neg_pairs += neg_count
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total positive pairs: {total_pos_pairs}")
    print(f"Total negative pairs: {total_neg_pairs}")
    print(f"Ratio (neg:pos): {total_neg_pairs/max(1, total_pos_pairs):.1f}:1")
    print(f"Hard negative ratio: {hard_negative_ratio:.1%}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Output directory: {pairs_output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    dataset_root = "/path/to/your/dataset_root"  # Change this to your dataset path
    
    # Configuration
    NUM_NEG_PER_POS = 3          # Keep 3:1 ratio as requested
    HARD_NEGATIVE_RATIO = 0.5    # 50% of negatives will be hard negatives
    MAX_WORKERS = None           # Use all available CPU cores
    CLEAR_OUTPUT = True          # Clear output directory before generation
    
    main(dataset_root, 
         num_neg_per_pos=NUM_NEG_PER_POS,
         hard_negative_ratio=HARD_NEGATIVE_RATIO, 
         max_workers=MAX_WORKERS,
         clear_output=CLEAR_OUTPUT)