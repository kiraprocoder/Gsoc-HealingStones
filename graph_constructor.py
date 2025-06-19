import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data, Batch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import networkx as nx

class GraphConstructor:
    def __init__(self, features_dir='features', graphs_dir='graphs', device='cpu'):
        self.features_dir = features_dir
        self.graphs_dir = graphs_dir
        self.device = device
        self.fragment_cache = {}  # Cache loaded fragments
        os.makedirs(self.graphs_dir, exist_ok=True)
    
    def load_fragment_features(self, fragment_id):
        """Load fragment features from JSON file"""
        if fragment_id in self.fragment_cache:
            return self.fragment_cache[fragment_id]
        
        filepath = os.path.join(self.features_dir, f"{fragment_id}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fragment features not found: {filepath}")
        
        with open(filepath, 'r') as f:
            fragment_data = json.load(f)
        
        # Convert lists back to numpy arrays
        fragment_data['points'] = np.array(fragment_data['points'])
        fragment_data['geometric_features'] = np.array(fragment_data['geometric_features'])
        if fragment_data['normals'] is not None:
            fragment_data['normals'] = np.array(fragment_data['normals'])
        
        # Cache the loaded data
        self.fragment_cache[fragment_id] = fragment_data
        return fragment_data
    
    def list_available_fragments(self):
        """List all available fragment IDs in the features directory"""
        if not os.path.exists(self.features_dir):
            return []
        
        fragment_ids = []
        for filename in os.listdir(self.features_dir):
            if filename.endswith('.json'):
                fragment_ids.append(filename[:-5])  # Remove .json extension
        
        return sorted(fragment_ids)
    
    def create_knn_graph(self, points, features, k=8):
        """Create KNN graph for a single fragment"""
        # Build KNN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Create edge list (exclude self-connections)
        edge_list = []
        edge_weights = []
        
        for i in range(len(points)):
            for j in range(1, k+1):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                edge_list.append([i, neighbor_idx])
                edge_weights.append(1.0 / (1.0 + distances[i][j]))  # Distance-based weight
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_attr
    
    def create_fully_connected_graph(self, features):
        """Create fully connected graph (for small fragments)"""
        n_nodes = features.shape[0]
        
        # Create all possible edges
        edge_list = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Compute edge weights based on feature similarity
        edge_weights = []
        for edge in edge_list:
            i, j = edge
            # Cosine similarity between node features
            similarity = np.dot(features[i], features[j]) / (
                np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-8
            )
            edge_weights.append(similarity)
        
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_attr
    
    def create_inter_fragment_edges(self, points1, points2, features1, features2, 
                                   max_distance=None, top_k=5):
        """Create edges between two fragments based on proximity and feature similarity"""
        # Compute pairwise distances between fragments
        distances = cdist(points1, points2)
        
        # Find potential matching points
        inter_edges = []
        inter_weights = []
        
        n1, n2 = len(points1), len(points2)
        
        for i in range(n1):
            # Find closest points in fragment 2
            closest_indices = np.argsort(distances[i])[:top_k]
            
            for j in closest_indices:
                dist = distances[i][j]
                
                # Skip if too far (if max_distance is set)
                if max_distance and dist > max_distance:
                    continue
                
                # Compute feature similarity
                feat_sim = np.dot(features1[i], features2[j]) / (
                    np.linalg.norm(features1[i]) * np.linalg.norm(features2[j]) + 1e-8
                )
                
                # Combine distance and feature similarity
                weight = feat_sim * np.exp(-dist)  # Gaussian kernel
                
                if weight > 0.1:  # Threshold for meaningful connections
                    inter_edges.append([i, n1 + j])  # Offset indices for fragment 2
                    inter_weights.append(weight)
        
        return inter_edges, inter_weights
    
    def create_fragment_graph(self, fragment_id, graph_type='knn', k=8):
        """Create graph for a single fragment by loading from features folder"""
        fragment_data = self.load_fragment_features(fragment_id)
        
        points = fragment_data['points']
        features = fragment_data['geometric_features']
        
        # Normalize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        # Create graph based on type
        if graph_type == 'knn':
            edge_index, edge_attr = self.create_knn_graph(points, features, k)
        elif graph_type == 'fully_connected':
            edge_index, edge_attr = self.create_fully_connected_graph(features)
        else:
            raise ValueError("graph_type must be 'knn' or 'fully_connected'")
        
        # Convert to tensors
        x = torch.tensor(features, dtype=torch.float)
        pos = torch.tensor(points, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            fragment_id=fragment_data['fragment_id']
        )
        
        return data
    
    def create_dual_fragment_graph(self, fragment_id1, fragment_id2, include_inter_edges=True):
        """Create combined graph for two fragments by loading from features folder"""
        # Load fragment data
        fragment1_data = self.load_fragment_features(fragment_id1)
        fragment2_data = self.load_fragment_features(fragment_id2)
        
        # Create individual graphs
        graph1 = self.create_fragment_graph(fragment_id1, 'knn')
        graph2 = self.create_fragment_graph(fragment_id2, 'knn')
        
        # Combine node features
        combined_x = torch.cat([graph1.x, graph2.x], dim=0)
        combined_pos = torch.cat([graph1.pos, graph2.pos], dim=0)
        
        # Adjust edge indices for fragment 2
        n1 = graph1.x.shape[0]
        graph2_edges = graph2.edge_index + n1
        
        # Combine edges
        combined_edge_index = torch.cat([graph1.edge_index, graph2_edges], dim=1)
        combined_edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=0)
        
        # Add inter-fragment edges if requested
        if include_inter_edges:
            inter_edges, inter_weights = self.create_inter_fragment_edges(
                fragment1_data['points'], 
                fragment2_data['points'],
                fragment1_data['geometric_features'],
                fragment2_data['geometric_features']
            )
            
            if inter_edges:
                inter_edge_index = torch.tensor(inter_edges, dtype=torch.long).t().contiguous()
                inter_edge_attr = torch.tensor(inter_weights, dtype=torch.float)
                
                combined_edge_index = torch.cat([combined_edge_index, inter_edge_index], dim=1)
                combined_edge_attr = torch.cat([combined_edge_attr, inter_edge_attr], dim=0)
        
        # Create fragment labels (0 for fragment1, 1 for fragment2)
        fragment_labels = torch.cat([
            torch.zeros(n1, dtype=torch.long),
            torch.ones(graph2.x.shape[0], dtype=torch.long)
        ])
        
        # Create combined graph
        combined_data = Data(
            x=combined_x,
            edge_index=combined_edge_index,
            edge_attr=combined_edge_attr,
            pos=combined_pos,
            fragment_labels=fragment_labels,
            fragment_id1=fragment_id1,
            fragment_id2=fragment_id2,
            n_fragment1=n1
        )
        
        return combined_data
    
    def create_dataset_from_pairs(self, fragment_pairs, ground_truth_labels):
        """Create dataset from list of fragment ID pairs with labels
        
        Args:
            fragment_pairs: List of tuples like [('frag1', 'frag2'), ('frag3', 'frag4'), ...]
            ground_truth_labels: List of labels [1.0, 0.0, ...] where 1.0 = match, 0.0 = no match
        """
        dataset = []
        
        for i, (frag1_id, frag2_id) in enumerate(fragment_pairs):
            print(f"Creating graph for pair {i+1}/{len(fragment_pairs)}: {frag1_id} vs {frag2_id}")
            
            # Create combined graph
            graph_data = self.create_dual_fragment_graph(frag1_id, frag2_id)
            
            # Add ground truth label
            graph_data.y = torch.tensor([ground_truth_labels[i]], dtype=torch.float)
            
            # Add pair index
            graph_data.pair_idx = i
            
            dataset.append(graph_data)
        
        return dataset
    
    def create_all_pairs_dataset(self, fragment_list=None, max_pairs=None):
        """Create dataset with all possible fragment pairs from features folder
        
        Args:
            fragment_list: List of fragment IDs to use. If None, uses all available fragments
            max_pairs: Maximum number of pairs to create (for testing)
        """
        if fragment_list is None:
            fragment_list = self.list_available_fragments()
        
        if len(fragment_list) < 2:
            raise ValueError("Need at least 2 fragments to create pairs")
        
        # Create all possible pairs
        fragment_pairs = []
        for i in range(len(fragment_list)):
            for j in range(i + 1, len(fragment_list)):
                fragment_pairs.append((fragment_list[i], fragment_list[j]))
                
                if max_pairs and len(fragment_pairs) >= max_pairs:
                    break
            if max_pairs and len(fragment_pairs) >= max_pairs:
                break
        
        print(f"Creating {len(fragment_pairs)} fragment pairs...")
        
        # For this example, we'll create dummy labels (you'd replace this with actual ground truth)
        # Labels would come from your ground truth JSON file
        dummy_labels = [0.0] * len(fragment_pairs)  # Assume no matches for now
        
        return self.create_dataset_from_pairs(fragment_pairs, dummy_labels)
    
    def load_ground_truth_labels(self, ground_truth_file, fragment_pairs):
        """Load ground truth labels for fragment pairs from JSON file"""
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        labels = []
        for frag1_id, frag2_id in fragment_pairs:
            # This is a placeholder - you'd need to adapt this based on your ground truth format
            # For example, if ground truth has fragment relationships:
            label = 0.0  # Default to no match
            
            # Check if fragments are related in ground truth
            # This depends on your ground truth JSON structure
            # Example logic:
            # if frag1_id in ground_truth and frag2_id in ground_truth[frag1_id]['matches']:
            #     label = 1.0
            
            labels.append(label)
        
        return labels
    
    # IMPROVED SERIALIZATION METHODS
    def graph_to_dict(self, graph_data):
        """Convert PyTorch Geometric Data object to serializable dictionary"""
        data_dict = {
            'x': graph_data.x.numpy().tolist(),
            'edge_index': graph_data.edge_index.numpy().tolist(),
            'pos': graph_data.pos.numpy().tolist(),
        }
        
        # Optional attributes
        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            data_dict['edge_attr'] = graph_data.edge_attr.numpy().tolist()
        
        if hasattr(graph_data, 'y') and graph_data.y is not None:
            data_dict['y'] = graph_data.y.numpy().tolist()
        
        if hasattr(graph_data, 'fragment_labels') and graph_data.fragment_labels is not None:
            data_dict['fragment_labels'] = graph_data.fragment_labels.numpy().tolist()
        
        # String attributes
        string_attrs = ['fragment_id', 'fragment_id1', 'fragment_id2']
        for attr in string_attrs:
            if hasattr(graph_data, attr):
                data_dict[attr] = getattr(graph_data, attr)
        
        # Integer attributes
        int_attrs = ['n_fragment1', 'pair_idx']
        for attr in int_attrs:
            if hasattr(graph_data, attr):
                data_dict[attr] = int(getattr(graph_data, attr))
        
        return data_dict
    
    def dict_to_graph(self, data_dict):
        """Convert dictionary back to PyTorch Geometric Data object"""
        # Required tensors
        x = torch.tensor(data_dict['x'], dtype=torch.float)
        edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
        pos = torch.tensor(data_dict['pos'], dtype=torch.float)
        
        # Create Data object
        graph_data = Data(x=x, edge_index=edge_index, pos=pos)
        
        # Optional tensors
        if 'edge_attr' in data_dict:
            graph_data.edge_attr = torch.tensor(data_dict['edge_attr'], dtype=torch.float)
        
        if 'y' in data_dict:
            graph_data.y = torch.tensor(data_dict['y'], dtype=torch.float)
        
        if 'fragment_labels' in data_dict:
            graph_data.fragment_labels = torch.tensor(data_dict['fragment_labels'], dtype=torch.long)
        
        # String and integer attributes
        other_attrs = ['fragment_id', 'fragment_id1', 'fragment_id2', 'n_fragment1', 'pair_idx']
        for attr in other_attrs:
            if attr in data_dict:
                setattr(graph_data, attr, data_dict[attr])
        
        return graph_data
    
    def save_graph(self, graph_data, filename):
        """Save graph data to JSON file (more portable than pickle)"""
        filepath = os.path.join(self.graphs_dir, f"{filename}.json")
        
        # Convert to dictionary
        data_dict = self.graph_to_dict(graph_data)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        print(f"Graph saved to {filepath}")
    
    def load_graph(self, filename):
        """Load graph data from JSON file"""
        filepath = os.path.join(self.graphs_dir, f"{filename}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        # Load JSON
        with open(filepath, 'r') as f:
            data_dict = json.load(f)
        
        # Convert back to PyTorch Geometric Data
        return self.dict_to_graph(data_dict)
    
    def save_dataset(self, dataset, filename):
        """Save entire dataset to JSON file"""
        filepath = os.path.join(self.graphs_dir, f"{filename}_dataset.json")
        
        # Convert all graphs to dictionaries
        dataset_dicts = []
        for graph_data in dataset:
            dataset_dicts.append(self.graph_to_dict(graph_data))
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(dataset_dicts, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filename):
        """Load dataset from JSON file"""
        filepath = os.path.join(self.graphs_dir, f"{filename}_dataset.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        # Load JSON
        with open(filepath, 'r') as f:
            dataset_dicts = json.load(f)
        
        # Convert back to PyTorch Geometric Data objects
        dataset = []
        for data_dict in dataset_dicts:
            dataset.append(self.dict_to_graph(data_dict))
        
        return dataset
    
    # PYTORCH TENSOR SERIALIZATION (Alternative method)
    def save_graph_tensors(self, graph_data, filename):
        """Save graph using PyTorch's tensor serialization (more efficient for large graphs)"""
        filepath = os.path.join(self.graphs_dir, f"{filename}.pt")
        
        # Create a dictionary with all tensors and metadata
        save_dict = {
            'x': graph_data.x,
            'edge_index': graph_data.edge_index,
            'pos': graph_data.pos,
        }
        
        # Optional tensors
        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            save_dict['edge_attr'] = graph_data.edge_attr
        
        if hasattr(graph_data, 'y') and graph_data.y is not None:
            save_dict['y'] = graph_data.y
        
        if hasattr(graph_data, 'fragment_labels') and graph_data.fragment_labels is not None:
            save_dict['fragment_labels'] = graph_data.fragment_labels
        
        # Metadata (non-tensor attributes)
        metadata = {}
        string_attrs = ['fragment_id', 'fragment_id1', 'fragment_id2']
        for attr in string_attrs:
            if hasattr(graph_data, attr):
                metadata[attr] = getattr(graph_data, attr)
        
        int_attrs = ['n_fragment1', 'pair_idx']
        for attr in int_attrs:
            if hasattr(graph_data, attr):
                metadata[attr] = getattr(graph_data, attr)
        
        save_dict['metadata'] = metadata
        
        # Save using PyTorch
        torch.save(save_dict, filepath)
        print(f"Graph saved to {filepath}")
    
    def load_graph_tensors(self, filename):
        """Load graph from PyTorch tensor file"""
        filepath = os.path.join(self.graphs_dir, f"{filename}.pt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        # Load using PyTorch
        save_dict = torch.load(filepath, map_location=self.device)
        
        # Create Data object
        graph_data = Data(
            x=save_dict['x'],
            edge_index=save_dict['edge_index'],
            pos=save_dict['pos']
        )
        
        # Optional tensors
        if 'edge_attr' in save_dict:
            graph_data.edge_attr = save_dict['edge_attr']
        
        if 'y' in save_dict:
            graph_data.y = save_dict['y']
        
        if 'fragment_labels' in save_dict:
            graph_data.fragment_labels = save_dict['fragment_labels']
        
        # Metadata
        if 'metadata' in save_dict:
            for key, value in save_dict['metadata'].items():
                setattr(graph_data, key, value)
        
        return graph_data
    
    def visualize_fragment_graph(self, graph_data, title="Fragment Graph", 
                                save_path=None, show_edges=True, max_edges=1000):
        """Visualize a single fragment graph in 3D"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get positions and convert to numpy if needed
        if isinstance(graph_data.pos, torch.Tensor):
            positions = graph_data.pos.numpy()
        else:
            positions = graph_data.pos
        
        # Plot nodes
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='blue', s=20, alpha=0.6, label='Nodes')
        
        # Plot edges (sample for performance)
        if show_edges and graph_data.edge_index.shape[1] > 0:
            edge_index = graph_data.edge_index.numpy() if isinstance(graph_data.edge_index, torch.Tensor) else graph_data.edge_index
            
            # Sample edges if too many
            n_edges = edge_index.shape[1]
            if n_edges > max_edges:
                edge_indices = np.random.choice(n_edges, max_edges, replace=False)
                edge_index = edge_index[:, edge_indices]
            
            for i in range(edge_index.shape[1]):
                start_idx, end_idx = edge_index[0, i], edge_index[1, i]
                start_pos = positions[start_idx]
                end_pos = positions[end_idx]
                
                ax.plot3D([start_pos[0], end_pos[0]], 
                         [start_pos[1], end_pos[1]], 
                         [start_pos[2], end_pos[2]], 
                         'gray', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title}\nNodes: {positions.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_dual_fragment_graph(self, graph_data, title="Dual Fragment Graph", 
                                     save_path=None, show_inter_edges=True, max_edges=2000):
        """Visualize combined graph with two fragments in different colors"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get positions and fragment labels
        if isinstance(graph_data.pos, torch.Tensor):
            positions = graph_data.pos.numpy()
        else:
            positions = graph_data.pos
            
        if isinstance(graph_data.fragment_labels, torch.Tensor):
            fragment_labels = graph_data.fragment_labels.numpy()
        else:
            fragment_labels = graph_data.fragment_labels
        
        # Plot nodes for each fragment in different colors
        frag1_mask = fragment_labels == 0
        frag2_mask = fragment_labels == 1
        
        ax.scatter(positions[frag1_mask, 0], positions[frag1_mask, 1], positions[frag1_mask, 2], 
                  c='blue', s=20, alpha=0.6, label=f'Fragment 1 ({np.sum(frag1_mask)} nodes)')
        ax.scatter(positions[frag2_mask, 0], positions[frag2_mask, 1], positions[frag2_mask, 2], 
                  c='red', s=20, alpha=0.6, label=f'Fragment 2 ({np.sum(frag2_mask)} nodes)')
        
        # Plot edges
        if graph_data.edge_index.shape[1] > 0:
            edge_index = graph_data.edge_index.numpy() if isinstance(graph_data.edge_index, torch.Tensor) else graph_data.edge_index
            
            # Identify inter-fragment edges
            n1 = graph_data.n_fragment1
            inter_edges = []
            intra_edges = []
            
            for i in range(edge_index.shape[1]):
                start_idx, end_idx = edge_index[0, i], edge_index[1, i]
                if (start_idx < n1 and end_idx >= n1) or (start_idx >= n1 and end_idx < n1):
                    inter_edges.append(i)
                else:
                    intra_edges.append(i)
            
            # Sample edges for performance
            if len(intra_edges) > max_edges:
                intra_edges = np.random.choice(intra_edges, max_edges, replace=False)
            
            # Plot intra-fragment edges
            for i in intra_edges:
                start_idx, end_idx = edge_index[0, i], edge_index[1, i]
                start_pos = positions[start_idx]
                end_pos = positions[end_idx]
                
                ax.plot3D([start_pos[0], end_pos[0]], 
                         [start_pos[1], end_pos[1]], 
                         [start_pos[2], end_pos[2]], 
                         'gray', alpha=0.2, linewidth=0.3)
            
            # Plot inter-fragment edges in different color
            if show_inter_edges and inter_edges:
                print(f"Found {len(inter_edges)} inter-fragment edges")
                for i in inter_edges:
                    start_idx, end_idx = edge_index[0, i], edge_index[1, i]
                    start_pos = positions[start_idx]
                    end_pos = positions[end_idx]
                    
                    ax.plot3D([start_pos[0], end_pos[0]], 
                             [start_pos[1], end_pos[1]], 
                             [start_pos[2], end_pos[2]], 
                             'green', alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title}\nTotal Nodes: {positions.shape[0]}, Total Edges: {graph_data.edge_index.shape[1]}")
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def create_graph_summary(self, graph_data):
        """Create a summary of graph statistics"""
        summary = {
            'num_nodes': graph_data.x.shape[0],
            'num_edges': graph_data.edge_index.shape[1],
            'num_features': graph_data.x.shape[1],
            'has_edge_attr': hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None,
            'has_pos': hasattr(graph_data, 'pos') and graph_data.pos is not None,
        }
        
        if hasattr(graph_data, 'fragment_labels'):
            summary['is_dual_fragment'] = True
            summary['fragment1_nodes'] = int(torch.sum(graph_data.fragment_labels == 0))
            summary['fragment2_nodes'] = int(torch.sum(graph_data.fragment_labels == 1))
        else:
            summary['is_dual_fragment'] = False
        
        # Fixed: Check if graph_data.y exists and is not None before accessing
        if hasattr(graph_data, 'y') and graph_data.y is not None:
            summary['has_label'] = True
            summary['label'] = float(graph_data.y.item())
        else:
            summary['has_label'] = False
        
        return summary

# Example usage with improved serialization
if __name__ == "__main__":
    # Create graph constructor that reads from features/ folder
    graph_constructor = GraphConstructor(features_dir="features", graphs_dir="graphs")
    
    # List available fragments
    available_fragments = graph_constructor.list_available_fragments()
    print(f"Available fragments: {available_fragments}")
    
    if len(available_fragments) >= 2:
        # Create individual graphs from saved features
        frag1_id = available_fragments[0]
        frag2_id = available_fragments[1]
        
        print("\n=== Creating Individual Graphs ===")
        graph1 = graph_constructor.create_fragment_graph(frag1_id)
        graph2 = graph_constructor.create_fragment_graph(frag2_id)
        
        print(f"Graph 1 ({frag1_id}) - Nodes: {graph1.x.shape[0]}, Edges: {graph1.edge_index.shape[1]}")
        print(f"Graph 2 ({frag2_id}) - Nodes: {graph2.x.shape[0]}, Edges: {graph2.edge_index.shape[1]}")
        
        # Save individual graphs using JSON (more portable)
        print("\n=== Saving Graphs (JSON format) ===")
        graph_constructor.save_graph(graph1, f"single_{frag1_id}")
        graph_constructor.save_graph(graph2, f"single_{frag2_id}")
        
        # Alternative: Save using PyTorch tensors (more efficient for large graphs)
        print("\n=== Saving Graphs (PyTorch format) ===")
        graph_constructor.save_graph_tensors(graph1, f"single_{frag1_id}_pt")
        graph_constructor.save_graph_tensors(graph2, f"single_{frag2_id}_pt")
        
        print("\n=== Creating Combined Graph ===")
        # Create combined graph
        combined_graph = graph_constructor.create_dual_fragment_graph(frag1_id, frag2_id)
        print(f"Combined graph - Nodes: {combined_graph.x.shape[0]}, Edges: {combined_graph.edge_index.shape[1]}")
        
        # Save combined graph (both formats)
        graph_constructor.save_graph(combined_graph, f"dual_{frag1_id}_{frag2_id}")
        graph_constructor.save_graph_tensors(combined_graph, f"dual_{frag1_id}_{frag2_id}_pt")
        
        # Print graph summary
        print("\n=== Graph Summary ===")
        summary = graph_constructor.create_graph_summary(combined_graph)
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("\n=== Testing Graph Loading ===")
        # Test loading graphs
        loaded_graph_json = graph_constructor.load_graph(f"single_{frag1_id}")
        loaded_graph_pt = graph_constructor.load_graph_tensors(f"single_{frag1_id}_pt")
        
        print(f"Loaded from JSON - Nodes: {loaded_graph_json.x.shape[0]}, Edges: {loaded_graph_json.edge_index.shape[1]}")
        print(f"Loaded from PyTorch - Nodes: {loaded_graph_pt.x.shape[0]}, Edges: {loaded_graph_pt.edge_index.shape[1]}")
        
        # Verify they're the same
        json_equal = torch.allclose(loaded_graph_json.x, graph1.x, atol=1e-6)
        pt_equal = torch.allclose(loaded_graph_pt.x, graph1.x, atol=1e-6)
        print(f"JSON loading preserves data: {json_equal}")
        print(f"PyTorch loading preserves data: {pt_equal}")
        
        print("\n=== Creating Dataset ===")
        # Create dataset with specific pairs
        fragment_pairs = [(frag1_id, frag2_id)]
        labels = [1.0]  # 1.0 if they match, 0.0 if they don't
        
        dataset = graph_constructor.create_dataset_from_pairs(fragment_pairs, labels)
        print(f"Dataset size: {len(dataset)}")
        print(f"First sample label: {dataset[0].y}")
        
        # Save dataset (JSON format)
        graph_constructor.save_dataset(dataset, "fragment_pairs")
        
        # Test loading dataset
        loaded_dataset = graph_constructor.load_dataset("fragment_pairs")
        print(f"Loaded dataset size: {len(loaded_dataset)}")
        print(f"Dataset loading preserves data: {torch.allclose(loaded_dataset[0].x, dataset[0].x, atol=1e-6)}")
        
        print("\n=== Visualizing Graphs ===")
        # Visualize individual fragment (sample points for better performance)
        print("Visualizing individual fragment...")
        graph_constructor.visualize_fragment_graph(
            graph1, 
            title=f"Fragment {frag1_id}",
            save_path=f"graphs/fragment_{frag1_id}_vis.png"
        )
        
        # Visualize combined graph
        print("Visualizing combined fragments...")
        graph_constructor.visualize_dual_fragment_graph(
            combined_graph,
            title=f"Combined: {frag1_id} + {frag2_id}",
            save_path=f"graphs/combined_{frag1_id}_{frag2_id}_vis.png"
        )
        
        print("\n=== File Summary ===")
        print("Generated files:")
        print(f"- graphs/single_{frag1_id}.json (JSON format)")
        print(f"- graphs/single_{frag1_id}_pt.pt (PyTorch format)")
        print(f"- graphs/single_{frag2_id}.json (JSON format)")
        print(f"- graphs/single_{frag2_id}_pt.pt (PyTorch format)")
        print(f"- graphs/dual_{frag1_id}_{frag2_id}.json (JSON format)")
        print(f"- graphs/dual_{frag1_id}_{frag2_id}_pt.pt (PyTorch format)")
        print(f"- graphs/fragment_pairs_dataset.json (Dataset in JSON format)")
        print(f"- graphs/fragment_{frag1_id}_vis.png (Visualization)")
        print(f"- graphs/combined_{frag1_id}_{frag2_id}_vis.png (Visualization)")
        
        print("\n=== Serialization Benefits ===")
        print("JSON format:")
        print("  ✓ Human readable")
        print("  ✓ Cross-platform compatible")
        print("  ✓ Version independent")
        print("  ✓ Can be opened in any text editor")
        print("  - Larger file size")
        print("  - Slower for very large graphs")
        
        print("\nPyTorch format:")
        print("  ✓ Smaller file size")
        print("  ✓ Faster loading for large graphs")
        print("  ✓ Preserves tensor types exactly")
        print("  - Requires PyTorch to load")
        print("  - May have version compatibility issues")
        
        # Create dataset with all possible pairs (limited for testing)
        # all_pairs_dataset = graph_constructor.create_all_pairs_dataset(max_pairs=5)
        # print(f"All pairs dataset size: {len(all_pairs_dataset)}")
    else:
        print("Need at least 2 fragments in the features/ folder to create graphs")
        print("Run the FragmentFeatureExtractor first to generate feature files")