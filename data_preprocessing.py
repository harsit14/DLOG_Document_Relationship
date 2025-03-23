import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.utils as utils
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

class DataPreprocessor:
    def __init__(self, data_dir='./data'):
        """
        Initialize the data preprocessor.
        Args:
            data_dir (str): Directory to store the datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_cora(self):
        """
        Load the CORA dataset using PyTorch Geometric.
        Returns:
            torch_geometric.data.Data: CORA dataset
        """
        print("Loading CORA dataset...")
        dataset = Planetoid(root=self.data_dir, name='Cora', transform=NormalizeFeatures())
        data = dataset[0]
        
        print(f"CORA dataset statistics:")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Number of edges: {data.num_edges}")
        print(f"  Number of node features: {data.num_node_features}")
        print(f"  Number of classes: {dataset.num_classes}")
        print(f"  Number of training nodes: {data.train_mask.sum().item()}")
        print(f"  Number of validation nodes: {data.val_mask.sum().item()}")
        print(f"  Number of test nodes: {data.test_mask.sum().item()}")
        
        return data, dataset.num_classes
    
    def create_link_prediction_split(self, data, val_ratio=0.05, test_ratio=0.1, neg_sampling_ratio=1.0):
        """
        Create train/val/test splits for link prediction task.
        
        Args:
            data (torch_geometric.data.Data): Input data
            val_ratio (float): Ratio of edges to use for validation
            test_ratio (float): Ratio of edges to use for testing
            neg_sampling_ratio (float): Ratio of negative samples to positive samples
            
        Returns:
            tuple: (edge_index_train, edge_index_val, edge_index_test, 
                neg_edge_index_train, neg_edge_index_val, neg_edge_index_test)
        """
        # Use the new RandomLinkSplit transform from PyTorch Geometric
        from torch_geometric.transforms import RandomLinkSplit
        
        # Create a copy of the data to avoid modifying the original
        data_copy = data.clone()
        
        # Perform the link split
        transform = RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=neg_sampling_ratio
        )
        
        train_data, val_data, test_data = transform(data_copy)
        
        # Extract the edge indices
        edge_index_train = train_data.edge_index
        edge_index_val = val_data.edge_label_index[:, val_data.edge_label == 1]
        edge_index_test = test_data.edge_label_index[:, test_data.edge_label == 1]
        
        # Generate negative edges for each split
        n_train_edges = edge_index_train.size(1)
        n_val_edges = edge_index_val.size(1)
        n_test_edges = edge_index_test.size(1)
        
        # Get negative edges
        neg_edge_index_train = torch.zeros((2, int(n_train_edges * neg_sampling_ratio)), dtype=torch.long)
        if hasattr(train_data, 'edge_label_index'):
            neg_indices = train_data.edge_label_index[:, train_data.edge_label == 0]
            if neg_indices.size(1) > 0:
                neg_edge_index_train = neg_indices
        else:
            # Generate negative edges using negative sampling
            neg_edge_index_train = torch.stack([
                torch.randint(0, data.num_nodes, (int(n_train_edges * neg_sampling_ratio),)),
                torch.randint(0, data.num_nodes, (int(n_train_edges * neg_sampling_ratio),))
            ])
        
        # Val negative edges
        neg_edge_index_val = val_data.edge_label_index[:, val_data.edge_label == 0]
        
        # Test negative edges
        neg_edge_index_test = test_data.edge_label_index[:, test_data.edge_label == 0]
        
        print(f"Link prediction split statistics:")
        print(f"  Training: {edge_index_train.size(1)} positive edges, {neg_edge_index_train.size(1)} negative edges")
        print(f"  Validation: {edge_index_val.size(1)} positive edges, {neg_edge_index_val.size(1)} negative edges")
        print(f"  Testing: {edge_index_test.size(1)} positive edges, {neg_edge_index_test.size(1)} negative edges")
        
        return (edge_index_train, edge_index_val, edge_index_test,
                neg_edge_index_train, neg_edge_index_val, neg_edge_index_test)
    
    def visualize_graph(self, data, output_path='./results/cora_graph.png', max_nodes=100):
        """
        Visualize the graph structure.
        
        Args:
            data (torch_geometric.data.Data): Input data
            output_path (str): Path to save the visualization
            max_nodes (int): Maximum number of nodes to visualize
        """
        # Convert to networkx graph for visualization
        G = utils.to_networkx(data, to_undirected=True)
        
        # If graph is too large, take a subgraph
        if G.number_of_nodes() > max_nodes:
            nodes = list(G.nodes())[:max_nodes]
            G = G.subgraph(nodes)
        
        # Get node colors based on class labels
        if hasattr(data, 'y'):
            node_colors = data.y.numpy()[:max_nodes]
        else:
            node_colors = [0] * G.number_of_nodes()
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='tab10', node_size=50)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        plt.title(f'CORA Graph Visualization (showing {G.number_of_nodes()} nodes)')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {output_path}")

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load CORA dataset
    data, num_classes = preprocessor.load_cora()
    
    # Create link prediction splits
    edge_split = preprocessor.create_link_prediction_split(data)
    
    # Visualize the graph
    os.makedirs('./results', exist_ok=True)
    preprocessor.visualize_graph(data)
