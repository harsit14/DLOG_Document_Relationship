import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch_geometric.utils as utils


class DocumentClustering:
    """
    Class for document clustering using GNN embeddings.
    """
    def __init__(self, model, device=None):
        """
        Initialize the document clustering.
        
        Args:
            model (torch.nn.Module): Trained GNN model for generating embeddings
            device (torch.device): Device to run the model on
        """
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def generate_embeddings(self, data):
        """
        Generate document embeddings using the trained GNN model.
        
        Args:
            data (torch_geometric.data.Data): Input data
            
        Returns:
            numpy.ndarray: Document embeddings
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)
        
        return embeddings.cpu().numpy()
    
    def cluster_documents(self, embeddings, n_clusters=None, method='kmeans', 
                          labels=None, visualize=True, output_path=None):
        """
        Cluster documents based on their embeddings.
        
        Args:
            embeddings (numpy.ndarray): Document embeddings
            n_clusters (int): Number of clusters (if None, use the number of classes in labels)
            method (str): Clustering method ('kmeans' or 'hierarchical')
            labels (numpy.ndarray): Ground truth labels for evaluation
            visualize (bool): Whether to visualize the clusters
            output_path (str): Path to save the visualization
            
        Returns:
            tuple: (cluster_labels, metrics)
        """
        # Set number of clusters
        if n_clusters is None:
            if labels is not None:
                n_clusters = len(np.unique(labels))
            else:
                n_clusters = 7  # Default for CORA
        
        # Perform clustering
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Evaluate clustering if ground truth labels are provided
        metrics = {}
        if labels is not None:
            metrics['nmi'] = normalized_mutual_info_score(labels, cluster_labels)
            metrics['ari'] = adjusted_rand_score(labels, cluster_labels)
            
            print(f"Clustering metrics:")
            print(f"  Normalized Mutual Information (NMI): {metrics['nmi']:.4f}")
            print(f"  Adjusted Rand Index (ARI): {metrics['ari']:.4f}")
        
        # Visualize clusters
        if visualize:
            self.visualize_clusters(embeddings, cluster_labels, labels, output_path)
        
        return cluster_labels, metrics
    
    def visualize_clusters(self, embeddings, cluster_labels, true_labels=None, output_path=None):
        """
        Visualize document clusters using t-SNE.
        
        Args:
            embeddings (numpy.ndarray): Document embeddings
            cluster_labels (numpy.ndarray): Cluster assignments
            true_labels (numpy.ndarray): Ground truth labels
            output_path (str): Path to save the visualization
        """
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create figure with multiple plots
        if true_labels is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot predicted clusters
        scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='tab10',
                              s=20, alpha=0.7)
        ax1.set_title('Document Clusters (Predicted)')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        
        # Add a colorbar for cluster labels
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
        ax1.add_artist(legend1)
        
        # Plot true labels if provided
        if true_labels is not None:
            scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='tab10',
                                  s=20, alpha=0.7)
            ax2.set_title('Document Clusters (Ground Truth)')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            
            # Add a colorbar for true labels
            legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes")
            ax2.add_artist(legend2)
        
        plt.tight_layout()
        
        # Save figure if output path is specified
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Cluster visualization saved to {output_path}")
        
        plt.close()
    
    def build_similarity_network(self, embeddings, threshold=0.7, output_path=None):
        """
        Build a similarity network based on document embeddings.
        
        Args:
            embeddings (numpy.ndarray): Document embeddings
            threshold (float): Similarity threshold for adding edges
            output_path (str): Path to save the visualization
            
        Returns:
            networkx.Graph: Similarity network
        """
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(embeddings)):
            G.add_node(i)
        
        # Add edges based on similarity threshold
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        print(f"Similarity network statistics:")
        print(f"  Number of nodes: {G.number_of_nodes()}")
        print(f"  Number of edges: {G.number_of_edges()}")
        
        # Visualize the similarity network
        if output_path is not None:
            plt.figure(figsize=(12, 12))
            
            # Use a layout that reflects the structure of the network
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=30, alpha=0.7)
            
            # Draw edges with weights reflected in width
            edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)
            
            plt.title('Document Similarity Network')
            plt.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Similarity network visualization saved to {output_path}")
        
        return G
    
    def identify_document_communities(self, G, method='louvain', output_path=None):
        """
        Identify document communities in the similarity network.
        
        Args:
            G (networkx.Graph): Similarity network
            method (str): Community detection method ('louvain', 'label_prop', or 'greedy')
            output_path (str): Path to save the visualization
            
        Returns:
            dict: Community assignments
        """
        try:
            # Import community detection algorithms
            if method == 'louvain':
                from community import best_partition
                communities = best_partition(G)
            elif method == 'label_prop':
                communities = {node: idx for idx, com in enumerate(nx.algorithms.community.label_propagation.label_propagation_communities(G)) for node in com}
            elif method == 'greedy':
                communities = {node: idx for idx, com in enumerate(nx.algorithms.community.greedy_modularity_communities(G)) for node in com}
            else:
                raise ValueError(f"Unknown community detection method: {method}")
            
            # Count the number of communities
            num_communities = len(set(communities.values()))
            print(f"Number of detected communities: {num_communities}")
            
            # Visualize communities
            if output_path is not None:
                plt.figure(figsize=(12, 12))
                
                # Use a layout that reflects the structure of the network
                pos = nx.spring_layout(G, seed=42)
                
                # Draw nodes colored by community
                cmap = plt.cm.get_cmap('tab20', num_communities)
                nx.draw_networkx_nodes(G, pos, node_size=50, 
                                      node_color=[communities[node] for node in G.nodes()],
                                      cmap=cmap, alpha=0.8)
                
                # Draw edges
                nx.draw_networkx_edges(G, pos, alpha=0.2)
                
                plt.title(f'Document Communities ({method.capitalize()})')
                plt.axis('off')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Community visualization saved to {output_path}")
            
            return communities
            
        except ImportError:
            print("Warning: Community detection requires additional libraries.")
            print("Install python-louvain for Louvain method: pip install python-louvain")
            return None
