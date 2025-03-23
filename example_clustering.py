#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for running document clustering using GNN embeddings.
This script focuses on clustering documents based on their GNN embeddings
and analyzing document communities in the CORA dataset.
"""

import os
import argparse
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import DataPreprocessor
from gnn_models import GCN, GAT, GraphSAGE
from training_evaluation import NodeClassificationTrainer
from document_clustering import DocumentClustering


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Document Clustering with GNN Embeddings')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results/clustering', help='Results directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run the models')
    
    # Model settings
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage'],
                        help='GNN model type')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Clustering settings
    parser.add_argument('--n_clusters', type=int, default=None, help='Number of clusters')
    parser.add_argument('--clustering_method', type=str, default='kmeans', 
                        choices=['kmeans', 'hierarchical'], help='Clustering method')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, 
                        help='Similarity threshold for building the document network')
    parser.add_argument('--community_method', type=str, default='louvain', 
                        choices=['louvain', 'label_prop', 'greedy'], help='Community detection method')
    parser.add_argument('--save_embeddings', action='store_true', help='Save document embeddings')
    
    args = parser.parse_args()
    return args


def main():
    """Main function to run document clustering."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = DataPreprocessor(data_dir=args.data_dir)
    data, num_classes = preprocessor.load_cora()
    
    # Initialize GNN model
    print(f"\nInitializing {args.model.upper()} model...")
    input_dim = data.num_node_features
    hidden_dim = args.hidden_dim
    embedding_dim = args.embedding_dim
    
    # Initialize model based on the specified type
    if args.model == 'gcn':
        model = GCN(input_dim, hidden_dim, embedding_dim, 
                   num_layers=args.num_layers, dropout=args.dropout)
    elif args.model == 'gat':
        model = GAT(input_dim, hidden_dim, embedding_dim, 
                   num_layers=args.num_layers, dropout=args.dropout)
    elif args.model == 'sage':
        model = GraphSAGE(input_dim, hidden_dim, embedding_dim, 
                         num_layers=args.num_layers, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Print model summary
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model to get good document embeddings
    print("\nTraining model to generate document embeddings...")
    trainer = NodeClassificationTrainer(
        model, lr=args.lr, weight_decay=args.weight_decay, device=device
    )
    trainer.train(data, epochs=args.epochs, patience=args.patience)
    
    # Generate document embeddings
    print("\nGenerating document embeddings...")
    doc_clustering = DocumentClustering(model, device=device)
    embeddings = doc_clustering.generate_embeddings(data)
    
    # Save embeddings if requested
    if args.save_embeddings:
        embedding_path = os.path.join(args.results_dir, 'document_embeddings.npy')
        np.save(embedding_path, embeddings)
        print(f"Document embeddings saved to {embedding_path}")
    
    # Cluster documents
    print("\nClustering documents...")
    cluster_labels, metrics = doc_clustering.cluster_documents(
        embeddings, n_clusters=args.n_clusters, method=args.clustering_method,
        labels=data.y.cpu().numpy(), visualize=True,
        output_path=os.path.join(args.results_dir, 'document_clusters.png')
    )
    
    # Build similarity network
    print("\nBuilding document similarity network...")
    similarity_network = doc_clustering.build_similarity_network(
        embeddings, threshold=args.similarity_threshold,
        output_path=os.path.join(args.results_dir, 'similarity_network.png')
    )
    
    # Identify document communities
    print("\nIdentifying document communities...")
    try:
        communities = doc_clustering.identify_document_communities(
            similarity_network, method=args.community_method,
            output_path=os.path.join(args.results_dir, 'document_communities.png')
        )
        
        # Print community statistics
        if communities is not None:
            num_communities = len(set(communities.values()))
            print(f"Number of communities: {num_communities}")
            
            # Count documents in each community
            community_counts = {}
            for community_id in communities.values():
                community_counts[community_id] = community_counts.get(community_id, 0) + 1
            
            print("Documents per community:")
            for community_id, count in sorted(community_counts.items()):
                print(f"  Community {community_id}: {count} documents")
    except Exception as e:
        print(f"Warning: Community detection failed: {e}")
        print("Make sure python-louvain is installed for community detection.")
    
    print("\nDocument clustering completed successfully!")
    print(f"Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
