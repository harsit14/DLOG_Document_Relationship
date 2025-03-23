#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for Document Relationship Modeling using Graph Neural Networks.
This script implements the pipeline described in the project proposal:
1. Load and preprocess data
2. Train GNN models for document classification
3. Train GNN models for link prediction
4. Perform document clustering
5. Visualize results
"""

import os
import argparse
import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Import custom modules
import sys
sys.path.append('./doc_gnn')
from data_preprocessing import DataPreprocessor
from gnn_models import GCN, GAT, GraphSAGE, LinkPredictionModel
from training_evaluation import NodeClassificationTrainer, LinkPredictionTrainer
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
    parser = argparse.ArgumentParser(description='Document Relationship Modeling with GNNs')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
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
    
    # Task settings
    parser.add_argument('--tasks', nargs='+', default=['classification', 'link_prediction', 'clustering'],
                        help='Tasks to perform')
    
    # Clustering settings
    parser.add_argument('--n_clusters', type=int, default=None, help='Number of clusters')
    parser.add_argument('--clustering_method', type=str, default='kmeans', 
                        choices=['kmeans', 'hierarchical'], help='Clustering method')
    parser.add_argument('--community_method', type=str, default='louvain', 
                        choices=['louvain', 'label_prop', 'greedy'], help='Community detection method')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, 
                        help='Similarity threshold for building network')
    
    args = parser.parse_args()
    return args


def main():
    """Main function to run the pipeline."""
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
    print("\n1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor(data_dir=args.data_dir)
    data, num_classes = preprocessor.load_cora()
    
    # Create link prediction split if needed
    edge_split = None
    if 'link_prediction' in args.tasks:
        edge_split = preprocessor.create_link_prediction_split(data)
    
    # Visualize the graph
    preprocessor.visualize_graph(data, output_path=os.path.join(args.results_dir, 'cora_graph.png'))
    
    # Initialize GNN model
    print(f"\n2. Initializing {args.model.upper()} model...")
    input_dim = data.num_node_features
    hidden_dim = args.hidden_dim
    output_dim = num_classes
    
    if args.model == 'gcn':
        model = GCN(input_dim, hidden_dim, output_dim, 
                   num_layers=args.num_layers, dropout=args.dropout)
    elif args.model == 'gat':
        model = GAT(input_dim, hidden_dim, output_dim, 
                   num_layers=args.num_layers, dropout=args.dropout)
    elif args.model == 'sage':
        model = GraphSAGE(input_dim, hidden_dim, output_dim, 
                         num_layers=args.num_layers, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Print model summary
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Perform document classification
    if 'classification' in args.tasks:
        print("\n3. Performing document classification...")
        classification_trainer = NodeClassificationTrainer(
            model, lr=args.lr, weight_decay=args.weight_decay, device=device
        )
        best_val_acc, test_acc = classification_trainer.train(
            data, epochs=args.epochs, patience=args.patience
        )
        classification_trainer.plot_training_metrics(
            output_path=os.path.join(args.results_dir, 'classification_metrics.png')
        )
    
    # Perform link prediction
    if 'link_prediction' in args.tasks:
        print("\n4. Performing link prediction...")
        # Initialize link prediction model
        link_pred_model = LinkPredictionModel(
            args.model, input_dim, hidden_dim, args.embedding_dim, dropout=args.dropout
        )
        
        # Unpack edge split
        train_pos_edge_index, val_pos_edge_index, test_pos_edge_index, \
        train_neg_edge_index, val_neg_edge_index, test_neg_edge_index = edge_split
        
        # Train link prediction model
        link_prediction_trainer = LinkPredictionTrainer(
            link_pred_model, lr=args.lr, weight_decay=args.weight_decay, device=device
        )
        best_val_auc, test_metrics = link_prediction_trainer.train(
            data, train_pos_edge_index, train_neg_edge_index,
            val_pos_edge_index, val_neg_edge_index,
            test_pos_edge_index, test_neg_edge_index,
            epochs=args.epochs, patience=args.patience
        )
        link_prediction_trainer.plot_training_metrics(
            output_path=os.path.join(args.results_dir, 'link_prediction_metrics.png')
        )
    
    # Perform document clustering
    if 'clustering' in args.tasks:
        print("\n5. Performing document clustering...")
        # Use the trained model from classification if available, otherwise initialize a new one
        if 'classification' not in args.tasks:
            if args.model == 'gcn':
                model = GCN(input_dim, hidden_dim, args.embedding_dim, 
                           num_layers=args.num_layers, dropout=args.dropout)
            elif args.model == 'gat':
                model = GAT(input_dim, hidden_dim, args.embedding_dim, 
                           num_layers=args.num_layers, dropout=args.dropout)
            elif args.model == 'sage':
                model = GraphSAGE(input_dim, hidden_dim, args.embedding_dim, 
                                 num_layers=args.num_layers, dropout=args.dropout)
            
            # Train the model for embedding generation
            classification_trainer = NodeClassificationTrainer(
                model, lr=args.lr, weight_decay=args.weight_decay, device=device
            )
            classification_trainer.train(data, epochs=args.epochs, patience=args.patience)
        
        # Initialize document clustering
        doc_clustering = DocumentClustering(model, device=device)
        
        # Generate document embeddings
        embeddings = doc_clustering.generate_embeddings(data)
        
        # Cluster documents
        cluster_labels, metrics = doc_clustering.cluster_documents(
            embeddings, n_clusters=args.n_clusters, method=args.clustering_method,
            labels=data.y.cpu().numpy(), visualize=True,
            output_path=os.path.join(args.results_dir, 'document_clusters.png')
        )
        
        # Build similarity network
        similarity_network = doc_clustering.build_similarity_network(
            embeddings, threshold=args.similarity_threshold,
            output_path=os.path.join(args.results_dir, 'similarity_network.png')
        )
        
        # Identify document communities
        communities = doc_clustering.identify_document_communities(
            similarity_network, method=args.community_method,
            output_path=os.path.join(args.results_dir, 'document_communities.png')
        )
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
