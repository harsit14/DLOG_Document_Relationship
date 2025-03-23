#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for running document classification using GNNs.
This script focuses on training and evaluating GNN models for the 
document classification task on the CORA dataset.
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
    parser = argparse.ArgumentParser(description='Document Classification with GNNs')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results/classification', help='Results directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run the models')
    
    # Model settings
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage'],
                        help='GNN model type')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    
    args = parser.parse_args()
    return args


def main():
    """Main function to run document classification."""
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
    
    # Visualize the graph
    preprocessor.visualize_graph(data, output_path=os.path.join(args.results_dir, 'cora_graph.png'))
    
    # Initialize GNN model
    print(f"\nInitializing {args.model.upper()} model...")
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
    
    # Train and evaluate model
    print("\nTraining and evaluating model...")
    trainer = NodeClassificationTrainer(
        model, lr=args.lr, weight_decay=args.weight_decay, device=device
    )
    best_val_acc, test_acc = trainer.train(
        data, epochs=args.epochs, patience=args.patience
    )
    trainer.plot_training_metrics(
        output_path=os.path.join(args.results_dir, 'training_metrics.png')
    )
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.results_dir, f"{args.model}_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    print("\nDocument classification completed successfully!")
    print(f"Results saved to {args.results_dir}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
