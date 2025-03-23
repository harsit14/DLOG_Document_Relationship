#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for running link prediction using GNNs.
This script focuses on training and evaluating GNN models for predicting
citation links between documents in the CORA dataset.
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
from gnn_models import LinkPredictionModel
from training_evaluation import LinkPredictionTrainer


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
    parser = argparse.ArgumentParser(description='Link Prediction with GNNs')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results/link_prediction', help='Results directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run the models')
    
    # Model settings
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage'],
                        help='GNN model type')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Split settings
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--neg_sampling_ratio', type=float, default=1.0, 
                        help='Ratio of negative samples to positive samples')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    
    args = parser.parse_args()
    return args


def main():
    """Main function to run link prediction."""
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
    
    # Create link prediction split
    print("\nCreating link prediction split...")
    edge_split = preprocessor.create_link_prediction_split(
        data, val_ratio=args.val_ratio, test_ratio=args.test_ratio, 
        neg_sampling_ratio=args.neg_sampling_ratio
    )
    
    # Unpack edge split
    train_pos_edge_index, val_pos_edge_index, test_pos_edge_index, \
    train_neg_edge_index, val_neg_edge_index, test_neg_edge_index = edge_split
    
    # Initialize link prediction model
    print(f"\nInitializing {args.model.upper()} link prediction model...")
    input_dim = data.num_node_features
    hidden_dim = args.hidden_dim
    embedding_dim = args.embedding_dim
    
    link_pred_model = LinkPredictionModel(
        args.model, input_dim, hidden_dim, embedding_dim, dropout=args.dropout
    )
    
    # Print model summary
    print(link_pred_model)
    print(f"Number of parameters: {sum(p.numel() for p in link_pred_model.parameters())}")
    
    # Train and evaluate model
    print("\nTraining and evaluating link prediction model...")
    trainer = LinkPredictionTrainer(
        link_pred_model, lr=args.lr, weight_decay=args.weight_decay, device=device
    )
    best_val_auc, test_metrics = trainer.train(
        data, train_pos_edge_index, train_neg_edge_index,
        val_pos_edge_index, val_neg_edge_index,
        test_pos_edge_index, test_neg_edge_index,
        epochs=args.epochs, patience=args.patience
    )
    trainer.plot_training_metrics(
        output_path=os.path.join(args.results_dir, 'training_metrics.png')
    )
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.results_dir, f"{args.model}_link_pred_model.pt")
        torch.save(link_pred_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Save test metrics
    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nLink prediction completed successfully!")
    print(f"Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
