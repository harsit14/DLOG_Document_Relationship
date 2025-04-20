import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
import os
from datetime import datetime
import argparse
import wandb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import to_undirected

from arxiv_dataset import ArXivDataset
from arxiv_gnn import ArXivGNN

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Compute loss (node classification)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def test(model, loader, device, mask):
    model.eval()
    total_correct = 0
    total_nodes = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        total_correct += int((pred[mask] == data.y[mask]).sum())
        total_nodes += int(mask.sum())
    
    return total_correct / total_nodes

def main():
    parser = argparse.ArgumentParser(description='Train GNN models on arXiv dataset')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'gin'],
                      help='GNN model to use (default: gcn)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension (default: 256)')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='Number of GNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--heads', type=int, default=8,
                      help='Number of attention heads for GAT (default: 8)')
    parser.add_argument('--wandb_project', type=str, default='arxiv-gnn',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Weights & Biases entity name')
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )
    
    # Log code
    wandb.run.log_code(".")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create dataset
    dataset = ArXivDataset(root='./data/arxiv')
    data = dataset[0]
    
    # Create train/val/test split
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Random split (80/10/10)
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Create data loaders
    train_loader = DataLoader([data], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([data], batch_size=args.batch_size)
    test_loader = DataLoader([data], batch_size=args.batch_size)

    # Create model
    model = ArXivGNN(
        model_type=args.model,
        input_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=len(torch.unique(data.y)),
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train(model, train_loader, optimizer, device)
        
        # Evaluate
        train_acc = test(model, train_loader, device, data.train_mask)
        val_acc = test(model, val_loader, device, data.val_mask)
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join('models', f'{args.model}_best.pt'))
            # Log best model to wandb
            wandb.save(os.path.join('models', f'{args.model}_best.pt'))
        
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Test final model
    test_acc = test(model, test_loader, device, data.test_mask)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Log final test accuracy
    wandb.log({'test_accuracy': test_acc})
    
    # Save final model
    torch.save(model.state_dict(), os.path.join('models', f'{args.model}_final.pt'))
    wandb.save(os.path.join('models', f'{args.model}_final.pt'))
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 