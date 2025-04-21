import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch.optim import Adam
import os
from datetime import datetime
import argparse
import wandb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import to_undirected
import logging
from tqdm import tqdm

from arxiv_dataset import ArXivDataset
from gnn_models import GCN, GAT, GraphSAGE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def to_device(tensor, device):
    """Move tensor to device (cuda if available, else cpu)"""
    return tensor.to(device)

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Move batch components to device
        x = to_device(batch.x, device)
        edge_index = to_device(batch.edge_index, device)
        y = to_device(batch.y, device)
        if hasattr(batch, 'train_mask'):
            train_mask = to_device(batch.train_mask, device)
        
        optimizer.zero_grad()
        
        # Forward pass
        out = model(x, edge_index)
        
        # Compute loss (node classification)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * train_mask.sum().item()
        total_samples += train_mask.sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Clear memory
        del out, x, edge_index, y, train_mask, batch
        torch.cuda.empty_cache()
    
    logger.info("Training epoch completed")
    return total_loss / total_samples

@torch.no_grad()
def test(model, loader, device, mask_type):
    model.eval()
    total_correct = 0
    total_nodes = 0
    
    pbar = tqdm(loader, desc=f"Evaluating {mask_type} set")
    for batch in pbar:
        # Move batch components to device
        x = to_device(batch.x, device)
        edge_index = to_device(batch.edge_index, device)
        y = to_device(batch.y, device)
        if hasattr(batch, 'train_mask'):
            train_mask = to_device(batch.train_mask, device)
        if hasattr(batch, 'val_mask'):
            val_mask = to_device(batch.val_mask, device)
        if hasattr(batch, 'test_mask'):
            test_mask = to_device(batch.test_mask, device)
        
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        
        if mask_type == 'train':
            mask = train_mask
        elif mask_type == 'val':
            mask = val_mask
        else:
            mask = test_mask
            
        correct = int((pred[mask] == y[mask]).sum())
        nodes = int(mask.sum())
        total_correct += correct
        total_nodes += nodes
        
        # Update progress bar
        accuracy = correct / nodes if nodes > 0 else 0
        pbar.set_postfix({'accuracy': f'{accuracy:.4f}'})
        
        # Clear memory
        del out, x, edge_index, y, train_mask, val_mask, test_mask, batch
        torch.cuda.empty_cache()
    
    return total_correct / total_nodes

def main():
    logger.info("Starting arXiv GNN training script")
    parser = argparse.ArgumentParser(description='Train GNN models on arXiv dataset')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage'],
                      help='GNN model to use (default: gcn)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='Hidden dimension (default: 256)')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='Number of GNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size (default: 512)')
    parser.add_argument('--num_neighbors', type=int, default=10,
                      help='Number of neighbors to sample (default: 10)')
    parser.add_argument('--heads', type=int, default=8,
                      help='Number of attention heads for GAT (default: 8)')
    parser.add_argument('--wandb_project', type=str, default='arxiv-gnn',
                      help='Weights & Biases project name')
    parser.add_argument('--run_name', type=str, default=None,
                      help='Weights & Biases run name')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Initialize wandb
    logger.info("Initializing Weights & Biases")
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )
    wandb.run.log_code(".")

    # Create dataset
    logger.info("Loading arXiv dataset")
    dataset = ArXivDataset(
        root='./data/arxiv', 
        language_model='all-MiniLM-L6-v2', 
        data_path='./data/arxiv/processed/data.pt', 
        title_embeddings_path='./data/arxiv/processed/title_embeddings_all-MiniLM-L6-v2.pt', 
        abstract_embeddings_path='./data/arxiv/processed/abstract_embeddings_all-MiniLM-L6-v2.pt',
        device=device
    )
    data, title_embeddings, abstract_embeddings = dataset.data, dataset.title_embeddings, dataset.abstract_embeddings

    # Move data to the correct device
    data = to_device(data, device)
    if title_embeddings is not None:
        title_embeddings = to_device(torch.from_numpy(title_embeddings), device)
    if abstract_embeddings is not None:
        abstract_embeddings = to_device(torch.from_numpy(abstract_embeddings), device)

    del data.title
    del data.abstract
    del data.paper_id_to_idx

    # Create train/val/test split
    logger.info("Creating train/val/test splits")
    num_nodes = data.num_nodes
    train_mask = to_device(torch.zeros(num_nodes, dtype=torch.bool), device)
    val_mask = to_device(torch.zeros(num_nodes, dtype=torch.bool), device)
    test_mask = to_device(torch.zeros(num_nodes, dtype=torch.bool), device)
    
    # Random split (80/10/10)
    indices = to_device(torch.randperm(num_nodes), device)
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Create data loaders with neighbor sampling
    train_loader = NeighborLoader(
        data,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        batch_size=args.batch_size,
        input_nodes=data.train_mask,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        batch_size=args.batch_size,
        input_nodes=data.val_mask
    )
    
    test_loader = NeighborLoader(
        data,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        batch_size=args.batch_size,
        input_nodes=data.test_mask
    )

    # Create model based on the specified type
    logger.info(f"Creating {args.model.upper()} model")
    if args.model == 'gcn':
        model = GCN(
            input_dim=data.x.size(1),
            hidden_dim=args.hidden_dim,
            output_dim=len(torch.unique(data.y)),
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model == 'gat':
        model = GAT(
            input_dim=data.x.size(1),
            hidden_dim=args.hidden_dim,
            output_dim=len(torch.unique(data.y)),
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout
        )
    elif args.model == 'sage':
        model = GraphSAGE(
            input_dim=data.x.size(1),
            hidden_dim=args.hidden_dim,
            output_dim=len(torch.unique(data.y)),
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    # Move model to device
    model = to_device(model, device)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train(model, train_loader, optimizer, device)
        
        # Evaluate
        train_acc = test(model, train_loader, device, 'train')
        val_acc = test(model, val_loader, device, 'val')
        
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
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            torch.save(model.state_dict(), os.path.join('models', f'{args.model}_best.pt'))
            wandb.save(os.path.join('models', f'{args.model}_best.pt'))
        
        logger.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # Clear memory after each epoch
        torch.cuda.empty_cache()
    
    # Test final model
    logger.info("Starting final model evaluation")
    test_acc = test(model, test_loader, device, 'test')
    logger.info(f'Test Accuracy: {test_acc:.4f}')
    
    # Log final test accuracy
    wandb.log({'test_accuracy': test_acc})
    
    # Save final model
    logger.info("Saving final model")
    torch.save(model.state_dict(), os.path.join('models', f'{args.model}_final.pt'))
    wandb.save(os.path.join('models', f'{args.model}_final.pt'))
    
    # Finish wandb run
    logger.info("Training completed, finishing Weights & Biases run")
    wandb.finish()

if __name__ == "__main__":
    main() 