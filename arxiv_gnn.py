import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn as nn

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Batch normalization layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(BatchNorm1d(hidden_dim))
        
    def forward(self, x, edge_index, batch=None):
        # Apply GCN layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class GATEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=8, dropout=0.5):
        super(GATEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))
        
        # Batch normalization layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(BatchNorm1d(hidden_dim * heads))
        
    def forward(self, x, edge_index, batch=None):
        # Apply GAT layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class GINEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(GINEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(
            Sequential(
                Linear(input_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            )
        ))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(
                Sequential(
                    Linear(hidden_dim, hidden_dim),
                    BatchNorm1d(hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                )
            ))
        self.convs.append(GINConv(
            Sequential(
                Linear(hidden_dim, output_dim),
                BatchNorm1d(output_dim),
                ReLU(),
            )
        ))
        
    def forward(self, x, edge_index, batch=None):
        # Apply GIN layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class ArXivGNN(torch.nn.Module):
    def __init__(self, model_type='gcn', input_dim=128, hidden_dim=256, output_dim=128, 
                 num_layers=3, heads=8, dropout=0.5):
        super(ArXivGNN, self).__init__()
        
        # Choose the encoder based on model_type
        if model_type.lower() == 'gcn':
            self.encoder = GCNEncoder(input_dim, hidden_dim, output_dim, num_layers, dropout)
        elif model_type.lower() == 'gat':
            self.encoder = GATEncoder(input_dim, hidden_dim, output_dim, num_layers, heads, dropout)
        elif model_type.lower() == 'gin':
            self.encoder = GINEncoder(input_dim, hidden_dim, output_dim, num_layers, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    def forward(self, x, edge_index, batch=None):
        # Initialize parameters if they're uninitialized
        if not hasattr(self, '_initialized'):
            with torch.no_grad():
                # Create dummy input
                dummy_x = torch.zeros(1, x.size(1), device=x.device)
                dummy_edge_index = torch.zeros(2, 1, dtype=torch.long, device=edge_index.device)
                # Forward pass to initialize parameters
                self.encoder(dummy_x, dummy_edge_index)
            self._initialized = True
            
        return self.encoder(x, edge_index, batch)

def main():
    # Example usage
    model = ArXivGNN(model_type='gcn')
    print(model)
    
    # Test with random data
    x = torch.randn(10, 128)  # 10 nodes, 128 features
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    
    output = model(x, edge_index)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main() 