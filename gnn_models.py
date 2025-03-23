import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) implementation as described in the paper:
    "Semi-supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        """
        Initialize the GCN model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output features (num classes for classification)
            num_layers (int): Number of GCN layers
            dropout (float): Dropout probability
        """
        super(GCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index):
        """
        Forward pass through the GCN.
        
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges]
            
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, output_dim]
        """
        # Apply GCN layers with ReLU activation and dropout
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (without activation for classification)
        x = self.convs[-1](x, edge_index)
        
        return x


class GAT(nn.Module):
    """
    Graph Attention Network (GAT) implementation as described in the paper:
    "Graph Attention Networks" (Veličković et al., 2018)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 heads=8, output_heads=1, dropout=0.5):
        """
        Initialize the GAT model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output features (num classes for classification)
            num_layers (int): Number of GAT layers
            heads (int): Number of attention heads in hidden layers
            output_heads (int): Number of attention heads in output layer
            dropout (float): Dropout probability
        """
        super(GAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        
        # Output layer
        self.convs.append(
            GATConv(hidden_dim * heads, output_dim, heads=output_heads, concat=False, dropout=dropout)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass through the GAT.
        
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges]
            
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, output_dim]
        """
        # Apply GAT layers with ELU activation and dropout
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (without activation for classification)
        x = self.convs[-1](x, edge_index)
        
        return x


class GraphSAGE(nn.Module):
    """
    GraphSAGE implementation as described in the paper:
    "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, 
                 aggregator='mean'):
        """
        Initialize the GraphSAGE model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output features (num classes for classification)
            num_layers (int): Number of GraphSAGE layers
            dropout (float): Dropout probability
            aggregator (str): Aggregation function ('mean', 'max', or 'lstm')
        """
        super(GraphSAGE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
    
    def forward(self, x, edge_index):
        """
        Forward pass through GraphSAGE.
        
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges]
            
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, output_dim]
        """
        # Apply GraphSAGE layers with ReLU activation and dropout
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (without activation for classification)
        x = self.convs[-1](x, edge_index)
        
        return x


class LinkPredictionModel(nn.Module):
    """
    GNN-based model for link prediction.
    """
    def __init__(self, gnn_model, input_dim, hidden_dim, embedding_dim, dropout=0.5):
        """
        Initialize the link prediction model.
        
        Args:
            gnn_model (str): GNN model type ('gcn', 'gat', or 'sage')
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            embedding_dim (int): Dimension of node embeddings
            dropout (float): Dropout probability
        """
        super(LinkPredictionModel, self).__init__()
        
        # Initialize the GNN encoder based on the specified model type
        if gnn_model == 'gcn':
            self.encoder = GCN(input_dim, hidden_dim, embedding_dim, num_layers=2, dropout=dropout)
        elif gnn_model == 'gat':
            self.encoder = GAT(input_dim, hidden_dim, embedding_dim, num_layers=2, dropout=dropout)
        elif gnn_model == 'sage':
            self.encoder = GraphSAGE(input_dim, hidden_dim, embedding_dim, num_layers=2, dropout=dropout)
        else:
            raise ValueError(f"Unknown GNN model type: {gnn_model}")
        
        # MLP for predicting link existence
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, edge_index, edge_label_index):
        """
        Forward pass for link prediction.
        
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges] for message passing
            edge_label_index (torch.Tensor): Edge indices for prediction of shape [2, num_edges_to_predict]
            
        Returns:
            torch.Tensor: Probability of edge existence for each edge in edge_label_index
        """
        # Get node embeddings using the GNN encoder
        embeddings = self.encoder(x, edge_index)
        
        # Extract source and target node embeddings
        src, dst = edge_label_index
        src_embeddings = embeddings[src]
        dst_embeddings = embeddings[dst]
        
        # Concatenate the source and target embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict the probability of edge existence
        pred = self.decoder(edge_embeddings)
        
        return pred.squeeze()
