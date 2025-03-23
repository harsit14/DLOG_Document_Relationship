import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm


class NodeClassificationTrainer:
    """
    Trainer for node classification tasks.
    """
    def __init__(self, model, lr=0.01, weight_decay=5e-4, device=None):
        """
        Initialize the trainer.
        
        Args:
            model (torch.nn.Module): The GNN model
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            device (torch.device): Device to run the training on
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.test_accs = []
    
    def train(self, data, epochs=200, patience=20, log_interval=10):
        """
        Train the model.
        
        Args:
            data (torch_geometric.data.Data): Input data
            epochs (int): Number of training epochs
            patience (int): Early stopping patience
            log_interval (int): Interval for logging metrics
            
        Returns:
            tuple: (best_val_acc, test_acc)
        """
        # Move data to device
        data = data.to(self.device)
        
        # Initialize early stopping variables
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        
        # Start training
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Training step
            train_loss, train_acc = self._train_step(data)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validation step
            val_loss, val_acc = self._val_step(data)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Log progress
            if epoch % log_interval == 0:
                print(f'Epoch: {epoch:03d}, '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Check for early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch} epochs")
                    break
        
        # Training completed
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Load best model and evaluate on test set
        self.model.load_state_dict(best_model_state)
        _, test_acc = self._test_step(data)
        self.test_accs.append(test_acc)
        
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        
        return best_val_acc, test_acc
    
    def _train_step(self, data):
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct = pred[data.train_mask] == data.y[data.train_mask]
        acc = correct.sum().item() / data.train_mask.sum().item()
        
        return loss.item(), acc
    
    def _val_step(self, data):
        """Perform one validation step."""
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            out = self.model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct = pred[data.val_mask] == data.y[data.val_mask]
            acc = correct.sum().item() / data.val_mask.sum().item()
            
        return loss.item(), acc
    
    def _test_step(self, data):
        """Evaluate on test set."""
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            out = self.model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask])
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask] == data.y[data.test_mask]
            acc = correct.sum().item() / data.test_mask.sum().item()
            
        return loss.item(), acc
    
    def plot_training_metrics(self, output_path='./results/training_metrics.png'):
        """
        Plot training and validation metrics.
        
        Args:
            output_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.axhline(y=self.test_accs[-1], color='r', linestyle='--', label=f'Test Accuracy: {self.test_accs[-1]:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training metrics plot saved to {output_path}")


class LinkPredictionTrainer:
    """
    Trainer for link prediction tasks.
    """
    def __init__(self, model, lr=0.01, weight_decay=5e-4, device=None):
        """
        Initialize the trainer.
        
        Args:
            model (torch.nn.Module): The link prediction model
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            device (torch.device): Device to run the training on
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.test_aucs = []
    
    def train(self, data, train_pos_edge_index, train_neg_edge_index, 
              val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index,
              epochs=200, patience=20, log_interval=10):
        """
        Train the link prediction model.
        
        Args:
            data (torch_geometric.data.Data): Input data
            train_pos_edge_index (torch.Tensor): Positive training edges
            train_neg_edge_index (torch.Tensor): Negative training edges
            val_pos_edge_index (torch.Tensor): Positive validation edges
            val_neg_edge_index (torch.Tensor): Negative validation edges
            test_pos_edge_index (torch.Tensor): Positive test edges
            test_neg_edge_index (torch.Tensor): Negative test edges
            epochs (int): Number of training epochs
            patience (int): Early stopping patience
            log_interval (int): Interval for logging metrics
            
        Returns:
            tuple: (best_val_auc, test_metrics)
        """
        # Move data to device
        data = data.to(self.device)
        train_pos_edge_index = train_pos_edge_index.to(self.device)
        train_neg_edge_index = train_neg_edge_index.to(self.device)
        val_pos_edge_index = val_pos_edge_index.to(self.device)
        val_neg_edge_index = val_neg_edge_index.to(self.device)
        test_pos_edge_index = test_pos_edge_index.to(self.device)
        test_neg_edge_index = test_neg_edge_index.to(self.device)
        
        # Initialize early stopping variables
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0
        
        # Start training
        print("Starting link prediction training...")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Training step
            train_loss = self._train_step(data, train_pos_edge_index, train_neg_edge_index)
            self.train_losses.append(train_loss)
            
            # Validation step
            val_loss, val_auc = self._val_step(data, val_pos_edge_index, val_neg_edge_index)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
            # Log progress
            if epoch % log_interval == 0:
                print(f'Epoch: {epoch:03d}, '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
            
            # Check for early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch} epochs")
                    break
        
        # Training completed
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Load best model and evaluate on test set
        self.model.load_state_dict(best_model_state)
        test_metrics = self._test_step(data, test_pos_edge_index, test_neg_edge_index)
        self.test_aucs.append(test_metrics['auc'])
        
        print(f"Best validation AUC: {best_val_auc:.4f}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        
        return best_val_auc, test_metrics
    
    def _train_step(self, data, pos_edge_index, neg_edge_index):
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Create positive and negative labels
        pos_labels = torch.ones(pos_edge_index.size(1), device=self.device)
        neg_labels = torch.zeros(neg_edge_index.size(1), device=self.device)
        
        # Combine positive and negative examples
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_labels = torch.cat([pos_labels, neg_labels])
        
        # Forward pass
        pred = self.model(data.x, data.edge_index, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, edge_labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _val_step(self, data, pos_edge_index, neg_edge_index):
        """Perform one validation step."""
        self.model.eval()
        
        with torch.no_grad():
            # Create positive and negative labels
            pos_labels = torch.ones(pos_edge_index.size(1), device=self.device)
            neg_labels = torch.zeros(neg_edge_index.size(1), device=self.device)
            
            # Combine positive and negative examples
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([pos_labels, neg_labels])
            
            # Forward pass
            pred = self.model(data.x, data.edge_index, edge_label_index)
            loss = F.binary_cross_entropy_with_logits(pred, edge_labels)
            
            # Calculate AUC
            pred_probs = torch.sigmoid(pred).cpu().numpy()
            labels = edge_labels.cpu().numpy()
            auc = roc_auc_score(labels, pred_probs)
            
        return loss.item(), auc
    
    def _test_step(self, data, pos_edge_index, neg_edge_index):
        """Evaluate on test set."""
        self.model.eval()
        
        with torch.no_grad():
            # Create positive and negative labels
            pos_labels = torch.ones(pos_edge_index.size(1), device=self.device)
            neg_labels = torch.zeros(neg_edge_index.size(1), device=self.device)
            
            # Combine positive and negative examples
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([pos_labels, neg_labels])
            
            # Forward pass
            pred = self.model(data.x, data.edge_index, edge_label_index)
            loss = F.binary_cross_entropy_with_logits(pred, edge_labels)
            
            # Calculate metrics
            pred_probs = torch.sigmoid(pred).cpu().numpy()
            pred_binary = (pred_probs > 0.5).astype(int)
            labels = edge_labels.cpu().numpy()
            
            metrics = {
                'loss': loss.item(),
                'auc': roc_auc_score(labels, pred_probs),
                'precision': precision_score(labels, pred_binary),
                'recall': recall_score(labels, pred_binary),
                'f1': f1_score(labels, pred_binary)
            }
            
        return metrics
    
    def plot_training_metrics(self, output_path='./results/link_prediction_metrics.png'):
        """
        Plot training and validation metrics.
        
        Args:
            output_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot AUC
        plt.subplot(1, 2, 2)
        plt.plot(self.val_aucs, label='Validation AUC')
        plt.axhline(y=self.test_aucs[-1], color='r', linestyle='--', label=f'Test AUC: {self.test_aucs[-1]:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Validation AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Link prediction metrics plot saved to {output_path}")