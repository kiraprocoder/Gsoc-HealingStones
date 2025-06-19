import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np
from datetime import datetime
import logging
from model import FragmentMatchingGNN  # Your model class
from graph_constructor import GraphConstructor  # Your graph constructor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graphs_dir = 'graphs'  # Where your .json dataset is saved
dataset_file = 'fragment_pairs_dataset.json'  # Change this if needed
input_dim = 10  # Set this to match your feature size!

# Hyperparameters
hidden_dim = 64
num_layers = 4
dropout_rate = 0.2
gnn_type = 'gat'
lr = 1e-3
batch_size = 4
num_epochs = 50
patience = 10  # Early stopping patience
min_delta = 1e-4  # Minimum change for early stopping

# Loss function selection
LOSS_TYPE = 'focal'  # Options: 'bce', 'focal', 'contrastive', 'infonce'

# Advanced Loss Functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        # Assume output is similarity score, target is 1 for similar, 0 for dissimilar
        euclidean_distance = torch.sigmoid(output)  # Convert to [0,1] range
        loss_contrastive = torch.mean((target) * torch.pow(euclidean_distance, 2) +
                                      (1-target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Simplified InfoNCE for binary classification
        # This is a basic implementation - you may need to adapt based on your specific use case
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        log_prob = similarity_matrix - torch.log(sum_exp_sim)
        
        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / torch.sum(mask, dim=1)
        loss = -mean_log_prob_pos.mean()
        
        return loss

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# Model Checkpointing Class
class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='min'):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        else:
            self.monitor_op = np.greater
            self.best = -np.Inf

    def __call__(self, current, model, optimizer, epoch, additional_info=None):
        if not self.save_best_only or self.monitor_op(current, self.best):
            self.best = current
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': self.best,
                'additional_info': additional_info or {}
            }
            torch.save(checkpoint, self.filepath)
            logger.info(f'Model checkpoint saved to {self.filepath}')

# Dataset Loader
def load_dataset(json_file):
    filepath = os.path.join(graphs_dir, json_file)
    with open(filepath, 'r') as f:
        dataset_json = json.load(f)

    dataset = []
    for data_dict in dataset_json:
        # Import torch_geometric here to avoid the missing import
        import torch_geometric.data
        
        # Reconstruct Data object
        data = torch_geometric.data.Data(
            x=torch.tensor(data_dict['x'], dtype=torch.float),
            edge_index=torch.tensor(data_dict['edge_index'], dtype=torch.long),
            pos=torch.tensor(data_dict['pos'], dtype=torch.float),
        )
        if 'edge_attr' in data_dict:
            data.edge_attr = torch.tensor(data_dict['edge_attr'], dtype=torch.float)
        if 'y' in data_dict:
            data.y = torch.tensor(data_dict['y'], dtype=torch.float)
        if 'fragment_labels' in data_dict:
            data.fragment_labels = torch.tensor(data_dict['fragment_labels'], dtype=torch.long)

        dataset.append(data)
    
    return dataset

# Get loss function
def get_loss_function(loss_type):
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        return FocalLoss(alpha=1, gamma=2)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(margin=1.0)
    elif loss_type == 'infonce':
        return InfoNCELoss(temperature=0.07)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Training Function
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        
        if LOSS_TYPE == 'infonce':
            loss = criterion(output, batch.y.view(-1))
        else:
            loss = criterion(output.view(-1), batch.y.view(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        with torch.no_grad():
            if LOSS_TYPE != 'infonce':
                preds = torch.sigmoid(output).view(-1) > 0.5
                correct += (preds.float() == batch.y.view(-1)).sum().item()
                total_samples += batch.y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy

# Evaluation Function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            
            if LOSS_TYPE == 'infonce':
                loss = criterion(output, batch.y.view(-1))
            else:
                loss = criterion(output.view(-1), batch.y.view(-1))
            
            total_loss += loss.item()
            
            # Calculate accuracy
            if LOSS_TYPE != 'infonce':
                preds = torch.sigmoid(output).view(-1) > 0.5
                correct += (preds.float() == batch.y.view(-1)).sum().item()
                total_samples += batch.y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy

def main():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'logs/experiment_{timestamp}')
    
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_file)
    
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    logger.info("Initializing model...")
    model = FragmentMatchingGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        dropout_rate=dropout_rate
    ).to(device)

    # Loss function and optimizer
    criterion = get_loss_function(LOSS_TYPE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping and checkpointing
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    checkpoint_callback = ModelCheckpoint(
        filepath=f'checkpoints/best_model_{timestamp}.pth',
        monitor='val_loss',
        save_best_only=True
    )

    logger.info(f"Starting training with {LOSS_TYPE} loss...")
    logger.info(f"Device: {device}")
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        logger.info(f"Epoch {epoch}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Model checkpointing
        additional_info = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'learning_rate': current_lr
        }
        checkpoint_callback(val_loss, model, optimizer, epoch, additional_info)
        
        # Save checkpoint every epoch
        epoch_checkpoint_path = f'checkpoints/epoch_{epoch}_{timestamp}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, epoch_checkpoint_path)
        
        # Early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Save final model
    final_model_path = f'final_model_{timestamp}.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()