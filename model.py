import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm
from torch_geometric.data import Batch
import math

class AttentionPooling(nn.Module):
    """Attention-based pooling for graph-level representations"""
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, batch):
        # Compute attention weights
        attention_weights = self.attention(x)  # [N, 1]
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Apply attention pooling
        weighted_features = x * attention_weights  # [N, hidden_dim]
        
        # Pool by batch
        pooled = global_add_pool(weighted_features, batch)
        return pooled

class ContrastiveBranch(nn.Module):
    """Contrastive learning branch for embedding generation"""
    def __init__(self, input_dim, embedding_dim=128):
        super(ContrastiveBranch, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projector(x), dim=1)

class MultiTaskHead(nn.Module):
    """Multi-task head for classification and regression"""
    def __init__(self, input_dim, num_classes=2):
        super(MultiTaskHead, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head (no sigmoid - using BCEWithLogitsLoss)
        self.classifier = nn.Linear(input_dim // 4, 1)
        
        # Regression head (for similarity score)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()  # Output similarity score [0, 1]
        )
        
        # Multi-class classification head (optional)
        self.multi_classifier = nn.Linear(input_dim // 4, num_classes)
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        # Binary classification logits (no sigmoid)
        classification_logits = self.classifier(shared_features)
        
        # Regression output
        regression_output = self.regressor(shared_features)
        
        # Multi-class classification logits
        multi_class_logits = self.multi_classifier(shared_features)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'multi_class_logits': multi_class_logits
        }

class FragmentMatchingGNN(nn.Module):
    """
    Advanced GNN for fragment matching with multiple features:
    - Graph Attention Networks
    - Attention Pooling
    - Contrastive Learning Branch
    - Multi-task Learning (Classification + Regression)
    """
    def __init__(self, 
                 node_features=10, 
                 hidden_dim=64, 
                 num_layers=3,
                 heads=4,
                 embedding_dim=128,
                 num_classes=2,
                 dropout=0.1,
                 use_attention_pooling=True,
                 use_contrastive=True,
                 use_multitask=True):
        
        super(FragmentMatchingGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.use_attention_pooling = use_attention_pooling
        self.use_contrastive = use_contrastive
        self.use_multitask = use_multitask
        
        # Graph Attention Layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(node_features, hidden_dim // heads, heads=heads, dropout=dropout)
        )
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            )
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Last layer
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        )
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Pooling layer
        if use_attention_pooling:
            self.pooling = AttentionPooling(hidden_dim)
        else:
            # Default pooling (combination of mean, max, add)
            self.pooling = None
        
        # Fragment pair interaction
        self.fragment_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Contrastive learning branch
        if use_contrastive:
            self.contrastive_branch = ContrastiveBranch(hidden_dim, embedding_dim)
        
        # Multi-task head
        if use_multitask:
            self.multitask_head = MultiTaskHead(hidden_dim // 2, num_classes)
        else:
            # Simple binary classification head
            self.classifier = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward_single_graph(self, data):
        """Forward pass for a single graph"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph attention layers
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x = gat_layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Graph-level pooling
        if self.use_attention_pooling:
            graph_embedding = self.pooling(x, batch)
        else:
            # Combine multiple pooling strategies
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            add_pool = global_add_pool(x, batch)
            graph_embedding = torch.cat([mean_pool, max_pool, add_pool], dim=1)
            
            # Project to correct dimension
            if not hasattr(self, 'pool_projection'):
                self.pool_projection = nn.Linear(self.hidden_dim * 3, self.hidden_dim).to(x.device)
            graph_embedding = self.pool_projection(graph_embedding)
        
        return graph_embedding
    
    def forward_dual_graph(self, data):
        """Forward pass for dual fragment graph"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        fragment_labels = data.fragment_labels
        
        # Graph attention layers
        for i, (gat_layer, batch_norm) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x = gat_layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Separate fragments
        fragment1_mask = fragment_labels == 0
        fragment2_mask = fragment_labels == 1
        
        fragment1_features = x[fragment1_mask]
        fragment2_features = x[fragment2_mask]
        
        # Create pseudo-batch indices for each fragment
        fragment1_batch = batch[fragment1_mask]
        fragment2_batch = batch[fragment2_mask]
        
        # Pool each fragment separately
        if self.use_attention_pooling:
            fragment1_embedding = self.pooling(fragment1_features, fragment1_batch)
            fragment2_embedding = self.pooling(fragment2_features, fragment2_batch)
        else:
            fragment1_embedding = global_mean_pool(fragment1_features, fragment1_batch)
            fragment2_embedding = global_mean_pool(fragment2_features, fragment2_batch)
        
        return fragment1_embedding, fragment2_embedding
    
    def forward(self, data):
        """
        Forward pass supporting both single graphs and dual fragment graphs
        """
        # Check if this is a dual fragment graph
        if hasattr(data, 'fragment_labels') and data.fragment_labels is not None:
            # Dual fragment mode
            fragment1_emb, fragment2_emb = self.forward_dual_graph(data)
            
            # Combine fragment embeddings
            combined_embedding = torch.cat([fragment1_emb, fragment2_emb], dim=1)
            
        else:
            # Single graph mode - assume batch contains pairs of graphs
            # This would require custom batching logic
            raise NotImplementedError("Single graph mode requires custom batching for pairs")
        
        # Fragment interaction
        interaction_features = self.fragment_interaction(combined_embedding)
        
        # Generate outputs
        outputs = {}
        
        # Contrastive embeddings
        if self.use_contrastive:
            frag1_contrastive = self.contrastive_branch(fragment1_emb)
            frag2_contrastive = self.contrastive_branch(fragment2_emb)
            outputs['fragment1_embedding'] = frag1_contrastive
            outputs['fragment2_embedding'] = frag2_contrastive
        
        # Multi-task predictions
        if self.use_multitask:
            task_outputs = self.multitask_head(interaction_features)
            outputs.update(task_outputs)
        else:
            # Simple binary classification
            outputs['classification_logits'] = self.classifier(interaction_features)
        
        return outputs

class FragmentMatchingLoss(nn.Module):
    """
    Combined loss function for multi-task learning with contrastive loss
    """
    def __init__(self, 
                 classification_weight=1.0,
                 regression_weight=0.5,
                 contrastive_weight=0.3,
                 multi_class_weight=0.2,
                 temperature=0.07):
        super(FragmentMatchingLoss, self).__init__()
        
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.contrastive_weight = contrastive_weight
        self.multi_class_weight = multi_class_weight
        self.temperature = temperature
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def contrastive_loss(self, embeddings1, embeddings2, labels):
        """
        InfoNCE contrastive loss
        """
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Create labels for contrastive learning
        batch_size = embeddings1.shape[0]
        contrastive_labels = torch.arange(batch_size).to(embeddings1.device)
        
        # Symmetric contrastive loss
        loss_1to2 = self.ce_loss(similarity_matrix, contrastive_labels)
        loss_2to1 = self.ce_loss(similarity_matrix.T, contrastive_labels)
        
        return (loss_1to2 + loss_2to1) / 2
    
    def forward(self, outputs, targets):
        """
        Compute combined loss
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Dictionary containing ground truth values
                - 'binary_labels': Binary classification labels [0, 1]
                - 'regression_targets': Regression targets [0, 1]
                - 'multi_class_labels': Multi-class labels
        """
        total_loss = 0.0
        loss_components = {}
        
        # Binary classification loss (using BCEWithLogitsLoss for numerical stability)
        if 'classification_logits' in outputs and 'binary_labels' in targets:
            binary_labels = targets['binary_labels'].float().unsqueeze(1)
            classification_loss = self.bce_loss(outputs['classification_logits'], binary_labels)
            total_loss += self.classification_weight * classification_loss
            loss_components['classification_loss'] = classification_loss
        
        # Regression loss
        if 'regression_output' in outputs and 'regression_targets' in targets:
            regression_targets = targets['regression_targets'].float().unsqueeze(1)
            regression_loss = self.mse_loss(outputs['regression_output'], regression_targets)
            total_loss += self.regression_weight * regression_loss
            loss_components['regression_loss'] = regression_loss
        
        # Multi-class classification loss
        if 'multi_class_logits' in outputs and 'multi_class_labels' in targets:
            multi_class_loss = self.ce_loss(outputs['multi_class_logits'], targets['multi_class_labels'])
            total_loss += self.multi_class_weight * multi_class_loss
            loss_components['multi_class_loss'] = multi_class_loss
        
        # Contrastive loss
        if ('fragment1_embedding' in outputs and 
            'fragment2_embedding' in outputs and 
            'binary_labels' in targets):
            
            contrastive_loss = self.contrastive_loss(
                outputs['fragment1_embedding'],
                outputs['fragment2_embedding'],
                targets['binary_labels']
            )
            total_loss += self.contrastive_weight * contrastive_loss
            loss_components['contrastive_loss'] = contrastive_loss
        
        loss_components['total_loss'] = total_loss
        return total_loss, loss_components

# Example usage and model initialization
if __name__ == "__main__":
    # Model configuration
    model_config = {
        'node_features': 10,
        'hidden_dim': 128,
        'num_layers': 4,
        'heads': 8,
        'embedding_dim': 64,
        'num_classes': 3,
        'dropout': 0.1,
        'use_attention_pooling': True,
        'use_contrastive': True,
        'use_multitask': True
    }
    
    # Initialize model
    model = FragmentMatchingGNN(**model_config)
    
    # Initialize loss function
    loss_fn = FragmentMatchingLoss(
        classification_weight=1.0,
        regression_weight=0.5,
        contrastive_weight=0.3,
        multi_class_weight=0.2,
        temperature=0.07
    )
    
    print("Model initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Model summary
    print("\nModel Architecture:")
    print(model)