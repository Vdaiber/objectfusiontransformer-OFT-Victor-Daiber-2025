# File: src/oft/transformer/training/losses/classification_losses.py
"""
Classification Loss Functions for Object Fusion Transformer Training Pipeline.

This module implements individual classification loss functions for object class prediction,
attribute classification, and duplicate detection in the Object Fusion Transformer pipeline.
The functions operate on Hungarian-matched predictions and ground truth labels, providing
modular loss computation for multi-task learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance.
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter to down-weight easy examples (default: 2.0)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute focal loss between inputs and targets.
        
        Args:
            inputs: Predicted logits [N, C]
            targets: Ground truth class indices [N]
            weight: Optional class weights [C]
            
        Returns:
            Focal loss value (scalar tensor)
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class LossClass(nn.Module):
    """Classification loss with support for both CrossEntropy and FocalLoss.
    
    Implements classification loss for object class prediction with configurable
    loss function type. Supports both standard CrossEntropy loss and FocalLoss
    for handling class imbalance scenarios.
    
    Args:
        cfg: Configuration dictionary containing model parameters and loss coefficients
        eos_coef: End-of-sequence coefficient for handling no-object cases
    """
    
    def __init__(self, cfg: Dict[str, Any], eos_coef: float = 0.1, **kwargs):
        """Initialize the classification loss module.
        
        Args:
            cfg: Configuration dictionary containing model parameters and loss coefficients
            eos_coef: End-of-sequence coefficient for handling no-object cases
            **kwargs: Additional keyword arguments (unused but maintained for compatibility)
        """
        super().__init__()
        # Store configuration dictionary for parameter access during forward pass
        self.cfg = cfg
        
        # Load EOS coefficient from configuration with fallback to default value
        self.eos_coef = cfg['loss']['class_eos_coefficient']
        
        # Extract number of classes from configuration with fallback to default
        self.num_classes = cfg['model']['decoder']['num_classes'] + 1  # Add no-object class to total count
        
        # Load classification loss type from configuration
        self.classification_loss_type = cfg['loss']['classification_loss']
        
        # Initialize FocalLoss if specified in configuration
        if self.classification_loss_type == 'FocalLoss':
            # Strictly require class-specific focal params (no fallback)
            focal_alpha = cfg['loss']['focal_loss_alpha_class']
            focal_gamma = cfg['loss']['focal_loss_gamma_class']
            self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.focal_loss_fn = None
            
        # Load manual class weights for WeightedCrossEntropy
        self.manual_class_weights = self._load_manual_class_weights(cfg)
        
    def _load_manual_class_weights(self, cfg: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Load manual class weights from configuration.
        
        Args:
            cfg: Configuration dictionary containing manual_class_weights section
            
        Returns:
            torch.Tensor: Class weights tensor with shape [num_classes + 1] or None if disabled
        """
        manual_cfg = cfg['loss']['manual_class_weights']
        if not manual_cfg['enabled']:
            return None
            
        # Get class names from dataset config (no_object weight is handled automatically)
        dataset_class_names = cfg['dataset']['class_names']
        all_class_names = dataset_class_names + ['no_object']  # For display only
        
        # Extract manual weights from config (only for real classes)
        manual_weights_dict = manual_cfg['weights']
        
        # Build weights tensor: manual weights for real classes + placeholder for no_object
        weights = []
        for class_name in dataset_class_names:  # Only real classes
            weight = manual_weights_dict[class_name]
            weights.append(weight)
        weights.append(1.0)  # No-object class weight (will be overridden by eos_coef during loss computation)
            
        weights_tensor = torch.tensor(weights, dtype=torch.float64)
        
        print(f"✓ Manual class weights loaded:")
        for i, (name, weight) in enumerate(zip(all_class_names, weights)):
            print(f"  [{i:2d}] {name:15s}: {weight:6.2f}")
            
        return weights_tensor
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, Any], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: torch.Tensor) -> torch.Tensor:
        """Compute classification loss for ALL predictions (matched + unmatched).
        
        In the new Hungarian-based approach, ALL predictions have targets:
        - Matched predictions (Hungarian winners) get actual GT class labels
        - Unmatched predictions (Hungarian losers) get no_object class label
        
        Args:
            predictions: Model predictions containing classification logits
            targets: Ground truth data containing class labels (modified with no_object assignments)
            indices: Hungarian matching indices (used for debugging only)
            num_boxes: Number of ground truth boxes for normalization
        
        Returns:
            torch.Tensor: Classification loss value (scalar tensor with gradient support)
        """
        # Extract logits and targets - now ALL predictions have corresponding targets
        src_logits = predictions['pred_class_logits_batch']  # [B, N, C]
        target_classes = targets['gt_labels_b']  # [B, N] - includes no_object for unmatched
        
        # Flatten tensors for loss computation
        src_logits_flat = src_logits.flatten(0, 1)  # [B*N, C]
        target_classes_flat = target_classes.flatten(0, 1)  # [B*N]
        
        # Create class weighting tensor with UNIFIED logic
        if self.manual_class_weights is not None and self.classification_loss_type == 'WeightedCrossEntropy':
            # Use manual class weights for real classes + unified no_object weight via eos_coef
            class_weights = self.manual_class_weights.to(device=src_logits.device, dtype=src_logits.dtype)
            class_weights[-1] = self.eos_coef
        else:
            # Fallback: All classes get weight 1.0, with no_object controlled by eos_coef
            class_weights = torch.ones(src_logits.shape[-1], device=src_logits.device, dtype=src_logits.dtype)
            class_weights[-1] = self.eos_coef
        
        # Compute loss based on configured loss type over ALL predictions
        if self.classification_loss_type == 'FocalLoss' and self.focal_loss_fn is not None:
            # Use FocalLoss for class imbalance scenarios
            loss = self.focal_loss_fn(src_logits_flat, target_classes_flat, weight=class_weights)
        elif self.classification_loss_type == 'WeightedCrossEntropy':
            # Use Weighted CrossEntropy with manual class weights
            loss = F.cross_entropy(src_logits_flat, target_classes_flat, weight=class_weights)
        else:
            # Use standard CrossEntropy loss  
            loss = F.cross_entropy(src_logits_flat, target_classes_flat, weight=class_weights)
        
       # # DEBUG: Print classification analysis for monitoring
       # pred_classes_flat = src_logits_flat.argmax(dim=-1)
       # accuracy = (pred_classes_flat == target_classes_flat).float().mean()
       # no_object_class_id = src_logits.shape[-1] - 1
        
        # Count class distribution for analysis
        #num_no_object_targets = (target_classes_flat == no_object_class_id).sum().item()
        #num_real_object_targets = len(target_classes_flat) - num_no_object_targets
        #num_no_object_preds = (pred_classes_flat == no_object_class_id).sum().item()
        
        # print(f"[DEBUG] UNIFIED CLASS LOSS ({self.classification_loss_type}): {loss.item():.6f}")
        # print(f"  - Total predictions: {len(src_logits_flat)}")
        # print(f"  - Real object targets: {num_real_object_targets}, no_object targets: {num_no_object_targets}")
        # print(f"  - no_object predictions: {num_no_object_preds}")
        # print(f"  - Overall accuracy: {accuracy.item():.4f}")
        
        return loss


class LossAttributes(nn.Module):
    """CrossEntropy or FocalLoss for object attribute prediction."""
    
    def __init__(self, cfg: Dict[str, Any], **kwargs):
        """Initialize the attribute classification loss module."""
        super().__init__()
        self.cfg = cfg
        self.num_attribute_classes = cfg['model']['decoder']['num_attribute_classes'] + 1
        
        # Allow configurable loss type for attributes
        self.attribute_loss_type = cfg['loss'].get('attribute_loss', 'CrossEntropy') # Default to CE
        
        if self.attribute_loss_type == 'FocalLoss':
            # Strictly require attribute-specific focal params (no fallback)
            focal_alpha = cfg['loss']['focal_loss_alpha_attr']
            focal_gamma = cfg['loss']['focal_loss_gamma_attr']
            self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.focal_loss_fn = None
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, Any], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: torch.Tensor) -> torch.Tensor:
        """Compute attribute loss for Hungarian-matched predictions.
        
        Args:
            predictions: Model predictions containing attribute logits
            targets: Ground truth data containing attribute labels
            indices: Hungarian matching indices as list of (src_idx, tgt_idx) tuples
            num_boxes: Number of ground truth boxes for normalization
        
        Returns:
            torch.Tensor: Attribute classification loss value (scalar tensor with gradient support)
        """
        # Extract source permutation indices from Hungarian matching results
        batch_idx, src_idx = _get_src_permutation_idx(indices)
        
        # Validate indices to prevent out-of-bounds access in tensor indexing
        batch_size, num_queries, num_attributes = predictions['pred_attributes_logits_batch'].shape
        max_idx = batch_size * num_queries - 1
        valid_mask = src_idx <= max_idx
        
        # Handle out-of-bounds indices by filtering and logging warning
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum().item()
            print(f"WARNING: Found {invalid_count} indices out of bounds. Max index: {max_idx}, Max src_idx: {src_idx.max().item()}")
            src_idx = src_idx[valid_mask]
            batch_idx = batch_idx[valid_mask]
        
        # Return differentiable zero loss if no valid matches exist
        if len(src_idx) == 0:
            return torch.tensor(0.0, device=predictions['pred_attributes_logits_batch'].device, dtype=predictions['pred_attributes_logits_batch'].dtype, requires_grad=True)
        
        # Convert flat indices to batch and query indices for tensor indexing
        batch_indices = src_idx // num_queries
        query_indices = src_idx % num_queries
        pred_attributes = predictions['pred_attributes_logits_batch'][batch_indices, query_indices]
        
        # Extract target attributes from ground truth using Hungarian matching indices
        target_attributes = torch.cat([t[i] for t, (_, i) in zip(targets['gt_attributes_b'], indices)], dim=0)
        
        # Ensure target attributes match the number of valid source indices
        if len(src_idx) < len(target_attributes):
            target_attributes = target_attributes[:len(src_idx)]
        
        # Compute CrossEntropyLoss for multi-class attribute classification with CONSISTENT weighting
        attr_eos_coef = self.cfg['loss']['attribute_eos_coefficient']
        attr_weights = torch.ones(pred_attributes.shape[-1], device=pred_attributes.device, dtype=pred_attributes.dtype)
        attr_weights[-1] = attr_eos_coef

        if self.attribute_loss_type == 'FocalLoss' and self.focal_loss_fn is not None:
            loss = self.focal_loss_fn(pred_attributes, target_attributes, weight=attr_weights)
        else:
            loss = F.cross_entropy(pred_attributes, target_attributes, reduction='mean', weight=attr_weights)
        
        # # DEBUG: Print attribute analysis for GT-input testing
        # if len(pred_attributes) > 0:
        #     pred_attrs = pred_attributes.argmax(dim=-1)
        #     accuracy = (pred_attrs == target_attributes).float().mean()
        #     print(f"[DEBUG] ATTRIBUTE LOSS (CrossEntropy): {loss.item():.6f}")
        #     print(f"  - Num matched: {len(pred_attributes)}")
        #     print(f"  - Accuracy: {accuracy.item():.4f}")
            
        #     # Show RANDOM 5 samples instead of first 5 to avoid systematic bias
        #     if len(pred_attributes) >= 5:
        #         import random
        #         sample_indices = random.sample(range(len(pred_attributes)), 5)
        #         sample_indices.sort()  # Keep order for readability
        #         print(f"  - Pred attributes (random sample): {pred_attrs[sample_indices]}")
        #         print(f"  - Target attributes (random sample): {target_attributes[sample_indices]}")
        #         print(f"  - Sample indices from {len(pred_attributes)}: {sample_indices}")
        #     else:
        #         print(f"  - Pred attributes (all): {pred_attrs}")
        #         print(f"  - Target attributes (all): {target_attributes}")
        
        return loss


def _get_src_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract source permutation indices from Hungarian matching results.
    
    Converts Hungarian matching indices into flat batch and source indices for
    efficient tensor indexing across all batch elements.
    
    Args:
        indices: Hungarian matching indices as list of (src_idx, tgt_idx) tuples
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing batch indices and source indices
    """
    # Create batch indices by repeating batch number for each source index
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    
    # Concatenate all source indices into a single tensor
    src_idx = torch.cat([src for (src, _) in indices])
    
    return batch_idx, src_idx 