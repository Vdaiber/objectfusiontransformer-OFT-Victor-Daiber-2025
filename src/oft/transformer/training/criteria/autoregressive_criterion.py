# File: src/oft/transformer/training/criteria/autoregressive_criterion.py
"""
SetCriterion for Autoregressive Object Fusion Transformer Pipeline.

This module implements the SetCriterion class, orchestrating all loss functions for
the autoregressive pipeline with uncertainty weighting and Hungarian matching. The
criterion coordinates multiple loss components including classification, regression,
attribute prediction, and geometric similarity losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from ..matchers.hungarian_matcher import HungarianMatcher
from ..losses.classification_losses import LossClass, LossAttributes
from ..losses.regression_losses import LossCenter, LossSize, LossAngle, LossVelocity


class SetCriterion(nn.Module):
    """Comprehensive loss orchestrator for autoregressive transformer-based object detection.
    
    Implements the complete multi-task loss formulation combining:
    - Classification loss with end-of-sequence (no-object) handling
    - Huber regression losses for center, size, angle, and velocity offsets  
    - Attribute prediction losses for object properties
    - Hungarian bipartite matching for optimal assignment between predictions and ground truth
    
    The criterion addresses the set prediction problem where the number of predictions
    may vary from ground truth objects. Hungarian matching ensures optimal one-to-one
    assignment before loss computation, essential for transformer-based detection models.
    
    Mathematical formulation:
        L_total = Σ_i w_i * L_i(Hungarian_match(pred, gt))
    
    where L_i are individual loss components and w_i are learned or configured weights.
    
    Args:
        weight_dict: Loss component weights for multi-task balancing
        losses: Active loss function identifiers for modular training
        matcher: Hungarian matcher for optimal bipartite assignment
        eos_coef: No-object class weight for classification imbalance
        cfg: Configuration containing loss-specific hyperparameters
    """
    
    def __init__(self, weight_dict: Dict[str, float], losses: List[str], matcher: HungarianMatcher = None, eos_coef: float = 0.1, cfg: Dict[str, Any] = None):
        """Initialize the set criterion with loss configuration and Hungarian matcher.
        
        Args:
            weight_dict: Dictionary mapping loss names to their corresponding weights for loss balancing
            losses: List of loss function names to compute during training
            matcher: Hungarian matcher instance for optimal bipartite matching between predictions and ground truth
            eos_coef: End-of-sequence coefficient for handling no-object cases in classification losses
            cfg: Configuration dictionary containing model parameters and loss-specific configurations
        """
        super().__init__()
        
        # Store configuration parameters for loss computation and Hungarian matching
        self.weight_dict = weight_dict
        self.losses_to_compute = losses
        self.eos_coef = eos_coef
        self.cfg = cfg
        
        # Define mapping from loss names to their corresponding loss classes for modular initialization
        loss_class_map = {
            'center': LossCenter,      # Regression: Center offset prediction (x, y, z) (EGO, normalized)
            'size': LossSize,          # Regression: Size offset prediction (w, l, h) (EGO, normalized)  
            'angle': LossAngle,        # Regression: Angle offset prediction (sin, cos) (EGO, normalized)
            'velocity': LossVelocity,  # Regression: Velocity offset prediction (vx, vy) (EGO, normalized)
            'class': LossClass,        # Classification: Object class prediction (logits) (unnormalized)
            'attributes': LossAttributes,  # Classification: Object attributes prediction (logits) (unnormalized)
        }
        
        # Initialize loss modules dynamically based on configuration
        self.loss_modules = nn.ModuleDict({})
        for loss_name in self.losses_to_compute:
            if loss_name in loss_class_map:
                # Special handling for classification loss requiring eos_coef parameter
                if loss_name == 'class':
                    self.loss_modules[loss_name] = loss_class_map[loss_name](
                        cfg=self.cfg, eos_coef=self.eos_coef
                    )
                else:
                    # All other loss modules use the standard configuration interface
                    self.loss_modules[loss_name] = loss_class_map[loss_name](cfg=self.cfg)
        
        
        # Create or update the matcher with cost weights (no loss functions needed)
        if matcher is None:
            # Create new matcher using config cost weights
            cost_center = self.cfg['loss']['cost_center']
            cost_class = self.cfg['loss']['cost_class']
            cost_size = self.cfg['loss'].get('cost_size', 0.0)
            cost_giou_bev = self.cfg['loss'].get('cost_giou_bev', 0.0)
            cost_angle = self.cfg['loss'].get('cost_angle', 0.0)  # Angle cost from config
            
            print("  > Matcher Costs:")
            print(f"    - cost_center: {cost_center}")
            print(f"    - cost_class: {cost_class}")
            print(f"    - cost_size: {cost_size}")
            print(f"    - cost_giou_bev: {cost_giou_bev}")
            print(f"    - cost_angle: {cost_angle}")  # Log angle cost
            print("----------------------------------------------------")

            self.matcher = HungarianMatcher(
                cost_center=cost_center,
                cost_class=cost_class,
                cost_size=cost_size,
                cost_giou_bev=cost_giou_bev,
                cost_angle=cost_angle,  # Pass angle cost to matcher
                cfg=self.cfg
            )
        else:
            # Use provided matcher instance
            self.matcher = matcher

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute weighted losses for the autoregressive object fusion model.
        
        Performs bipartite matching between predictions and ground truth using the Hungarian algorithm,
        then computes individual losses for each task and returns their weighted sum. Creates a unified
        target structure where matched predictions get GT classes and unmatched predictions get no_object.
        
        Args:
            predictions: Model predictions containing classification logits, box offsets, and attributes
            targets: Ground truth data containing labels, boxes, attributes, and valid masks
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing computed losses with standardized naming
        """
        # Check if pre-computed Hungarian matching indices are available
        if 'indices' in targets:
            # Use pre-computed indices (e.g., from previous forward pass or external matcher)
            indices = targets['indices']
        else:
            # Compute Hungarian matching indices using the matcher
            # Matcher returns ((indices, matched_masks), avg_costs)
            (indices, matched_masks), avg_costs = self.matcher(predictions, targets)
        
        # Create target_classes_with_no_object
        # Hungarian winners get GT class, unmatched predictions get no_object
        batch_size = predictions['pred_class_logits_batch'].shape[0]
        num_queries = predictions['pred_class_logits_batch'].shape[1]
        num_classes = predictions['pred_class_logits_batch'].shape[2]
        no_object_class_id = num_classes - 1  # Last class is no_object (background)
        
        # Initialize all predictions as no_object class
        target_classes_with_no_object = torch.full(
            (batch_size, num_queries), 
            no_object_class_id, 
            dtype=torch.long, 
            device=predictions['pred_class_logits_batch'].device
        )
        
        # For matched predictions (Hungarian winners), set actual GT class
        # *** VECTORIZED IMPLEMENTATION: No explicit batch loops for better GPU parallelization ***
        
        # Collect all batch_indices, src_indices, and corresponding GT classes
        all_batch_indices = []
        all_src_indices = []
        all_gt_classes = []
        
        for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
            if len(src_indices) > 0:
                # Create batch indices tensor for this batch
                batch_indices_for_this_batch = torch.full_like(src_indices, batch_idx, 
                                                             device=target_classes_with_no_object.device)
                
                # Extract GT classes for matched targets
                if isinstance(targets['gt_labels_b'], list):
                    gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
                else:
                    gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
                
                # Collect indices and GT classes
                all_batch_indices.append(batch_indices_for_this_batch)
                all_src_indices.append(src_indices)
                all_gt_classes.append(gt_classes)
        
        # Vectorized assignment: process all batches simultaneously
        if len(all_batch_indices) > 0:
            flat_batch_indices = torch.cat(all_batch_indices)  # [total_matches]
            flat_src_indices = torch.cat(all_src_indices)      # [total_matches]
            flat_gt_classes = torch.cat(all_gt_classes)        # [total_matches]
            
            # Single vectorized assignment (replaces all individual batch assignments)
            target_classes_with_no_object[flat_batch_indices, flat_src_indices] = flat_gt_classes
        
        # Create modified targets with unified class structure
        targets_modified = targets.copy()
        targets_modified['gt_labels_b'] = target_classes_with_no_object
        
        # Calculate total number of ground truth boxes across all batch elements
        if isinstance(targets['gt_labels_b'], list):
            num_boxes = sum(len(t) for t in targets['gt_labels_b'])
        else:
            # For tensor format, count valid entries (not no_object for original targets)
            num_boxes = sum((targets['gt_valid_mask_b'][i]).sum().item() for i in range(batch_size))
        
        # Convert to tensor with gradient support for differentiable loss computation
        num_boxes = torch.tensor([num_boxes], dtype=torch.float, device=predictions['pred_class_logits_batch'].device, requires_grad=True)
        
        # Handle edge case: empty batch with no ground truth objects
        if num_boxes.item() < 1:
            # Compute classification loss against all-no_object targets to learn background properly
            losses = {}
            loss_values = []
            # Only class loss uses targets_modified (contains no_object for all unmatched)
            if 'class' in self.loss_modules:
                class_loss = self.loss_modules['class'](predictions, targets_modified, [], num_boxes)
                losses['loss_class'] = class_loss
                loss_values.append(self.weight_dict.get('loss_class', 1.0) * class_loss)
            # Other losses contribute differentiable zero
            device = predictions['pred_class_logits_batch'].device
            dtype = predictions['pred_class_logits_batch'].dtype
            zero = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
            if 'center' in self.loss_modules:
                losses['loss_center'] = zero
            if 'size' in self.loss_modules:
                losses['loss_size'] = zero
            if 'angle' in self.loss_modules:
                losses['loss_angle'] = zero
            if 'velocity' in self.loss_modules:
                losses['loss_velocity'] = zero
            if 'attributes' in self.loss_modules:
                losses['loss_attributes'] = zero
            # Total loss = weighted class loss only
            losses['loss_total'] = sum(loss_values) if len(loss_values) > 0 else zero
            return losses
        
        # Initialize dictionaries to store computed losses and their values
        losses = {}
        loss_values = []
        
        # Compute individual loss components and apply configured weights exactly once

        for loss_name in self.losses_to_compute:
            if loss_name in self.loss_modules:
                if loss_name == 'class':
                    loss_val = self.loss_modules[loss_name](predictions, targets_modified, indices, num_boxes)
                else:
                    loss_val = self.loss_modules[loss_name](predictions, targets, indices, num_boxes)

                key_name = 'loss_' + loss_name
                losses[key_name] = loss_val  # keep unweighted for transparency
                
                weight = self.weight_dict.get(key_name, 1.0)
                loss_values.append(weight * loss_val)
        
        # Compute weighted total loss using fixed weights from configuration
        if len(loss_values) > 0:
            total_loss = sum(loss_values)
            losses['loss_total'] = total_loss
        else:
            # Fallback: sum all losses if no individual losses were computed
            losses['loss_total'] = sum(losses.values())
        
        # Return complete loss dictionary for training loop
        return losses 