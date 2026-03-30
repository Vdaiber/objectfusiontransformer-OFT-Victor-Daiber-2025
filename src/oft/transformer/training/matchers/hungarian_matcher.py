# File: src/oft/transformer/training/matchers/hungarian_matcher.py
"""
Hungarian matcher for optimal assignment between predictions and ground truth in the Object Fusion Transformer pipeline.

Computes cost matrices in normalized ego coordinates and applies the Hungarian algorithm for bipartite matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple
import logging
import math
from oft.transformer.training.losses.matcher_costs import (
    center_cost_matrix_huber, 
    class_cost_matrix, 
    size_cost_matrix_l1, 
    giou_cost_matrix,
    angle_cost_matrix_huber
)

class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matcher using the Hungarian algorithm for prediction-to-ground-truth assignment.

    Args:
        loss_functions: Dict with required loss functions (must include 'center')
        cost_center: Weight for center cost
        cost_class: Weight for class cost
        cfg: Configuration dictionary
    """
    def __init__(self, cost_center: float = 1.0, cost_class: float = 1.0, cost_size: float = 0.0, cost_giou_bev: float = 0.0, cost_angle: float = 0.0, cfg: dict = None):
        """
        Initialize matcher with cost weights.

        Args:
            cost_center: Weight for center cost
            cost_class: Weight for class cost
            cost_size: Weight for size cost
            cost_giou_bev: Weight for GIoU cost (DETR-style geometric matching)
            cost_angle: Weight for angle cost (sin/cos Huber loss)
            cfg: Configuration dictionary
        """
        super().__init__()
        
        self.cost_center = cost_center
        self.cost_class = cost_class
        self.cost_size = cost_size
        self.cost_giou_bev = cost_giou_bev
        self.cost_angle = cost_angle
        self.cfg = cfg or {}
        
        # Store cost weights in a dictionary for easier access by diagnostics
        self.cost_weights = {
            'cost_center': self.cost_center,
            'cost_class': self.cost_class,
            'cost_size': self.cost_size,
            'cost_giou_bev': self.cost_giou_bev,
            'cost_angle': self.cost_angle,
        }
        
        if (self.cost_center == 0 and self.cost_class == 0 and self.cost_size == 0 and 
            self.cost_giou_bev == 0 and self.cost_angle == 0):
            raise ValueError("All cost weights cannot be zero")

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Perform optimal assignment between predictions and ground truth for each batch element.

        Args:
            predictions: Dict with 'pred_class_logits_batch' [B, N, C], 'pred_boxes_normalized' [B, N, 10]
            targets: Dict with 'gt_labels_b' (List[Tensor]), 'gt_boxes_normalized' [B, N_gt, 10], 'gt_valid_mask_b' (List[Tensor])
        Returns:
            Tuple: 
                - (assignment indices, matched masks)
                - Dict with average cost values for diagnostics
        """
        batch_size = len(targets['gt_labels_b']) if isinstance(targets['gt_labels_b'], list) else targets['gt_labels_b'].shape[0]
        indices = []
        matched_masks = []
        
        # Diagnostics: store costs for analysis
        all_costs_diag = {
            "cost_center": [], "cost_class": [], "cost_size": [], 
            "cost_giou_bev": [], "cost_angle": []
        }
        # End diagnostics
        
        for i in range(batch_size):
            # Extract predictions for current batch element
            pred_logits = predictions['pred_class_logits_batch'][i]  # Shape: [num_queries, num_classes] (unitless)
            pred_bbox = predictions['pred_boxes_normalized'][i]  # Shape: [num_queries, 10] (EGO, normalized)
            
            # Extract ground truth data for current batch element
            gt_class = targets['gt_labels_b'][i]  # Shape: [num_gt] (class indices)
            gt_bbox_normalized = targets['gt_boxes_b_normalized'][i]  # Shape: [num_gt, 10] (EGO, normalized)
            gt_valid_mask = targets['gt_valid_mask_b'][i]  # Shape: [num_gt] (boolean)

            # Filter valid ground truth indices (exclude padding)
            # Only valid ground truth objects participate in matching
            valid_gt_indices = torch.where(gt_valid_mask)[0]  # Valid ground truth indices (indices)
            
            # Handle empty ground truth case (no valid objects to match)
            # Return empty assignments and all-false matched mask
            if len(valid_gt_indices) == 0:
                indices.append((torch.empty(0, dtype=torch.long, device=pred_logits.device),
                              torch.empty(0, dtype=torch.long, device=pred_logits.device)))
                matched_masks.append(torch.zeros(pred_logits.shape[0], dtype=torch.bool, device=pred_logits.device))
                continue

            # Extract valid ground truth data for cost computation
            gt_class_valid = gt_class[valid_gt_indices]  # Shape: [num_valid_gt] (class indices)
            gt_bbox_valid_normalized = gt_bbox_normalized[valid_gt_indices]  # Shape: [num_valid_gt, 10] (EGO, normalized)

            # Get dimensions for cost matrix computation
            num_queries = pred_logits.shape[0]  # Number of model predictions (count)
            num_valid_gt = len(valid_gt_indices)  # Number of valid ground truth objects (count)

            # Store the original number of queries for consistent tensor shapes
            # This preserves the query dimension for proper tensor indexing later
            original_num_queries = predictions['pred_class_logits_batch'].shape[1]  # Total number of queries (count)

            # Compute center cost matrix using Huber loss (same function as LossCenter)
            # This ensures perfect consistency with training loss computation
            delta = self.cfg['loss']['huber']['delta']
            center_cost = center_cost_matrix_huber(pred_bbox, gt_bbox_valid_normalized, delta=delta)  # Shape: [num_queries, num_valid_gt] (EGO, normalized)

            # Compute class cost matrix using DETR's approach (negative log probabilities)
            class_cost = class_cost_matrix(pred_logits, gt_class_valid)  # Shape: [num_queries, num_valid_gt]

            # Compute size cost matrix using L1 distance
            size_cost = size_cost_matrix_l1(pred_bbox[:, 3:6], gt_bbox_valid_normalized[:, 3:6])  # Shape: [num_queries, num_valid_gt]
            
            # Compute GIoU cost matrix ONLY if weight > 0 for performance optimization
            if self.cost_giou_bev > 0:
                giou_cost = giou_cost_matrix(pred_bbox, gt_bbox_valid_normalized)  # Shape: [num_queries, num_valid_gt]
            else:
                # Skip expensive GIoU computation when weight is 0 - use zero matrix with correct shape
                giou_cost = torch.zeros_like(center_cost)  # Shape: [num_queries, num_valid_gt]
            
            # Compute angle cost matrix ONLY if weight > 0 for performance optimization
            if self.cost_angle > 0:
                angle_cost = angle_cost_matrix_huber(pred_bbox, gt_bbox_valid_normalized, delta=delta)  # Shape: [num_queries, num_valid_gt]
            else:
                # Skip computation when weight is 0 - use zero matrix with correct shape
                angle_cost = torch.zeros_like(center_cost)  # Shape: [num_queries, num_valid_gt]

            # COMPLETE cost matrix with ALL components (weights of 0.0 = no influence)
            cost_matrix = (self.cost_center * center_cost + 
                          self.cost_class * class_cost + 
                          self.cost_size * size_cost +
                          self.cost_giou_bev * giou_cost +
                          self.cost_angle * angle_cost)  # Shape: [num_queries, num_valid_gt] (unitless)
            
            # Diagnostics: real computed values
            all_costs_diag["cost_center"].append(center_cost.mean())
            all_costs_diag["cost_class"].append(class_cost.mean())
            all_costs_diag["cost_size"].append(size_cost.mean())
            all_costs_diag["cost_giou_bev"].append(giou_cost.mean())
            all_costs_diag["cost_angle"].append(angle_cost.mean())  # Will be 0.0 if cost_angle = 0
            # End diagnostics
            
            # Perform Hungarian algorithm for optimal bipartite assignment
            # This finds the assignment that minimizes total cost across all pairs
            src_indices, tgt_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())  # Optimal assignment (indices)
            src_indices = torch.from_numpy(src_indices).to(pred_logits.device)  # Convert to tensor (indices)
            tgt_indices = torch.from_numpy(tgt_indices).to(pred_logits.device)  # Convert to tensor (indices)
            tgt_indices = valid_gt_indices[tgt_indices]  # Map back to original ground truth indices (indices)

            # Ensure proper tensor types and handle edge cases
            # This prevents type errors and ensures consistent return format
            if not isinstance(src_indices, torch.Tensor):
                src_indices = torch.tensor(src_indices, dtype=torch.long, device=pred_logits.device)  # Convert to tensor (indices)
            if not isinstance(tgt_indices, torch.Tensor):
                tgt_indices = torch.tensor(tgt_indices, dtype=torch.long, device=pred_logits.device)  # Convert to tensor (indices)
            if src_indices.numel() == 0:
                src_indices = torch.empty(0, dtype=torch.long, device=pred_logits.device)  # Empty tensor (indices)
            if tgt_indices.numel() == 0:
                tgt_indices = torch.empty(0, dtype=torch.long, device=pred_logits.device)  # Empty tensor (indices)

            # Store assignment results for current batch element
            indices.append((src_indices, tgt_indices))  # Store optimal assignment (prediction indices, ground truth indices)
            
            # Create matched mask indicating which predictions were assigned
            # This is used by downstream loss computation to identify matched predictions
            matched_mask = torch.zeros(original_num_queries, dtype=torch.bool, device=pred_logits.device)  # Initialize mask (boolean)
            if src_indices.numel() > 0:
                matched_mask[src_indices] = True  # Mark matched predictions as True (boolean)
            matched_masks.append(matched_mask)  # Store matched mask for current batch element (boolean)
            
        # Diagnostics: average costs over batch
        avg_costs_diag = {k: torch.mean(torch.stack(v)) if v else torch.tensor(0.0) for k, v in all_costs_diag.items()}
        # End diagnostics
            
        return (indices, matched_masks), avg_costs_diag
