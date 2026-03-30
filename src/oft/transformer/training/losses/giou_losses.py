# File: src/oft/transformer/training/losses/giou_losses.py
"""
Geometric IoU Loss Functions for Object Detection and Tracking in Object Fusion Transformer.

This module provides GIoU (Generalized Intersection over Union) loss implementations
for object detection tasks in the Object Fusion Transformer pipeline. GIoU extends 
the standard IoU metric by considering the minimum enclosing rectangle of two boxes, 
providing a more robust similarity measure that penalizes boxes that are far apart, 
even when their IoU is zero.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Any, Optional
from ...utils.iou import generalized_box_iou_bev
from ...utils.geometry_utils import sin_cos_to_yaw

def _get_src_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract source permutation indices from Hungarian matching results.
    
    Args:
        indices: Hungarian matching indices as list of (src_idx, tgt_idx) tuples
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing batch indices and source indices
    """
    # Create batch indices by repeating batch number for each source index
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    # Concatenate all source indices into a single tensor for parallel processing
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract target permutation indices from Hungarian matching results.
    
    Args:
        indices: Hungarian matching indices as list of (src_idx, tgt_idx) tuples
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing batch indices and target indices
    """
    # Create batch indices by repeating batch number for each target index
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    # Concatenate all target indices into a single tensor for parallel processing
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

class LossGIoUBEV(nn.Module):
    """
    Computes the Generalized IoU (GIoU) loss between predicted and target bounding boxes in Bird's-Eye-View (BEV).
    
    This loss function operates in the denormalized ego coordinate system to ensure physical
    consistency in loss computation. It converts normalized model predictions back to physical
    units before computing GIoU.
    
    Args:
        cfg: Configuration dictionary containing loss parameters
    """
    def __init__(self, cfg: Dict[str, Any], **kwargs):
        """Initialize the LossGIoUBEV module with configuration parameters.
        
        Args:
            cfg: Configuration dictionary containing loss parameters
        """
        super().__init__()
        # Extract IoU calculation mode from configuration
        self.use_exact_iou = cfg["loss"]["iou"]["use_exact_iou"]

    def _convert_to_7d_box(self, box_10d: torch.Tensor) -> torch.Tensor:
        """Converts a 10D box representation to a 7D box for IoU computation.
        
        Args:
            box_10d: 10D box representation with shape [..., 10]
                
        Returns:
            torch.Tensor: 7D box representation with shape [..., 7]
        """
        # Use only the first 8 dimensions for box conversion (ignore velocity for IoU)
        box_8d = box_10d[..., :8]
        centers = box_8d[..., :3]  # [..., 3] - center coordinates
        dims = box_8d[..., 3:6]    # [..., 3] - dimensions in log-space
        sin_yaw = box_8d[..., 6]   # [..., 1] - sin(yaw)
        cos_yaw = box_8d[..., 7]   # [..., 1] - cos(yaw)
        
        # Convert sin/cos yaw representation to single yaw angle using atan2
        yaw = torch.atan2(sin_yaw, cos_yaw)  # [..., 1] - yaw angle in radians
        
        # Combine all components into 7D box representation
        return torch.cat([centers, dims, yaw.unsqueeze(-1)], dim=-1)

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, Any], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute the Generalized IoU loss between predicted and target boxes in BEV (Bird's Eye View).
        
        Args:
            predictions: Model prediction dictionary containing normalized box predictions
            targets: Ground truth target dictionary containing physical ground truth boxes
            indices: Hungarian matching results as list of (src_idx, tgt_idx) tuples
            num_boxes: Number of valid boxes for loss normalization
                
        Returns:
            torch.Tensor: GIoU loss value (scalar) computed as (1 - GIoU) averaged over matched boxes
        """
        # Extract source permutation indices from Hungarian matching results
        batch_idx, src_idx = _get_src_permutation_idx(indices)
        batch_idx_tgt, tgt_idx = _get_tgt_permutation_idx(indices)

        # Handle empty matching case (no valid matches found)
        if len(src_idx) == 0:
            return torch.tensor(0.0, device=predictions['pred_boxes_normalized'].device, dtype=predictions['pred_boxes_normalized'].dtype, requires_grad=True)

        # --- DENORMALIZATION OF PREDICTED BOXES ---
        # Extract matched predictions from model predictions (EGO, normalized)
        src_boxes_norm = predictions['pred_boxes_normalized'][batch_idx, src_idx]  # [num_matched, 10] (EGO, normalized)
        coords_norm = src_boxes_norm[:, :3]    # [num_matched, 3] - center coordinates (EGO, normalized, [0,1] range)
        dims_norm = src_boxes_norm[:, 3:6]     # [num_matched, 3] - dimensions in log-space (EGO, normalized)
        sin_yaw = src_boxes_norm[:, 6]         # [num_matched] - sin(yaw) (EGO, normalized)
        cos_yaw = src_boxes_norm[:, 7]         # [num_matched] - cos(yaw) (EGO, normalized)

        # Import denormalization utilities for coordinate and dimension conversion
        from oft.transformer.utils.normalization_utils import denormalize_coordinates, denormalize_dimensions, get_global_normalizer
        from oft.transformer.utils.geometry_utils import sin_cos_to_yaw
        from oft.transformer.utils.iou import generalized_box_iou_bev

        # Extract point_cloud_range from global normalizer for coordinate denormalization
        point_cloud_range = torch.tensor(get_global_normalizer().stats['metadata']['point_cloud_range'], 
                                        device=coords_norm.device, dtype=coords_norm.dtype)
        
        # Denormalize coordinates from [0,1] range to physical meters in ego frame
        coords_meter = denormalize_coordinates(coords_norm, point_cloud_range)  # [num_matched, 3] (EGO, unnormalized, meters)
        
        # Denormalize dimensions from log-space to physical meters in ego frame
        dims_meter = denormalize_dimensions(dims_norm)  # [num_matched, 3] (EGO, unnormalized, meters)
        
        # Convert sin/cos yaw representation to single yaw angle in radians
        #Vectorized operation instead of inefficient list comprehension
        yaw = torch.atan2(sin_yaw, cos_yaw)  # [num_matched] (EGO, unnormalized, radians)
        
        # Combine denormalized components into 7D box representation for IoU computation
        pred_boxes_7d = torch.cat([coords_meter, dims_meter, yaw.unsqueeze(-1)], dim=-1)  # [num_matched, 7] (EGO, unnormalized)

        # --- GROUND TRUTH BOXES: PHYSICAL, EGO-METER ---
        # Extract matched ground truth boxes in physical ego coordinate system
        gt_boxes_phys = targets['gt_boxes_b_physical'][batch_idx_tgt, tgt_idx, :7]  # [num_matched, 7] (EGO, unnormalized, meters/radians)

        # --- GIoU COMPUTATION ---
        # Compute Generalized IoU between denormalized predictions and physical ground truth
        giou = generalized_box_iou_bev(pred_boxes_7d, gt_boxes_phys)  # [num_matched] (unnormalized, [0,1] range)
        
        # Compute final loss as (1 - GIoU) averaged over all matched boxes
        loss_giou = (1 - giou).sum() / num_boxes  # scalar (unnormalized)
        return loss_giou 