# File: src/oft/transformer/training/losses/regression_losses.py
"""
Regression loss functions for bounding box parameters in the Object Fusion Transformer pipeline.

Implements Huber losses for center, size, angle, and velocity regression in normalized ego coordinates.
All losses operate on Hungarian-matched predictions and targets, supporting robust training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
import math

class LossCenter(nn.Module):
    """
    Huber loss for bounding box center prediction (x, y, z) in normalized ego coordinates.

    Args:
        cfg: Configuration dictionary with loss parameters.
    """
    def __init__(self, cfg: Dict[str, Any], **kwargs):
        """
        Initialize center regression loss module.

        Args:
            cfg: Configuration dictionary with loss parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.huber_delta = cfg['loss']['huber']['delta']
        
        # Centralized normalizer for consistent normalization across loss functions
        from oft.transformer.utils.normalization_utils import get_global_normalizer
        self.normalizer = get_global_normalizer()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, Any], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss for matched bounding box centers (x, y, z).

        Args:
            predictions: Dict with 'pred_boxes_normalized' [B, N, 10]
            targets: Dict with 'gt_boxes_b_normalized' [B, N_gt, 10]
            indices: List of (src_idx, tgt_idx) tuples from Hungarian matcher
            num_boxes: Number of boxes (unused)
        Returns:
            torch.Tensor: Scalar center loss
        """
        batch_idx, src_idx = _get_src_permutation_idx(indices)
        batch_size, num_queries, _ = predictions['pred_boxes_normalized'].shape
        max_idx = batch_size * num_queries - 1
        valid_mask = src_idx <= max_idx
        if not valid_mask.all():
            src_idx = src_idx[valid_mask]
            batch_idx = batch_idx[valid_mask]
        if len(src_idx) == 0:
            return torch.tensor(0.0, device=predictions['pred_boxes_normalized'].device, dtype=predictions['pred_boxes_normalized'].dtype, requires_grad=True)
        
        # Convert flat indices to batch and query indices for tensor indexing
        # This enables efficient extraction of matched predictions from batched tensors
        batch_indices = src_idx // num_queries  # Batch index for each matched prediction
        query_indices = src_idx % num_queries   # Query index for each matched prediction
        
        # Extract center coordinates (x, y, z) from predicted boxes (normalized)
        # Format: [x_norm, y_norm, z_norm] (EGO, normalized) - range [-1, 1]
        src_boxes_normalized = predictions['pred_boxes_normalized'][batch_indices, query_indices][:, :3]  # Shape: [num_matched, 3]
        
        # Extract target center coordinates using target indices (normalized)
        # Format: [x_norm, y_norm, z_norm] (EGO, normalized) - range [-1, 1]
        batch_idx_tgt, tgt_idx = _get_tgt_permutation_idx(indices)  # Extract target indices
        target_boxes_normalized = targets['gt_boxes_b_normalized'][batch_idx_tgt, tgt_idx][:, :3]  # Shape: [num_matched, 3]
        
        # Compute Huber loss between normalized predictions and targets
        # Huber loss provides robustness against outliers while maintaining smooth gradients
        # Delta parameter controls transition between L1 and L2 loss regimes
        loss_center = F.huber_loss(src_boxes_normalized, target_boxes_normalized, reduction='mean', delta=self.huber_delta)
        
        # # DEBUG: Print loss analysis for GT-input testing
        # if len(src_boxes_normalized) > 0:
        #     diff = (src_boxes_normalized - target_boxes_normalized).abs()
        #     print(f"[DEBUG] CENTER LOSS: {loss_center.item():.6f}")
        #     print(f"  - Num matched: {len(src_boxes_normalized)}")
        #     print(f"  - Max diff: {diff.max().item():.6f}")
        #     print(f"  - Mean diff: {diff.mean().item():.6f}")
        #     if len(src_boxes_normalized) >= 1:
        #         print(f"  - First pred: {src_boxes_normalized[0]}")
        #         print(f"  - First tgt:  {target_boxes_normalized[0]}")
        
        # Apply regression scalar from configuration
        regression_scalar = self.cfg['loss']['regression_scalar']
        return loss_center * regression_scalar


class LossSize(nn.Module):
    """
    Huber loss for bounding box size prediction (w, l, h) in normalized ego coordinates.

    Args:
        cfg: Configuration dictionary with loss parameters.
    """
    def __init__(self, cfg: Dict[str, Any], **kwargs):
        """
        Initialize size regression loss module.

        Args:
            cfg: Configuration dictionary with loss parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.huber_delta = cfg['loss']['huber']['delta']
        from oft.transformer.utils.normalization_utils import get_global_normalizer
        self.normalizer = get_global_normalizer()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, Any], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss for matched bounding box sizes (w, l, h).

        Args:
            predictions: Dict with 'pred_boxes_normalized' [B, N, 10]
            targets: Dict with 'gt_boxes_b_normalized' [B, N_gt, 10]
            indices: List of (src_idx, tgt_idx) tuples from Hungarian matcher
            num_boxes: Number of boxes (unused)
        Returns:
            torch.Tensor: Scalar size loss
        """
        batch_idx, src_idx = _get_src_permutation_idx(indices)
        batch_size, num_queries, _ = predictions['pred_boxes_normalized'].shape
        max_idx = batch_size * num_queries - 1
        valid_mask = src_idx <= max_idx
        if not valid_mask.all():
            src_idx = src_idx[valid_mask]
        if len(src_idx) == 0:
            return torch.tensor(0.0, device=predictions['pred_boxes_normalized'].device, dtype=predictions['pred_boxes_normalized'].dtype, requires_grad=True)
        
        # Convert flat indices to batch and query indices for tensor indexing
        # This enables efficient extraction of matched predictions from the full tensor
        batch_indices = src_idx // num_queries  # Batch index for each matched prediction
        query_indices = src_idx % num_queries   # Query index within each batch
        
        # Extract size/dimension components (w, l, h) from predicted boxes (log-space, normalized)
        # Format: [w_log_norm, l_log_norm, h_log_norm] (EGO, normalized) - log-normalized dimensions
        src_boxes_normalized = predictions['pred_boxes_normalized'][batch_indices, query_indices][:, 3:6]  # Shape: [num_matched, 3]
        
        # Extract target size coordinates using target indices (normalized)
        # Format: [w_norm, l_norm, h_norm] (EGO, normalized) - log-space normalization, range [-1, 1]
        batch_idx_tgt, tgt_idx = _get_tgt_permutation_idx(indices)  # Extract target indices
        target_boxes_normalized = targets['gt_boxes_b_normalized'][batch_idx_tgt, tgt_idx][:, 3:6]  # Shape: [num_matched, 3]
        
        # Compute Huber loss between predictions and targets
        # Huber loss provides robustness against outliers while maintaining smooth gradients
        # Delta parameter controls transition between L1 and L2 loss regimes
        loss_size = F.huber_loss(src_boxes_normalized, target_boxes_normalized, reduction='mean', delta=self.huber_delta)
        
        # Apply regression scalar from configuration
        regression_scalar = self.cfg['loss']['regression_scalar']
        return loss_size * regression_scalar


class LossAngle(nn.Module):
    """
    Huber loss for bounding box angle prediction (sin, cos) in normalized ego coordinates.

    Args:
        cfg: Configuration dictionary with loss parameters.
    """
    def __init__(self, cfg: Dict[str, Any], **kwargs):
        """
        Initialize angle regression loss module.

        Args:
            cfg: Configuration dictionary with loss parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.huber_delta = cfg['loss']['huber']['delta']

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, Any], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss for matched bounding box angles (sin, cos).

        Args:
            predictions: Dict with 'pred_boxes_normalized' [B, N, 10]
            targets: Dict with 'gt_boxes_b_normalized' [B, N_gt, 10]
            indices: List of (src_idx, tgt_idx) tuples from Hungarian matcher
            num_boxes: Number of boxes (unused)
        Returns:
            torch.Tensor: Scalar angle loss
        """
        batch_idx, src_idx = _get_src_permutation_idx(indices)
        
        # Validate indices to prevent out-of-bounds access in tensor indexing
        # This ensures that all source indices are within valid tensor dimensions
        batch_size, num_queries, _ = predictions['pred_boxes_normalized'].shape  # Extract tensor dimensions
        max_idx = batch_size * num_queries - 1  # Maximum valid index in flattened tensor
        valid_mask = src_idx <= max_idx  # Boolean mask for valid indices
        
        # Handle out-of-bounds indices by filtering and logging warning
        # This prevents runtime errors while maintaining gradient flow
        if not valid_mask.all():
            src_idx = src_idx[valid_mask]
            batch_idx = batch_idx[valid_mask]
        if len(src_idx) == 0:
            return torch.tensor(0.0, device=predictions['pred_boxes_normalized'].device, dtype=predictions['pred_boxes_normalized'].dtype, requires_grad=True)
        
        # Convert flat indices to batch and query indices for tensor indexing
        # This enables efficient extraction of matched predictions from batched tensors
        batch_indices = src_idx // num_queries  # Batch index for each matched prediction
        query_indices = src_idx % num_queries   # Query index for each matched prediction
        
        # Extract angle coordinates (sin, cos) from predicted boxes (normalized)
        # Format: [sin(yaw), cos(yaw)] (EGO, normalized) - range [-1, 1]
        src_boxes_normalized = predictions['pred_boxes_normalized'][batch_indices, query_indices][:, 6:8]  # Shape: [num_matched, 2]
        
        # Extract target angle coordinates using target indices (normalized)
        # Format: [sin(yaw), cos(yaw)] (EGO, normalized) - range [-1, 1]
        batch_idx_tgt, tgt_idx = _get_tgt_permutation_idx(indices)  # Extract target indices
        target_boxes_normalized = targets['gt_boxes_b_normalized'][batch_idx_tgt, tgt_idx][:, 6:8]  # Shape: [num_matched, 2]
        
        # Compute Huber loss between predictions and targets for consistency with other regression losses
        loss_angle = F.huber_loss(src_boxes_normalized, target_boxes_normalized, reduction='mean', delta=self.huber_delta)
        
        # Apply regression scalar from configuration
        regression_scalar = self.cfg['loss']['regression_scalar']
        return loss_angle * regression_scalar


class LossVelocity(nn.Module):
    """
    Huber loss for velocity prediction (vx, vy) in normalized ego coordinates.

    Args:
        cfg: Configuration dictionary with loss parameters.
    """
    def __init__(self, cfg: Dict[str, Any], **kwargs):
        """
        Initialize velocity regression loss module.

        Args:
            cfg: Configuration dictionary with loss parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.huber_delta = cfg['loss']['huber']['delta']
        from oft.transformer.utils.normalization_utils import get_global_normalizer
        self.normalizer = get_global_normalizer()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, Any], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss for matched velocity components (vx, vy).

        Args:
            predictions: Dict with 'pred_boxes_normalized' [B, N, 10]
            targets: Dict with 'gt_boxes_b_normalized' [B, N_gt, 10]
            indices: List of (src_idx, tgt_idx) tuples from Hungarian matcher
            num_boxes: Number of boxes (unused)
        Returns:
            torch.Tensor: Scalar velocity loss
        """
        batch_idx, src_idx = _get_src_permutation_idx(indices)
        batch_size, num_queries, _ = predictions['pred_boxes_normalized'].shape
        max_idx = batch_size * num_queries - 1
        valid_mask = src_idx <= max_idx
        if not valid_mask.all():
            src_idx = src_idx[valid_mask]
            batch_idx = batch_idx[valid_mask]
        if len(src_idx) == 0:
            return torch.tensor(0.0, device=predictions['pred_boxes_normalized'].device, dtype=predictions['pred_boxes_normalized'].dtype, requires_grad=True)
        
        # Convert flat indices to batch and query indices for tensor indexing
        # This enables efficient extraction of matched predictions from batched tensors
        batch_indices = src_idx // num_queries  # Batch index for each matched prediction
        query_indices = src_idx % num_queries   # Query index for each matched prediction
        
        # Extract velocity components (vx_norm, vy_norm) from predicted velocities (normalized)
        # Format: [vx_norm, vy_norm] (EGO, normalized) - range [-1, 1]
        src_velocities = predictions['pred_boxes_normalized'][batch_indices, query_indices][:, 8:10]  # Shape: [num_matched, 2]
        
        # Extract target velocity components using target indices (normalized)
        # Format: [vx_norm, vy_norm] (EGO, normalized) - range [-1, 1]
        batch_idx_tgt, tgt_idx = _get_tgt_permutation_idx(indices)  # Extract target indices
        target_velocities = targets['gt_boxes_b_normalized'][batch_idx_tgt, tgt_idx][:, 8:10]  # Shape: [num_matched, 2]
        
        # Compute Huber loss between predictions and targets
        # Huber loss provides robustness against outliers while maintaining smooth gradients
        # Delta parameter controls transition between L1 and L2 loss regimes
        loss_velocity = F.huber_loss(src_velocities, target_velocities, reduction='mean', delta=self.huber_delta)
        
        # Apply regression scalar from configuration
        regression_scalar = self.cfg['loss']['regression_scalar']
        return loss_velocity * regression_scalar


def _get_src_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract source permutation indices from Hungarian matching results for tensor indexing.

    Args:
        indices: List of (src_idx, tgt_idx) tuples from Hungarian matcher
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: batch indices, source indices
    """
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract target permutation indices from Hungarian matching results for tensor indexing.

    Args:
        indices: List of (src_idx, tgt_idx) tuples from Hungarian matcher
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: batch indices, target indices
    """
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx 