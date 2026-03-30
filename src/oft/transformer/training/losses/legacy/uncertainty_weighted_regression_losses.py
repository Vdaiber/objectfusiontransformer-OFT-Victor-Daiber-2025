# File: src/oft/transformer/training/losses/regression_losses.py
"""
Regression loss functions for object detection.

This module provides regression loss implementations for object detection tasks,
including uncertainty-weighted regression loss according to Kendall & Gal (2017).

Author: Object Fusion Transformer Team
Year: 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ...utils.normalization_utils import create_offset_normalizer

def _get_src_permutation_idx(indices):
    """Extract source permutation indices from Hungarian matching results.
    
    Args:
        indices: Hungarian matching indices.
        
    Returns:
        Tuple of (batch_idx, src_idx) for indexing.
    """
    # OBJECT-LEVEL FUSION: Validate indices are within bounds
    valid_indices = []
    for i, (src, tgt) in enumerate(indices):
        if len(src) > 0:
            # Ensure src indices are valid (within bounds)
            max_valid_idx = src.max().item() if len(src) > 0 else -1
            if max_valid_idx >= 0:
                valid_indices.append((src, tgt, i))
    
    if not valid_indices:
        # Return empty tensors if no valid indices
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    
    batch_idx = torch.cat([torch.full_like(src, i) for src, _, i in valid_indices])
    src_idx = torch.cat([src for src, _, _ in valid_indices])
    return batch_idx, src_idx

class UncertaintyWeightedRegressionLoss(nn.Module):
    """Uncertainty-weighted regression loss according to Kendall & Gal (2017).
    
    Implements homoscedastic uncertainty weighting for multi-task regression,
    allowing the model to learn task-specific uncertainties and automatically
    balance different regression objectives without manual weight tuning.
    
    The loss is computed as:
    L_reg = Σ_i (exp(-s_i) * L1(p_i, gt_i) + s_i)
    
    where:
    - p_i is the predicted value for target i
    - gt_i is the ground truth value for target i  
    - s_i is the predicted log-variance (uncertainty) for target i
    - exp(-s_i) acts as a learned weighting factor
    - s_i acts as a regularizer preventing infinite uncertainty
    
    Reference: Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian 
    deep learning for computer vision? Advances in neural information processing systems, 30.
    """
    
    def __init__(self, cfg, **kwargs):
        """Initialize the uncertainty-weighted regression loss.
        
        Args:
            cfg: Configuration dictionary containing dataset parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.cfg = cfg  # Store config for point_cloud_range access
        
        # Initialize dynamic offset normalizer
        normalizer_type = cfg.get('normalization', {}).get('offset_normalizer', 'dynamic')
        self.offset_normalizer = create_offset_normalizer(
            normalizer_type=normalizer_type,
            fallback_std=cfg.get('normalization', {}).get('fallback_std', 1.0),
            min_samples=cfg.get('normalization', {}).get('min_samples', 10)
        )

    def forward(self, outputs, targets, indices, num_boxes):
        """Compute uncertainty-weighted regression loss.
        
        Args:
            outputs: Model predictions containing 'pred_box_offsets' (8D), 'pred_box_log_var', 
                    'pred_velocities', and 'pred_velocity_log_var'.
            targets: Ground truth data including boxes and velocities.
            indices: Hungarian matching indices.
            num_boxes: Number of boxes for normalization.
            
        Returns:
            Uncertainty-weighted regression loss.
        """
        idx = _get_src_permutation_idx(indices)
        
        # Extract predictions and their uncertainties
        pred_offsets_bounded = outputs['pred_box_offsets'][idx]  # [num_matched, 8] (x,y,z,w,l,h,yaw_sin,yaw_cos)
        pred_box_log_var = outputs['pred_box_log_var'][idx]      # [num_matched, 8]
        pred_velocities = outputs['pred_velocities'][idx]        # [num_matched, 2]
        pred_velocity_log_var = outputs['pred_velocity_log_var'][idx]  # [num_matched, 2]
        
        # Get matched ground truth data
        target_boxes_matched = torch.cat([t[i] for t, (_, i) in zip(targets['gt_boxes_b_physical'], indices)], dim=0)
        initial_boxes_matched = torch.cat([t[src] for t, (src, _) in zip(targets['initial_sensor_boxes'], indices)], dim=0)
        
        # Get point_cloud_range from config
        point_cloud_range = torch.tensor(self.cfg['dataset'].get('point_cloud_range', [-150.0, -150.0, -5.0, 150.0, 150.0, 3.0]), 
                                        device=pred_offsets_bounded.device, dtype=torch.float64)
        x_min, y_min, z_min = point_cloud_range[:3]
        x_max, y_max, z_max = point_cloud_range[3:]
        center_range = torch.stack([x_max - x_min, y_max - y_min, z_max - z_min])
        
        # Calculate target offsets in physical space
        target_offset_center = target_boxes_matched[:, :3] - initial_boxes_matched[:, :3]
        target_offset_dims_log = torch.log(target_boxes_matched[:, 3:6].clamp(min=1e-6)) - torch.log(initial_boxes_matched[:, 3:6].clamp(min=1e-6))
        target_offset_yaw = (target_boxes_matched[:, 6:7] - initial_boxes_matched[:, 6:7] + math.pi) % (2 * math.pi) - math.pi
        
        # Convert yaw to sin/cos representation for target
        target_yaw_sin = torch.sin(target_offset_yaw.squeeze(-1))
        target_yaw_cos = torch.cos(target_offset_yaw.squeeze(-1))
        target_yaw_sin_cos = torch.stack([target_yaw_sin, target_yaw_cos], dim=-1)
        
        # FIXED: Use dynamic offset normalization instead of point_cloud_range
        normalized_target_offset_center = self.offset_normalizer.normalize_offsets(
            target_offset_center, initial_boxes_matched, target_boxes_matched
        )
        
        # For dimensions, use log-space directly (no additional scaling needed)
        normalized_target_offset_dims = target_offset_dims_log
        
        # DEBUG: Target offset computation validation - FIXED: Using dynamic normalization
        print(f"DEBUG: Target offset computation (DYNAMIC):")
        print(f"  - Target boxes range: [{target_boxes_matched[:, :3].min():.2f}, {target_boxes_matched[:, :3].max():.2f}]")
        print(f"  - Initial boxes range: [{initial_boxes_matched[:, :3].min():.2f}, {initial_boxes_matched[:, :3].max():.2f}]")
        print(f"  - Raw target offsets range: [{target_offset_center.min():.4f}, {target_offset_center.max():.4f}]")
        print(f"  - Normalized target offsets range: [{normalized_target_offset_center.min():.4f}, {normalized_target_offset_center.max():.4f}]")
        print(f"  - Using dynamic normalization with fallback_std: {self.offset_normalizer.fallback_std}")
        print(f"  - Expected normalized range: [-1, 1]")
        
        # Extract predictions for different components
        pred_center = pred_offsets_bounded[:, :3]      # [num_matched, 3] (x, y, z)
        pred_dims = pred_offsets_bounded[:, 3:6]       # [num_matched, 3] (w, l, h)
        pred_yaw_sin_cos = pred_offsets_bounded[:, 6:8] # [num_matched, 2] (yaw_sin, yaw_cos)
        
        # Extract uncertainties for different components
        pred_center_log_var = pred_box_log_var[:, :3]      # [num_matched, 3]
        pred_dims_log_var = pred_box_log_var[:, 3:6]       # [num_matched, 3]
        pred_yaw_log_var = pred_box_log_var[:, 6:8]        # [num_matched, 2]
        
        # Get target velocities (if available)
        if 'gt_velocities_b' in targets and targets['gt_velocities_b'] is not None:
            target_velocities = torch.cat([t[i] for t, (_, i) in zip(targets['gt_velocities_b'], indices)], dim=0)
        else:
            target_velocities = torch.zeros_like(pred_velocities)
        
        # Compute uncertainty-weighted losses for each component
        loss_center = self._uncertainty_weighted_l1_loss(
            pred_center, normalized_target_offset_center, pred_center_log_var
        )
        
        loss_dims = self._uncertainty_weighted_l1_loss(
            pred_dims, normalized_target_offset_dims, pred_dims_log_var
        )
        
        # Yaw loss using L2 on sin/cos representation
        loss_yaw = self._uncertainty_weighted_l2_loss(
            pred_yaw_sin_cos, target_yaw_sin_cos, pred_yaw_log_var
        )
        
        # Velocity loss
        loss_velocity = self._uncertainty_weighted_l1_loss(
            pred_velocities, target_velocities, pred_velocity_log_var
        )
        
        # Combine all regression losses
        total_regression_loss = loss_center + loss_dims + loss_yaw + loss_velocity
        
        # DEBUG: Check for NaN/Inf in targets and predictions
        if torch.isnan(pred_center).any() or torch.isinf(pred_center).any():
            print("[ERROR] NaN/Inf detected in pred_center!")
        if torch.isnan(normalized_target_offset_center).any() or torch.isinf(normalized_target_offset_center).any():
            print("[ERROR] NaN/Inf detected in normalized_target_offset_center!")
        if torch.isnan(pred_dims).any() or torch.isinf(pred_dims).any():
            print("[ERROR] NaN/Inf detected in pred_dims!")
        if torch.isnan(normalized_target_offset_dims).any() or torch.isinf(normalized_target_offset_dims).any():
            print("[ERROR] NaN/Inf detected in normalized_target_offset_dims!")
        if torch.isnan(pred_yaw_sin_cos).any() or torch.isinf(pred_yaw_sin_cos).any():
            print("[ERROR] NaN/Inf detected in pred_yaw_sin_cos!")
        if torch.isnan(target_yaw_sin_cos).any() or torch.isinf(target_yaw_sin_cos).any():
            print("[ERROR] NaN/Inf detected in target_yaw_sin_cos!")
        # Print value ranges
        print(f"[DEBUG] pred_center: min={pred_center.min():.4f}, max={pred_center.max():.4f}, mean={pred_center.mean():.4f}")
        print(f"[DEBUG] normalized_target_offset_center: min={normalized_target_offset_center.min():.4f}, max={normalized_target_offset_center.max():.4f}, mean={normalized_target_offset_center.mean():.4f}")
        print(f"[DEBUG] pred_dims: min={pred_dims.min():.4f}, max={pred_dims.max():.4f}, mean={pred_dims.mean():.4f}")
        print(f"[DEBUG] normalized_target_offset_dims: min={normalized_target_offset_dims.min():.4f}, max={normalized_target_offset_dims.max():.4f}, mean={normalized_target_offset_dims.mean():.4f}")
        print(f"[DEBUG] pred_yaw_sin_cos: min={pred_yaw_sin_cos.min():.4f}, max={pred_yaw_sin_cos.max():.4f}, mean={pred_yaw_sin_cos.mean():.4f}")
        print(f"[DEBUG] target_yaw_sin_cos: min={target_yaw_sin_cos.min():.4f}, max={target_yaw_sin_cos.max():.4f}, mean={target_yaw_sin_cos.mean():.4f}")
        # Print all loss components
        print(f"[DEBUG] Loss components: center={loss_center:.4f}, dims={loss_dims:.4f}, yaw={loss_yaw:.4f}, velocity={loss_velocity:.4f}")
        
        return total_regression_loss / num_boxes

    def _uncertainty_weighted_l1_loss(self, pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty-weighted L1 loss.
        
        Args:
            pred: Predicted values [N, D].
            target: Target values [N, D].
            log_var: Log-variance predictions [N, D].
            
        Returns:
            Uncertainty-weighted L1 loss.
        """
        # Prevent log-variance from becoming too extreme
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        
        # Compute L1 loss
        l1_loss = F.l1_loss(pred, target, reduction='none')  # [N, D]
        
        # Apply uncertainty weighting: exp(-s_i) * L1_loss + s_i
        weighted_loss = torch.exp(-log_var) * l1_loss + log_var
        
        return weighted_loss.sum()
    
    def _uncertainty_weighted_l2_loss(self, pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty-weighted L2 loss.
        
        Args:
            pred: Predicted values [N, D].
            target: Target values [N, D].
            log_var: Log-variance predictions [N, D].
            
        Returns:
            Uncertainty-weighted L2 loss.
        """
        # Prevent log-variance from becoming too extreme
        log_var = torch.clamp(log_var, min=-0.0, max=5.0)
        
        # Compute L2 loss
        l2_loss = F.mse_loss(pred, target, reduction='none')  # [N, D]
        
        # Apply uncertainty weighting: exp(-s_i) * L2_loss + s_i
        weighted_loss = torch.exp(-log_var) * l2_loss + log_var
        
        return weighted_loss.sum() 