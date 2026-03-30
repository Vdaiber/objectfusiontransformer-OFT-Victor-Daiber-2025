# src/oft/transformer/utils/box_reconstruction.py
"""
Centralized box reconstruction utilities for consistent normalization.

This module provides a unified box reconstruction function that ensures
all components (training, evaluation, loss functions) use the same
normalization and reconstruction logic.

Author: Object Fusion Transformer Team
Year: 2025
"""

import torch
import math
import numpy as np
from typing import Optional, Tuple
from pyquaternion import Quaternion as PyQuaternion


def transform_box_vehicle_to_world(box_7d_vehicle, ego_translation_world, ego_rotation_world):
    """Transform a 7D bounding box from vehicle to world coordinates.
    
    Args:
        box_7d_vehicle: (7,) np.ndarray or torch.Tensor (cx, cy, cz, w, l, h, yaw) in vehicle frame.
        ego_translation_world: (3,) np.ndarray or torch.Tensor, ego vehicle translation in world frame.
        ego_rotation_world: PyQuaternion, rotation from vehicle to world frame.
        
    Returns:
        box_7d_world: (7,) np.ndarray (cx, cy, cz, w, l, h, yaw) in world frame.
    """
    if isinstance(box_7d_vehicle, torch.Tensor):
        box_7d_vehicle = box_7d_vehicle.detach().cpu().numpy()
    if isinstance(ego_translation_world, torch.Tensor):
        ego_translation_world = ego_translation_world.detach().cpu().numpy()
    if hasattr(ego_rotation_world, 'elements') and not isinstance(ego_rotation_world, PyQuaternion):
        ego_rotation_world = PyQuaternion(ego_rotation_world)

    box_center_vehicle = box_7d_vehicle[:3]
    box_dims = box_7d_vehicle[3:6]
    box_yaw_vehicle = box_7d_vehicle[6]

    # Transform center coordinates
    box_center_world = ego_rotation_world.rotate(box_center_vehicle) + ego_translation_world
    # Transform yaw angle
    ego_yaw_world = ego_rotation_world.yaw_pitch_roll[0]
    box_yaw_world = (box_yaw_vehicle + ego_yaw_world + np.pi) % (2 * np.pi) - np.pi

    box_7d_world = np.array([*box_center_world, *box_dims, box_yaw_world], dtype=np.float64)
    return box_7d_world


def reconstruct_boxes_consistent(
    pred_offsets: torch.Tensor,
    initial_boxes: torch.Tensor,
    normalization_factor: Optional[torch.Tensor] = None,
    use_fallback_normalization: bool = False,
    center_abs_99p: Optional[torch.Tensor] = None,
    log_size_abs_99p: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Centralized box reconstruction function for consistent normalization.
    
    This function ensures that all components use the same reconstruction logic:
    1. Denormalize center offsets using the provided normalization factor OR normalization_stats.yaml
    2. Reconstruct dimensions using log-space exponential
    3. Reconstruct yaw using sin/cos to angle conversion
    4. Combine into 7D boxes
    
    Args:
        pred_offsets: Predicted box offsets with shape [num_boxes, 8] (x,y,z,w,l,h,yaw_sin,yaw_cos)
                     Can be either normalized [-1, 1] or already denormalized
        initial_boxes: Initial reference boxes with shape [num_boxes, 7] (x,y,z,w,l,h,yaw)
        normalization_factor: Normalization factor from Center Loss [3] (x,y,z ranges) - DEPRECATED
        use_fallback_normalization: DEPRECATED
        center_abs_99p: Center normalization factors [3] from normalization_stats.yaml (NEW)
        log_size_abs_99p: Size normalization factors [3] from normalization_stats.yaml (NEW)
        
    Returns:
        Reconstructed boxes with shape [num_boxes, 7] (x,y,z,w,l,h,yaw)
    """
    # NEU: Check if offsets are normalized and need denormalization
    if center_abs_99p is not None and log_size_abs_99p is not None:
        # Offsets are normalized, need denormalization
        from .normalization_utils import denormalize_offsets
        pred_offsets = denormalize_offsets(pred_offsets)
    
    # Extract components from 8D offsets (now denormalized)
    pred_center_offsets = pred_offsets[:, :3]      # [num_boxes, 3] (x, y, z) - physical meters
    pred_dims_offsets = pred_offsets[:, 3:6]       # [num_boxes, 3] (w, l, h) - log-space
    pred_yaw_sin_cos = pred_offsets[:, 6:8]        # [num_boxes, 2] (yaw_sin, yaw_cos)
    
    # Reconstruct centers directly
    recon_center = initial_boxes[:, :3] + pred_center_offsets  # [num_boxes, 3]
    
    # Dimensions: Log-space, direct exponential with safety bounds
    # pred_dims_offsets are raw log-space offsets (not differences)
    # initial_boxes are sensor boxes, we want to predict GT boxes
    # GT = sensor * exp(log_offset) where log_offset = log(GT/sensor)
    
    # FIXED: Add safety bounds to prevent non-positive dimensions
    # Clamp log offsets to prevent exp() from producing very small values
    safe_dims_offsets = torch.clamp(pred_dims_offsets, min=-5.0, max=5.0)  # exp(-5) ≈ 0.007, exp(5) ≈ 148
    recon_dims = initial_boxes[:, 3:6] * torch.exp(safe_dims_offsets)  # [num_boxes, 3]
    
    # Additional safety check: ensure minimum dimensions
    min_dims = torch.tensor([0.1, 0.1, 0.1], device=recon_dims.device, dtype=recon_dims.dtype)  # 10cm minimum
    recon_dims = torch.maximum(recon_dims, min_dims)
    
    # Yaw: Convert sin/cos back to angle and add to initial yaw
    yaw_sin = pred_yaw_sin_cos[:, 0]  # [num_boxes]
    yaw_cos = pred_yaw_sin_cos[:, 1]  # [num_boxes]
    yaw_offset = torch.atan2(yaw_sin, yaw_cos)  # [num_boxes]
    yaw_offset = (yaw_offset + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]
    recon_yaw = (initial_boxes[:, 6:7] + yaw_offset.unsqueeze(-1) + math.pi) % (2 * math.pi) - math.pi  # [num_boxes, 1]
    
    # Combine into 7D reconstructed boxes
    reconstructed_boxes = torch.cat([recon_center, recon_dims, recon_yaw], dim=-1)  # [num_boxes, 7]
    
    if torch.isnan(reconstructed_boxes).any() or torch.isinf(reconstructed_boxes).any():
        nan_mask = torch.isnan(reconstructed_boxes)
        inf_mask = torch.isinf(reconstructed_boxes)
        error_msg = f"Numerical instability detected in box reconstruction.\n"
        error_msg += f"NaNs found at indices: {torch.where(nan_mask)}\n"
        error_msg += f"Infs found at indices: {torch.where(inf_mask)}\n"
        error_msg += f"Problematic initial boxes:\n{initial_boxes[nan_mask | inf_mask]}"
        raise RuntimeError(error_msg)
    
    return reconstructed_boxes 