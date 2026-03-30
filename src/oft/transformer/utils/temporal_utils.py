# src/oft/transformer/utils/temporal_utils.py
"""
Temporal Processing Utilities for Autoregressive Transformers.

This module contains all temporal-related functions for the Object Fusion Transformer,
including physics-based extrapolation, ego-motion compensation, and coordinate 
transformations between different frames.

FUTURE WORK - TEMPORAL PROCESSING PIPELINE:
These temporal components are currently DISABLED in single-frame mode but fully
implemented for future multi-frame temporal processing.
ACTIVATION: Set autoregressive.enabled=true and memory_enabled=true in config.

The pipeline follows: normalize → physical → physics operations → renormalize
for mathematically correct temporal processing.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict, Optional
from pyquaternion import Quaternion as PyQuaternion

from .normalization_utils import (
    get_global_normalizer, 
    normalize_coordinates, 
    denormalize_coordinates,
    normalize_dimensions,
    denormalize_dimensions
)


def compute_ego_motion_transform(
    ego_pose_t: Dict[str, torch.Tensor], 
    ego_pose_t_plus_1: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Compute relative ego-motion transformation between two consecutive world poses.
    
    Args:
        ego_pose_t: Ego pose at previous time step t:
            - 'translation': Position (x, y, z) in world frame (meters)
            - 'rotation': Quaternion (w, x, y, z) in world frame
        ego_pose_t_plus_1: Ego pose at current time step t+1 (same structure)
            
    Returns:
        torch.Tensor: 4×4 homogeneous transformation matrix T_{t→t+1} in ego frame
    """
    # Convert pose components to NumPy arrays for PyQuaternion operations
    trans_t = ego_pose_t['translation']
    rot_t = ego_pose_t['rotation']
    trans_t_plus_1 = ego_pose_t_plus_1['translation']
    rot_t_plus_1 = ego_pose_t_plus_1['rotation']

    if isinstance(trans_t, torch.Tensor):
        trans_t = trans_t.detach().cpu().numpy()
    if isinstance(rot_t, torch.Tensor):
        rot_t = rot_t.detach().cpu().numpy()
    if isinstance(trans_t_plus_1, torch.Tensor):
        trans_t_plus_1 = trans_t_plus_1.detach().cpu().numpy()
    if isinstance(rot_t_plus_1, torch.Tensor):
        rot_t_plus_1 = rot_t_plus_1.detach().cpu().numpy()

    # Ensure correct shapes for quaternion and translation vectors
    if rot_t.ndim == 1 and rot_t.shape[0] == 4:
        rot_t = rot_t.flatten()
    if rot_t_plus_1.ndim == 1 and rot_t_plus_1.shape[0] == 4:
        rot_t_plus_1 = rot_t_plus_1.flatten()
    if trans_t.ndim == 1 and trans_t.shape[0] == 3:
        trans_t = trans_t.flatten()
    if trans_t_plus_1.ndim == 1 and trans_t_plus_1.shape[0] == 3:
        trans_t_plus_1 = trans_t_plus_1.flatten()

    # Create PyQuaternion objects from world rotations
    quat_t = PyQuaternion(rot_t)
    quat_t_plus_1 = PyQuaternion(rot_t_plus_1)

    # Compute relative transformation in world frame
    quat_relative = quat_t_plus_1 * quat_t.inverse
    
    # FIX: For ego-frame transformation, we need the inverse transformation
    # When ego moves forward, objects appear to move backward in ego frame
    quat_relative_ego = quat_relative.inverse
    trans_relative_ego = -quat_relative_ego.rotate(trans_t_plus_1 - trans_t)

    # Assemble 4x4 homogeneous transformation matrix
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = quat_relative_ego.rotation_matrix
    transform_matrix[:3, 3] = trans_relative_ego

    return torch.from_numpy(transform_matrix)


def apply_ego_motion_transform_physical(
    anchor_boxes: torch.Tensor, 
    ego_transform: torch.Tensor
) -> torch.Tensor:
    """Apply ego-motion transformation to physical anchor boxes.
    
    Args:
        anchor_boxes: Physical box parameters [x, y, z, length, width, height, yaw] (B, N, 7)
        ego_transform: 4×4 homogeneous transformation matrix T_{t-1→t}
            
    Returns:
        torch.Tensor: Transformed anchor boxes (B, N, 7) in new ego frame
    """
    batch_size, num_objects, _ = anchor_boxes.shape

    # Decompose anchor boxes into components
    centers = anchor_boxes[:, :, :3]  # Centers (x,y,z) in meters
    dims = anchor_boxes[:, :, 3:6]    # Dimensions (w,l,h) in meters
    yaws = anchor_boxes[:, :, 6:7]    # Yaw angles in radians

    # Convert centers to homogeneous coordinates for matrix multiplication
    centers_homogeneous = torch.cat([
        centers,
        torch.ones(batch_size, num_objects, 1, device=anchor_boxes.device)
    ], dim=-1)  # Shape: (B, N, 4)

    # Ensure transformation matrix is on correct device
    ego_transform_device = ego_transform.to(anchor_boxes.device)

    # Reshape for efficient batch matrix multiplication
    centers_homogeneous_reshaped = centers_homogeneous.view(-1, 4)  # Shape: (B*N, 4)
    ego_transform_batch = ego_transform_device.unsqueeze(0).expand(batch_size * num_objects, 4, 4)

    # Apply 4x4 transformation to object centers
    transformed_centers_homogeneous = torch.bmm(
        ego_transform_batch,
        centers_homogeneous_reshaped.unsqueeze(-1)
    ).squeeze(-1)  # Shape: (B*N, 4)

    # Convert back from homogeneous to 3D coordinates
    transformed_centers = transformed_centers_homogeneous[:, :3].view(batch_size, num_objects, 3)

    # Transform yaw angles by extracting yaw component from rotation matrix
    rotation_matrix = ego_transform_device[:3, :3]
    yaw_change = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Add yaw change to original yaw angles
    transformed_yaws = yaws + yaw_change

    # Normalize resulting yaw angle to range [-π, π]
    transformed_yaws = (transformed_yaws + math.pi) % (2 * math.pi) - math.pi

    # Reassemble transformed anchor boxes
    transformed_anchor_boxes = torch.cat([
        transformed_centers,
        dims,                # Invariant
        transformed_yaws
    ], dim=-1)

    return transformed_anchor_boxes


def compute_positional_encoding_from_boxes(
    anchor_boxes: torch.Tensor, 
    point_cloud_range: torch.Tensor
) -> torch.Tensor:
    """Generate normalized positional encodings from physical anchor box centers.
    
    Args:
        anchor_boxes: Physical box parameters [x, y, z, l, w, h, yaw] (B, N, 7)
        point_cloud_range: Physical boundaries [x_min, y_min, z_min, x_max, y_max, z_max] (6,)
            
    Returns:
        torch.Tensor: Normalized center coordinates (x, y, z) in range [0, 1] (B, N, 3)
    """
    # Extract physical center coordinates from anchor boxes
    centers = anchor_boxes[:, :, :3]  # Shape: (B, N, 3)

    # Retrieve physical point cloud boundaries
    x_min, y_min, z_min = point_cloud_range[:3]
    x_max, y_max, z_max = point_cloud_range[3:]
    
    # Normalize center coordinates to [0, 1] range using point cloud boundaries
    normalized_centers = torch.zeros_like(centers)
    normalized_centers[:, :, 0] = (centers[:, :, 0] - x_min) / (x_max - x_min)
    normalized_centers[:, :, 1] = (centers[:, :, 1] - y_min) / (y_max - y_min)
    normalized_centers[:, :, 2] = (centers[:, :, 2] - z_min) / (z_max - z_min)

    # Clamp values to [0, 1] range to handle objects outside defined range
    normalized_centers = torch.clamp(normalized_centers, 0.0, 1.0)

    return normalized_centers


def convert_12d_features_to_10d_memory_format(features_12d: torch.Tensor) -> torch.Tensor:
    """Convert 12D sensor features to standardized 10D memory format.
    
    Args:
        features_12d: 12D sensor features (B, N, 12) 
                     [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin_yaw, cos_yaw, 
                      vx_norm, vy_norm, sensor_class, sensor_attr]
                     (EGO, normalized)
        
    Returns:
        torch.Tensor: 10D memory format (B, N, 10)
                     [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin_yaw, cos_yaw, vx_norm, vy_norm]
                     (EGO, normalized)
    """
    # Extract relevant components, skip class/attribute information [10:12]
    memory_10d = torch.zeros(features_12d.shape[0], features_12d.shape[1], 10, 
                            device=features_12d.device, dtype=features_12d.dtype)
    
    memory_10d[..., :10] = features_12d[..., :10]  # Take first 10 dimensions
    
    return memory_10d


def denormalize_memory_to_physical(
    memory_10d_norm: torch.Tensor, 
    point_cloud_range: torch.Tensor,
    normalizer = None
) -> torch.Tensor:
    """Convert 10D normalized memory to physical coordinates for temporal operations.
    
    Args:
        memory_10d_norm: 10D normalized memory (B, N, 10) (EGO, normalized)
                        [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin_yaw, cos_yaw, vx_norm, vy_norm]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] (6,) (meters)
        normalizer: Global normalizer instance for velocity denormalization
                        
    Returns:
        torch.Tensor: 10D physical format (B, N, 10) (EGO, physical)
                     [x_phys, y_phys, z_phys, w_phys, l_phys, h_phys, yaw_phys, 0, vx_phys, vy_phys]
    """
    if normalizer is None:
        normalizer = get_global_normalizer()
    
    # Extract normalized components
    coords_norm = memory_10d_norm[..., :3]      # [B, N, 3] [0,1]
    dims_norm = memory_10d_norm[..., 3:6]       # [B, N, 3] [-1,1]  
    sin_yaw = memory_10d_norm[..., 6]           # [B, N] sin(yaw)
    cos_yaw = memory_10d_norm[..., 7]           # [B, N] cos(yaw)
    vel_norm = memory_10d_norm[..., 8:10]       # [B, N, 2] [-1,1]
    
    # Denormalize to physical units
    coords_phys = denormalize_coordinates(coords_norm, point_cloud_range.to(coords_norm.device))  # meters
    dims_phys = denormalize_dimensions(dims_norm)                                                  # meters
    yaw_phys = torch.atan2(sin_yaw, cos_yaw)                                                      # radians (tensor-aware)
    vel_phys = normalizer.denormalize_velocity_absolute(vel_norm)                                  # m/s
    
    # Create 10D physical format: [x,y,z,w,l,h,yaw,0,vx,vy]
    memory_10d_phys = torch.zeros_like(memory_10d_norm)
    memory_10d_phys[..., :3] = coords_phys              # [B,N,3] meters
    memory_10d_phys[..., 3:6] = dims_phys               # [B,N,3] meters
    memory_10d_phys[..., 6] = yaw_phys                  # [B,N] radians
    memory_10d_phys[..., 7] = 0.0                       # [B,N] padding
    memory_10d_phys[..., 8:10] = vel_phys               # [B,N,2] m/s
    
    return memory_10d_phys


def renormalize_physical_to_memory(
    memory_10d_phys: torch.Tensor,
    point_cloud_range: torch.Tensor,
    normalizer = None
) -> torch.Tensor:
    """Convert 10D physical coordinates back to normalized memory format.
    
    Args:
        memory_10d_phys: 10D physical format (B, N, 10) (EGO, physical)
                        [x_phys, y_phys, z_phys, w_phys, l_phys, h_phys, yaw_phys, 0, vx_phys, vy_phys]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] (6,) (meters)
        normalizer: Global normalizer instance for velocity normalization
        
    Returns:
        torch.Tensor: 10D normalized memory format (B, N, 10) (EGO, normalized)
                     [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin_yaw, cos_yaw, vx_norm, vy_norm]
    """
    if normalizer is None:
        normalizer = get_global_normalizer()
    
    # Extract physical components
    coords_phys = memory_10d_phys[..., :3]       # [B, N, 3] meters
    dims_phys = memory_10d_phys[..., 3:6]        # [B, N, 3] meters  
    yaw_phys = memory_10d_phys[..., 6]           # [B, N] radians
    vel_phys = memory_10d_phys[..., 8:10]        # [B, N, 2] m/s
    
    # Renormalize to system format
    coords_norm = normalize_coordinates(coords_phys, point_cloud_range.to(coords_phys.device))     # [0,1]
    dims_norm = normalize_dimensions(dims_phys)                                                     # [-1,1]
    sin_yaw, cos_yaw = torch.sin(yaw_phys), torch.cos(yaw_phys)                                   # sin/cos (tensor-aware)
    vel_norm = normalizer.normalize_velocity_absolute(vel_phys)                                     # [-1,1]
    
    # Create 10D normalized format
    memory_10d_norm = torch.zeros_like(memory_10d_phys)
    memory_10d_norm[..., :3] = coords_norm                              # [B,N,3] [0,1]
    memory_10d_norm[..., 3:6] = dims_norm                               # [B,N,3] [-1,1]  
    memory_10d_norm[..., 6] = sin_yaw                                   # [B,N] sin(yaw)
    memory_10d_norm[..., 7] = cos_yaw                                   # [B,N] cos(yaw)
    memory_10d_norm[..., 8:10] = vel_norm                               # [B,N,2] [-1,1]
    
    return memory_10d_norm


def extrapolate_memory_with_physics_physical(
    memory_10d_phys: torch.Tensor, 
    dt: torch.Tensor
) -> torch.Tensor:
    """Physics-based extrapolation in physical coordinate space (MATHEMATICALLY CORRECT).
    
    Args:
        memory_10d_phys: 10D physical memory (B, N, 10) (EGO, physical)
                        [x_phys, y_phys, z_phys, w_phys, l_phys, h_phys, yaw_phys, 0, vx_phys, vy_phys]
        dt: Time delta between frames (seconds, physical)
        
    Returns:
        torch.Tensor: Extrapolated physical memory (B, N, 10) (EGO, physical)
    """
    # If dt is zero or negative, no extrapolation is needed
    if dt.nelement() == 0 or dt.item() <= 0:
        return memory_10d_phys

    # Extract physical velocities (m/s) and positions (meters)
    memory_velocities_phys = memory_10d_phys[..., 8:10]  # [B, N, 2] m/s
    
    # Ensure dt is on the same device
    dt_device = dt.to(memory_velocities_phys.device)

    # CORRECT physics: velocity [m/s] * time [s] = displacement [m]
    delta_xy_phys = memory_velocities_phys * dt_device.unsqueeze(0).unsqueeze(0)  # [B, N, 2] meters
    
    # Apply displacement to physical positions (meters + meters = meters)
    extrapolated_memory = memory_10d_phys.clone()
    extrapolated_memory[..., :2] += delta_xy_phys  # MATHEMATICALLY CORRECT
    
    return extrapolated_memory


def apply_ego_motion_compensation_physical(
    memory_10d_phys: torch.Tensor,
    ego_transform: torch.Tensor
) -> torch.Tensor:
    """Ego motion compensation in physical coordinate space (MATHEMATICALLY CORRECT).
    
    Args:
        memory_10d_phys: 10D physical memory (B, N, 10) (EGO, physical)
        ego_transform: 4x4 transformation matrix from compute_ego_motion_transform (physical)
        
    Returns:
        torch.Tensor: Ego-compensated physical memory (B, N, 10) (EGO, physical)  
    """
    # Extract 7D boxes for transformation [x,y,z,w,l,h,yaw] in physical units
    boxes_7d_phys = memory_10d_phys[..., :7]  # [B, N, 7] meters/radians
    
    # Apply transformation in physical space (MATHEMATICALLY CORRECT!)
    transformed_boxes_7d_phys = apply_ego_motion_transform_physical(
        boxes_7d_phys, ego_transform
    )  # meters → meters, radians → radians
    
    # Reconstruct 10D format, preserving velocities
    compensated_memory = memory_10d_phys.clone()
    compensated_memory[..., :7] = transformed_boxes_7d_phys           # Updated positions/orientations
    # Keep velocities unchanged: compensated_memory[..., 8:10] = memory_10d_phys[..., 8:10]
    
    return compensated_memory


def extrapolate_memory_with_physics_legacy(
    memory_anchor_boxes: torch.Tensor, 
    dt: torch.Tensor
) -> torch.Tensor:
    """LEGACY: Extrapolate memory box positions using constant velocity model (normalized space).
    
    DEPRECATED: This function operates in normalized space, which is mathematically incorrect.
    Use the physical space pipeline instead: 
    denormalize → extrapolate_memory_with_physics_physical → renormalize
    
    Args:
        memory_anchor_boxes: 10-D anchor box representation (B, N, 10) (EGO, normalized)
        dt: Time delta between frames (seconds)
        
    Returns:
        torch.Tensor: Extrapolated anchor boxes (B, N, 10) (EGO, normalized)
    """
    # If dt is zero or negative, no extrapolation is needed.
    if dt.nelement() == 0 or dt.item() <= 0:
        return memory_anchor_boxes

    # Extract the normalized 2D velocities (vx_norm, vy_norm) from the anchor boxes.
    # These are at indices 8 and 9 of the 10-D vector. (EGO, normalized)
    memory_velocities = memory_anchor_boxes[..., 8:10] # Shape: [B, N, 2]

    # Ensure dt is on the same device as the tensor.
    dt_device = dt.to(memory_velocities.device)

    # Calculate positional displacement in normalized space
    # Correct broadcasting for velocity [B, N, 2] * dt [scalar]
    delta_xy = memory_velocities * dt_device.unsqueeze(0).unsqueeze(0)  # Shape: [B, N, 2]

    # Apply displacement to normalized box centers
    extrapolated_boxes = memory_anchor_boxes.clone()
    extrapolated_boxes[..., :2] += delta_xy
    
    return extrapolated_boxes
