# File: src/oft/transformer/training/losses/matcher_costs.py
"""
Cost functions for Hungarian matcher in the Object Fusion Transformer pipeline.

Provides cost matrix computation for optimal assignment between predictions and ground truth
in normalized ego coordinate space. Used for robust bipartite matching during training.
"""

import torch
import torch.nn.functional as F

def center_cost_matrix_huber(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    """
    Compute Huber loss cost matrix for bounding box center distances in normalized space.

    Args:
        pred_boxes: Predicted boxes, shape [num_queries, D], D >= 3 (EGO, normalized)
        gt_boxes: Ground truth boxes, shape [num_gt, D], D >= 3 (EGO, normalized)
        delta: Huber loss delta parameter (default: 0.01)
    Returns:
        torch.Tensor: Cost matrix [num_queries, num_gt] (lower is better)
    """
    # Pairwise center differences in normalized ego coordinates
    diff = pred_boxes[:, None, :3] - gt_boxes[None, :, :3]
    abs_diff = diff.abs()
    # Huber loss: quadratic for small diffs, linear for large
    loss = torch.where(
        abs_diff < delta,
        0.5 * (abs_diff ** 2) / delta,  # Quadratic region: 0.5 * (x^2 / delta)
        abs_diff - 0.5 * delta          # Linear region: |x| - 0.5 * delta
    )
    # Sum over x, y, z
    return loss.sum(-1)


def class_cost_matrix(pred_logits: torch.Tensor, gt_classes: torch.Tensor) -> torch.Tensor:
    """
    Compute class cost matrix using direct probabilities (DETR Original).
    
    DETR Paper: "we use probabilities instead of log-probabilities"
    Cost = 1.0 - P(y_gt) for each prediction-GT pair.
    
    Args:
        pred_logits: Predicted class logits, shape [num_queries, num_classes]
        gt_classes: Ground truth class indices, shape [num_gt]
    Returns:
        torch.Tensor: Cost matrix [num_queries, num_gt] (lower is better)
    """
    # Convert logits to probabilities
    pred_probs = F.softmax(pred_logits, dim=-1)  # [num_queries, num_classes]
    
    # Extract probabilities for GT classes
    # gt_classes: [num_gt] -> expand to [num_queries, num_gt]
    gt_classes_expanded = gt_classes.unsqueeze(0).expand(pred_logits.shape[0], -1)  # [num_queries, num_gt]
    
    # Gather probabilities for GT classes
    pred_probs_for_gt = torch.gather(
        pred_probs.unsqueeze(1).expand(-1, len(gt_classes), -1),  # [num_queries, num_gt, num_classes]
        dim=-1, 
        index=gt_classes_expanded.unsqueeze(-1)  # [num_queries, num_gt, 1]
    ).squeeze(-1)  # [num_queries, num_gt]
    
    # Use direct probabilities (DETR Original standard)
    # DETR Paper: "we use probabilities instead of log-probabilities"
    class_cost = 1.0 - pred_probs_for_gt
    
    return class_cost


def size_cost_matrix_l1(pred_size_offsets: torch.Tensor, gt_size_offsets: torch.Tensor) -> torch.Tensor:
    """
    Compute size cost matrix using L1 distance.
    
    Args:
        pred_size_offsets: Predicted size offsets, shape [num_queries, 3] (w, l, h)
        gt_size_offsets: Ground truth size offsets, shape [num_gt, 3] (w, l, h)
    Returns:
        torch.Tensor: Cost matrix [num_queries, num_gt] (lower is better)
    """
    # Compute pairwise L1 distance
    return torch.cdist(pred_size_offsets, gt_size_offsets, p=1)




def angle_cost_matrix_huber(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    """
    Compute Huber loss cost matrix for bounding box angle differences (sin, cos representation).
    
    This function computes pairwise angle costs using the same Huber loss as LossAngle
    to ensure perfect consistency between matcher and training loss.
    
    Args:
        pred_boxes: Predicted boxes, shape [num_queries, D], D >= 8 (x,y,z,w,l,h,sin_yaw,cos_yaw,...)
        gt_boxes: Ground truth boxes, shape [num_gt, D], D >= 8 (x,y,z,w,l,h,sin_yaw,cos_yaw,...)
        delta: Huber loss delta parameter (default: 0.1, same as LossAngle)
    Returns:
        torch.Tensor: Cost matrix [num_queries, num_gt] (lower is better)
    """
    # Extract angle components (sin, cos) from normalized boxes
    pred_angles = pred_boxes[:, 6:8]  # [num_queries, 2] - sin(yaw), cos(yaw) in range [-1, 1]
    gt_angles = gt_boxes[:, 6:8]      # [num_gt, 2] - sin(yaw), cos(yaw) in range [-1, 1]
    
    # Compute pairwise angle differences for all prediction-GT pairs
    # Shape: [num_queries, num_gt, 2] (broadcasting: pred[:, None, :] - gt[None, :, :])
    diff = pred_angles[:, None, :] - gt_angles[None, :, :]
    abs_diff = diff.abs()
    
    # Apply Huber loss element-wise (same as LossAngle for consistency)
    # Huber loss: quadratic for small differences, linear for large differences
    loss = torch.where(
        abs_diff < delta,
        0.5 * (abs_diff ** 2) / delta,  # Quadratic region: 0.5 * (x^2 / delta)
        abs_diff - 0.5 * delta          # Linear region: |x| - 0.5 * delta
    )
    
    # Sum over sin and cos components to get total angle cost per pair
    # Shape: [num_queries, num_gt]
    return loss.sum(-1)


def giou_cost_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute GIoU cost matrix for better geometric matching in Hungarian algorithm.
    
    CRITICAL FIX: This function now properly handles normalized vs denormalized boxes.
    
    Args:
        pred_boxes: Predicted boxes, shape [num_queries, 10] (x,y,z,w,l,h,sin_yaw,cos_yaw,vx,vy) NORMALIZED [0,1]
        gt_boxes: Ground truth boxes, shape [num_gt, 10] (x,y,z,w,l,h,sin_yaw,cos_yaw,vx,vy) NORMALIZED [0,1]
    Returns:
        torch.Tensor: Cost matrix [num_queries, num_gt] (lower is better)
    """
    # IMPORT DEPENDENCIES
    from oft.transformer.utils.normalization_utils import get_global_normalizer, denormalize_coordinates, denormalize_dimensions
    from oft.transformer.utils.iou import generalized_box_iou_bev
    
    # EXTRACT COMPONENTS (Both inputs are NORMALIZED)
    pred_coords_norm = pred_boxes[:, :3]       # [num_queries, 3] - normalized coordinates [0,1]
    pred_dims_norm = pred_boxes[:, 3:6]        # [num_queries, 3] - normalized log dimensions
    pred_sin_yaw = pred_boxes[:, 6]            # [num_queries] - sin(yaw)
    pred_cos_yaw = pred_boxes[:, 7]            # [num_queries] - cos(yaw)
    
    gt_coords_norm = gt_boxes[:, :3]           # [num_gt, 3] - normalized coordinates [0,1] 
    gt_dims_norm = gt_boxes[:, 3:6]            # [num_gt, 3] - normalized log dimensions
    gt_sin_yaw = gt_boxes[:, 6]                # [num_gt] - sin(yaw)
    gt_cos_yaw = gt_boxes[:, 7]                # [num_gt] - cos(yaw)
    
    # DENORMALIZE TO PHYSICAL UNITS (meters, radians)
    # This is CRITICAL for GIoU to work correctly!
    
    # Get point cloud range for coordinate denormalization
    point_cloud_range = torch.tensor(
        get_global_normalizer().stats['metadata']['point_cloud_range'], 
        device=pred_boxes.device, 
        dtype=pred_boxes.dtype
    )
    
    # Denormalize coordinates from [0,1] to physical meters
    pred_coords_meter = denormalize_coordinates(pred_coords_norm, point_cloud_range)  # [num_queries, 3] meters
    gt_coords_meter = denormalize_coordinates(gt_coords_norm, point_cloud_range)      # [num_gt, 3] meters
    
    # Denormalize dimensions from log-space to physical meters
    pred_dims_meter = denormalize_dimensions(pred_dims_norm)  # [num_queries, 3] meters
    gt_dims_meter = denormalize_dimensions(gt_dims_norm)      # [num_gt, 3] meters
    
    # Convert sin/cos to yaw angles in radians
    pred_yaw = torch.atan2(pred_sin_yaw, pred_cos_yaw)  # [num_queries] radians
    gt_yaw = torch.atan2(gt_sin_yaw, gt_cos_yaw)        # [num_gt] radians
    
    # CONSTRUCT 7D BOXES FOR GIOU (x, y, z, w, l, h, yaw) in PHYSICAL units
    pred_boxes_7d = torch.cat([
        pred_coords_meter, 
        pred_dims_meter, 
        pred_yaw.unsqueeze(-1)
    ], dim=-1)  # [num_queries, 7] physical units
    
    gt_boxes_7d = torch.cat([
        gt_coords_meter, 
        gt_dims_meter, 
        gt_yaw.unsqueeze(-1)
    ], dim=-1)  # [num_gt, 7] physical units
    
    # COMPUTE GIOU MATRIX
    # Note: We use exact=False for speed during training (Hungarian matching)
    giou_matrix = generalized_box_iou_bev(pred_boxes_7d, gt_boxes_7d, exact=False)  # [num_queries, num_gt]
    
    # CONVERT TO COST: cost = 1 - GIoU
    # GIoU = 1 → cost = 0 (perfect match)
    # GIoU = 0 → cost = 1 (no overlap) 
    # GIoU = -1 → cost = 2 (boxes far apart)
    cost_matrix = 1.0 - giou_matrix
    
    return cost_matrix
