# File: src/oft/transformer/evaluation/prediction_utils.py
"""
Prediction utilities for the Object Fusion Transformer pipeline.

Contains functions for reconstructing and converting model predictions to evaluation format,
including coordinate transformations and normalization inversions for autoregressive models.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from pyquaternion import Quaternion

from oft.transformer.datasets.truckscenes.dataset import ATTRIBUTE_VOCAB
from oft.transformer.datasets.truckscenes.transforms import transform_relative_velocity_ego_to_world_frame
from oft.transformer.utils.normalization_utils import get_global_normalizer
from oft.transformer.utils.geometry_utils import sin_cos_to_yaw


def reconstruct_and_convert_predictions_autoregressive(
    batch_dict: Dict[str, Any], 
    predictions: Dict[str, torch.Tensor], 
    cfg: Dict[str, Any],
    ego_pose_current: Optional[Dict[str, torch.Tensor]] = None
) -> List[Dict[str, Any]]:
    """Denormalize and convert predicted boxes from autoregressive model to world coordinates.
    
    Args:
        batch_dict: Batch data containing ego pose and sample information:
            - 'sample_tokens': List of unique sample identifiers
            - 'ego_translation_world': Ego translations in world frame (B, 3) (meters)
            - 'ego_rotation_world_quat': Ego rotations as quaternions (B, 4)
            - 'ego_motion': Ego motion data with cabin velocity (B, 3) (m/s)
            - 'gt_valid_mask_b': Ground truth valid mask for adaptive filtering
        predictions: Model outputs from autoregressive model:
            - 'pred_class_logits_batch': Classification logits (B, N, num_classes+1) including no_object class
            - 'pred_boxes_normalized': Normalized 10D boxes (B, N, 10) (EGO, normalized)
            - 'pred_velocities_normalized': Normalized 2D velocities (B, N, 2) (EGO, normalized)
            - 'pred_attributes_logits_batch': Attribute logits (B, N, num_attrs+1)
        cfg: Configuration with evaluation and dataset parameters:
            - 'evaluation': Contains 'conf_th_eval' confidence threshold
            - 'dataset': Contains 'class_names' for label mapping
        ego_pose_current: Optional current ego pose override:
            - 'translation': Ego translation (3,) (WORLD, meters)
            - 'rotation': Ego rotation quaternion (4,)
            
    Returns:
        List of prediction dictionaries for each sample, each containing:
            - 'sample_token': Unique sample identifier
            - 'predictions': List of detection predictions with:
                - 'translation': 3D center [x, y, z] (WORLD, meters)
                - 'size': Dimensions [width, length, height] (WORLD, meters)
                - 'rotation': Quaternion [w, x, y, z] (WORLD)
                - 'velocity': 2D velocity [vx, vy] (WORLD, m/s)
                - 'detection_name': Object class name
                - 'detection_score': Confidence score [0, 1]
                - 'attribute_name': Predicted attribute
    """
    # Extract prediction components from model outputs
    pred_logits = predictions['pred_class_logits_batch']  # [B, N, num_classes+1]
    pred_boxes_normalized = predictions['pred_boxes_normalized']  # [B, N, 10] (EGO, normalized)
    pred_velocities_normalized = predictions['pred_velocities_normalized']  # [B, N, 2] (EGO, normalized)
    pred_attributes = predictions.get('pred_attributes_logits_batch', None)  # [B, N, num_attrs+1]

    batch_size = pred_logits.shape[0]
    results = []

    # Initialize global normalizer for denormalization
    normalizer = get_global_normalizer()
    
    # Extract confidence threshold for filtering
    conf_threshold = cfg['evaluation']['conf_th_eval']
    num_real_classes = pred_logits.shape[-1] - 1  # Exclude background class

    # Extract point cloud range for coordinate denormalization
    point_cloud_range = torch.tensor(
        normalizer.stats['metadata']['point_cloud_range'],
        device=pred_logits.device,
        dtype=torch.float64,
    )  # [x_min, y_min, z_min, x_max, y_max, z_max]

    # Process each sample in the batch
    for i in range(batch_size):
        sample_token = batch_dict['sample_tokens'][i]

        # Compute softmax probabilities and extract confidence scores and class labels
        probs_i = pred_logits[i].softmax(-1)  # (N, num_classes)
        scores_i, labels_i = probs_i.max(-1)  # (N,) - confidence scores and class labels
        
    # Unified filtering: keep only predictions that are not the no_object class
    # Duplicate suppression is handled by no_object assignment from Hungarian matching
        keep_mask = (labels_i < num_real_classes) & (scores_i > conf_threshold)  # (N,) - boolean mask

        # Handle case where no predictions meet the criteria
        if keep_mask.sum() == 0:
            results.append({'sample_token': sample_token, 'predictions': []})
            continue

        # Extract filtered predictions
        final_scores = scores_i[keep_mask]  # (N_valid,) - confidence scores
        final_labels = labels_i[keep_mask]  # (N_valid,) - class labels
        final_boxes_normalized = pred_boxes_normalized[i][keep_mask]  # (N_valid, 10) - normalized boxes
        final_velocities_normalized = pred_velocities_normalized[i][keep_mask]  # (N_valid, 2) - normalized velocities
        
        # Extract attribute predictions
        _, final_attr_labels = pred_attributes[i][keep_mask].softmax(-1).max(-1)  # (N_valid,) - attribute labels

        # Denormalize boxes and velocities
        # Extract components from normalized boxes: [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin(yaw), cos(yaw), vx_norm, vy_norm]
        coords_normalized = final_boxes_normalized[:, :3]  # (N_valid, 3) - [x_norm, y_norm, z_norm]
        dims_normalized = final_boxes_normalized[:, 3:6]   # (N_valid, 3) - [w_norm, l_norm, h_norm]
        yaw_sin_cos = final_boxes_normalized[:, 6:8]       # (N_valid, 2) - [sin(yaw), cos(yaw)]
        
        # Ensure unit-norm for sin/cos pairs to prevent numerical instability
        norm = torch.clamp(torch.sqrt((yaw_sin_cos ** 2).sum(dim=-1, keepdim=True)), min=1e-6)  # (N_valid, 1)
        yaw_sin_cos_unit = yaw_sin_cos / norm  # (N_valid, 2) - unit-normalized sin/cos pairs
        
        # Import denormalization functions
        from oft.transformer.utils.normalization_utils import denormalize_coordinates, denormalize_dimensions
        from oft.transformer.utils.geometry_utils import sin_cos_to_yaw
        
        # Optional high-precision denormalization in eval path
        use_high_precision = True if 'evaluation' not in cfg else cfg['evaluation'].get('high_precision_denorm', True)

        if use_high_precision:
            coords_denormalized = denormalize_coordinates(coords_normalized.to(torch.float64), point_cloud_range)  # (N_valid, 3)
            dims_denormalized = denormalize_dimensions(dims_normalized.to(torch.float64))  # (N_valid, 3)
        else:
            coords_denormalized = denormalize_coordinates(coords_normalized, point_cloud_range)
            dims_denormalized = denormalize_dimensions(dims_normalized)
        velocities_denormalized = normalizer.denormalize_velocity_absolute(final_velocities_normalized)  # (N_valid, 2) - [vx, vy] (EGO, m/s)
        
        # Convert sin/cos yaw to single yaw angle
        yaw_angles = torch.tensor([sin_cos_to_yaw(sin_val.item(), cos_val.item()) 
                                  for sin_val, cos_val in zip(yaw_sin_cos_unit[:, 0], yaw_sin_cos_unit[:, 1])], 
                                 device=yaw_sin_cos.device, dtype=torch.float64)  # (N_valid,) - yaw angles (radians)
        
        # Reconstruct denormalized boxes: [x, y, z, w, l, h, yaw]
        denormalized_boxes = torch.cat([coords_denormalized, dims_denormalized, yaw_angles.unsqueeze(-1)], dim=-1)  # (N_valid, 7)
        # Cast back to float32 for downstream consistency if desired (numerically negligible for output lists)
        if not use_high_precision:
            denormalized_boxes = denormalized_boxes.to(torch.float32)

        # Extract ego vehicle pose for coordinate transformation
        if ego_pose_current is not None:
            ego_translation = ego_pose_current['translation'].cpu().numpy()  # (3,) - ego translation (WORLD, meters)
            ego_rotation = Quaternion(ego_pose_current['rotation'].cpu().numpy())  # Ego rotation quaternion
        else:
            ego_translation = batch_dict['ego_translation_world'][i].cpu().numpy()  # (3,) - ego translation (WORLD, meters)
            ego_rotation = Quaternion(batch_dict['ego_rotation_world_quat'][i].cpu().numpy())  # Ego rotation quaternion
            
        # Extract ego vehicle 2D velocity for relative velocity transformation
        ego_velocity_world_phys_2d = batch_dict['ego_motion']['cabin']['velocity'][i, :2].cpu().numpy()  # (2,) - ego velocity (WORLD, m/s)

        # Process each valid prediction and transform to world coordinates
        sample_predictions = []
        for idx, (box_ego, vel_ego, label, score, attr_label) in enumerate(zip(denormalized_boxes, velocities_denormalized, final_labels, final_scores, final_attr_labels)):
            box_ego_np = box_ego.cpu().numpy()  # (7,) - [x, y, z, w, l, h, yaw] (EGO)

            # Transform box center from ego to world coordinates
            center_ego = box_ego_np[:3]  # (3,) - center coordinates (EGO, meters)
            center_ego_rotated = ego_rotation.rotate(center_ego)  # (3,) - rotated center (WORLD, meters)
            center_world = center_ego_rotated + ego_translation  # (3,) - final world center (WORLD, meters)

            # Transform relative velocity from ego to absolute world velocity
            vel_world_absolute = transform_relative_velocity_ego_to_world_frame(
                relative_velocity_ego_2d=vel_ego.cpu().numpy(), 
                ego_rotation=ego_rotation, 
                ego_velocity_world_phys_2d=ego_velocity_world_phys_2d
            )  # (2,) - absolute velocity (WORLD, m/s)
            
            # Map numerical attribute index to semantic attribute name
            attr_name = ATTRIBUTE_VOCAB[attr_label.item()] if 0 <= attr_label.item() < len(ATTRIBUTE_VOCAB) else ""
            
            # Extract yaw angle from denormalized box
            yaw_ego = box_ego_np[6]  # Single yaw angle in radians (EGO)
            
            # Transform yaw angle from ego to world coordinate system
            ego_yaw_world = ego_rotation.yaw_pitch_roll[0]  # Ego yaw in world frame (radians)
            
            # Direct yaw angle addition with proper angle wrapping to [-π, π]
            box_yaw_world = (yaw_ego + ego_yaw_world + np.pi) % (2 * np.pi) - np.pi  # (WORLD, radians)
            
            # Create clean quaternion with only yaw rotation (Roll=0, Pitch=0)
            world_rotation = Quaternion(axis=[0, 0, 1], radians=box_yaw_world)  # Box rotation quaternion (WORLD)

            # Validate values before .tolist() conversion
            if np.any(np.isnan(center_world)) or np.any(np.isinf(center_world)):
                print(f"🚨 NAN/INF in center_world: {center_world}")
            if np.any(np.isnan(box_ego_np[3:6])) or np.any(np.isinf(box_ego_np[3:6])):
                print(f"🚨 NAN/INF in box_ego_np size: {box_ego_np[3:6]}")
            if np.any(np.isnan(world_rotation.elements)) or np.any(np.isinf(world_rotation.elements)):
                print(f"🚨 NAN/INF in world_rotation: {world_rotation.elements}")
            if np.any(np.isnan(vel_world_absolute)) or np.any(np.isinf(vel_world_absolute)):
                print(f"🚨 NAN/INF in vel_world_absolute: {vel_world_absolute}")
            
            # Construct prediction dictionary in evaluation format
            prediction_obj = {
                "sample_token": str(sample_token),  # Required by TruckScenes DevKit!
                "translation": center_world.tolist(),  # (3,) - center coordinates (WORLD, meters)
                "size": box_ego_np[3:6].tolist(),  # (3,) - dimensions [w, l, h] (WORLD, meters)
                "rotation": world_rotation.elements.tolist(),  # (4,) - quaternion [w, x, y, z] (WORLD)
                "velocity": vel_world_absolute.tolist(),  # (2,) - velocity [vx, vy] (WORLD, m/s)
                "detection_name": cfg['dataset']['class_names'][label],  # Semantic class name
                "detection_score": score.item(),  # Confidence score [0, 1]
                "attribute_name": attr_name  # Semantic attribute name
            }
            sample_predictions.append(prediction_obj)

        # Add sample predictions to results list
        results.append({'sample_token': sample_token, 'predictions': sample_predictions})

    return results


__all__ = [
    'reconstruct_and_convert_predictions_autoregressive',
] 