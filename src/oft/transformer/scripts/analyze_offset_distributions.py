#/app/src/oft/transformer/scripts/analyze_offset_distributions.py
"""
Offset Distribution Analysis Script for Object Fusion Transformer

This script iterates over the entire training dataset to analyze the distribution
of offsets between initial_sensor_boxes and gt_boxes_b_physical. It calculates
percentiles to determine appropriate normalization bounds for the model.

The script separates one-sided scaling (center, size) from two-sided scaling (velocity)
as requested by the user.

Author: Object Fusion Transformer Team
Year: 2025
"""

import hydra
from omegaconf import DictConfig
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm
import math
from typing import Dict, List, Any, Tuple
import torch

# Add the project src directory to Python path
project_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_src_root not in sys.path:
    sys.path.insert(0, project_src_root)

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive


def calculate_offset_percentiles(cfg: DictConfig) -> Dict[str, Any]:
    """
    Iterate over the entire training dataset to calculate offset distribution percentiles.
    
    Args:
        cfg: Configuration dictionary containing dataset parameters.
        
    Returns:
        Dictionary containing percentile statistics for normalization.
    """
    print("=" * 80)
    print("OFFSET DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Initialize dataset
    train_split = cfg.training.train_split_name
    print(f"Using training split: '{train_split}'")
    
    dataset = ObjectFusionGTDatasetStaged(
        dataroot=cfg.dataset.dataroot,
        version=cfg.dataset.version,
        split_name=train_split,
        pipeline_config=cfg,
        verbose=False
    )
    
    print(f"Dataset initialized with {len(dataset)} samples")
    print(f"Cutoff distance: {dataset.cutoff_dist}m")
    print(f"Point cloud range: {dataset.point_cloud_range}")
    print("-" * 50)
    
    # Collections for offset statistics
    all_center_offsets = []      # [dx, dy, dz]
    all_log_size_offsets = []    # [d(log_w), d(log_l), d(log_h)]
    all_velocity_offsets = []    # [dvx, dvy]
    
    print("Iterating over dataset to collect offset statistics...")
    
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        try:
            # Get sample data
            sample_data = dataset[i]
            
            # Get ground truth boxes with actual dimensions
            gt_targets = sample_data['ground_truth_boxes']
            
            if not gt_targets:
                continue
                
            # Collect sensor detection boxes from all sensors
            all_sensor_detection_boxes = []
            for sensor_name, sensor_data in sample_data['sensor_data'].items():
                sensor_detection_boxes = sensor_data['boxes']  # Neue Struktur: 'boxes' statt 'sensor_detection_boxes'
                if sensor_detection_boxes.shape[0] > 0:
                    all_sensor_detection_boxes.extend(sensor_detection_boxes)
            
            if not all_sensor_detection_boxes:
                continue
                
            # Convert to numpy arrays for easier processing
            gt_boxes = np.array([gt['box_7d'] for gt in gt_targets])
            gt_velocities = np.array([gt.get('velocity_2d', [0.0, 0.0]) for gt in gt_targets])
            sensor_detection_boxes = np.array(all_sensor_detection_boxes)
            
            # Match GT boxes with sensor detection boxes using simple distance-based matching
            matched_pairs = match_boxes_simple(gt_boxes, sensor_detection_boxes)
            
            # Calculate offsets for matched pairs
            for gt_idx, sensor_detection_idx in matched_pairs:
                gt_box = gt_boxes[gt_idx]
                gt_velocity = gt_velocities[gt_idx]
                sensor_detection_box = sensor_detection_boxes[sensor_detection_idx]
                
                # Center offsets: gt_center - sensor_detection_center
                center_offset = gt_box[:3] - sensor_detection_box[:3]
                all_center_offsets.append(center_offset)
                
                # Log size offsets: log(gt_size) - log(sensor_detection_size)
                gt_sizes = np.maximum(gt_box[3:6], 1e-6)  # Avoid log(0)
                sensor_detection_sizes = np.maximum(sensor_detection_box[3:6], 1e-6)
                log_size_offset = np.log(gt_sizes) - np.log(sensor_detection_sizes)
                all_log_size_offsets.append(log_size_offset)
                
                # Velocity offsets: gt_velocity - sensor_detection_velocity
                if sensor_detection_box.shape[0] >= 9:  # 9D box includes velocity
                    sensor_detection_velocity = sensor_detection_box[7:9]
                    velocity_offset = gt_velocity - sensor_detection_velocity
                    all_velocity_offsets.append(velocity_offset)
                    
        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            continue
    
    print(f"\nCollected statistics:")
    print(f"  - Center offset pairs: {len(all_center_offsets)}")
    print(f"  - Log size offset pairs: {len(all_log_size_offsets)}")
    print(f"  - Velocity offset pairs: {len(all_velocity_offsets)}")
    
    # Calculate percentiles
    normalization_stats = {}
    
    if all_center_offsets:
        center_offsets_array = np.array(all_center_offsets)
        # One-sided scaling: use 99th percentile of absolute values
        center_abs_99p = np.percentile(np.abs(center_offsets_array), 99, axis=0)
        normalization_stats['center_abs_99p'] = center_abs_99p.tolist()
        print(f"  - Center offsets 99th percentile (abs): {center_abs_99p}")
    
    if all_log_size_offsets:
        log_size_offsets_array = np.array(all_log_size_offsets)
        # One-sided scaling: use 99th percentile of absolute values
        log_size_abs_99p = np.percentile(np.abs(log_size_offsets_array), 99, axis=0)
        normalization_stats['log_size_abs_99p'] = log_size_abs_99p.tolist()
        print(f"  - Log size offsets 99th percentile (abs): {log_size_abs_99p}")
    
    if all_velocity_offsets:
        velocity_offsets_array = np.array(all_velocity_offsets)
        # Two-sided scaling: use 1st and 99th percentiles
        velocity_1p = np.percentile(velocity_offsets_array, 1, axis=0)
        velocity_99p = np.percentile(velocity_offsets_array, 99, axis=0)
        normalization_stats['velocity_percentiles'] = {
            '1p': velocity_1p.tolist(),
            '99p': velocity_99p.tolist()
        }
        print(f"  - Velocity offsets 1st percentile: {velocity_1p}")
        print(f"  - Velocity offsets 99th percentile: {velocity_99p}")
    
    # Add metadata
    normalization_stats['metadata'] = {
        'dataset_split': train_split,
        'total_samples': len(dataset),
        'center_offset_pairs': len(all_center_offsets),
        'log_size_offset_pairs': len(all_log_size_offsets),
        'velocity_offset_pairs': len(all_velocity_offsets),
        'cutoff_distance': dataset.cutoff_dist,
        'point_cloud_range': dataset.point_cloud_range.tolist()
    }
    
    return normalization_stats


def match_boxes_simple(gt_boxes: np.ndarray, sensor_detection_boxes: np.ndarray, 
                      distance_threshold: float = 5.0) -> List[Tuple[int, int]]:
    """
    Simple distance-based matching between GT boxes and sensor detection boxes.
    
    Args:
        gt_boxes: Ground truth boxes [N, 7] (x, y, z, w, l, h, yaw)
        sensor_detection_boxes: Sensor detection boxes [M, 9] (x, y, z, w, l, h, yaw, vx, vy)
        distance_threshold: Maximum distance for matching (meters)
        
    Returns:
        List of matched (gt_idx, sensor_detection_idx) pairs
    """
    matched_pairs = []
    used_gt_indices = set()
    used_sensor_detection_indices = set()
    
    # Calculate pairwise distances between centers
    gt_centers = gt_boxes[:, :3]
    sensor_detection_centers = sensor_detection_boxes[:, :3]
    
    # Compute distance matrix
    distances = np.linalg.norm(gt_centers[:, np.newaxis, :] - sensor_detection_centers[np.newaxis, :, :], axis=2)
    
    # Find closest pairs within threshold
    while True:
        if len(used_gt_indices) >= len(gt_boxes) or len(used_sensor_detection_indices) >= len(sensor_detection_boxes):
            break
            
        # Find minimum distance among unused pairs
        min_dist = float('inf')
        best_gt_idx = -1
        best_sensor_detection_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in used_gt_indices:
                continue
            for sensor_detection_idx in range(len(sensor_detection_boxes)):
                if sensor_detection_idx in used_sensor_detection_indices:
                    continue
                    
                dist = distances[gt_idx, sensor_detection_idx]
                if dist < min_dist and dist <= distance_threshold:
                    min_dist = dist
                    best_gt_idx = gt_idx
                    best_sensor_detection_idx = sensor_detection_idx
        
        if best_gt_idx == -1:  # No more valid matches
            break
            
        # Add the match
        matched_pairs.append((best_gt_idx, best_sensor_detection_idx))
        used_gt_indices.add(best_gt_idx)
        used_sensor_detection_indices.add(best_sensor_detection_idx)
    
    return matched_pairs


def save_normalization_stats(stats: Dict[str, Any], output_path: str = "config/normalization_stats.yaml"):
    """
    Save normalization statistics to YAML file.
    
    Args:
        stats: Dictionary containing normalization statistics
        output_path: Path to save the YAML file
    """
    # Ensure config directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False, indent=2)
    
    print(f"\n✅ Normalization statistics saved to: {output_path}")
    
    # Also print a summary
    print("\n" + "=" * 50)
    print("NORMALIZATION STATISTICS SUMMARY")
    print("=" * 50)
    
    if 'center_abs_99p' in stats:
        print(f"Center offsets (99th percentile abs):")
        print(f"  X: {stats['center_abs_99p'][0]:.4f} m")
        print(f"  Y: {stats['center_abs_99p'][1]:.4f} m")
        print(f"  Z: {stats['center_abs_99p'][2]:.4f} m")
    
    if 'log_size_abs_99p' in stats:
        print(f"Log size offsets (99th percentile abs):")
        print(f"  Width:  {stats['log_size_abs_99p'][0]:.4f}")
        print(f"  Length: {stats['log_size_abs_99p'][1]:.4f}")
        print(f"  Height: {stats['log_size_abs_99p'][2]:.4f}")
    
    if 'velocity_percentiles' in stats:
        print(f"Velocity offsets:")
        print(f"  1st percentile:  [{stats['velocity_percentiles']['1p'][0]:.4f}, {stats['velocity_percentiles']['1p'][1]:.4f}] m/s")
        print(f"  99th percentile: [{stats['velocity_percentiles']['99p'][0]:.4f}, {stats['velocity_percentiles']['99p'][1]:.4f}] m/s")
    
    print(f"\nDataset info:")
    metadata = stats.get('metadata', {})
    print(f"  Split: {metadata.get('dataset_split', 'unknown')}")
    print(f"  Total samples: {metadata.get('total_samples', 0)}")
    print(f"  Matched pairs: {metadata.get('center_offset_pairs', 0)}")
    print("=" * 50)


@hydra.main(config_path="../../../../config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the offset distribution analysis.
    
    Args:
        cfg: Hydra configuration object
    """
    print("Starting offset distribution analysis...")
    
    # Temporarily disable dropout but keep noise for realistic offset analysis
    for sensor in cfg.dataset.virtual_sensors:
        if hasattr(sensor, 'dropout_rate'):
            print(f"Temporarily setting dropout_rate for '{sensor.name}' to 0.0 for analysis")
            sensor.dropout_rate = 0.0
        # Keep noise enabled to get realistic offsets
        print(f"Keeping noise enabled for '{sensor.name}' to get realistic offset distributions")
    
    # Calculate normalization statistics
    normalization_stats = calculate_offset_percentiles(cfg)
    
    # Save results
    save_normalization_stats(normalization_stats)
    
    print("\n🎉 Offset distribution analysis completed successfully!")


if __name__ == '__main__':
    main() 