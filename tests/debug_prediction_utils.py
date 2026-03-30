#!/usr/bin/env python3
"""
Simple debug script for prediction_utils.py to identify ATE and box count issues.
"""

import torch
import numpy as np
from typing import Dict, Any, List
from pyquaternion import Quaternion

# Import the function under test
from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive
from oft.transformer.utils.normalization_utils import get_global_normalizer


def setup_test_data():
    """Setup realistic test data for debugging"""
    # Setup minimal config
    cfg = {
        'evaluation': {'conf_th_eval': 0.3},
        'dataset': {
            'class_names': ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 
                           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 
                           'barrier', 'animal', 'traffic_sign']
        }
    }
    
    # Setup batch_dict with B=2 samples
    batch_dict = {
        'sample_tokens': ['sample_001', 'sample_002'],
        'ego_translation_world': torch.tensor([
            [100.0, 200.0, 1.5],  # Sample 0
            [105.0, 205.0, 1.5]   # Sample 1  
        ], dtype=torch.float64),
        'ego_rotation_world_quat': torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Sample 0 - no rotation
            [0.9996, 0.0, 0.0, 0.0283]  # Sample 1 - small rotation
        ], dtype=torch.float64),
        'ego_motion': {
            'cabin': {
                'velocity': torch.tensor([
                    [10.0, 0.0, 0.0],  # Sample 0 - moving forward
                    [8.0, 2.0, 0.0]   # Sample 1 - moving diagonal
                ], dtype=torch.float64)
            }
        },
        'gt_valid_mask_b': torch.tensor([
            [True, True, False, False],  # Sample 0 - 2 valid GT
            [True, True, True, False]    # Sample 1 - 3 valid GT
        ])
    }
    
    # Setup predictions with N=4 max detections per sample
    N = 4
    B = 2
    num_classes = len(cfg['dataset']['class_names'])
    
    predictions = {
        # Classification: [car, truck, no_object, no_object] for sample 0
        'pred_class_logits_batch': torch.tensor([
            [
                [5.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -1.0],  # car
                [-2.0, 4.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -1.0],  # truck
                [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0],  # no_object
                [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0]   # no_object
            ],
            [
                [4.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -1.0],  # car
                [-3.0, -3.0, -3.0, -3.0, -3.0, 5.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -1.0],  # pedestrian
                [-2.0, 3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -1.0],  # truck
                [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0]   # no_object
            ]
        ], dtype=torch.float32),
        
        # Normalized 10D boxes [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy] in EGO frame
        'pred_boxes_normalized': torch.tensor([
            [
                [0.6, 0.5, 0.5, 0.3, 0.7, 0.4, 0.0, 1.0, 0.1, 0.0],  # car box
                [0.4, 0.6, 0.5, 0.5, 0.9, 0.6, 0.1, 0.995, 0.0, 0.1],  # truck box
                [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0],  # no_object (should be filtered)
                [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0]   # no_object (should be filtered)
            ],
            [
                [0.7, 0.4, 0.5, 0.3, 0.7, 0.4, 0.05, 0.999, 0.2, 0.0],  # car box
                [0.3, 0.7, 0.5, 0.2, 0.4, 0.8, 0.0, 1.0, 0.0, 0.0],  # pedestrian box
                [0.5, 0.3, 0.5, 0.6, 1.0, 0.7, -0.1, 0.995, 0.1, 0.1],  # truck box
                [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0]   # no_object (should be filtered)
            ]
        ], dtype=torch.float32),
        
        # Normalized velocities (already handled in boxes above)
        'pred_velocities_normalized': torch.tensor([
            [[0.1, 0.0], [0.0, 0.1], [0.0, 0.0], [0.0, 0.0]],
            [[0.2, 0.0], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
        ], dtype=torch.float32),
        
        # Attribute logits (dummy - not critical for coordinate debugging)
        'pred_attributes_logits_batch': torch.zeros(B, N, 12, dtype=torch.float32)
    }
    
    return cfg, batch_dict, predictions


def test_basic_functionality():
    """Test that the function runs without errors and returns expected structure"""
    cfg, batch_dict, predictions = setup_test_data()
    
    print("🔍 TESTING BASIC FUNCTIONALITY")
    print("-" * 40)
    
    # Test with ego_pose_current = None (per-sample poses)
    result = reconstruct_and_convert_predictions_autoregressive(
        batch_dict, predictions, cfg, ego_pose_current=None
    )
    
    print(f"✅ Result is list: {isinstance(result, list)}")
    print(f"✅ Number of samples: {len(result)}")
    
    for i, sample_result in enumerate(result):
        required_keys = ['sample_token', 'name', 'score', 'translation', 'size', 'rotation', 'velocity', 'attribute_name']
        missing_keys = [key for key in required_keys if key not in sample_result]
        
        print(f"  Sample {i}:")
        print(f"    Missing keys: {missing_keys if missing_keys else 'None'}")
        print(f"    Number of predictions: {len(sample_result['name'])}")
        print(f"    Classes: {sample_result['name']}")
        print(f"    Scores: {[f'{s:.3f}' for s in sample_result['score']]}")


def test_coordinate_transformation_consistency():
    """Test coordinate transformation consistency between ego_pose_current modes"""
    cfg, batch_dict, predictions = setup_test_data()
    
    print("\n🔍 TESTING COORDINATE TRANSFORMATION CONSISTENCY")
    print("-" * 50)
    
    # Test with ego_pose_current = None (per-sample)
    result_per_sample = reconstruct_and_convert_predictions_autoregressive(
        batch_dict, predictions, cfg, ego_pose_current=None
    )
    
    # Test with ego_pose_current from first sample (original behavior)
    ego_pose_current = {
        'translation': batch_dict['ego_translation_world'][0],
        'rotation': batch_dict['ego_rotation_world_quat'][0]
    }
    result_single_pose = reconstruct_and_convert_predictions_autoregressive(
        batch_dict, predictions, cfg, ego_pose_current=ego_pose_current
    )
    
    print("=== COORDINATE TRANSFORMATION COMPARISON ===")
    for i in range(2):
        per_sample = result_per_sample[i]
        single_pose = result_single_pose[i]
        
        print(f"\nSample {i}:")
        print(f"  Per-sample mode: {len(per_sample['translation'])} predictions")
        print(f"  Single-pose mode: {len(single_pose['translation'])} predictions")
        
        if len(per_sample['translation']) > 0 and len(single_pose['translation']) > 0:
            # Compare first prediction coordinates
            per_trans = np.array(per_sample['translation'][0])
            single_trans = np.array(single_pose['translation'][0])
            diff = np.linalg.norm(per_trans - single_trans)
            
            print(f"  First prediction translation diff: {diff:.6f} meters")
            print(f"    Per-sample: {per_trans}")
            print(f"    Single-pose: {single_trans}")
            
            if i == 0:
                # Sample 0 should be identical (same ego pose)
                print(f"    ✅ Sample 0 identical (expected): {diff < 1e-6}")
            elif i == 1:
                # Sample 1 should be different (different ego pose)
                print(f"    ✅ Sample 1 different (expected): {diff > 1e-3}")


def test_denormalization_step_by_step():
    """Test denormalization step by step to find precision issues"""
    cfg, batch_dict, predictions = setup_test_data()
    
    print("\n🔍 TESTING DENORMALIZATION STEP BY STEP")
    print("-" * 45)
    
    # Get normalizer
    normalizer = get_global_normalizer()
    
    # Extract point cloud range
    point_cloud_range = torch.tensor(
        normalizer.stats['metadata']['point_cloud_range'],
        dtype=torch.float64
    )
    
    print(f"Point cloud range: {point_cloud_range}")
    
    # Test denormalization with different precisions
    test_box_norm = predictions['pred_boxes_normalized'][0, 0]  # First prediction
    
    print(f"Normalized box: {test_box_norm}")
    
    # Extract components
    coords_norm = test_box_norm[:3]
    dims_norm = test_box_norm[3:6]
    sin_yaw_norm = test_box_norm[6]
    cos_yaw_norm = test_box_norm[7]
    vel_norm = test_box_norm[8:10]
    
    print(f"Coords normalized: {coords_norm}")
    print(f"Dims normalized: {dims_norm}")
    print(f"Yaw sin/cos normalized: {sin_yaw_norm}, {cos_yaw_norm}")
    print(f"Velocity normalized: {vel_norm}")
    
    # Test with different precisions
    for dtype in [torch.float32, torch.float64]:
        print(f"\n--- Testing with {dtype} ---")
        
        # Denormalize coordinates
        from oft.transformer.utils.normalization_utils import denormalize_coordinates, denormalize_dimensions
        
        coords_denorm = denormalize_coordinates(
            coords_norm.to(dtype), 
            point_cloud_range.to(dtype)
        )
        dims_denorm = denormalize_dimensions(dims_norm.to(dtype))
        
        print(f"Coords denormalized ({dtype}): {coords_denorm}")
        print(f"Dims denormalized ({dtype}): {dims_denorm}")
        
        # Test yaw conversion
        yaw_angle = torch.atan2(sin_yaw_norm, cos_yaw_norm)
        print(f"Yaw angle ({dtype}): {yaw_angle} rad = {yaw_angle * 180 / np.pi} deg")


def test_confidence_filtering():
    """Test that confidence filtering works correctly"""
    cfg, batch_dict, predictions = setup_test_data()
    
    print("\n🔍 TESTING CONFIDENCE FILTERING")
    print("-" * 35)
    
    # Test with different confidence thresholds
    for conf_th in [0.1, 0.3, 0.5, 0.9]:
        cfg['evaluation']['conf_th_eval'] = conf_th
        result = reconstruct_and_convert_predictions_autoregressive(
            batch_dict, predictions, cfg, ego_pose_current=None
        )
        
        print(f"\nConfidence threshold: {conf_th}")
        for i, sample_result in enumerate(result):
            scores = sample_result['score']
            below_threshold = [s for s in scores if s < conf_th]
            
            print(f"  Sample {i}: {len(scores)} predictions, scores: {[f'{s:.3f}' for s in scores]}")
            if below_threshold:
                print(f"    ❌ Found scores below threshold: {below_threshold}")
            else:
                print(f"    ✅ All scores above threshold")


def main():
    """Run all debug tests"""
    print("🔬 PREDICTION_UTILS DEBUG ANALYSIS")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_coordinate_transformation_consistency()
        test_denormalization_step_by_step()
        test_confidence_filtering()
        
        print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
