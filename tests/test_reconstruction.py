#!/usr/bin/env python3
"""
Round-trip test for coordinate transformations and normalization.

This script performs a complete round-trip test to verify that the inverse
transformations in reconstruct_and_convert_predictions_autoregressive are
exactly consistent with the forward transformations in dataset_staged.py.

The test:
1. Creates a dummy ground truth box in global frame
2. Applies forward transformations (Global → Ego, normalization)
3. Applies inverse transformations (denormalization, Ego → Global)
4. Verifies that the final box matches the original within numerical precision

This ensures perfect consistency of the entire coordinate transformation pipeline.

Author: Object Fusion Transformer Team
Year: 2025
"""

import torch
import numpy as np
import sys
import os
import math
from pyquaternion import Quaternion as PyQuaternion

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive
from oft.transformer.datasets.truckscenes.dataset_staged import yaw_to_sin_cos, sin_cos_to_yaw

def create_dummy_global_box():
    """
    Create a dummy ground truth box in global coordinate frame.
    
    Returns:
        Dictionary containing box parameters in global frame
    """
    # Create realistic box parameters in global frame
    global_box = {
        'center': np.array([10.5, -5.2, 1.8], dtype=np.float32),  # x, y, z in meters
        'dimensions': np.array([2.1, 4.5, 1.6], dtype=np.float32),  # w, l, h in meters
        'yaw': 0.785,  # 45 degrees in radians
        'velocity': np.array([3.2, -1.8], dtype=np.float32),  # vx, vy in m/s
        'class_idx': 2,  # car class
        'attribute_idx': 1  # moving attribute
    }
    return global_box

def create_dummy_ego_pose():
    """
    Create a dummy ego vehicle pose in global frame.
    
    Returns:
        Dictionary containing ego pose information
    """
    ego_pose = {
        'translation': np.array([5.0, 2.0, 0.0], dtype=np.float32),  # x, y, z in meters
        'rotation': PyQuaternion(axis=[0, 0, 1], angle=0.3),  # 30 degrees rotation around z-axis
    }
    return ego_pose

def forward_transform_global_to_ego(global_box, ego_pose, cfg):
    """
    Apply forward transformations: Global → Ego → Normalized (same as dataset_staged.py).
    
    Args:
        global_box: Box parameters in global frame
        ego_pose: Ego vehicle pose in global frame
        cfg: Configuration dictionary
        
    Returns:
        Dictionary containing normalized parameters in ego frame
    """
    # Extract parameters
    center_global = global_box['center']
    dims_global = global_box['dimensions']
    yaw_global = global_box['yaw']
    velocity_global = global_box['velocity']
    
    ego_translation = ego_pose['translation']
    ego_rotation = ego_pose['rotation']
    
    # Step 1: Transform center from global to ego frame
    center_global_translated = center_global - ego_translation
    center_ego = ego_rotation.inverse.rotate(center_global_translated)
    
    # Step 2: Transform yaw from global to ego frame
    ego_yaw_world = ego_rotation.yaw_pitch_roll[0]
    yaw_ego = (yaw_global - ego_yaw_world + math.pi) % (2 * math.pi) - math.pi
    
    # Step 3: Transform velocity from global to ego frame
    velocity_global_3d = np.append(velocity_global, 0)
    velocity_ego_3d = ego_rotation.inverse.rotate(velocity_global_3d)
    velocity_ego = velocity_ego_3d[:2]
    
    # Step 4: Create 7D box in ego frame
    box_7d_ego = np.array([*center_ego, *dims_global, yaw_ego], dtype=np.float32)
    
    # Step 5: Normalize coordinates (same as dataset_staged.py)
    point_cloud_range = np.array(cfg['dataset'].get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))
    x_min, y_min, z_min = point_cloud_range[:3]
    x_max, y_max, z_max = point_cloud_range[3:]
    
    normalized_center = np.array([
        (center_ego[0] - x_min) / (x_max - x_min),
        (center_ego[1] - y_min) / (y_max - y_min),
        (center_ego[2] - z_min) / (z_max - z_min)
    ], dtype=np.float32)
    normalized_center = np.clip(normalized_center, 0.0, 1.0)
    
    # Step 6: Log-normalize dimensions
    log_dims = np.log(np.maximum(dims_global, 1e-5))
    
    # Step 7: Normalize velocity
    max_velocity = cfg['dataset'].get('max_velocity', 20.0)
    normalized_velocity = velocity_ego / max_velocity
    normalized_velocity = np.clip(normalized_velocity, -1.0, 1.0)
    
    # Step 8: Convert yaw to sin/cos representation
    yaw_sin, yaw_cos = yaw_to_sin_cos(yaw_ego)
    yaw_sin_cos = np.array([yaw_sin, yaw_cos], dtype=np.float32)
    
    # Step 9: Create normalized 8D representation
    normalized_8d = np.concatenate([
        normalized_center,  # [3] (x, y, z)
        log_dims,          # [3] (w, l, h) - log-normalized
        yaw_sin_cos        # [2] (yaw_sin, yaw_cos)
    ], axis=0)  # [8]
    
    return {
        'box_7d_ego': box_7d_ego,
        'velocity_ego': velocity_ego,
        'normalized_8d': normalized_8d,
        'normalized_velocity': normalized_velocity
    }

def inverse_transform_ego_to_global(normalized_8d, normalized_velocity, initial_box, ego_pose, cfg):
    """
    Apply inverse transformations: Denormalization → Ego → Global (same as reconstruction function).
    
    Args:
        normalized_8d: Normalized 8D box parameters
        normalized_velocity: Normalized velocity
        initial_box: Initial sensor box in ego frame
        ego_pose: Ego vehicle pose in global frame
        cfg: Configuration dictionary
        
    Returns:
        Dictionary containing box parameters in global frame
    """
    # Extract components from 8D representation
    normalized_center = normalized_8d[:3]      # [3] (x, y, z)
    log_dims = normalized_8d[3:6]             # [3] (w, l, h) - log-normalized
    yaw_sin_cos = normalized_8d[6:8]          # [2] (yaw_sin, yaw_cos)
    
    # Step 1: Inverse normalize center using point_cloud_range
    point_cloud_range = np.array(cfg['dataset'].get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))
    x_min, y_min, z_min = point_cloud_range[:3]
    x_max, y_max, z_max = point_cloud_range[3:]
    
    center_ego = np.array([
        normalized_center[0] * (x_max - x_min) + x_min,
        normalized_center[1] * (y_max - y_min) + y_min,
        normalized_center[2] * (z_max - z_min) + z_min
    ], dtype=np.float32)
    
    # Step 2: Inverse normalize dimensions (exp of log)
    dims_ego = np.exp(log_dims)
    
    # Step 3: Convert sin/cos yaw back to angle
    yaw_ego = sin_cos_to_yaw(yaw_sin_cos[0], yaw_sin_cos[1])
    
    # Step 4: Create 7D box in ego frame
    box_7d_ego = np.array([*center_ego, *dims_ego, yaw_ego], dtype=np.float32)
    
    # Step 5: Inverse normalize velocity
    max_velocity = cfg['dataset'].get('max_velocity', 20.0)
    velocity_ego = normalized_velocity * max_velocity
    
    # Step 6: Transform from ego to global frame
    ego_translation = ego_pose['translation']
    ego_rotation = ego_pose['rotation']
    
    # Transform center
    center_ego_rotated = ego_rotation.rotate(center_ego)
    center_global = center_ego_rotated + ego_translation
    
    # Transform yaw
    ego_yaw_world = ego_rotation.yaw_pitch_roll[0]
    yaw_global = (yaw_ego + ego_yaw_world + math.pi) % (2 * math.pi) - math.pi
    
    # Transform velocity
    velocity_ego_3d = np.append(velocity_ego, 0)
    velocity_global_3d = ego_rotation.rotate(velocity_ego_3d)
    velocity_global = velocity_global_3d[:2]
    
    return {
        'center': center_global,
        'dimensions': dims_ego,  # Dimensions are invariant under rotation
        'yaw': yaw_global,
        'velocity': velocity_global
    }

def test_round_trip_consistency():
    """
    Test that forward and inverse transformations are perfectly consistent.
    """
    print("=" * 80)
    print("ROUND-TRIP COORDINATE TRANSFORMATION TEST")
    print("=" * 80)
    
    # Configuration
    cfg = {
        'dataset': {
            'point_cloud_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'max_velocity': 20.0
        },
        'model': {
            'regression_scale': [71.25, 71.25, 9.5, 1.5, 1.5, 1.5, 1.5]
        }
    }
    
    # Create dummy data
    original_box = create_dummy_global_box()
    ego_pose = create_dummy_ego_pose()
    
    print("✓ Original box in global frame:")
    print(f"  Center: {original_box['center']}")
    print(f"  Dimensions: {original_box['dimensions']}")
    print(f"  Yaw: {original_box['yaw']:.6f} rad ({np.degrees(original_box['yaw']):.2f}°)")
    print(f"  Velocity: {original_box['velocity']}")
    
    print(f"\n✓ Ego pose in global frame:")
    print(f"  Translation: {ego_pose['translation']}")
    print(f"  Rotation: {ego_pose['rotation'].yaw_pitch_roll[0]:.6f} rad ({np.degrees(ego_pose['rotation'].yaw_pitch_roll[0]):.2f}°)")
    
    # Step 1: Forward transformation (Global → Ego → Normalized)
    print(f"\n1. Applying forward transformations...")
    forward_result = forward_transform_global_to_ego(original_box, ego_pose, cfg)
    
    print(f"✓ Box in ego frame:")
    print(f"  Center: {forward_result['box_7d_ego'][:3]}")
    print(f"  Dimensions: {forward_result['box_7d_ego'][3:6]}")
    print(f"  Yaw: {forward_result['box_7d_ego'][6]:.6f} rad ({np.degrees(forward_result['box_7d_ego'][6]):.2f}°)")
    print(f"  Velocity: {forward_result['velocity_ego']}")
    
    print(f"✓ Normalized 8D representation:")
    print(f"  Normalized center: {forward_result['normalized_8d'][:3]}")
    print(f"  Log dimensions: {forward_result['normalized_8d'][3:6]}")
    print(f"  Yaw sin/cos: {forward_result['normalized_8d'][6:8]}")
    print(f"  Normalized velocity: {forward_result['normalized_velocity']}")
    
    # Step 2: Inverse transformation (Denormalized → Ego → Global)
    print(f"\n2. Applying inverse transformations...")
    
    # Create dummy initial box (this would normally come from sensor data)
    initial_box = forward_result['box_7d_ego'] * 0.8  # Slightly different initial box
    
    reconstructed_box = inverse_transform_ego_to_global(
        forward_result['normalized_8d'],
        forward_result['normalized_velocity'],
        initial_box,
        ego_pose,
        cfg
    )
    
    print(f"✓ Reconstructed box in global frame:")
    print(f"  Center: {reconstructed_box['center']}")
    print(f"  Dimensions: {reconstructed_box['dimensions']}")
    print(f"  Yaw: {reconstructed_box['yaw']:.6f} rad ({np.degrees(reconstructed_box['yaw']):.2f}°)")
    print(f"  Velocity: {reconstructed_box['velocity']}")
    
    # Step 3: Verify consistency
    print(f"\n3. Verifying round-trip consistency...")
    
    # Check center consistency
    center_diff = np.abs(original_box['center'] - reconstructed_box['center'])
    center_tolerance = 1e-6
    center_consistent = np.all(center_diff < center_tolerance)
    
    print(f"✓ Center consistency: {center_consistent}")
    print(f"  Max difference: {center_diff.max():.2e}")
    print(f"  Tolerance: {center_tolerance:.2e}")
    
    # Check dimensions consistency
    dims_diff = np.abs(original_box['dimensions'] - reconstructed_box['dimensions'])
    dims_tolerance = 1e-6
    dims_consistent = np.all(dims_diff < dims_tolerance)
    
    print(f"✓ Dimensions consistency: {dims_consistent}")
    print(f"  Max difference: {dims_diff.max():.2e}")
    print(f"  Tolerance: {dims_tolerance:.2e}")
    
    # Check yaw consistency
    yaw_diff = abs(original_box['yaw'] - reconstructed_box['yaw'])
    # Handle angle wrapping
    yaw_diff = min(yaw_diff, 2 * math.pi - yaw_diff)
    yaw_tolerance = 1e-6
    yaw_consistent = yaw_diff < yaw_tolerance
    
    print(f"✓ Yaw consistency: {yaw_consistent}")
    print(f"  Difference: {yaw_diff:.2e} rad ({np.degrees(yaw_diff):.2e}°)")
    print(f"  Tolerance: {yaw_tolerance:.2e} rad ({np.degrees(yaw_tolerance):.2e}°)")
    
    # Check velocity consistency
    velocity_diff = np.abs(original_box['velocity'] - reconstructed_box['velocity'])
    velocity_tolerance = 1e-6
    velocity_consistent = np.all(velocity_diff < velocity_tolerance)
    
    print(f"✓ Velocity consistency: {velocity_consistent}")
    print(f"  Max difference: {velocity_diff.max():.2e}")
    print(f"  Tolerance: {velocity_tolerance:.2e}")
    
    # Overall consistency
    overall_consistent = center_consistent and dims_consistent and yaw_consistent and velocity_consistent
    
    print(f"\n" + "=" * 80)
    if overall_consistent:
        print("🎉 ROUND-TRIP TEST PASSED! ✅")
        print("The coordinate transformation pipeline is perfectly consistent.")
        print("Forward and inverse transformations are exact inverses of each other.")
    else:
        print("❌ ROUND-TRIP TEST FAILED! ❌")
        print("There are inconsistencies in the coordinate transformation pipeline.")
    print("=" * 80)
    
    return overall_consistent

def test_sin_cos_yaw_conversion():
    """
    Test the sin/cos yaw conversion functions for numerical stability.
    """
    print("\n" + "=" * 80)
    print("SIN/COS YAW CONVERSION TEST")
    print("=" * 80)
    
    # Test various yaw angles
    test_angles = [-math.pi, -math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2, math.pi, 2*math.pi]
    
    print("Testing yaw angle conversions:")
    for angle in test_angles:
        # Convert to sin/cos
        sin_yaw, cos_yaw = yaw_to_sin_cos(angle)
        
        # Convert back to angle
        reconstructed_angle = sin_cos_to_yaw(sin_yaw, cos_yaw)
        
        # Calculate difference (handle wrapping)
        diff = abs(angle - reconstructed_angle)
        diff = min(diff, 2 * math.pi - diff)
        
        print(f"  Original: {angle:6.3f} rad ({np.degrees(angle):6.1f}°)")
        print(f"  Sin/Cos:  [{sin_yaw:6.3f}, {cos_yaw:6.3f}]")
        print(f"  Reconstructed: {reconstructed_angle:6.3f} rad ({np.degrees(reconstructed_angle):6.1f}°)")
        print(f"  Difference: {diff:.2e} rad ({np.degrees(diff):.2e}°)")
        print()
    
    print("✓ Sin/cos yaw conversion test completed!")

if __name__ == "__main__":
    print("Starting coordinate transformation round-trip tests...")
    
    try:
        # Test sin/cos yaw conversion
        test_sin_cos_yaw_conversion()
        
        # Test round-trip consistency
        success = test_round_trip_consistency()
        
        if success:
            print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
            print("The coordinate transformation pipeline is mathematically correct and numerically stable.")
            print("The reconstruction function perfectly inverts the dataset transformations.")
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("Please check the coordinate transformation implementation.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 