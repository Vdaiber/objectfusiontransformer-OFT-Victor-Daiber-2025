#!/usr/bin/env python3
"""
Analysis and fix for camera coordinate system issue.
The problem: Camera is pointing backwards, causing many objects to appear behind it.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from truckscenes import TruckScenes
from pyquaternion import Quaternion


def analyze_camera_coordinate_system():
    """Analyze the camera coordinate system issue."""
    
    print("🔍 CAMERA COORDINATE SYSTEM ANALYSIS")
    print("=" * 80)
    
    ts = TruckScenes(version="v1.0-mini", dataroot="/data", verbose=False)
    sample_token = "f1c03220990143e19b983bf3da478764"
    camera_channel = "CAMERA_LEFT_FRONT"
    
    # Get camera setup
    sample_record = ts.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = ts.getclosest('ego_pose', sample_record['timestamp'])
    
    print(f"Sample: {sample_token}")
    print(f"Camera: {camera_channel}")
    print(f"Timestamp consistency: {abs(sample_record['timestamp'] - pose_record['timestamp']) / 1000:.2f} ms")
    
    # Analyze coordinate frames
    print(f"\n📐 COORDINATE FRAME ANALYSIS:")
    
    # Camera extrinsics
    cam_translation = np.array(cs_record['translation'])
    cam_rotation = Quaternion(cs_record['rotation'])
    
    print(f"Camera translation (ego → camera): {cam_translation}")
    print(f"Camera rotation quaternion: {cs_record['rotation']}")
    
    # Test coordinate system orientation
    # Standard camera coordinates: X=right, Y=down, Z=forward
    ego_axes = {
        'forward': np.array([1, 0, 0]),  # Ego X-axis (forward)
        'left': np.array([0, 1, 0]),     # Ego Y-axis (left)
        'up': np.array([0, 0, 1])        # Ego Z-axis (up)
    }
    
    print(f"\n🧭 COORDINATE TRANSFORMATION:")
    print("Ego frame axes transformed to camera frame:")
    
    for axis_name, ego_axis in ego_axes.items():
        # Transform ego axis to camera frame
        cam_axis = cam_rotation.rotate(ego_axis)
        print(f"  Ego {axis_name} {ego_axis} → Camera {cam_axis}")
    
    # Check what direction the camera is pointing
    ego_forward = np.array([1, 0, 0])  # Forward in ego coordinates
    camera_forward_in_ego = cam_rotation.inverse.rotate(np.array([0, 0, 1]))  # Camera Z-axis in ego frame
    
    print(f"\n📷 CAMERA POINTING DIRECTION:")
    print(f"Camera Z-axis (forward) in ego frame: {camera_forward_in_ego}")
    
    # Dot product to check alignment
    dot_product = np.dot(ego_forward, camera_forward_in_ego)
    print(f"Dot product with ego forward: {dot_product:.3f}")
    
    if dot_product > 0.5:
        print("✅ Camera pointing forward")
    elif dot_product < -0.5:
        print("❌ Camera pointing backward!")
    else:
        print("⚠️ Camera pointing sideways")
    
    # Analyze why objects are behind camera
    print(f"\n🎯 OBJECT PLACEMENT ANALYSIS:")
    
    # Load some predictions to see their distribution
    with open('tests/visualization_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    transforms = analysis['transformations']['transformations']
    
    # Check object positions relative to ego vehicle
    ego_pos = np.array(pose_record['translation'])
    
    object_positions = []
    for transform in transforms[:10]:  # First 10 objects
        world_pos = np.array(transform['world_center'])
        relative_pos = world_pos - ego_pos
        
        # Transform to ego frame
        ego_rotation = Quaternion(pose_record['rotation'])
        ego_relative = ego_rotation.inverse.rotate(relative_pos)
        
        object_positions.append({
            'name': transform['detection_name'],
            'score': transform['detection_score'],
            'world_pos': world_pos,
            'ego_relative': ego_relative,
            'camera_pos': transform['camera_center'],
            'camera_z': transform['camera_z_depth']
        })
    
    print("Sample object positions:")
    print("Name         Score  Ego_X   Ego_Y   Ego_Z   Cam_Z")
    print("-" * 50)
    for obj in object_positions:
        print(f"{obj['name']:<12} {obj['score']:.2f}  "
              f"{obj['ego_relative'][0]:6.1f}  {obj['ego_relative'][1]:6.1f}  {obj['ego_relative'][2]:6.1f}  "
              f"{obj['camera_z']:6.1f}")
    
    # Check if there's a systematic issue
    positive_z = sum(1 for obj in object_positions if obj['camera_z'] > 0)
    negative_z = sum(1 for obj in object_positions if obj['camera_z'] < 0)
    
    print(f"\nCamera Z distribution in sample:")
    print(f"  Positive Z (in front): {positive_z}")
    print(f"  Negative Z (behind): {negative_z}")
    
    return {
        'camera_translation': cam_translation.tolist(),
        'camera_rotation': cs_record['rotation'],
        'camera_forward_in_ego': camera_forward_in_ego.tolist(),
        'forward_alignment': float(dot_product),
        'objects_in_front': positive_z,
        'objects_behind': negative_z
    }


def propose_fix():
    """Propose a fix for the camera coordinate system issue."""
    
    print(f"\n🔧 PROPOSED FIXES:")
    print("=" * 50)
    
    print("ISSUE IDENTIFIED:")
    print("  • Camera coordinate transformation is causing objects to appear behind camera")
    print("  • Forward direction in camera frame: [0, 0, -1] (negative Z)")
    print("  • This suggests camera Z-axis is flipped or rotation is incorrect")
    
    print(f"\nPOSSIBLE CAUSES:")
    print("  1. Camera rotation quaternion is incorrect/inverted")
    print("  2. Camera coordinate system convention mismatch")
    print("  3. Ego-to-camera transformation order is wrong")
    print("  4. DevKit coordinate system assumption is incorrect")
    
    print(f"\nSUGGESTED FIXES:")
    print("  1. Check TruckScenes DevKit documentation for coordinate conventions")
    print("  2. Verify camera calibration data in dataset")
    print("  3. Test with inverted camera rotation:")
    print("     cam_rotation.inverse or conjugate")
    print("  4. Check if camera Z-axis should be negated")
    print("  5. Compare with working samples from other cameras")
    
    print(f"\nIMPACT:")
    print("  • Current: 20/31 objects behind camera (depth fail)")
    print("  • Expected: Most objects should be in front (positive Z)")
    print("  • Fix should increase visible objects from 11 to ~25-30")


def test_alternative_transformations():
    """Test alternative transformation approaches."""
    
    print(f"\n🧪 TESTING ALTERNATIVE TRANSFORMATIONS:")
    print("=" * 60)
    
    ts = TruckScenes(version="v1.0-mini", dataroot="/data", verbose=False)
    sample_token = "f1c03220990143e19b983bf3da478764"
    camera_channel = "CAMERA_LEFT_FRONT"
    
    # Load test prediction
    with open('tests/visualization_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    sample_predictions = analysis['predictions']['sample_predictions']
    first_pred = sample_predictions[0]  # Test with first prediction
    
    # Get transformation components
    sample_record = ts.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = ts.getclosest('ego_pose', sample_record['timestamp'])
    
    # Original world position
    world_pos = np.array(first_pred['translation'])
    world_rot = Quaternion(first_pred['rotation'])
    
    print(f"Testing with: {first_pred['detection_name']} (score: {first_pred['detection_score']:.2f})")
    print(f"World position: {world_pos}")
    
    # Test different transformation approaches
    approaches = {
        'original': {},
        'invert_camera_rotation': {'invert_camera': True},
        'flip_z_axis': {'flip_z': True},
        'both_fixes': {'invert_camera': True, 'flip_z': True}
    }
    
    for approach_name, params in approaches.items():
        print(f"\n--- {approach_name.upper()} ---")
        
        # World → Ego
        ego_pos = world_pos - np.array(pose_record['translation'])
        ego_rot = Quaternion(pose_record['rotation']).inverse
        ego_pos_rotated = ego_rot.rotate(ego_pos)
        
        # Ego → Camera
        cam_pos = ego_pos_rotated - np.array(cs_record['translation'])
        cam_rot = Quaternion(cs_record['rotation'])
        
        if params.get('invert_camera', False):
            cam_rot = cam_rot.inverse
            
        cam_pos_rotated = cam_rot.rotate(cam_pos)
        
        if params.get('flip_z', False):
            cam_pos_rotated[2] = -cam_pos_rotated[2]
        
        print(f"Camera position: {cam_pos_rotated}")
        print(f"Camera Z: {cam_pos_rotated[2]:.2f} ({'FRONT' if cam_pos_rotated[2] > 0 else 'BEHIND'})")
    
    return approaches


if __name__ == "__main__":
    # Run analysis
    analysis = analyze_camera_coordinate_system()
    
    # Propose fixes
    propose_fix()
    
    # Test alternatives
    test_alternatives = test_alternative_transformations()
    
    # Save results
    results = {
        'analysis': analysis,
        'test_alternatives': test_alternatives
    }
    
    with open('tests/camera_coordinate_fix_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: tests/camera_coordinate_fix_analysis.json")
