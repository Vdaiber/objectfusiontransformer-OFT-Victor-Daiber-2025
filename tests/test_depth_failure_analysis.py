#!/usr/bin/env python3
"""
Specific analysis of depth failures in visualization pipeline.
Analyzes why 20 out of 31 predicted boxes fail the depth test (negative Z values).
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


def analyze_depth_failures():
    """Analyze why predicted boxes have negative Z values in camera coordinates."""
    
    # Load analysis results
    with open('tests/visualization_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    print("🔍 DEPTH FAILURE ANALYSIS")
    print("=" * 80)
    
    # Extract failed boxes
    fov_analysis = analysis['fov_filtering']
    failed_boxes = [box for box in fov_analysis['box_analysis'] if not box['depth_pass']]
    passed_boxes = [box for box in fov_analysis['box_analysis'] if box['depth_pass']]
    
    print(f"Total boxes: {len(fov_analysis['box_analysis'])}")
    print(f"Failed depth test: {len(failed_boxes)}")
    print(f"Passed depth test: {len(passed_boxes)}")
    
    # Analyze failure patterns
    print(f"\n📊 DEPTH FAILURE PATTERNS:")
    
    # Group by detection name
    failure_by_class = {}
    for box in failed_boxes:
        class_name = box['name']
        if class_name not in failure_by_class:
            failure_by_class[class_name] = []
        failure_by_class[class_name].append(box)
    
    print(f"\nFailures by class:")
    for class_name, boxes in failure_by_class.items():
        depths = [box['depth_range'] for box in boxes]
        min_depths = [d[0] for d in depths]
        max_depths = [d[1] for d in depths]
        print(f"  • {class_name}: {len(boxes)} boxes")
        print(f"    - Depth range: [{min(min_depths):.1f}, {max(max_depths):.1f}]")
        print(f"    - Avg min depth: {np.mean(min_depths):.1f}")
        print(f"    - Avg max depth: {np.mean(max_depths):.1f}")
    
    # Analyze transformations leading to negative depths
    print(f"\n🔄 TRANSFORMATION ANALYSIS:")
    transform_analysis = analysis['transformations']
    
    if 'transformations' in transform_analysis:
        transforms = transform_analysis['transformations']
        
        # Check world coordinates
        world_coords = [(t['world_center'][0], t['world_center'][1], t['world_center'][2]) 
                       for t in transforms]
        camera_coords = [(t['camera_center'][0], t['camera_center'][1], t['camera_center'][2]) 
                        for t in transforms]
        
        print(f"Sample of world coordinates (X, Y, Z):")
        for i, (world, camera) in enumerate(zip(world_coords[:5], camera_coords[:5])):
            print(f"  Box {i}: World {world} → Camera {camera}")
            
        # Analyze coordinate ranges
        world_x = [c[0] for c in world_coords]
        world_y = [c[1] for c in world_coords]
        world_z = [c[2] for c in world_coords]
        
        camera_x = [c[0] for c in camera_coords]
        camera_y = [c[1] for c in camera_coords]
        camera_z = [c[2] for c in camera_coords]
        
        print(f"\nCoordinate ranges:")
        print(f"  World X: [{min(world_x):.1f}, {max(world_x):.1f}]")
        print(f"  World Y: [{min(world_y):.1f}, {max(world_y):.1f}]")
        print(f"  World Z: [{min(world_z):.1f}, {max(world_z):.1f}]")
        print(f"  Camera X: [{min(camera_x):.1f}, {max(camera_x):.1f}]")
        print(f"  Camera Y: [{min(camera_y):.1f}, {max(camera_y):.1f}]")
        print(f"  Camera Z: [{min(camera_z):.1f}, {max(camera_z):.1f}]")
        
        # Identify boxes behind camera
        behind_camera = [i for i, z in enumerate(camera_z) if z < 0]
        print(f"\nBoxes behind camera (negative Z): {len(behind_camera)}")
        
        for i in behind_camera[:10]:  # Show first 10
            transform = transforms[i]
            print(f"  Box {i} ({transform['detection_name']}, score={transform['detection_score']:.2f}):")
            print(f"    World: {transform['world_center']}")
            print(f"    Camera: {transform['camera_center']} (Z={transform['camera_z_depth']:.1f})")
    
    # Analyze camera setup
    print(f"\n📷 CAMERA SETUP ANALYSIS:")
    ts = TruckScenes(version="v1.0-mini", dataroot="/data", verbose=False)
    sample_token = analysis['test_sample']
    camera_channel = analysis['camera_channel']
    
    sample_record = ts.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = ts.getclosest('ego_pose', sample_record['timestamp'])
    
    print(f"Camera intrinsics:")
    K = np.array(cs_record['camera_intrinsic'])
    print(f"  {K}")
    
    print(f"Camera extrinsics (translation):")
    print(f"  {cs_record['translation']}")
    
    print(f"Camera extrinsics (rotation):")
    print(f"  {cs_record['rotation']}")
    
    print(f"Ego pose (translation):")
    print(f"  {pose_record['translation']}")
    
    print(f"Ego pose (rotation):")
    print(f"  {pose_record['rotation']}")
    
    # Check if camera is pointing backwards
    camera_rot = Quaternion(cs_record['rotation'])
    print(f"\nCamera orientation analysis:")
    print(f"  Rotation quaternion: {cs_record['rotation']}")
    
    # Transform forward vector (0, 1, 0) by camera rotation
    forward_ego = np.array([0, 1, 0])  # Forward in ego coordinates
    forward_camera = camera_rot.rotate(forward_ego)
    print(f"  Forward direction in camera frame: {forward_camera}")
    
    # Check camera coordinate system
    # In camera coordinates: X=right, Y=down, Z=forward
    # If Z is negative, objects are behind the camera
    
    return {
        'total_boxes': len(fov_analysis['box_analysis']),
        'depth_failures': len(failed_boxes),
        'depth_passes': len(passed_boxes),
        'failure_by_class': failure_by_class,
        'coordinate_analysis': {
            'world_ranges': {
                'x': [min(world_x), max(world_x)],
                'y': [min(world_y), max(world_y)],
                'z': [min(world_z), max(world_z)]
            },
            'camera_ranges': {
                'x': [min(camera_x), max(camera_x)],
                'y': [min(camera_y), max(camera_y)],
                'z': [min(camera_z), max(camera_z)]
            },
            'behind_camera_count': len(behind_camera)
        }
    }


if __name__ == "__main__":
    analysis = analyze_depth_failures()
    
    print(f"\n💾 Saving detailed depth failure analysis...")
    with open('tests/depth_failure_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"Analysis saved to: tests/depth_failure_analysis.json")
