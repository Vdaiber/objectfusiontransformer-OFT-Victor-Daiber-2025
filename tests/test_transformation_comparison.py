#!/usr/bin/env python3
"""
Direct comparison: Original transformation (11 visible) vs "fixed" transformation (6 visible).
Maybe the original was actually better!
"""

import os
import sys
import json
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevkitBox
from truckscenes.utils.geometry_utils import BoxVisibility, box_in_image
from pyquaternion import Quaternion


def test_both_transformations():
    """Test both original and 'fixed' transformations side by side."""
    
    print("🔍 TRANSFORMATION COMPARISON: ORIGINAL vs 'FIXED'")
    print("=" * 80)
    
    ts = TruckScenes(version="v1.0-mini", dataroot="/data", verbose=False)
    sample_token = "f1c03220990143e19b983bf3da478764"
    camera_channel = "CAMERA_LEFT_FRONT"
    
    # Get setup
    sample_record = ts.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = ts.getclosest('ego_pose', sample_record['timestamp'])
    
    # Load predictions
    with open('tests/visualization_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    sample_predictions = analysis['predictions']['sample_predictions']
    
    # Camera setup
    K = np.array(cs_record['camera_intrinsic'], dtype=np.float64).reshape(3, 3)
    import cv2
    image_path = os.path.join(ts.dataroot, sd_record['filename'])
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    imsize = (w, h)
    
    print(f"Testing with {len(sample_predictions)} predictions")
    
    results = {}
    
    # Test both transformations
    transformations = {
        'ORIGINAL (.inverse)': True,
        'FIXED (no .inverse)': False
    }
    
    for transform_name, use_inverse in transformations.items():
        print(f"\n--- {transform_name} ---")
        
        camera_boxes = []
        
        for i, pred in enumerate(sample_predictions):
            center_world = np.array(pred['translation'])
            size_world = np.array(pred['size'])
            rotation_world_quat = Quaternion(pred['rotation'])
            
            # Create world box
            world_box = DevkitBox(
                center=center_world,
                size=size_world,
                orientation=rotation_world_quat,
                name=pred['detection_name'],
                score=pred['detection_score'],
                token=f'pred_{i}'
            )
            
            # Transform to camera
            # World → Ego
            ego_box = world_box.copy()
            ego_box.translate(-np.array(pose_record['translation']))
            ego_box.rotate(Quaternion(pose_record['rotation']).inverse)
            
            # Ego → Camera - TEST BOTH WAYS
            camera_box = ego_box.copy()
            camera_box.translate(-np.array(cs_record['translation']))
            
            if use_inverse:
                camera_box.rotate(Quaternion(cs_record['rotation']).inverse)  # ORIGINAL
            else:
                camera_box.rotate(Quaternion(cs_record['rotation']))  # "FIXED"
            
            camera_boxes.append(camera_box)
        
        # Count results with different filters
        
        # 1. Custom FOV filter (original visualization logic)
        def custom_fov_filter(boxes):
            visible = []
            for box in boxes:
                corners_cam = box.corners()
                if np.any(corners_cam[2, :] <= 0):  # Behind camera
                    continue
                # Project to image
                homog_pts = K @ corners_cam
                u = homog_pts[0, :] / homog_pts[2, :]
                v = homog_pts[1, :] / homog_pts[2, :]
                # Check if any corner in image
                in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if np.any(in_img):
                    visible.append(box)
            return visible
        
        # 2. DevKit filter
        def devkit_filter(boxes):
            visible = []
            for box in boxes:
                if box_in_image(box, K, imsize, vis_level=BoxVisibility.ANY):
                    visible.append(box)
            return visible
        
        custom_visible = custom_fov_filter(camera_boxes)
        devkit_visible = devkit_filter(camera_boxes)
        
        print(f"Total camera boxes: {len(camera_boxes)}")
        print(f"Custom FOV filter: {len(custom_visible)} visible")
        print(f"DevKit filter: {len(devkit_visible)} visible")
        
        # Analyze depths
        depths = []
        behind_camera = 0
        for box in camera_boxes:
            corners = box.corners()
            min_depth = np.min(corners[2, :])
            depths.append(min_depth)
            if min_depth <= 0:
                behind_camera += 1
        
        print(f"Depth analysis:")
        print(f"  Behind camera (Z <= 0): {behind_camera}")
        print(f"  Depth range: [{min(depths):.1f}, {max(depths):.1f}]")
        
        results[transform_name] = {
            'total_boxes': len(camera_boxes),
            'custom_fov_visible': len(custom_visible),
            'devkit_visible': len(devkit_visible),
            'behind_camera': behind_camera,
            'depth_range': [min(depths), max(depths)]
        }
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    print("Transformation        Total  Custom_FOV  DevKit  Behind_Cam")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<20} {data['total_boxes']:5d}  {data['custom_fov_visible']:9d}  {data['devkit_visible']:6d}  {data['behind_camera']:10d}")
    
    # Check which one matches the original 11 visible
    original_custom = results['ORIGINAL (.inverse)']['custom_fov_visible']
    fixed_custom = results['FIXED (no .inverse)']['custom_fov_visible']
    
    print(f"\n🎯 ANALYSIS:")
    print(f"Original (.inverse) + Custom FOV: {original_custom} visible")
    print(f"Fixed (no .inverse) + Custom FOV: {fixed_custom} visible")
    print(f"From logs we know original visualization showed: 11 visible")
    
    if original_custom == 11:
        print("✅ ORIGINAL transformation matches logged results!")
        print("❌ 'Fixed' transformation is actually WORSE")
    elif fixed_custom == 11:
        print("✅ 'Fixed' transformation matches logged results!")
        print("❌ Original transformation was wrong")
    else:
        print("⚠️ Neither transformation exactly matches logged 11 visible")
        print("There might be another factor (confidence threshold, etc.)")
    
    return results


def analyze_devkit_coordinate_convention():
    """Check TruckScenes coordinate system documentation."""
    
    print(f"\n📚 DEVKIT COORDINATE SYSTEM CHECK:")
    print("=" * 50)
    
    # Check if we can find coordinate system info in DevKit
    ts = TruckScenes(version="v1.0-mini", dataroot="/data", verbose=False)
    sample_token = "f1c03220990143e19b983bf3da478764"
    camera_channel = "CAMERA_LEFT_FRONT"
    
    sample_record = ts.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    
    print(f"Calibrated sensor record:")
    print(f"  Translation: {cs_record['translation']}")
    print(f"  Rotation: {cs_record['rotation']}")
    
    # Check camera coordinate system by looking at how DevKit transforms GT
    _, gt_boxes_cam, _ = ts.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    
    print(f"\nGT boxes from DevKit (already in camera coords):")
    print(f"  Count: {len(gt_boxes_cam)}")
    
    if gt_boxes_cam:
        sample_gt = gt_boxes_cam[0]
        print(f"  Sample GT center: {sample_gt.center}")
        print(f"  Sample GT Z (depth): {sample_gt.center[2]}")
    
    # The key insight: DevKit's get_sample_data already gives us boxes in camera coordinates
    # So we should compare our transformation result with DevKit's result
    
    print(f"\n💡 KEY INSIGHT:")
    print("DevKit's get_sample_data() already returns boxes in camera coordinates.")
    print("We should check which of our transformations produces similar results.")


if __name__ == "__main__":
    # Test both transformations
    results = test_both_transformations()
    
    # Check DevKit convention
    analyze_devkit_coordinate_convention()
    
    # Save results
    with open('tests/transformation_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: tests/transformation_comparison.json")
    
    print(f"\n🎯 CONCLUSION:")
    print("Sie haben recht - wenn Original 11 sichtbare Boxen liefert")
    print("und 'Fixed' nur 6, dann war Original besser!")
    print("Das .inverse zu entfernen war möglicherweise FALSCH.")
