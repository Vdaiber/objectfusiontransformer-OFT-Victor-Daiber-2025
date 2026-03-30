#!/usr/bin/env python3
"""
Match DevKit's exact visibility logic for predictions.
The goal: Get exactly 14 visible predicted boxes, same as GT boxes.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevkitBox
from truckscenes.utils.geometry_utils import BoxVisibility, box_in_image, view_points
from pyquaternion import Quaternion


def apply_devkit_visibility_filter(boxes: List[DevkitBox], K: np.ndarray, img_shape: Tuple[int, int]) -> List[DevkitBox]:
    """Apply exact DevKit visibility logic like GT boxes."""
    h, w = img_shape[:2]
    imsize = (w, h)  # DevKit expects (width, height)
    
    visible_boxes = []
    for box in boxes:
        # Use DevKit's exact box_in_image function
        if box_in_image(box, K, imsize, vis_level=BoxVisibility.ANY):
            visible_boxes.append(box)
    
    return visible_boxes


def analyze_devkit_visibility_logic():
    """Analyze DevKit's exact visibility requirements."""
    
    print("🔍 DEVKIT VISIBILITY LOGIC ANALYSIS")
    print("=" * 80)
    
    # DevKit's box_in_image function requirements:
    print("DevKit's box_in_image() requirements for BoxVisibility.ANY:")
    print("1. At least one corner visible in image bounds")
    print("2. All corners must be at least 0.1m in front of camera (Z > 0.1)")
    print("3. Additional constraint: corners_3d[2, :] > 1 for visibility")
    
    # Load our analysis data
    with open('tests/visualization_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    transform_analysis = analysis['transformations']
    
    if 'camera_boxes' not in transform_analysis:
        print("❌ No camera boxes found in analysis")
        return
    
    # Get DevKit setup
    ts = TruckScenes(version="v1.0-mini", dataroot="/data", verbose=False)
    sample_token = analysis['test_sample']
    camera_channel = analysis['camera_channel']
    
    sample_record = ts.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    
    # Camera setup
    K = np.array(cs_record['camera_intrinsic'], dtype=np.float64).reshape(3, 3)
    image_path = os.path.join(ts.dataroot, sd_record['filename'])
    
    import cv2
    image = cv2.imread(image_path)
    img_shape = image.shape
    h, w = img_shape[:2]
    imsize = (w, h)
    
    print(f"\nImage dimensions: {img_shape}")
    print(f"Camera intrinsics: {K}")
    
    # Test with corrected transformation (no camera rotation inversion)
    print(f"\n🔄 TESTING CORRECTED TRANSFORMATION:")
    
    # Load predictions
    sample_predictions = analysis['predictions']['sample_predictions']
    pose_record = ts.getclosest('ego_pose', sample_record['timestamp'])
    
    # Create camera boxes with CORRECT transformation
    corrected_camera_boxes = []
    
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
        
        # CORRECTED TRANSFORMATION: World → Ego → Camera
        # Step 1: World → Ego
        ego_box = world_box.copy()
        ego_box.translate(-np.array(pose_record['translation']))
        ego_box.rotate(Quaternion(pose_record['rotation']).inverse)
        
        # Step 2: Ego → Camera (WITHOUT .inverse - this was the bug!)
        camera_box = ego_box.copy()
        camera_box.translate(-np.array(cs_record['translation']))
        camera_box.rotate(Quaternion(cs_record['rotation']))  # CORRECTED: No .inverse!
        
        corrected_camera_boxes.append(camera_box)
    
    print(f"Created {len(corrected_camera_boxes)} camera boxes with corrected transformation")
    
    # Apply DevKit visibility filter
    visible_corrected = apply_devkit_visibility_filter(corrected_camera_boxes, K, img_shape)
    
    print(f"\n📊 VISIBILITY RESULTS:")
    print(f"Original predictions: {len(sample_predictions)}")
    print(f"After corrected transformation: {len(corrected_camera_boxes)}")
    print(f"After DevKit visibility filter: {len(visible_corrected)}")
    print(f"Target GT boxes visible: 14")
    
    # Detailed analysis of visibility filtering
    print(f"\n🔎 DETAILED VISIBILITY ANALYSIS:")
    
    failed_details = {
        'depth_shallow': 0,  # Z <= 0.1
        'depth_negative': 0,  # Z <= 0  
        'depth_visibility': 0,  # Z <= 1 (DevKit's additional constraint)
        'outside_image': 0,   # No corners in image bounds
        'passed': 0
    }
    
    for i, box in enumerate(corrected_camera_boxes):
        corners_3d = box.corners()
        corners_img = view_points(corners_3d, K, normalize=True)[:2, :]
        
        # Check depth constraints
        min_depth = np.min(corners_3d[2, :])
        all_in_front_01 = np.all(corners_3d[2, :] > 0.1)
        all_in_front_1 = np.all(corners_3d[2, :] > 1)
        
        # Check image bounds
        visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < w)
        visible = np.logical_and(visible, corners_img[1, :] < h)
        visible = np.logical_and(visible, corners_img[1, :] > 0)
        visible = np.logical_and(visible, corners_3d[2, :] > 1)  # DevKit's constraint
        any_visible = np.any(visible)
        
        # DevKit's full check
        devkit_visible = box_in_image(box, K, imsize, vis_level=BoxVisibility.ANY)
        
        if not all_in_front_01:
            failed_details['depth_shallow'] += 1
        elif not all_in_front_1:
            failed_details['depth_visibility'] += 1
        elif not any_visible:
            failed_details['outside_image'] += 1
        elif devkit_visible:
            failed_details['passed'] += 1
        
        if i < 10:  # Show details for first 10
            print(f"  Box {i} ({box.name}, score={getattr(box, 'score', 0):.2f}):")
            print(f"    Min depth: {min_depth:.2f}, All > 0.1: {all_in_front_01}, All > 1: {all_in_front_1}")
            print(f"    Any visible: {any_visible}, DevKit result: {devkit_visible}")
    
    print(f"\nFailure breakdown:")
    for reason, count in failed_details.items():
        print(f"  {reason}: {count}")
    
    # Compare with GT boxes
    print(f"\n🎯 COMPARISON WITH GT BOXES:")
    _, gt_boxes_cam, _ = ts.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    print(f"GT boxes from DevKit: {len(gt_boxes_cam)}")
    
    # Check GT box depths
    gt_depths = []
    for gt_box in gt_boxes_cam:
        gt_corners = gt_box.corners()
        gt_min_depth = np.min(gt_corners[2, :])
        gt_depths.append(gt_min_depth)
    
    print(f"GT box depth range: [{min(gt_depths):.1f}, {max(gt_depths):.1f}]")
    
    # Check predicted box depths
    pred_depths = []
    for pred_box in visible_corrected:
        pred_corners = pred_box.corners()
        pred_min_depth = np.min(pred_corners[2, :])
        pred_depths.append(pred_min_depth)
    
    if pred_depths:
        print(f"Visible pred box depth range: [{min(pred_depths):.1f}, {max(pred_depths):.1f}]")
    
    return {
        'original_predictions': len(sample_predictions),
        'corrected_camera_boxes': len(corrected_camera_boxes),
        'visible_after_devkit_filter': len(visible_corrected),
        'target_gt_boxes': 14,
        'failure_breakdown': failed_details,
        'gt_depth_range': [min(gt_depths), max(gt_depths)] if gt_depths else None,
        'pred_depth_range': [min(pred_depths), max(pred_depths)] if pred_depths else None
    }


def create_fixed_visualization_function():
    """Create a corrected version of the visualization function."""
    
    print(f"\n🔧 CREATING FIXED VISUALIZATION FUNCTION:")
    
    fixed_code = '''
def _filter_boxes_with_devkit_logic(boxes: List[DevkitBox], K: np.ndarray, img_shape) -> List[DevkitBox]:
    """Filter boxes using exact DevKit visibility logic to match GT box count."""
    from truckscenes.utils.geometry_utils import box_in_image, BoxVisibility
    
    h, w = img_shape[:2]
    imsize = (w, h)  # DevKit expects (width, height)
    
    visible_boxes = []
    for box in boxes:
        # Use DevKit's exact box_in_image function
        if box_in_image(box, K, imsize, vis_level=BoxVisibility.ANY):
            visible_boxes.append(box)
    
    return visible_boxes

# CRITICAL FIX in epoch_visualization_workflow.py:
# Line ~171: Remove .inverse from camera rotation
# OLD (BUGGY):
# camera_box.rotate(Quaternion(cs_record['rotation']).inverse)
# NEW (CORRECT):
# camera_box.rotate(Quaternion(cs_record['rotation']))

# Line ~179: Replace custom FOV filter with DevKit logic
# OLD:
# fused_boxes = _filter_boxes_in_camera_fov(fused_boxes, K, image.shape)
# NEW:
# fused_boxes = _filter_boxes_with_devkit_logic(fused_boxes, K, image.shape)
'''
    
    print(fixed_code)
    
    # Save the fixed function
    with open('tests/fixed_visualization_function.py', 'w') as f:
        f.write(fixed_code)
    
    print(f"💾 Fixed function saved to: tests/fixed_visualization_function.py")


if __name__ == "__main__":
    # Run analysis
    results = analyze_devkit_visibility_logic()
    
    # Create fix
    create_fixed_visualization_function()
    
    # Save results
    with open('tests/devkit_visibility_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: tests/devkit_visibility_analysis.json")
    
    print(f"\n🎯 SUMMARY:")
    print(f"Current visible predictions: {results['visible_after_devkit_filter']}")
    print(f"Target GT boxes: {results['target_gt_boxes']}")
    if results['visible_after_devkit_filter'] == results['target_gt_boxes']:
        print("✅ PERFECT MATCH! Fix successful.")
    else:
        print(f"❌ Still {abs(results['visible_after_devkit_filter'] - results['target_gt_boxes'])} boxes difference")
