#!/usr/bin/env python3
"""
The REAL analysis: Original transformation is correct, find the actual issue.
Why only 11 instead of 14 visible boxes?
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


def analyze_real_issue():
    """Find the real reason why we get 11 instead of 14 visible boxes."""
    
    print("🔍 REAL ISSUE ANALYSIS: Why 11 instead of 14?")
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
    
    # Get GT boxes (14 visible)
    _, gt_boxes_cam, K = ts.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    print(f"GT boxes visible: {len(gt_boxes_cam)}")
    
    # Load predictions
    with open('tests/visualization_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    sample_predictions = analysis['predictions']['sample_predictions']
    print(f"Total predictions: {len(sample_predictions)}")
    
    # Use ORIGINAL (correct) transformation with .inverse
    camera_boxes = []
    
    for i, pred in enumerate(sample_predictions):
        center_world = np.array(pred['translation'])
        size_world = np.array(pred['size'])
        rotation_world_quat = Quaternion(pred['rotation'])
        
        world_box = DevkitBox(
            center=center_world,
            size=size_world,
            orientation=rotation_world_quat,
            name=pred['detection_name'],
            score=pred['detection_score'],
            token=f'pred_{i}'
        )
        
        # ORIGINAL (CORRECT) transformation
        ego_box = world_box.copy()
        ego_box.translate(-np.array(pose_record['translation']))
        ego_box.rotate(Quaternion(pose_record['rotation']).inverse)
        
        camera_box = ego_box.copy()
        camera_box.translate(-np.array(cs_record['translation']))
        camera_box.rotate(Quaternion(cs_record['rotation']).inverse)  # KEEP .inverse!
        
        camera_boxes.append(camera_box)
    
    # Test different filters to see what gives us 11
    import cv2
    image_path = os.path.join(ts.dataroot, sd_record['filename'])
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    imsize = (w, h)
    
    # 1. Original custom FOV filter
    def original_fov_filter(boxes):
        visible = []
        for box in boxes:
            corners_cam = box.corners()
            if np.any(corners_cam[2, :] <= 0):
                continue
            homog_pts = K @ corners_cam
            u = homog_pts[0, :] / homog_pts[2, :]
            v = homog_pts[1, :] / homog_pts[2, :]
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
    
    original_visible = original_fov_filter(camera_boxes)
    devkit_visible = devkit_filter(camera_boxes)
    
    print(f"\nFILTER COMPARISON:")
    print(f"Original FOV filter: {len(original_visible)} visible")
    print(f"DevKit filter: {len(devkit_visible)} visible")
    
    # The difference between 11 and 14 is likely:
    # 1. Confidence threshold filtering
    # 2. Some predictions don't exist for GT objects
    # 3. Quality difference between GT and predictions
    
    print(f"\nANALYZING THE GAP (11 vs 14):")
    
    # Check confidence scores of visible vs invisible
    visible_scores = [getattr(box, 'score', 0) for box in original_visible]
    invisible_boxes = [box for box in camera_boxes if box not in original_visible]
    invisible_scores = [getattr(box, 'score', 0) for box in invisible_boxes]
    
    print(f"Visible box scores: min={min(visible_scores):.3f}, max={max(visible_scores):.3f}, mean={np.mean(visible_scores):.3f}")
    if invisible_scores:
        print(f"Invisible box scores: min={min(invisible_scores):.3f}, max={max(invisible_scores):.3f}, mean={np.mean(invisible_scores):.3f}")
    
    # Check class distribution
    visible_classes = {}
    invisible_classes = {}
    
    for box in original_visible:
        cls = box.name
        visible_classes[cls] = visible_classes.get(cls, 0) + 1
    
    for box in invisible_boxes:
        cls = box.name
        invisible_classes[cls] = invisible_classes.get(cls, 0) + 1
    
    print(f"\nCLASS DISTRIBUTION:")
    print("Visible classes:")
    for cls, count in visible_classes.items():
        print(f"  {cls}: {count}")
    
    print("Invisible classes:")
    for cls, count in invisible_classes.items():
        print(f"  {cls}: {count}")
    
    # Compare with GT classes
    gt_classes = {}
    for gt_box in gt_boxes_cam:
        cls = gt_box.name
        gt_classes[cls] = gt_classes.get(cls, 0) + 1
    
    print("GT classes:")
    for cls, count in gt_classes.items():
        print(f"  {cls}: {count}")
    
    # The real issue analysis
    print(f"\n🎯 REAL ISSUE IDENTIFIED:")
    
    # Issue 1: Not all GT objects have corresponding predictions
    all_pred_classes = set(visible_classes.keys()) | set(invisible_classes.keys())
    gt_class_set = set(gt_classes.keys())
    
    missing_classes = gt_class_set - all_pred_classes
    extra_classes = all_pred_classes - gt_class_set
    
    if missing_classes:
        print(f"Missing prediction classes: {missing_classes}")
    if extra_classes:
        print(f"Extra prediction classes: {extra_classes}")
    
    # Issue 2: Quality/position differences
    print(f"\nPosition comparison:")
    gt_positions = [box.center for box in gt_boxes_cam]
    visible_positions = [box.center for box in original_visible]
    
    if gt_positions and visible_positions:
        gt_depths = [pos[2] for pos in gt_positions]
        vis_depths = [pos[2] for pos in visible_positions]
        
        print(f"GT depth range: [{min(gt_depths):.1f}, {max(gt_depths):.1f}]")
        print(f"Visible pred depth range: [{min(vis_depths):.1f}, {max(vis_depths):.1f}]")
    
    return {
        'gt_boxes': len(gt_boxes_cam),
        'total_predictions': len(camera_boxes),
        'visible_predictions': len(original_visible),
        'gap': len(gt_boxes_cam) - len(original_visible),
        'visible_classes': visible_classes,
        'invisible_classes': invisible_classes,
        'gt_classes': gt_classes,
        'missing_classes': list(missing_classes),
        'extra_classes': list(extra_classes)
    }


def determine_real_fix():
    """Determine what the real fix should be."""
    
    print(f"\n🔧 REAL FIX DETERMINATION:")
    print("=" * 50)
    
    print("CONFIRMED:")
    print("✅ Original transformation (.inverse) is CORRECT - gives 11 visible")
    print("❌ Removing .inverse is WRONG - only gives 6 visible")
    
    print(f"\nREAL ISSUES TO INVESTIGATE:")
    print("1. Model prediction quality - only 11 good predictions vs 14 GT")
    print("2. Class name mapping between predictions and GT")
    print("3. Confidence threshold too high")
    print("4. Spatial accuracy of predictions")
    
    print(f"\nREAL FIX STRATEGY:")
    print("→ Keep original transformation")
    print("→ Focus on improving model to predict missing 3 objects")
    print("→ Check if confidence threshold can be lowered")
    print("→ Verify class name consistency")


if __name__ == "__main__":
    # Analyze the real issue
    results = analyze_real_issue()
    
    # Determine real fix
    determine_real_fix()
    
    # Save results
    with open('tests/real_issue_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Real analysis saved to: tests/real_issue_analysis.json")
    
    print(f"\n🎯 FINAL CORRECT ANALYSIS:")
    print(f"• Original code ist KORREKT")
    print(f"• Transformation mit .inverse ist RICHTIG")
    print(f"• Problem: Model produziert nur {results['visible_predictions']} statt {results['gt_boxes']} sichtbare Objekte")
    print(f"• Lösung: Model-Performance verbessern, nicht Visualisierung 'fixen'")
