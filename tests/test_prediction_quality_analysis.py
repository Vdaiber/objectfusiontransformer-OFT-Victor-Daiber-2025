#!/usr/bin/env python3
"""
Analyze prediction quality vs GT boxes.
Why do we only get 6 visible predictions when there are 14 GT boxes?
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
from truckscenes.utils.geometry_utils import BoxVisibility, box_in_image
from pyquaternion import Quaternion


def analyze_prediction_vs_gt_quality():
    """Analyze why predictions don't match GT visibility."""
    
    print("🔍 PREDICTION QUALITY vs GT ANALYSIS")
    print("=" * 80)
    
    ts = TruckScenes(version="v1.0-mini", dataroot="/data", verbose=False)
    sample_token = "f1c03220990143e19b983bf3da478764"
    camera_channel = "CAMERA_LEFT_FRONT"
    
    # Get GT setup
    sample_record = ts.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    
    # Get GT boxes (14 visible ones)
    _, gt_boxes_cam, K = ts.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    
    print(f"Sample: {sample_token}")
    print(f"GT boxes visible: {len(gt_boxes_cam)}")
    
    # Get all annotations for comparison
    all_annotations = []
    for ann_token in sample_record['anns']:
        ann_record = ts.get('sample_annotation', ann_token)
        all_annotations.append(ann_record)
    
    print(f"Total annotations: {len(all_annotations)}")
    
    # Analyze GT box distribution
    print(f"\n📊 GT BOX ANALYSIS:")
    gt_classes = {}
    gt_positions = []
    
    for gt_box in gt_boxes_cam:
        class_name = gt_box.name
        if class_name not in gt_classes:
            gt_classes[class_name] = 0
        gt_classes[class_name] += 1
        gt_positions.append(gt_box.center)
    
    print("GT box classes:")
    for class_name, count in gt_classes.items():
        print(f"  {class_name}: {count}")
    
    # Analyze GT positions
    gt_positions = np.array(gt_positions)
    print(f"\nGT box positions in camera frame:")
    print(f"  X range: [{np.min(gt_positions[:, 0]):.1f}, {np.max(gt_positions[:, 0]):.1f}]")
    print(f"  Y range: [{np.min(gt_positions[:, 1]):.1f}, {np.max(gt_positions[:, 1]):.1f}]")
    print(f"  Z range: [{np.min(gt_positions[:, 2]):.1f}, {np.max(gt_positions[:, 2]):.1f}]")
    
    # Load predictions
    with open('tests/visualization_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    sample_predictions = analysis['predictions']['sample_predictions']
    pose_record = ts.getclosest('ego_pose', sample_record['timestamp'])
    
    # Create corrected prediction boxes
    print(f"\n🎯 PREDICTION ANALYSIS:")
    pred_classes = {}
    pred_positions = []
    visible_pred_classes = {}
    
    import cv2
    image_path = os.path.join(ts.dataroot, sd_record['filename'])
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    imsize = (w, h)
    
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
        
        # Transform to camera (corrected)
        ego_box = world_box.copy()
        ego_box.translate(-np.array(pose_record['translation']))
        ego_box.rotate(Quaternion(pose_record['rotation']).inverse)
        
        camera_box = ego_box.copy()
        camera_box.translate(-np.array(cs_record['translation']))
        camera_box.rotate(Quaternion(cs_record['rotation']))  # No .inverse!
        
        # Count by class
        class_name = pred['detection_name']
        if class_name not in pred_classes:
            pred_classes[class_name] = 0
        pred_classes[class_name] += 1
        
        pred_positions.append(camera_box.center)
        
        # Check visibility
        if box_in_image(camera_box, K, imsize, vis_level=BoxVisibility.ANY):
            if class_name not in visible_pred_classes:
                visible_pred_classes[class_name] = 0
            visible_pred_classes[class_name] += 1
    
    print("All prediction classes:")
    for class_name, count in pred_classes.items():
        visible_count = visible_pred_classes.get(class_name, 0)
        print(f"  {class_name}: {count} total, {visible_count} visible")
    
    # Analyze prediction positions
    pred_positions = np.array(pred_positions)
    print(f"\nPrediction positions in camera frame:")
    print(f"  X range: [{np.min(pred_positions[:, 0]):.1f}, {np.max(pred_positions[:, 0]):.1f}]")
    print(f"  Y range: [{np.min(pred_positions[:, 1]):.1f}, {np.max(pred_positions[:, 1]):.1f}]")
    print(f"  Z range: [{np.min(pred_positions[:, 2]):.1f}, {np.max(pred_positions[:, 2]):.1f}]")
    
    # Compare class distributions
    print(f"\n📈 CLASS DISTRIBUTION COMPARISON:")
    all_classes = set(gt_classes.keys()) | set(pred_classes.keys())
    
    print("Class         GT    Pred_Total  Pred_Visible")
    print("-" * 45)
    for class_name in sorted(all_classes):
        gt_count = gt_classes.get(class_name, 0)
        pred_total = pred_classes.get(class_name, 0)
        pred_visible = visible_pred_classes.get(class_name, 0)
        print(f"{class_name:<12} {gt_count:3d}   {pred_total:3d}        {pred_visible:3d}")
    
    # Distance analysis
    print(f"\n📏 DISTANCE ANALYSIS:")
    gt_distances = [np.linalg.norm(pos[:2]) for pos in gt_positions]  # XY distance
    pred_distances = [np.linalg.norm(pos[:2]) for pos in pred_positions]
    
    print(f"GT box distances (XY plane):")
    print(f"  Range: [{min(gt_distances):.1f}, {max(gt_distances):.1f}]")
    print(f"  Mean: {np.mean(gt_distances):.1f}")
    
    print(f"Prediction distances (XY plane):")
    print(f"  Range: [{min(pred_distances):.1f}, {max(pred_distances):.1f}]")
    print(f"  Mean: {np.mean(pred_distances):.1f}")
    
    # Check if predictions are too far away
    far_predictions = sum(1 for d in pred_distances if d > max(gt_distances))
    print(f"Predictions beyond GT range: {far_predictions}")
    
    # Score analysis
    print(f"\n⭐ SCORE ANALYSIS:")
    scores = [pred['detection_score'] for pred in sample_predictions]
    print(f"Prediction scores:")
    print(f"  Range: [{min(scores):.3f}, {max(scores):.3f}]")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  High confidence (>0.8): {sum(1 for s in scores if s > 0.8)}")
    print(f"  Medium confidence (0.5-0.8): {sum(1 for s in scores if 0.5 <= s <= 0.8)}")
    print(f"  Low confidence (<0.5): {sum(1 for s in scores if s < 0.5)}")
    
    return {
        'gt_boxes': len(gt_boxes_cam),
        'total_predictions': len(sample_predictions),
        'visible_predictions': sum(visible_pred_classes.values()),
        'gt_classes': gt_classes,
        'pred_classes': pred_classes,
        'visible_pred_classes': visible_pred_classes,
        'gt_distance_range': [min(gt_distances), max(gt_distances)],
        'pred_distance_range': [min(pred_distances), max(pred_distances)],
        'score_stats': {
            'range': [min(scores), max(scores)],
            'mean': np.mean(scores),
            'high_conf': sum(1 for s in scores if s > 0.8),
            'medium_conf': sum(1 for s in scores if 0.5 <= s <= 0.8),
            'low_conf': sum(1 for s in scores if s < 0.5)
        }
    }


def suggest_improvements():
    """Suggest improvements to match GT box count."""
    
    print(f"\n💡 SUGGESTED IMPROVEMENTS:")
    print("=" * 50)
    
    print("ISSUE ANALYSIS:")
    print("  • Model predicts 31 objects but only 6 are visible")
    print("  • GT has 14 visible objects")
    print("  • Many predictions are outside image bounds")
    print("  • Some predictions are behind camera (depth < 0.1)")
    
    print(f"\nPOSSIBLE CAUSES:")
    print("  1. Model training issue: Poor localization accuracy")
    print("  2. Coordinate system issue: Still some transformation error")
    print("  3. Prediction filtering: Too strict confidence thresholds")
    print("  4. Dataset mismatch: Training vs evaluation coordinate frame")
    print("  5. Hungarian matching: Poor assignment during training")
    
    print(f"\nRECOMMENDED FIXES:")
    print("  1. Check confidence threshold - lower it to get more predictions")
    print("  2. Verify training coordinate system matches evaluation")
    print("  3. Analyze Hungarian matcher cost weights")
    print("  4. Check if denormalization is correct")
    print("  5. Compare with other working samples")
    
    print(f"\nQUICK TEST:")
    print("  • Lower confidence threshold from default to 0.1")
    print("  • Check if more predictions become visible")
    print("  • If yes: training/inference threshold mismatch")
    print("  • If no: fundamental localization issue")


if __name__ == "__main__":
    # Run analysis
    results = analyze_prediction_vs_gt_quality()
    
    # Suggest improvements
    suggest_improvements()
    
    # Save results
    with open('tests/prediction_quality_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: tests/prediction_quality_analysis.json")
    
    print(f"\n🎯 SUMMARY:")
    print(f"GT boxes: {results['gt_boxes']}")
    print(f"Visible predictions: {results['visible_predictions']}")
    print(f"Gap: {results['gt_boxes'] - results['visible_predictions']} boxes")
    
    if results['visible_predictions'] < results['gt_boxes']:
        print("❌ Predictions not covering all GT objects")
        print("→ This suggests a model performance/training issue, not visualization bug")
