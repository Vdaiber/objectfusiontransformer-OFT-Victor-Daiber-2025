#!/usr/bin/env python3
"""
Test the class mapping fix to see improvement from 11 to 13 visible boxes.
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


# Import the fixed mapping
from oft.transformer.utils.epoch_visualization_workflow import PREDICTION_TO_GT_CLASS_MAPPING


def test_class_mapping_fix():
    """Test the class mapping fix."""
    
    print("🧪 TESTING CLASS MAPPING FIX")
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
    
    # Get GT boxes for comparison
    _, gt_boxes_cam, K = ts.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    print(f"GT boxes: {len(gt_boxes_cam)}")
    
    # Load predictions
    with open('tests/visualization_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    sample_predictions = analysis['predictions']['sample_predictions']
    print(f"Total predictions: {len(sample_predictions)}")
    
    # Test both: original and fixed class mapping
    import cv2
    image_path = os.path.join(ts.dataroot, sd_record['filename'])
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    imsize = (w, h)
    
    results = {}
    
    for use_mapping in [False, True]:
        mapping_name = "WITH MAPPING" if use_mapping else "WITHOUT MAPPING"
        print(f"\n--- {mapping_name} ---")
        
        camera_boxes = []
        class_distribution = {}
        
        for i, pred in enumerate(sample_predictions):
            center_world = np.array(pred['translation'])
            size_world = np.array(pred['size'])
            rotation_world_quat = Quaternion(pred['rotation'])
            
            # Apply class mapping if enabled
            pred_class = pred['detection_name']
            if use_mapping:
                gt_class = PREDICTION_TO_GT_CLASS_MAPPING.get(pred_class, pred_class)
            else:
                gt_class = pred_class
            
            # Count class distribution
            if gt_class not in class_distribution:
                class_distribution[gt_class] = 0
            class_distribution[gt_class] += 1
            
            world_box = DevkitBox(
                center=center_world,
                size=size_world,
                orientation=rotation_world_quat,
                name=gt_class,
                score=pred['detection_score'],
                token=f'pred_{i}'
            )
            
            # Transform to camera (original transformation with .inverse)
            ego_box = world_box.copy()
            ego_box.translate(-np.array(pose_record['translation']))
            ego_box.rotate(Quaternion(pose_record['rotation']).inverse)
            
            camera_box = ego_box.copy()
            camera_box.translate(-np.array(cs_record['translation']))
            camera_box.rotate(Quaternion(cs_record['rotation']).inverse)
            
            camera_boxes.append(camera_box)
        
        # Apply original FOV filter
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
        
        visible_boxes = original_fov_filter(camera_boxes)
        
        # Count visible classes
        visible_classes = {}
        for box in visible_boxes:
            class_name = box.name
            if class_name not in visible_classes:
                visible_classes[class_name] = 0
            visible_classes[class_name] += 1
        
        print(f"Total camera boxes: {len(camera_boxes)}")
        print(f"Visible boxes: {len(visible_boxes)}")
        print(f"Class distribution (all):")
        for class_name, count in sorted(class_distribution.items()):
            print(f"  {class_name}: {count}")
        print(f"Class distribution (visible):")
        for class_name, count in sorted(visible_classes.items()):
            print(f"  {class_name}: {count}")
        
        results[mapping_name] = {
            'total_boxes': len(camera_boxes),
            'visible_boxes': len(visible_boxes),
            'class_distribution': class_distribution,
            'visible_classes': visible_classes
        }
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    without = results['WITHOUT MAPPING']
    with_mapping = results['WITH MAPPING']
    
    print(f"Without mapping: {without['visible_boxes']} visible")
    print(f"With mapping: {with_mapping['visible_boxes']} visible")
    print(f"GT target: {len(gt_boxes_cam)}")
    print(f"Improvement: {with_mapping['visible_boxes'] - without['visible_boxes']}")
    
    # Show GT classes for reference
    gt_classes = {}
    for gt_box in gt_boxes_cam:
        class_name = gt_box.name
        if class_name not in gt_classes:
            gt_classes[class_name] = 0
        gt_classes[class_name] += 1
    
    print(f"\nGT classes for reference:")
    for class_name, count in sorted(gt_classes.items()):
        print(f"  {class_name}: {count}")
    
    return results


if __name__ == "__main__":
    results = test_class_mapping_fix()
    
    # Save results
    with open('tests/class_mapping_fix_test.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: tests/class_mapping_fix_test.json")
    
    improvement = results['WITH MAPPING']['visible_boxes'] - results['WITHOUT MAPPING']['visible_boxes']
    if improvement > 0:
        print(f"\n✅ SUCCESS: Class mapping fix improves by +{improvement} visible boxes!")
    else:
        print(f"\n❌ No improvement with class mapping fix")
