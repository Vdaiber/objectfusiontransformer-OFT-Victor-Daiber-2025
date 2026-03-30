#!/usr/bin/env python3
"""
Analyze exact class name differences between GT and predictions.
"""

import os
import sys
import json
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from truckscenes import TruckScenes


def analyze_class_mapping_differences():
    """Find exact class name differences."""
    
    print("🔍 CLASS MAPPING DIFFERENCES ANALYSIS")
    print("=" * 80)
    
    ts = TruckScenes(version="v1.0-mini", dataroot="/data", verbose=False)
    sample_token = "f1c03220990143e19b983bf3da478764"
    camera_channel = "CAMERA_LEFT_FRONT"
    
    # Get GT classes
    sample_record = ts.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    _, gt_boxes_cam, _ = ts.get_sample_data(cam_token, box_vis_level=1)  # ANY visibility
    
    print(f"Sample: {sample_token}")
    print(f"GT boxes: {len(gt_boxes_cam)}")
    
    # Extract GT class names
    gt_classes = {}
    gt_class_names = []
    for gt_box in gt_boxes_cam:
        class_name = gt_box.name
        gt_class_names.append(class_name)
        if class_name not in gt_classes:
            gt_classes[class_name] = 0
        gt_classes[class_name] += 1
    
    print(f"\n📊 GT CLASS DISTRIBUTION:")
    for class_name, count in sorted(gt_classes.items()):
        print(f"  '{class_name}': {count}")
    
    # Get prediction classes
    with open('tests/real_issue_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    visible_classes = analysis['visible_classes']
    invisible_classes = analysis['invisible_classes']
    
    print(f"\n📊 PREDICTION CLASS DISTRIBUTION:")
    print("Visible predictions:")
    for class_name, count in sorted(visible_classes.items()):
        print(f"  '{class_name}': {count}")
    
    print("Invisible predictions:")
    for class_name, count in sorted(invisible_classes.items()):
        print(f"  '{class_name}': {count}")
    
    # Compare directly
    print(f"\n🔄 DIRECT COMPARISON:")
    
    all_gt_classes = set(gt_classes.keys())
    all_pred_classes = set(visible_classes.keys()) | set(invisible_classes.keys())
    
    print(f"GT classes: {sorted(all_gt_classes)}")
    print(f"Prediction classes: {sorted(all_pred_classes)}")
    
    # Find exact mapping
    print(f"\n🎯 MAPPING ANALYSIS:")
    
    # Manual mapping based on semantic similarity
    potential_mappings = {
        # Prediction -> GT
        'pedestrian': 'human.pedestrian.adult',
        'car': 'vehicle.car', 
        'truck': 'vehicle.truck',
        'trailer': 'vehicle.trailer',
        'bicycle': 'vehicle.bicycle',  # Not in GT but might exist
        'bus': 'vehicle.bus'  # Not in GT but might exist
    }
    
    print("Potential mappings:")
    for pred_class, gt_class in potential_mappings.items():
        pred_count = visible_classes.get(pred_class, 0) + invisible_classes.get(pred_class, 0)
        gt_count = gt_classes.get(gt_class, 0)
        match_status = "✅" if gt_count > 0 else "❌"
        print(f"  {match_status} '{pred_class}' → '{gt_class}' | Pred: {pred_count}, GT: {gt_count}")
    
    # Check for unmatched classes
    print(f"\n⚠️  UNMATCHED CLASSES:")
    
    mapped_gt_classes = set(potential_mappings.values())
    unmapped_gt = all_gt_classes - mapped_gt_classes
    if unmapped_gt:
        print(f"GT classes without prediction mapping:")
        for class_name in sorted(unmapped_gt):
            print(f"  '{class_name}': {gt_classes[class_name]} instances")
    
    mapped_pred_classes = set(potential_mappings.keys())
    unmapped_pred = all_pred_classes - mapped_pred_classes
    if unmapped_pred:
        print(f"Prediction classes without GT mapping:")
        for class_name in sorted(unmapped_pred):
            pred_count = visible_classes.get(class_name, 0) + invisible_classes.get(class_name, 0)
            print(f"  '{class_name}': {pred_count} instances")
    
    # Calculate potential improvement
    print(f"\n📈 POTENTIAL IMPROVEMENT WITH MAPPING:")
    
    current_visible = sum(visible_classes.values())
    gt_target = len(gt_boxes_cam)
    
    # Count how many predictions could match GT if mapped correctly
    mappable_predictions = 0
    for pred_class, gt_class in potential_mappings.items():
        if gt_class in gt_classes:
            pred_total = visible_classes.get(pred_class, 0) + invisible_classes.get(pred_class, 0)
            gt_count = gt_classes[gt_class]
            # Take minimum (can't match more than GT has)
            mappable_predictions += min(pred_total, gt_count)
    
    print(f"Current visible: {current_visible}")
    print(f"GT target: {gt_target}")
    print(f"Potential with perfect mapping: {mappable_predictions}")
    print(f"Expected improvement: {mappable_predictions - current_visible}")
    
    return {
        'gt_classes': gt_classes,
        'visible_pred_classes': visible_classes,
        'invisible_pred_classes': invisible_classes,
        'potential_mappings': potential_mappings,
        'current_visible': current_visible,
        'gt_target': gt_target,
        'potential_with_mapping': mappable_predictions,
        'expected_improvement': mappable_predictions - current_visible
    }


def check_dataset_config():
    """Check what class names are configured in the dataset."""
    
    print(f"\n📋 DATASET CONFIGURATION CHECK:")
    print("=" * 50)
    
    config_file = "config/pipeline_staged.yaml"
    if os.path.exists(config_file):
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_classes = config.get('dataset', {}).get('class_names', [])
        print(f"Configured dataset classes ({len(dataset_classes)}):")
        for i, class_name in enumerate(dataset_classes):
            print(f"  {i}: '{class_name}'")
    else:
        print("❌ Config file not found")
    
    # Check if there's a mapping file
    potential_mapping_files = [
        "config/class_mapping.yaml",
        "config/class_names_mapping.yaml", 
        "src/oft/transformer/datasets/truckscenes/class_mapping.py"
    ]
    
    print(f"\nChecking for mapping files:")
    for mapping_file in potential_mapping_files:
        if os.path.exists(mapping_file):
            print(f"✅ Found: {mapping_file}")
        else:
            print(f"❌ Not found: {mapping_file}")


def suggest_implementation():
    """Suggest how to implement the class mapping."""
    
    print(f"\n🔧 IMPLEMENTATION SUGGESTION:")
    print("=" * 50)
    
    print("CLASS MAPPING IMPLEMENTATION:")
    print("""
# Add to epoch_visualization_workflow.py:

PREDICTION_TO_GT_CLASS_MAPPING = {
    'pedestrian': 'human.pedestrian.adult',
    'car': 'vehicle.car',
    'truck': 'vehicle.truck', 
    'trailer': 'vehicle.trailer',
    'bicycle': 'vehicle.bicycle',
    'bus': 'vehicle.bus',
    'barrier': 'movable_object.barrier',
    'traffic_cone': 'movable_object.trafficcone',
    'motorcycle': 'vehicle.motorcycle',
    'other_vehicle': 'vehicle.other',
    'animal': 'animal',
    'traffic_sign': 'static_object.trafficsign'
}

# In prediction processing:
pred_class = pred['detection_name']
gt_class = PREDICTION_TO_GT_CLASS_MAPPING.get(pred_class, pred_class)

world_box = DevkitBox(
    center=center_world,
    size=size_world,
    orientation=rotation_world_quat,
    name=gt_class,  # Use mapped class name
    score=pred['detection_score'],
    token=f'pred_{i}'
)
""")


if __name__ == "__main__":
    # Analyze class mapping
    results = analyze_class_mapping_differences()
    
    # Check dataset config
    check_dataset_config()
    
    # Suggest implementation
    suggest_implementation()
    
    # Save results
    with open('tests/class_mapping_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: tests/class_mapping_analysis.json")
    
    print(f"\n🎯 SUMMARY:")
    if results['expected_improvement'] > 0:
        print(f"✅ Class mapping can improve from {results['current_visible']} to {results['potential_with_mapping']} visible boxes")
        print(f"Expected improvement: +{results['expected_improvement']} boxes")
    else:
        print(f"⚠️ Class mapping alone won't solve the issue")
        print(f"Current: {results['current_visible']}, Target: {results['gt_target']}")
        print(f"Need to improve model predictions, not just mapping")
