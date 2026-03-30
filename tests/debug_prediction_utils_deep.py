#!/usr/bin/env python3
"""
Deep dive into prediction_utils structure
"""

import torch
import numpy as np
from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive

def deep_structure_analysis():
    # Minimal setup
    cfg = {
        'evaluation': {'conf_th_eval': 0.1},  # Low threshold to see more predictions
        'dataset': {
            'class_names': ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 
                           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 
                           'barrier', 'animal', 'traffic_sign']
        }
    }
    
    # Minimal batch_dict
    batch_dict = {
        'sample_tokens': ['sample_001'],
        'ego_translation_world': torch.tensor([[100.0, 200.0, 1.5]], dtype=torch.float64),
        'ego_rotation_world_quat': torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        'ego_motion': {
            'cabin': {
                'velocity': torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)
            }
        },
        'gt_valid_mask_b': torch.tensor([[True, False, False, False]])
    }
    
    # Predictions with high confidence car
    predictions = {
        'pred_class_logits_batch': torch.tensor([
            [[8.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -5.0],  # High conf car
             [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 1.0],   # Low conf no_object
             [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 1.0],   # Low conf no_object
             [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 1.0]]   # Low conf no_object
        ], dtype=torch.float32),
        'pred_boxes_normalized': torch.tensor([
            [[0.6, 0.5, 0.5, 0.3, 0.7, 0.4, 0.0, 1.0, 0.1, 0.0],  # car box
             [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0],  # no_object box
             [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0],  # no_object box
             [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0]]  # no_object box
        ], dtype=torch.float32),
        'pred_velocities_normalized': torch.tensor([
            [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ], dtype=torch.float32),
        'pred_attributes_logits_batch': torch.zeros(1, 4, 12, dtype=torch.float32)
    }
    
    print("🔍 DEEP STRUCTURE ANALYSIS")
    print("=" * 50)
    
    result = reconstruct_and_convert_predictions_autoregressive(
        batch_dict, predictions, cfg, ego_pose_current=None
    )
    
    print(f"Top level: {type(result)}, length: {len(result)}")
    
    for i, sample in enumerate(result):
        print(f"\nSample {i}:")
        print(f"  Type: {type(sample)}")
        print(f"  Keys: {list(sample.keys())}")
        
        for key, value in sample.items():
            if key == 'predictions':
                print(f"  {key}: {type(value)}, length: {len(value)}")
                
                for j, pred in enumerate(value):
                    print(f"    Prediction {j}:")
                    print(f"      Type: {type(pred)}")
                    if isinstance(pred, dict):
                        print(f"      Keys: {list(pred.keys())}")
                        for pred_key, pred_value in pred.items():
                            if isinstance(pred_value, (list, np.ndarray)):
                                print(f"        {pred_key}: {type(pred_value)}, length: {len(pred_value)}, content: {pred_value}")
                            else:
                                print(f"        {pred_key}: {type(pred_value)}, value: {pred_value}")
                    else:
                        print(f"      Value: {pred}")
            else:
                print(f"  {key}: {type(value)}, value: {value}")

if __name__ == "__main__":
    deep_structure_analysis()
