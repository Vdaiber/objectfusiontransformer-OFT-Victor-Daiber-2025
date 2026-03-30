#!/usr/bin/env python3
"""
Minimal debug to see what prediction_utils actually returns
"""

import torch
import numpy as np
from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive

def minimal_test():
    # Minimal setup
    cfg = {
        'evaluation': {'conf_th_eval': 0.3},
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
    
    # Minimal predictions
    predictions = {
        'pred_class_logits_batch': torch.tensor([
            [[5.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -1.0],
             [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0],
             [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0],
             [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 3.0]]
        ], dtype=torch.float32),
        'pred_boxes_normalized': torch.tensor([
            [[0.6, 0.5, 0.5, 0.3, 0.7, 0.4, 0.0, 1.0, 0.1, 0.0],
             [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0],
             [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0],
             [0.5, 0.5, 0.5, 0.2, 0.4, 0.3, 0.0, 1.0, 0.0, 0.0]]
        ], dtype=torch.float32),
        'pred_velocities_normalized': torch.tensor([
            [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ], dtype=torch.float32),
        'pred_attributes_logits_batch': torch.zeros(1, 4, 12, dtype=torch.float32)
    }
    
    print("🔍 CALLING prediction_utils...")
    
    try:
        result = reconstruct_and_convert_predictions_autoregressive(
            batch_dict, predictions, cfg, ego_pose_current=None
        )
        
        print(f"✅ Function returned successfully")
        print(f"Type: {type(result)}")
        print(f"Length: {len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
        
        if isinstance(result, list) and len(result) > 0:
            print(f"First element type: {type(result[0])}")
            if isinstance(result[0], dict):
                print(f"Keys in first element: {list(result[0].keys())}")
                for key, value in result[0].items():
                    print(f"  {key}: {type(value)} - {value if not isinstance(value, (list, np.ndarray)) else f'{type(value)} with length {len(value)}'}")
            else:
                print(f"First element: {result[0]}")
        else:
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"❌ Function failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    minimal_test()
