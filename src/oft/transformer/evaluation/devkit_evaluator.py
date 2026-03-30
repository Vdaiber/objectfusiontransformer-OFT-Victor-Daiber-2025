# File: src/oft/transformer/evaluation/devkit_evaluator.py
"""
TruckScenes evaluation module using the official devkit.

Provides evaluation functionality for TruckScenes dataset predictions using the official
TruckScenes evaluation framework. Handles prediction serialization, evaluation configuration,
and metric computation for object detection tasks.
"""
import os
import json
from truckscenes import TruckScenes
from truckscenes.eval.detection.evaluate import DetectionEval
from truckscenes.eval.common.config import config_factory
from ..utils.common import sanitize_for_json


def run_devkit_evaluation(raw_predictions_list: list, config: dict, output_dir: str):
    """Execute evaluation using the TruckScenes official evaluation framework.
    
    Args:
        raw_predictions_list: List of prediction dictionaries. Each dict contains:
            - 'sample_token' (str): Unique sample identifier
            - 'predictions' (list): List of detection predictions, each containing:
                - 'translation' (list[float]): 3D center [x, y, z] in world coordinates (meters)
                - 'size' (list[float]): 3D dimensions [width, length, height] in meters
                - 'rotation' (list[float]): Quaternion [w, x, y, z] in world frame
                - 'velocity' (list[float]): 2D velocity [vx, vy] in m/s
                - 'detection_name' (str): Object class name (e.g., 'car', 'truck')
                - 'detection_score' (float): Confidence score [0.0, 1.0]
        config: Configuration dict with required keys:
            - 'dataset': Contains 'version' (str) and 'dataroot' (str)
            - 'evaluation': Contains optional 'eval_config_name' and 'eval_split_name'
        output_dir: Directory path for storing evaluation results
            
    Returns:
        DetectionEval: Evaluation object with computed metrics (precision, recall, mAP)
    """
    # Create output directory for evaluation artifacts
    os.makedirs(output_dir, exist_ok=True)
    
    # 🔬 DEBUGGING: Check for None AND nan/inf values in raw predictions BEFORE DevKit processing
    print(f"🔍 DEVKIT DEBUG: Processing {len(raw_predictions_list)} prediction samples")
    import numpy as np
    none_count = 0
    nan_inf_count = 0
    
            
    # Count total predictions across all samples
    total_predictions = sum(len(item['predictions']) for item in raw_predictions_list)
    empty_samples = sum(1 for item in raw_predictions_list if len(item['predictions']) == 0)
    print(f"🔍 PREDICTION COUNTS:")
    print(f"   Total samples: {len(raw_predictions_list)}")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Empty samples: {empty_samples}")
    print(f"   Average predictions per sample: {total_predictions / len(raw_predictions_list) if len(raw_predictions_list) > 0 else 0:.2f}")
    
    for sample_idx, item in enumerate(raw_predictions_list):
        if item['predictions'] is None:
            print(f"🚨 PREDICTIONS ARRAY IS NONE in sample {sample_idx}")
            none_count += 1
            continue
            
        for pred_idx, pred in enumerate(item['predictions']):
            if pred is None:
                print(f"🚨 PREDICTION IS NONE in sample {sample_idx}, prediction {pred_idx}")
                none_count += 1
                continue
                
            for key, value in pred.items():
                if value is None:
                    print(f"🚨 NONE VALUE FOUND in raw_predictions_list[{sample_idx}][{pred_idx}]['{key}'] = None")
                    print(f"   FULL PREDICTION: {pred}")
                    none_count += 1
                elif isinstance(value, list):
                    if None in value:
                        print(f"🚨 NONE IN LIST FOUND in raw_predictions_list[{sample_idx}][{pred_idx}]['{key}'] = {value}")
                        none_count += 1
                    # Check for nan/inf in list elements
                    for val in value:
                        if isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val)):
                            print(f"🚨 NAN/INF IN LIST: sample[{sample_idx}][{pred_idx}]['{key}'] contains {val}")
                            nan_inf_count += 1
                            break
                elif isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    print(f"🚨 NAN/INF VALUE: sample[{sample_idx}][{pred_idx}]['{key}'] = {value}")
                    nan_inf_count += 1
    
    if none_count > 0:
        print(f"🚨 TOTAL NONE VALUES IN RAW PREDICTIONS: {none_count}")
    if nan_inf_count > 0:
        print(f"🚨 TOTAL NAN/INF VALUES IN RAW PREDICTIONS: {nan_inf_count}")
    if none_count == 0 and nan_inf_count == 0:
        print("✅ NO NONE OR NAN/INF VALUES in raw_predictions_list")
    
    # Convert list format to dictionary format required by devkit
    # Devkit expects {sample_token: predictions_list}
    results = {item['sample_token']: item['predictions'] for item in raw_predictions_list}
    
    # Create submission structure with empty metadata
    submission = {"meta": {}, "results": results}
    
    # Check sanitize_for_json output for None values
    sanitized_submission = sanitize_for_json(submission)
    print(f"🔍 CHECKING sanitize_for_json output...")
    sanitized_none_count = 0
    for sample_token, predictions in sanitized_submission.get('results', {}).items():
        for pred_idx, pred in enumerate(predictions):
            for key, value in pred.items():
                if value is None:
                    print(f"🚨 NONE AFTER SANITIZE: sample_token='{sample_token}', pred[{pred_idx}]['{key}'] = None")
                    sanitized_none_count += 1
                elif isinstance(value, list) and None in value:
                    print(f"🚨 NONE IN LIST AFTER SANITIZE: sample_token='{sample_token}', pred[{pred_idx}]['{key}'] = {value}")
                    sanitized_none_count += 1
    
    if sanitized_none_count > 0:
        print(f"🚨 SANITIZE_FOR_JSON CREATED {sanitized_none_count} NONE VALUES!")
    else:
        print("✅ NO NONE VALUES after sanitize_for_json")
    
    # Serialize predictions to JSON file for devkit input
    pred_path = os.path.join(output_dir, 'predictions.json')
    with open(pred_path, 'w') as f:
        # Handle non-serializable data types (e.g., numpy arrays)
        json.dump(sanitized_submission, f, indent=2)
        
    print(f"🔍 PREDICTIONS WRITTEN TO: {pred_path}")
    
    # Extract evaluation configuration with fallback defaults
    eval_cfg = config['evaluation']
    
    # Initialize TruckScenes dataset with specified version and data root
    ts = TruckScenes(version=config['dataset']['version'], 
                     dataroot=config['dataset']['dataroot'], 
                     verbose=False)
    
    # Wrap DevKit call to catch and analyze the error
    try:
        print(f"🔍 CREATING DetectionEval with pred_path: {pred_path}")
        
        # Create evaluation engine with configuration and prediction file
        eval_main = DetectionEval(ts, 
                                 config=config_factory(eval_cfg['eval_config_name']), 
                                 result_path=pred_path, 
                                 eval_set=eval_cfg['eval_split_name'], 
                                 output_dir=output_dir, 
                                 verbose=True)
        
        # Execute evaluation with minimal visualization for faster execution
        return eval_main.main(plot_examples=0, render_curves=False)
        
    except Exception as e:
        print(f"🚨 DEVKIT ERROR: {e}")
        print(f"🔍 ERROR TYPE: {type(e)}")
        
        # Try to read back the written predictions.json to see if it's corrupted
        print(f"🔍 READING BACK WRITTEN PREDICTIONS.JSON...")
        try:
            with open(pred_path, 'r') as f:
                written_data = json.load(f)
            print(f"✅ Successfully read predictions.json")
            print(f"   Meta keys: {written_data.get('meta', {}).keys()}")
            print(f"   Results samples: {len(written_data.get('results', {}))}")
            
            # Check for None values in written file
            for sample_token, predictions in written_data.get('results', {}).items():
                for pred_idx, pred in enumerate(predictions):
                    for key, value in pred.items():
                        if value is None:
                            print(f"🚨 NONE IN WRITTEN FILE: {sample_token}[{pred_idx}][{key}] = None")
                        elif isinstance(value, list) and None in value:
                            print(f"🚨 NONE IN LIST IN WRITTEN FILE: {sample_token}[{pred_idx}][{key}] = {value}")
                            
        except Exception as read_e:
            print(f"🚨 FAILED TO READ PREDICTIONS.JSON: {read_e}")
        
        raise e  # Re-raise the original exception 