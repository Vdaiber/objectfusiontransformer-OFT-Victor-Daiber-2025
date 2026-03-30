# File: src/oft/transformer/scripts/evaluate_virtual_sensors_baseline.py
"""
Baseline evaluation for individual virtual sensors using TruckScenes DevKit.

Uses the SAME evaluation system as the main pipeline, but evaluates raw sensor data
without transformer processing. This provides baseline metrics for comparison.
"""

import os
import sys
import yaml
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add src to Python path
sys.path.insert(0, '/app/src')

# Import the SAME evaluation system as main pipeline
from oft.transformer.evaluation.devkit_evaluator import run_devkit_evaluation
from oft.transformer.datasets.loaders.autoregressive_loader import build_autoregressive_dataloaders


class VirtualSensorBaselineEvaluator:
    """
    Evaluates individual virtual sensors using the same TruckScenes evaluation as main pipeline.
    """
    
    def __init__(self, config_path: str, output_dir: str = None, config_hash_override: str = None):
        """Initialize evaluator with config and output directory."""
        self.config_path = config_path
        self.config = self._load_config()
        self.config_hash_override = config_hash_override
        
        # Create output directory  
        if output_dir is None:
            output_dir = f"baseline_eval_{self.config['dataset']['version']}"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Get enabled virtual sensors
        self.virtual_sensors = [
            sensor['name'] for sensor in self.config['dataset']['virtual_sensors'] 
            if sensor.get('enabled', True)
        ]
        
        self.logger.info(f"Initialized baseline evaluator for sensors: {self.virtual_sensors}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the evaluator."""
        logger = logging.getLogger(f"VirtualSensorEvaluator")
        # Set to WARNING to avoid duplicate logging with dataset logger
        # Dataset logger already provides all necessary information
        logger.setLevel(logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def run_baseline_evaluation(self) -> Dict[str, Dict[str, float]]:
        """
        Run baseline evaluation for all virtual sensors.
        
        Returns:
            Dict mapping sensor names to their metrics
        """
        self.logger.info("🔍 Starting baseline evaluation for virtual sensors")
        self.logger.info(f"Using TruckScenes dataset: {self.config['dataset']['version']}")
        
        all_results = {}
        
        # Get dataset split for baseline evaluation
        eval_split = self.config.get('evaluation', {}).get('baseline_eval_split', 'mini_val')
        
        for sensor_name in self.virtual_sensors:
            self.logger.info(f"\n📊 Evaluating sensor: {sensor_name}")
            
            try:
                # Load raw sensor data using SAME dataloader as main pipeline
                sensor_predictions = self._extract_raw_sensor_predictions(sensor_name, eval_split)
                
                if not sensor_predictions:
                    self.logger.warning(f"No predictions found for {sensor_name}")
                    all_results[sensor_name] = {}
                    continue
                
                # Run evaluation using SAME system as main pipeline
                metrics = self._run_sensor_evaluation(sensor_name, sensor_predictions)
                all_results[sensor_name] = metrics
                
                self.logger.info(f"✅ Completed evaluation for {sensor_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to evaluate {sensor_name}: {e}")
                all_results[sensor_name] = {}
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _extract_raw_sensor_predictions(self, sensor_name: str, split_name: str) -> List[Dict[str, Any]]:
        """
        Extract raw sensor predictions from dataset.
        Uses the SAME dataloader as main pipeline.
        """
        self.logger.info(f"Loading raw data for {sensor_name} from {split_name} split")
        
        # Build dataloader using SAME system as main pipeline
        temp_config = self.config.copy()
        temp_config['training']['val_split_name'] = split_name
        temp_config['training']['batch_size'] = 1  # Process one sample at a time
        
        # FORCE training mode to enable sensor noise simulation!
        # This ensures virtual sensors apply realistic noise for baseline evaluation
        temp_config['_force_training_mode'] = True  # Special flag for baseline evaluation
        
        # DOE SUPPORT: Override config hash if provided to find correct preprocessed data
        if self.config_hash_override:
            temp_config['_config_hash_override'] = self.config_hash_override
            self.logger.info(f"🎯 DOE Mode: Using config hash override: {self.config_hash_override}")
        
        dataloaders = build_autoregressive_dataloaders(
            temp_config, 
            self.logger, 
            splits_to_build=['val']
        )
        
        val_dataloader = dataloaders['val']
        if val_dataloader is None:
            self.logger.error("Failed to create validation dataloader")
            return []
        
        all_predictions = []
        
        for batch_idx, batch_dict in enumerate(val_dataloader):
            try:
                # Extract raw sensor data for this sensor
                sensor_predictions = self._extract_sensor_data_from_batch(batch_dict, sensor_name)
                all_predictions.extend(sensor_predictions)
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed {batch_idx + 1} samples for {sensor_name}")
                    
            except Exception as e:
                self.logger.warning(f"Error processing batch {batch_idx} for {sensor_name}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(all_predictions)} predictions for {sensor_name}")
        return all_predictions
    
    def _extract_sensor_data_from_batch(self, batch_dict: Dict[str, Any], sensor_name: str) -> List[Dict[str, Any]]:
        """
        Extract predictions for a specific sensor from batch data.
        This converts raw sensor data to the same format as model predictions.
        """
        predictions = []
        
        # Extract sample tokens
        sample_tokens = batch_dict.get('sample_tokens', [])
        
        # Extract sensor data - this should be the raw, noisy sensor data
        sensor_data = batch_dict.get('sensor_data', {})
        
        if sensor_name not in sensor_data:
            return predictions
        
        sensor_info = sensor_data[sensor_name]  # Sensor data dict
        
        for sample_idx, sample_token in enumerate(sample_tokens):
            sample_predictions = []
            
            # Extract boxes and masks for this sample
            boxes = sensor_info['boxes'][sample_idx]        # [N, 9] - 9D physical boxes
            features = sensor_info['features'][sample_idx]  # [N, 12] - 12D normalized features  
            mask = sensor_info['mask'][sample_idx]          # [N] - padding mask
            
            # Filter out padded entries (mask=True means padding)
            valid_indices = ~mask
            valid_boxes = boxes[valid_indices]             # [N_valid, 9]
            valid_features = features[valid_indices]       # [N_valid, 12]
            
            # Convert each valid box to DevKit format
            for i in range(len(valid_boxes)):
                prediction = self._convert_raw_box_to_devkit_format(
                    valid_boxes[i],    # 9D physical box
                    valid_features[i], # 12D normalized features (for class/conf)
                    batch_dict, 
                    sample_idx,
                    sensor_name,
                    sample_token       # Add sample_token parameter
                )
                if prediction:
                    sample_predictions.append(prediction)
            
            predictions.append({
                'sample_token': sample_token,
                'predictions': sample_predictions
            })
        
        return predictions
    
    def _convert_raw_box_to_devkit_format(self, box_9d, features_12d, batch_dict, sample_idx, sensor_name, sample_token):
        """
        Convert raw sensor box to DevKit format using SAME logic as main pipeline.
        
        Args:
            box_9d: 9D physical box [x, y, z, w, l, h, yaw, vx, vy] (EGO, meters/radians)
            features_12d: 12D normalized features [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin_yaw, cos_yaw, vx_norm, vy_norm, sensor_class, sensor_attr]
            batch_dict: Batch containing ego pose information
            sample_idx: Index of sample in batch
            sensor_name: Name of sensor
        """
        try:
            # Import same utilities as main pipeline
            from pyquaternion import Quaternion
            from oft.transformer.datasets.truckscenes.transforms import transform_box_vehicle_to_world
            
            # Extract box components (EGO coordinates, physical units)
            center_ego = box_9d[:3].numpy()      # [x, y, z] meters
            size_ego = box_9d[3:6].numpy()       # [w, l, h] meters  
            yaw_ego = box_9d[6].item()           # yaw radians
            velocity_ego = box_9d[7:9].numpy()   # [vx, vy] m/s
            
            # Extract ego pose for coordinate transformation
            ego_translation_world = batch_dict['ego_translation_world'][sample_idx].numpy()  # [3]
            ego_rotation_quat = batch_dict['ego_rotation_world_quat'][sample_idx].numpy()    # [4]
            ego_rotation_world = Quaternion(ego_rotation_quat)
            
            # Transform box from EGO to WORLD coordinates (same as main pipeline)
            box_ego_7d = np.array([*center_ego, *size_ego, yaw_ego])  # [x,y,z,w,l,h,yaw]
            box_world_7d = transform_box_vehicle_to_world(
                box_ego_7d, 
                ego_translation_world, 
                ego_rotation_world
            )
            
            # Extract WORLD coordinates
            center_world = box_world_7d[:3]      # [x, y, z] WORLD meters
            size_world = box_world_7d[3:6]       # [w, l, h] meters (same in both frames)
            yaw_world = box_world_7d[6]          # yaw WORLD radians
            
            # Transform velocity to WORLD frame
            velocity_ego_3d = np.array([velocity_ego[0], velocity_ego[1], 0.0])  # Add z=0
            velocity_world_3d = ego_rotation_world.rotate(velocity_ego_3d)       # Rotate to WORLD
            velocity_world = velocity_world_3d[:2]  # [vx, vy] WORLD m/s
            
            # Create rotation quaternion (yaw-only rotation)
            rotation_world = Quaternion(axis=[0, 0, 1], radians=yaw_world)
            
            # Extract class information from features
            # Features 10-11 are sensor_class and sensor_attr
            sensor_class_idx = int(features_12d[10].item())  # Should map to class names
            class_names = self.config['dataset']['class_names']
            
            if 0 <= sensor_class_idx < len(class_names):
                detection_name = class_names[sensor_class_idx]
            else:
                detection_name = "car"  # Default fallback
            
            # Use realistic confidence scores based on sensor type (consistent with training data)
            sensor_confidence_map = {
                'virtual_lidar': 0.85,    # Matches training input confidence
                'virtual_camera': 0.75,   # Matches training input confidence  
                'virtual_radar': 0.65     # Matches training input confidence
            }
            detection_score = sensor_confidence_map.get(sensor_name, 0.75)
            
            # Map attribute index to TruckScenes attribute name
            sensor_attr_idx = int(features_12d[11].item())  # Feature 11 uses YOUR system's attribute indices
            
            # Use YOUR ATTRIBUTE_VOCAB (confirmed by test - Feature[11] matches GT attribute indices!)
            from oft.transformer.datasets.truckscenes.dataset import ATTRIBUTE_VOCAB
            attribute_name = ATTRIBUTE_VOCAB[sensor_attr_idx] if 0 <= sensor_attr_idx < len(ATTRIBUTE_VOCAB) else ""
            
            return {
                "sample_token": str(sample_token),                       # Required by TruckScenes DevKit
                "translation": center_world.tolist(),                    # [x, y, z] WORLD meters
                "size": size_world.tolist(),                             # [w, l, h] meters
                "rotation": rotation_world.elements.tolist(),            # [w, x, y, z] quaternion
                "velocity": velocity_world.tolist(),                     # [vx, vy] WORLD m/s
                "detection_name": detection_name,                        # Class name
                "detection_score": detection_score,                      # Confidence
                "attribute_name": attribute_name                         # Attribute
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to convert box for {sensor_name}: {e}")
            return None
    
    def _run_sensor_evaluation(self, sensor_name: str, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Run TruckScenes evaluation for a sensor using SAME system as main pipeline.
        """
        try:
            # Create sensor-specific output directory
            sensor_output_dir = os.path.join(self.output_dir, f"{sensor_name}_eval")
            os.makedirs(sensor_output_dir, exist_ok=True)
            
            self.logger.info(f"Running TruckScenes evaluation for {sensor_name}")
            
            # Use the SAME evaluation function as main pipeline (base_trainer.py:222-224)
            eval_result = run_devkit_evaluation(
                raw_predictions_list=predictions,
                config=self.config,
                output_dir=sensor_output_dir
            )
            
            # Extract metrics (same as what would be logged in main pipeline)
            metrics = {}
            if eval_result and hasattr(eval_result, 'nd_score'):
                metrics = {
                    'nd_score': getattr(eval_result, 'nd_score', 0.0),
                    'mean_ap': getattr(eval_result, 'mean_ap', 0.0),
                    'mean_ate': getattr(eval_result, 'mean_ate', 0.0),
                    'mean_aoe': getattr(eval_result, 'mean_aoe', 0.0),
                    'mean_ase': getattr(eval_result, 'mean_ase', 0.0),
                    'mean_ave': getattr(eval_result, 'mean_ave', 0.0)
                }
            
            # Save individual sensor metrics
            metrics_file = os.path.join(self.output_dir, f"{sensor_name}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f" Metrics saved to {metrics_file}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"TruckScenes evaluation failed for {sensor_name}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print evaluation summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info(" VIRTUAL SENSOR BASELINE EVALUATION RESULTS")
        self.logger.info("="*60)
        
        
        self.logger.info(f"\n Detailed results saved to: {self.output_dir}")
        self.logger.info("="*60)


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(description="Evaluate virtual sensors baseline performance")
    parser.add_argument('--config', type=str, default='config/pipeline_staged.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--config-hash', type=str, default=None,
                       help='Override config hash for DOE experiments (to find preprocessed data)')
    
    args = parser.parse_args()
    
    evaluator = VirtualSensorBaselineEvaluator(args.config, args.output_dir, args.config_hash)
    evaluator.run_baseline_evaluation()


if __name__ == "__main__":
    main()