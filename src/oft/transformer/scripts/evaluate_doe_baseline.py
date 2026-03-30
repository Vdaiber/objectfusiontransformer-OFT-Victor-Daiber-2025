#!/usr/bin/env python3
"""
DOE Baseline Evaluation Script

Evaluates preprocessed virtual sensor data using TruckScenes DevKit.
This script loads preprocessed JSON data (from DOE experiments) and runs
the official TruckScenes evaluation to compute NDS, mAP, mATE, etc.

Unlike the original baseline evaluator, this script:
1. Loads preprocessed JSON data directly (no DataLoader)
2. Converts sensor data to DevKit format
3. Runs official TruckScenes evaluation
4. Saves detailed metrics for DOE analysis
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

# Import evaluation system
from oft.transformer.evaluation.devkit_evaluator import run_devkit_evaluation
from oft.transformer.utils.config_hash_utils import generate_unified_config_hash


class DOEBaselineEvaluator:
    """
    Evaluates preprocessed virtual sensor data for DOE experiments.
    """
    
    def __init__(self, config_path: str, output_dir: str = None, config_hash_override: str = None):
        """Initialize evaluator with config and output directory."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Calculate config hash to find preprocessed data (or use override)
        if config_hash_override:
            self.config_hash = config_hash_override
        else:
            self.config_hash = generate_unified_config_hash(self.config)
        
        # Set up paths
        cache_root = self.config.get('preprocessing', {}).get('cache_dir', '/data/daiber_fent/virtual_sensor_cache')
        self.cache_dir = Path(cache_root) / f'config_{self.config_hash}'
        
        if output_dir is None:
            output_dir = str(self.cache_dir / 'baseline_evaluation_results')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Get validation split name
        self.val_split = self.config.get('training', {}).get('val_split_name', 'val')
        
        # Get enabled virtual sensors
        self.virtual_sensors = [
            sensor['name'] for sensor in self.config['dataset']['virtual_sensors'] 
            if sensor.get('enabled', True)
        ]
        
        self.logger.info(f"Initialized DOE baseline evaluator")
        self.logger.info(f"Config hash: {self.config_hash}")
        self.logger.info(f"Cache directory: {self.cache_dir}")
        self.logger.info(f"Validation split: {self.val_split}")
        self.logger.info(f"Virtual sensors: {self.virtual_sensors}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the evaluator."""
        logger = logging.getLogger('DOEBaselineEvaluator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_preprocessed_data(self) -> Dict[str, Any]:
        """Load preprocessed JSON data."""
        json_file = self.cache_dir / f'complete_dataset_{self.val_split}.json'
        
        if not json_file.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {json_file}")
        
        self.logger.info(f"Loading preprocessed data: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data['results'])} samples")
        return data
    
    def convert_sensor_data_to_devkit_format(self, sensor_data: Dict[str, Any], sensor_name: str, sample_token: str) -> List[Dict[str, Any]]:
        """Convert sensor data from preprocessed format to DevKit format."""
        if 'boxes' not in sensor_data or len(sensor_data['boxes']) == 0:
            return []
        
        predictions = []
        boxes = sensor_data['boxes']
        features = sensor_data.get('features', [])
        
        for i, box in enumerate(boxes):
            # box format: [x, y, z, w, l, h, yaw, vx, vy]
            if len(box) < 9:
                continue
                
            # Extract box components
            center = [float(box[0]), float(box[1]), float(box[2])]  # [x, y, z] world coordinates
            size = [float(box[3]), float(box[4]), float(box[5])]    # [w, l, h]
            yaw = float(box[6])                                      # yaw angle
            velocity = [float(box[7]), float(box[8])]               # [vx, vy]
            
            # Convert yaw to quaternion [w, x, y, z]
            # Yaw rotation around z-axis
            half_yaw = yaw / 2.0
            rotation = [
                np.cos(half_yaw),  # w
                0.0,               # x  
                0.0,               # y
                np.sin(half_yaw)   # z
            ]
            
            # Get class and confidence from features if available
            if i < len(features) and len(features[i]) >= 12:
                feature = features[i]
                class_idx = int(feature[10])  # class index
                confidence = float(feature[11]) if len(feature) > 11 else 1.0  # confidence or metadata
            else:
                class_idx = 0  # default to first class
                confidence = 1.0
            
            # Map class index to class name
            class_names = self.config['dataset']['class_names']
            if 0 <= class_idx < len(class_names):
                detection_name = class_names[class_idx]
            else:
                detection_name = class_names[0]  # fallback
            
            # Create prediction in DevKit format
            prediction = {
                "translation": center,
                "size": size,
                "rotation": rotation,
                "velocity": velocity,
                "detection_name": detection_name,
                "detection_score": confidence
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def run_baseline_evaluation(self) -> Dict[str, Dict[str, float]]:
        """Run baseline evaluation for all virtual sensors."""
        try:
            # Load preprocessed data
            data = self.load_preprocessed_data()
            
            all_results = {}
            
            # Evaluate each virtual sensor
            for sensor_name in self.virtual_sensors:
                self.logger.info(f"Evaluating {sensor_name}...")
                
                # Extract predictions for this sensor
                raw_predictions_list = []
                
                for sample_token, sample_data in data['results'].items():
                    if 'sensor_data' in sample_data and sensor_name in sample_data['sensor_data']:
                        sensor_data = sample_data['sensor_data'][sensor_name]
                        
                        # Convert to DevKit format
                        predictions = self.convert_sensor_data_to_devkit_format(
                            sensor_data, sensor_name, sample_token
                        )
                        
                        raw_predictions_list.append({
                            'sample_token': sample_token,
                            'predictions': predictions
                        })
                
                if not raw_predictions_list:
                    self.logger.warning(f"No predictions found for {sensor_name}")
                    continue
                
                # Run DevKit evaluation
                sensor_output_dir = self.output_dir / f'{sensor_name}_eval'
                sensor_output_dir.mkdir(exist_ok=True)
                
                self.logger.info(f"Running DevKit evaluation for {sensor_name}...")
                self.logger.info(f"Output directory: {sensor_output_dir}")
                
                try:
                    eval_result = run_devkit_evaluation(
                        raw_predictions_list=raw_predictions_list,
                        config=self.config,
                        output_dir=str(sensor_output_dir)
                    )
                    
                    # Extract metrics
                    if eval_result is not None:
                        # Get main metrics
                        metrics = {
                            'nds': eval_result.summary.get('nds', 0.0),
                            'mAP': eval_result.summary.get('mAP', 0.0),
                            'mATE': eval_result.summary.get('mATE', 0.0),
                            'mASE': eval_result.summary.get('mASE', 0.0),
                            'mAOE': eval_result.summary.get('mAOE', 0.0),
                            'mAVE': eval_result.summary.get('mAVE', 0.0),
                            'total_predictions': len(raw_predictions_list)
                        }
                        
                        all_results[sensor_name] = metrics
                        
                        # Save detailed metrics
                        metrics_file = sensor_output_dir / 'metrics_summary.json'
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2)
                        
                        self.logger.info(f"✅ {sensor_name} evaluation completed")
                        self.logger.info(f"   NDS: {metrics['nds']:.3f}")
                        self.logger.info(f"   mAP: {metrics['mAP']:.3f}")
                        self.logger.info(f"   mATE: {metrics['mATE']:.3f}")
                        
                    else:
                        self.logger.error(f"❌ DevKit evaluation failed for {sensor_name}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error evaluating {sensor_name}: {e}")
                    continue
            
            # Save overall summary
            summary_file = self.output_dir / 'evaluation_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            self.logger.info(f"💾 Evaluation summary saved: {summary_file}")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"❌ Baseline evaluation failed: {e}")
            return {}


def main():
    """Main entry point for DOE baseline evaluation."""
    parser = argparse.ArgumentParser(description="DOE baseline evaluation using preprocessed data")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: auto from config hash)')
    parser.add_argument('--config-hash', type=str, default=None,
                       help='Override config hash for finding preprocessed data (for DOE experiments)')
    
    args = parser.parse_args()
    
    evaluator = DOEBaselineEvaluator(args.config, args.output_dir, args.config_hash)
    results = evaluator.run_baseline_evaluation()
    
    if results:
        print("✅ DOE baseline evaluation completed successfully!")
        for sensor_name, metrics in results.items():
            print(f"📊 {sensor_name}: NDS={metrics.get('nds', 0):.3f}, mAP={metrics.get('mAP', 0):.3f}")
    else:
        print("❌ DOE baseline evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
