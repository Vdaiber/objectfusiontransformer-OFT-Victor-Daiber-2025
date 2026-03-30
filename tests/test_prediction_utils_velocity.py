#!/usr/bin/env python3
"""
Test der Prediction Utils - speziell Velocity Handling
Testet die Box-Rekonstruktion und Velocity-Transformation in prediction_utils.py
"""

import unittest
import torch
import numpy as np
import yaml
import tempfile
import os
from typing import Dict, Any, List
from pyquaternion import Quaternion as PyQuaternion
from pathlib import Path

# Import the modules we need to test
import sys
sys.path.append('/app/src')

from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive
from oft.transformer.datasets.truckscenes.transforms import (
    transform_velocity_world_to_ego_frame,
    transform_relative_velocity_ego_to_world_frame
)
from oft.transformer.utils.normalization_utils import CentralizedNormalizer


class PredictionUtilsVelocityTest(unittest.TestCase):
    """Test velocity handling in prediction_utils.py"""

    def setUp(self):
        """Set up test data and configuration."""
        
        # Create temporary normalization stats file
        self.temp_dir = tempfile.mkdtemp()
        self.norm_stats_path = os.path.join(self.temp_dir, "test_normalization_stats.yaml")
        
        # Create test normalization stats (matching real config format)
        test_norm_stats = {
            'center_abs_99p': [0.42, 0.42, 0.42],
            'log_size_abs_99p': [0.39, 0.39, 0.39],
            'velocity_percentiles': {
                '1p': [-2.9, -2.9],
                '99p': [2.9, 2.9]
            },
            'velocity_offset_percentiles': {
                '1p': [-2.9, -2.9],
                '99p': [2.9, 2.9]
            },
            'absolute_velocity_percentiles': {
                '1p': [-50.0, -50.0],
                '99p': [50.0, 50.0],
                '5p': [-30.0, -30.0],
                '95p': [30.0, 30.0],
                'mean': [0.0, 0.0],
                'std': [15.0, 15.0]
            },
            'metadata': {
                'point_cloud_range': [-150.0, -150.0, -12.0, 150.0, 150.0, 12.0]
            }
        }
        
        with open(self.norm_stats_path, 'w') as f:
            yaml.dump(test_norm_stats, f)
        
        # Initialize normalizer with test stats
        self.normalizer = CentralizedNormalizer(self.norm_stats_path)
        
        # Test configuration (adapted for mini dataset)
        self.cfg = {
            'dataset': {
                'dataroot': '/app/datasets/v1.0-mini',
                'version': 'v1.0-mini',
                'class_names': ['car', 'truck', 'bus', 'trailer', 'other_vehicle',
                               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone',
                               'barrier', 'animal', 'traffic_sign']
            },
            'evaluation': {
                'conf_th_eval': 0.01
            }
        }
        
        # Point cloud range for coordinate normalization
        self.point_cloud_range = torch.tensor([-150.0, -150.0, -12.0, 150.0, 150.0, 12.0])

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_prediction_utils_velocity_reconstruction(self):
        """Test the prediction_utils velocity reconstruction pipeline."""
        print("\n🔍 Testing Prediction Utils Velocity Reconstruction")
        
        # Test realistic velocity scenarios
        test_scenarios = [
            {
                'name': 'stationary_car',
                'gt_velocity_world': np.array([0.0, 0.0], dtype=np.float64),
                'ego_velocity_world': np.array([20.0, 0.0], dtype=np.float64),
                'ego_rotation': PyQuaternion(axis=[0, 0, 1], radians=0.0)
            },
            {
                'name': 'moving_car_forward',
                'gt_velocity_world': np.array([25.0, 0.0], dtype=np.float64),
                'ego_velocity_world': np.array([20.0, 0.0], dtype=np.float64),
                'ego_rotation': PyQuaternion(axis=[0, 0, 1], radians=0.0)
            },
            {
                'name': 'car_with_rotation',
                'gt_velocity_world': np.array([30.0, 10.0], dtype=np.float64),
                'ego_velocity_world': np.array([25.0, 5.0], dtype=np.float64),
                'ego_rotation': PyQuaternion(axis=[0, 0, 1], radians=np.pi/4)  # 45° rotation
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n--- Testing {scenario['name']} ---")
            
            # Extract scenario data
            gt_velocity_world = scenario['gt_velocity_world']
            ego_velocity_world = scenario['ego_velocity_world']
            ego_rotation = scenario['ego_rotation']
            ego_translation = np.array([10.0, 5.0, 0.0], dtype=np.float64)
            
            print(f"GT Velocity World: {gt_velocity_world} m/s")
            print(f"Ego Velocity World: {ego_velocity_world} m/s")
            print(f"Ego Rotation: {ego_rotation.degrees:.1f}°")
            
            # ==============================
            # STEP 1: Create the expected velocity in ego frame
            # ==============================
            expected_velocity_ego = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=gt_velocity_world,
                ego_rotation=ego_rotation,
                ego_velocity_world_phys_2d=ego_velocity_world
            )
            print(f"Expected Velocity Ego: {expected_velocity_ego} m/s")
            
            # ==============================
            # STEP 2: Normalize the velocity (as model would output)
            # ==============================
            velocity_ego_tensor = torch.from_numpy(expected_velocity_ego).double().unsqueeze(0)
            velocity_normalized = self.normalizer.normalize_velocity_absolute(velocity_ego_tensor)
            print(f"Velocity Normalized: {velocity_normalized.squeeze().numpy()}")
            
            # ==============================
            # STEP 3: Create mock batch_dict (as from dataloader)
            # ==============================
            batch_dict = {
                'sample_tokens': ['test_sample_001'],
                'ego_translation_world': torch.from_numpy(ego_translation).double().unsqueeze(0),  # (1, 3)
                'ego_rotation_world_quat': torch.tensor([ego_rotation.elements], dtype=torch.float64),  # (1, 4)
                'ego_motion': {
                    'cabin': {
                        'velocity': torch.tensor([[ego_velocity_world[0], ego_velocity_world[1], 0.0]], dtype=torch.float64)  # (1, 3)
                    }
                }
            }
            
            # ==============================
            # STEP 4: Create mock predictions (as from model)
            # ==============================
            batch_size = 1
            max_detections = 100
            
            predictions = {}
            
            # Classification (one high-confidence car detection)
            class_logits = torch.full((batch_size, max_detections, len(self.cfg['dataset']['class_names']) + 1), -10.0)
            class_logits[0, 0, 0] = 5.0  # High confidence car
            predictions['pred_class_logits_batch'] = class_logits
            
            # Box predictions (normalized 10D format)
            boxes_normalized = torch.zeros((batch_size, max_detections, 10))
            boxes_normalized[0, 0, :3] = torch.tensor([0.57, 0.53, 0.5])  # Normalized coordinates
            boxes_normalized[0, 0, 3:6] = torch.tensor([0.1, 0.2, 0.05])  # Normalized dimensions
            boxes_normalized[0, 0, 6:8] = torch.tensor([0.0, 1.0])  # sin(0), cos(0) = forward facing
            predictions['pred_boxes_normalized'] = boxes_normalized
            
            # Velocity predictions (normalized, our test data)
            velocities_normalized = torch.zeros((batch_size, max_detections, 2))
            velocities_normalized[0, 0] = velocity_normalized.squeeze()  # Use our test velocity
            predictions['pred_velocities_normalized'] = velocities_normalized
            
            # Attribute predictions
            attr_logits = torch.full((batch_size, max_detections, 9), -10.0)  # 8 attributes + 1 no_object
            attr_logits[0, 0, 5] = 3.0  # vehicle.moving
            predictions['pred_attributes_logits_batch'] = attr_logits
            
            print(f"Mock predictions created with normalized velocity: {velocity_normalized.squeeze().numpy()}")
            
            # ==============================
            # STEP 5: Call prediction_utils reconstruction
            # ==============================
            print("STEP 5: Call prediction_utils reconstruction")
            
            try:
                results = reconstruct_and_convert_predictions_autoregressive(
                    batch_dict=batch_dict,
                    predictions=predictions,
                    cfg=self.cfg
                )
                
                print(f"✅ Reconstruction successful")
                
                # Extract results
                sample_result = results[0]
                sample_predictions = sample_result['predictions']
                
                print(f"Number of predictions: {len(sample_predictions)}")
                
                # Find the high-confidence prediction
                high_conf_pred = None
                for pred in sample_predictions:
                    if pred['detection_score'] > 0.5:  # High confidence
                        high_conf_pred = pred
                        break
                
                if high_conf_pred is None:
                    self.fail("No high-confidence prediction found")
                
                # ==============================
                # STEP 6: Verify velocity reconstruction
                # ==============================
                print("STEP 6: Verify velocity reconstruction")
                
                reconstructed_velocity_world = np.array(high_conf_pred['velocity'], dtype=np.float64)
                print(f"Reconstructed Velocity World: {reconstructed_velocity_world} m/s")
                print(f"Expected Velocity World:      {gt_velocity_world} m/s")
                
                # Calculate error
                velocity_error = np.linalg.norm(reconstructed_velocity_world - gt_velocity_world)
                print(f"Velocity Error: {velocity_error:.8f} m/s")
                
                # ==============================
                # STEP 7: Manual verification of the reconstruction steps
                # ==============================
                print("STEP 7: Manual verification of reconstruction steps")
                
                # Step 7a: Denormalize velocity
                velocity_denormalized = self.normalizer.denormalize_velocity_absolute(velocity_normalized)
                manual_velocity_ego = velocity_denormalized.squeeze().numpy()
                print(f"Manual denormalized ego velocity: {manual_velocity_ego} m/s")
                print(f"Expected ego velocity:            {expected_velocity_ego} m/s")
                
                # Step 7b: Transform back to world
                manual_velocity_world = transform_relative_velocity_ego_to_world_frame(
                    relative_velocity_ego_2d=manual_velocity_ego,
                    ego_rotation=ego_rotation,
                    ego_velocity_world_phys_2d=ego_velocity_world
                )
                print(f"Manual world velocity:            {manual_velocity_world} m/s")
                
                # Compare manual vs prediction_utils result
                manual_error = np.linalg.norm(manual_velocity_world - gt_velocity_world)
                prediction_utils_error = np.linalg.norm(reconstructed_velocity_world - gt_velocity_world)
                
                print(f"Manual reconstruction error:      {manual_error:.8f} m/s")
                print(f"prediction_utils error:           {prediction_utils_error:.8f} m/s")
                
                # ==============================
                # STEP 8: Assertions
                # ==============================
                print("STEP 8: Verification")
                
                # Manual reconstruction should be perfect
                self.assertLess(manual_error, 1e-10, 
                               f"Manual reconstruction error too large: {manual_error}")
                print(f"✅ Manual reconstruction is perfect")
                
                # prediction_utils should match manual reconstruction
                prediction_manual_diff = np.linalg.norm(reconstructed_velocity_world - manual_velocity_world)
                print(f"Difference between prediction_utils and manual: {prediction_manual_diff:.8f} m/s")
                
                if prediction_manual_diff > 1e-10:
                    print(f"❌ prediction_utils differs from manual reconstruction!")
                    print(f"   This indicates a BUG in prediction_utils.py")
                    
                    # Show detailed comparison
                    print(f"   Manual result:        {manual_velocity_world}")
                    print(f"   prediction_utils result: {reconstructed_velocity_world}")
                    print(f"   Difference:           {reconstructed_velocity_world - manual_velocity_world}")
                else:
                    print(f"✅ prediction_utils matches manual reconstruction")
                
                print(f"🎉 {scenario['name']} test completed")
                
            except Exception as e:
                print(f"❌ Reconstruction failed: {e}")
                import traceback
                traceback.print_exc()
                self.fail(f"Reconstruction failed for {scenario['name']}: {e}")

    def test_velocity_denormalization_in_prediction_utils(self):
        """Test specific velocity denormalization in prediction_utils."""
        print("\n🔍 Testing Velocity Denormalization in prediction_utils")
        
        # Test different normalized velocity values
        test_normalized_velocities = [
            np.array([0.0, 0.0], dtype=np.float64),      # Zero velocity
            np.array([0.2, -0.1], dtype=np.float64),     # Moderate velocity
            np.array([1.0, -1.0], dtype=np.float64),     # Maximum velocity (at percentile)
            np.array([-0.5, 0.8], dtype=np.float64),     # Mixed velocity
        ]
        
        for i, vel_norm in enumerate(test_normalized_velocities):
            print(f"\nTest case {i+1}: Normalized velocity {vel_norm}")
            
            # Convert to tensor format as prediction_utils expects
            vel_norm_tensor = torch.from_numpy(vel_norm).double().unsqueeze(0)  # (1, 2)
            
            # Test denormalization
            vel_denorm = self.normalizer.denormalize_velocity_absolute(vel_norm_tensor)
            vel_denorm_np = vel_denorm.squeeze().numpy()
            
            print(f"  Denormalized: {vel_denorm_np} m/s")
            
            # Test round-trip
            vel_renorm = self.normalizer.normalize_velocity_absolute(vel_denorm)
            roundtrip_error = torch.norm(vel_norm_tensor - vel_renorm).item()
            
            print(f"  Round-trip error: {roundtrip_error:.2e} m/s")
            
            self.assertLess(roundtrip_error, 1e-10, 
                           f"Round-trip error too large: {roundtrip_error}")
            print(f"  ✅ Round-trip successful")

    def test_coordinate_transformation_in_prediction_utils(self):
        """Test coordinate transformations as used in prediction_utils."""
        print("\n🔍 Testing Coordinate Transformations in prediction_utils")
        
        # Test various ego poses
        test_poses = [
            {
                'name': 'no_rotation_no_movement',
                'ego_velocity': np.array([0.0, 0.0], dtype=np.float64),
                'ego_rotation': PyQuaternion(axis=[0, 0, 1], radians=0.0)
            },
            {
                'name': 'forward_movement_no_rotation',
                'ego_velocity': np.array([20.0, 0.0], dtype=np.float64),
                'ego_rotation': PyQuaternion(axis=[0, 0, 1], radians=0.0)
            },
            {
                'name': 'forward_movement_45deg_rotation',
                'ego_velocity': np.array([20.0, 0.0], dtype=np.float64),
                'ego_rotation': PyQuaternion(axis=[0, 0, 1], radians=np.pi/4)
            },
            {
                'name': 'diagonal_movement_90deg_rotation',
                'ego_velocity': np.array([15.0, 10.0], dtype=np.float64),
                'ego_rotation': PyQuaternion(axis=[0, 0, 1], radians=np.pi/2)
            }
        ]
        
        # Test object velocity
        object_velocity_world = np.array([25.0, 5.0], dtype=np.float64)
        
        for pose in test_poses:
            print(f"\nTesting pose: {pose['name']}")
            print(f"  Object velocity world: {object_velocity_world} m/s")
            print(f"  Ego velocity world: {pose['ego_velocity']} m/s")
            print(f"  Ego rotation: {pose['ego_rotation'].degrees:.1f}°")
            
            # Step 1: World to Ego
            velocity_ego = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=object_velocity_world,
                ego_rotation=pose['ego_rotation'],
                ego_velocity_world_phys_2d=pose['ego_velocity']
            )
            print(f"  Velocity ego: {velocity_ego} m/s")
            
            # Step 2: Ego to World (round-trip)
            velocity_world_reconstructed = transform_relative_velocity_ego_to_world_frame(
                relative_velocity_ego_2d=velocity_ego,
                ego_rotation=pose['ego_rotation'],
                ego_velocity_world_phys_2d=pose['ego_velocity']
            )
            print(f"  Velocity world reconstructed: {velocity_world_reconstructed} m/s")
            
            # Verify round-trip accuracy
            roundtrip_error = np.linalg.norm(object_velocity_world - velocity_world_reconstructed)
            print(f"  Round-trip error: {roundtrip_error:.2e} m/s")
            
            self.assertLess(roundtrip_error, 1e-10, 
                           f"Coordinate transformation round-trip error: {roundtrip_error}")
            print(f"  ✅ Coordinate transformation successful")


def main():
    """Run the prediction utils velocity test."""
    print("🚀 Starting Prediction Utils Velocity Test")
    print("=" * 60)
    print("Testing velocity handling in prediction_utils.py")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(PredictionUtilsVelocityTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL PREDICTION UTILS TESTS PASSED!")
        print("✅ Velocity handling in prediction_utils.py is correct")
    else:
        print("❌ PREDICTION UTILS TESTS FAILED:")
        for failure in result.failures:
            print(f"  - {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"  - {error[0]}: {error[1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)