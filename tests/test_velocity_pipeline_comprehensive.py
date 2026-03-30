#!/usr/bin/env python3
"""
Comprehensive Unit Test for Velocity Pipeline

Tests the complete velocity processing pipeline from original data loading through
data augmentation, normalization, denormalization, reconstruction, and DevKit evaluation format.

This test helps identify where velocity errors are introduced in the baseline evaluation.
"""

import unittest
import torch
import numpy as np
import yaml
import json
import tempfile
import os
from typing import Dict, Any, List, Tuple, Optional
from pyquaternion import Quaternion as PyQuaternion
from pathlib import Path

# Import the modules we need to test
import sys
sys.path.append('/app/src')

from oft.transformer.datasets.truckscenes.transforms import (
    transform_velocity_world_to_ego_frame,
    transform_relative_velocity_ego_to_world_frame
)
from oft.transformer.datasets.preprocessing.data_augmentation import add_noise_to_box
from oft.transformer.utils.normalization_utils import (
    CentralizedNormalizer,
    normalize_velocity_absolute,
    denormalize_velocity_absolute,
    normalize_velocity_offset,
    denormalize_velocity_offset
)
from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive
from oft.transformer.utils.geometry_utils import yaw_to_sin_cos, sin_cos_to_yaw


class VelocityPipelineTest(unittest.TestCase):
    """Comprehensive test suite for the velocity processing pipeline."""

    def setUp(self):
        """Set up test data and configuration."""
        
        # Check if mini dataset is available
        mini_dataset_path = Path('/app/datasets/v1.0-mini')
        self.mini_dataset_available = mini_dataset_path.exists()
        if not self.mini_dataset_available:
            print(f"⚠️  Mini dataset not found at {mini_dataset_path}")
            print("   Some tests will use mock data instead")
        
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
                'dataroot': '/app/datasets/v1.0-mini',  # Mini dataset path
                'version': 'v1.0-mini',  # Mini dataset version
                'class_names': ['car', 'truck', 'bus', 'trailer', 'other_vehicle',
                               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone',
                               'barrier', 'animal', 'traffic_sign'],
                'cutoff_dist': 150.0,
                'virtual_sensors': [{
                    'name': 'virtual_radar',
                    'velocity_noise_std': 0.078,
                    'pos_noise_std': [0.85, 0.72, 1.40],  # Required by add_noise_to_box
                    'dim_noise_std': 0.58,
                    'yaw_noise_std': 0.30,
                    'dropout_rate': 0.18,
                    'class_accuracy': 0.73,
                    'attribute_accuracy': 0.65
                }]
            },
            'evaluation': {
                'conf_th_eval': 0.01
            }
        }
        
        # Point cloud range for coordinate normalization
        self.point_cloud_range = torch.tensor([-150.0, -150.0, -12.0, 150.0, 150.0, 12.0])
        
        # Test data: realistic velocity values
        self.test_velocities_world = {
            'stationary_car': np.array([0.0, 0.0], dtype=np.float64),
            'moving_car_forward': np.array([25.0, 0.0], dtype=np.float64),  # 90 km/h forward
            'moving_car_diagonal': np.array([20.0, 10.0], dtype=np.float64),  # diagonal movement
            'moving_car_backward': np.array([-15.0, 5.0], dtype=np.float64),  # backing up with steering
            'fast_highway': np.array([35.0, -5.0], dtype=np.float64),  # 126 km/h with lane change
        }
        
        # Test ego motion
        self.ego_velocity_world = np.array([20.0, 0.0], dtype=np.float64)  # Ego moving at 72 km/h
        self.ego_rotation = PyQuaternion(axis=[0, 0, 1], radians=0.0)  # No rotation initially
        self.ego_translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Test boxes (7D format: [x, y, z, w, l, h, yaw])
        self.test_box_ego = np.array([10.0, 5.0, 0.0, 2.0, 4.5, 1.8, 0.0], dtype=np.float64)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_step1_temporal_velocity_calculation(self):
        """Test Step 1: Velocity calculation from temporal tracking."""
        print("\n🔍 Testing Step 1: Temporal Velocity Calculation")
        
        # Simulate temporal tracking data
        dt = 0.1  # 100ms between frames
        
        # Position at t=0
        pos_t0 = np.array([10.0, 5.0], dtype=np.float64)
        
        # Position at t=1 (after movement)
        velocity_true = np.array([25.0, 10.0], dtype=np.float64)  # 25 m/s forward, 10 m/s left
        pos_t1 = pos_t0 + velocity_true * dt
        
        # Calculate velocity from positions
        velocity_calculated = (pos_t1 - pos_t0) / dt
        
        # Verify calculation
        np.testing.assert_allclose(velocity_calculated, velocity_true, rtol=1e-10)
        print(f"✅ Temporal velocity calculation: {velocity_calculated} m/s")
        
        # Test edge cases
        # Zero movement
        velocity_zero = (pos_t0 - pos_t0) / dt
        np.testing.assert_allclose(velocity_zero, [0.0, 0.0], rtol=1e-10)
        print(f"✅ Zero velocity case: {velocity_zero} m/s")

    def test_step2_coordinate_transformations(self):
        """Test Step 2: Coordinate frame transformations."""
        print("\n🔍 Testing Step 2: Coordinate Frame Transformations")
        
        for name, vel_world in self.test_velocities_world.items():
            print(f"\n--- Testing {name}: {vel_world} m/s (world) ---")
            
            # Test 1: World to Ego transformation
            vel_ego_relative = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=vel_world,
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            print(f"World->Ego: {vel_world} -> {vel_ego_relative} m/s (relative)")
            
            # Test 2: Ego to World transformation (round-trip)
            vel_world_reconstructed = transform_relative_velocity_ego_to_world_frame(
                relative_velocity_ego_2d=vel_ego_relative,
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            print(f"Ego->World: {vel_ego_relative} -> {vel_world_reconstructed} m/s (absolute)")
            
            # Verify round-trip accuracy
            np.testing.assert_allclose(vel_world, vel_world_reconstructed, rtol=1e-10)
            print(f"✅ Round-trip transformation successful")
            
            # Test with rotation
            ego_rotation_45deg = PyQuaternion(axis=[0, 0, 1], radians=np.pi/4)  # 45° rotation
            
            vel_ego_rotated = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=vel_world,
                ego_rotation=ego_rotation_45deg,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            
            vel_world_from_rotated = transform_relative_velocity_ego_to_world_frame(
                relative_velocity_ego_2d=vel_ego_rotated,
                ego_rotation=ego_rotation_45deg,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            
            np.testing.assert_allclose(vel_world, vel_world_from_rotated, rtol=1e-10)
            print(f"✅ Rotated coordinate transformation successful")

    def test_step3_data_augmentation_velocity_noise(self):
        """Test Step 3: Data augmentation with velocity noise."""
        print("\n🔍 Testing Step 3: Data Augmentation - Velocity Noise")
        
        # Test configuration for radar sensor
        sensor_cfg = self.cfg['dataset']['virtual_sensors'][0]
        velocity_noise_std = sensor_cfg['velocity_noise_std']
        
        print(f"Testing velocity noise with std = {velocity_noise_std} m/s")
        
        for name, vel_ego in self.test_velocities_world.items():
            print(f"\n--- Testing {name}: {vel_ego} m/s ---")
            
            # Transform to ego frame first
            vel_ego_relative = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=vel_ego,
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            
            # Apply noise
            box_9d_noisy, vel_noisy = add_noise_to_box(
                sensor_box_ego_phys_7d=self.test_box_ego,
                sensor_velocity_ego_phys_2d=vel_ego_relative,
                sensor_noise_cfg=sensor_cfg,
                sample_token="test_sample_123",  # For deterministic noise
                global_seed=42
            )
            
            # Extract velocity from 9D box
            vel_from_box = box_9d_noisy[7:9]
            
            # Verify noise was applied
            noise_applied = vel_noisy - vel_ego_relative
            print(f"Original velocity: {vel_ego_relative} m/s")
            print(f"Noisy velocity: {vel_noisy} m/s")
            print(f"Applied noise: {noise_applied} m/s")
            print(f"Noise magnitude: {np.linalg.norm(noise_applied):.4f} m/s")
            
            # Verify consistency between outputs
            np.testing.assert_allclose(vel_noisy, vel_from_box, rtol=1e-10)
            print(f"✅ Velocity consistency between outputs")
            
            # Test deterministic behavior
            box_9d_noisy2, vel_noisy2 = add_noise_to_box(
                sensor_box_ego_phys_7d=self.test_box_ego,
                sensor_velocity_ego_phys_2d=vel_ego_relative,
                sensor_noise_cfg=sensor_cfg,
                sample_token="test_sample_123",  # Same token should give same noise
                global_seed=42
            )
            
            np.testing.assert_allclose(vel_noisy, vel_noisy2, rtol=1e-10)
            print(f"✅ Deterministic noise generation")

    def test_step4_velocity_normalization_denormalization(self):
        """Test Step 4: Velocity normalization and denormalization."""
        print("\n🔍 Testing Step 4: Velocity Normalization/Denormalization")
        
        for name, vel_world in self.test_velocities_world.items():
            print(f"\n--- Testing {name}: {vel_world} m/s ---")
            
            # Convert to ego frame
            vel_ego_relative = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=vel_world,
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            
            # Convert to tensor (use float64 for consistency)
            vel_tensor = torch.from_numpy(vel_ego_relative).double().unsqueeze(0)  # (1, 2)
            
            # Test absolute velocity normalization (used for dataset)
            vel_norm_absolute = self.normalizer.normalize_velocity_absolute(vel_tensor)
            vel_denorm_absolute = self.normalizer.denormalize_velocity_absolute(vel_norm_absolute)
            
            print(f"Absolute velocity normalization:")
            print(f"  Original: {vel_tensor.numpy()} m/s")
            print(f"  Normalized: {vel_norm_absolute.numpy()}")
            print(f"  Denormalized: {vel_denorm_absolute.numpy()} m/s")
            
            # Verify round-trip accuracy
            torch.testing.assert_close(vel_tensor, vel_denorm_absolute, rtol=1e-6, atol=1e-6)
            print(f"✅ Absolute velocity round-trip successful")
            
            # Test velocity offset normalization (used for object decoder)
            # Create some velocity offsets
            vel_offsets = torch.tensor([[0.1, -0.2], [1.5, 2.8], [-0.5, 0.3]], dtype=torch.float64)
            
            vel_offsets_norm = self.normalizer.normalize_velocity_offset(vel_offsets)
            vel_offsets_denorm = self.normalizer.denormalize_velocity_offset(vel_offsets_norm)
            
            print(f"Velocity offset normalization:")
            print(f"  Original offsets: {vel_offsets.numpy()} m/s")
            print(f"  Normalized offsets: {vel_offsets_norm.numpy()}")
            print(f"  Denormalized offsets: {vel_offsets_denorm.numpy()} m/s")
            
            torch.testing.assert_close(vel_offsets, vel_offsets_denorm, rtol=1e-6, atol=1e-6)
            print(f"✅ Velocity offset round-trip successful")

    def test_step5_sin_cos_yaw_handling(self):
        """Test Step 5: Sin/Cos yaw representation and conversion."""
        print("\n🔍 Testing Step 5: Sin/Cos Yaw Handling")
        
        test_yaws = [0.0, np.pi/4, np.pi/2, np.pi, -np.pi/2, -np.pi/4, 2*np.pi]
        
        for yaw in test_yaws:
            print(f"\n--- Testing yaw = {yaw:.4f} rad ({np.degrees(yaw):.1f}°) ---")
            
            # Convert to sin/cos
            sin_yaw, cos_yaw = yaw_to_sin_cos(yaw)
            print(f"yaw_to_sin_cos: {yaw:.4f} -> sin={sin_yaw:.4f}, cos={cos_yaw:.4f}")
            
            # Convert back to yaw
            yaw_reconstructed = sin_cos_to_yaw(sin_yaw, cos_yaw)
            print(f"sin_cos_to_yaw: sin={sin_yaw:.4f}, cos={cos_yaw:.4f} -> {yaw_reconstructed:.4f}")
            
            # Verify unit norm
            norm = np.sqrt(sin_yaw**2 + cos_yaw**2)
            self.assertAlmostEqual(norm, 1.0, places=10)
            print(f"✅ Unit norm: {norm:.10f}")
            
            # Verify round-trip (accounting for angle wrapping)
            yaw_diff = np.abs(yaw - yaw_reconstructed)
            yaw_diff_wrapped = min(yaw_diff, 2*np.pi - yaw_diff)
            self.assertLess(yaw_diff_wrapped, 1e-10)
            print(f"✅ Round-trip successful: diff = {yaw_diff_wrapped:.2e}")

    def test_step6_prediction_reconstruction(self):
        """Test Step 6: Prediction reconstruction to evaluation format."""
        print("\n🔍 Testing Step 6: Prediction Reconstruction")
        
        # Create mock predictions (normalized format)
        batch_size = 2
        max_detections = 100
        
        # Mock batch dict (adapted to expected format)
        batch_dict = {
            'sample_tokens': ['sample_001', 'sample_002'],
            'ego_translation_world': torch.tensor([[0.0, 0.0, 0.0], [10.0, 5.0, 0.0]], dtype=torch.float64),
            'ego_rotation_world_quat': torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9239, 0.0, 0.0, 0.3827]], dtype=torch.float64),  # 45° rotation
            'ego_motion': {
                'cabin': {
                    'velocity': torch.tensor([[20.0, 0.0, 0.0], [15.0, 5.0, 0.0]], dtype=torch.float64)  # (B, 3) format
                }
            }
        }
        
        # Mock predictions
        predictions = {}
        
        # Classification predictions (with high confidence for first few detections)
        class_logits = torch.full((batch_size, max_detections, len(self.cfg['dataset']['class_names']) + 1), -10.0)
        class_logits[:, 0, 0] = 5.0  # Car with high confidence
        class_logits[:, 1, 1] = 4.0  # Truck with medium confidence 
        predictions['pred_class_logits_batch'] = class_logits
        
        # Normalized box predictions (10D format)
        boxes_normalized = torch.zeros((batch_size, max_detections, 10))
        
        # Sample 1: Car at (20, 10, 0) in ego frame
        boxes_normalized[0, 0, :3] = torch.tensor([0.57, 0.53, 0.5])  # Normalized coordinates
        boxes_normalized[0, 0, 3:6] = torch.tensor([0.1, 0.2, 0.05])  # Normalized dimensions
        boxes_normalized[0, 0, 6:8] = torch.tensor([0.0, 1.0])  # sin(0), cos(0) = forward
        
        # Sample 2: Truck at (30, -5, 0) in ego frame  
        boxes_normalized[1, 1, :3] = torch.tensor([0.6, 0.47, 0.5])
        boxes_normalized[1, 1, 3:6] = torch.tensor([0.15, 0.3, 0.1])
        boxes_normalized[1, 1, 6:8] = torch.tensor([0.707, 0.707])  # sin(45°), cos(45°)
        
        predictions['pred_boxes_normalized'] = boxes_normalized
        
        # Normalized velocity predictions
        velocities_normalized = torch.zeros((batch_size, max_detections, 2))
        velocities_normalized[0, 0] = torch.tensor([0.1, -0.05])  # Normalized relative velocity
        velocities_normalized[1, 1] = torch.tensor([-0.2, 0.1])
        predictions['pred_velocities_normalized'] = velocities_normalized
        
        # Attribute predictions (using proper attribute vocab)
        attribute_vocab = [
            'pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing',
            'cycle.with_rider', 'cycle.without_rider', 'vehicle.moving', 'vehicle.parked',
            'traffic_sign.temporary'
        ]
        attr_logits = torch.full((batch_size, max_detections, len(attribute_vocab) + 1), -10.0)
        attr_logits[:, 0, 5] = 3.0  # vehicle.moving
        attr_logits[:, 1, 6] = 2.0  # vehicle.parked
        predictions['pred_attributes_logits_batch'] = attr_logits
        
        print(f"Mock predictions created:")
        print(f"  Batch size: {batch_size}")
        print(f"  Max detections: {max_detections}")
        print(f"  Sample tokens: {batch_dict['sample_tokens']}")
        
        # Test reconstruction
        try:
            results = reconstruct_and_convert_predictions_autoregressive(
                batch_dict=batch_dict,
                predictions=predictions,
                cfg=self.cfg
            )
            
            print(f"✅ Reconstruction successful, got {len(results)} samples")
            
            # Verify results structure
            for i, result in enumerate(results):
                sample_token = result['sample_token']
                sample_predictions = result['predictions']
                
                print(f"\nSample {i+1}: {sample_token}")
                print(f"  Number of predictions: {len(sample_predictions)}")
                
                for j, pred in enumerate(sample_predictions):
                    print(f"  Prediction {j+1}:")
                    print(f"    Translation (world): {pred['translation']} m")
                    print(f"    Size: {pred['size']} m") 
                    print(f"    Rotation: {pred['rotation']}")
                    print(f"    Velocity (world): {pred['velocity']} m/s")
                    print(f"    Detection name: {pred['detection_name']}")
                    print(f"    Detection score: {pred['detection_score']:.4f}")
                    print(f"    Attribute: {pred['attribute_name']}")
                    
                    # Verify required fields
                    required_fields = ['translation', 'size', 'rotation', 'velocity', 
                                     'detection_name', 'detection_score', 'attribute_name']
                    for field in required_fields:
                        self.assertIn(field, pred)
                    
                    # Verify data types and shapes
                    self.assertIsInstance(pred['translation'], list)
                    self.assertEqual(len(pred['translation']), 3)
                    self.assertIsInstance(pred['velocity'], list)
                    self.assertEqual(len(pred['velocity']), 2)
                    
                    # Check for NaN/Inf values
                    for val in pred['translation'] + pred['size'] + pred['rotation'] + pred['velocity']:
                        self.assertFalse(np.isnan(val), f"NaN found in prediction: {pred}")
                        self.assertFalse(np.isinf(val), f"Inf found in prediction: {pred}")
            
            print(f"✅ All predictions have valid structure and values")
            
        except Exception as e:
            self.fail(f"Reconstruction failed: {e}")

    def test_step7_devkit_format_validation(self):
        """Test Step 7: DevKit format validation."""
        print("\n🔍 Testing Step 7: DevKit Format Validation")
        
        # Create a sample prediction in DevKit format
        sample_prediction = {
            'sample_token': 'test_sample_001',
            'predictions': [
                {
                    'translation': [10.5, 5.2, 0.1],  # World coordinates
                    'size': [2.1, 4.8, 1.9],  # Width, length, height
                    'rotation': [0.9239, 0.0, 0.0, 0.3827],  # Quaternion [w, x, y, z]
                    'velocity': [25.3, -2.1],  # World velocity
                    'detection_name': 'car',
                    'detection_score': 0.8754,
                    'attribute_name': 'vehicle.moving'
                },
                {
                    'translation': [30.1, -8.7, 0.2],
                    'size': [2.8, 12.5, 3.2],
                    'rotation': [0.7071, 0.0, 0.0, 0.7071],  # 90° rotation
                    'velocity': [18.2, 5.5],
                    'detection_name': 'truck',
                    'detection_score': 0.6523,
                    'attribute_name': 'vehicle.moving'
                }
            ]
        }
        
        print(f"Validating DevKit format for sample: {sample_prediction['sample_token']}")
        
        # Validate structure
        self.assertIn('sample_token', sample_prediction)
        self.assertIn('predictions', sample_prediction)
        self.assertIsInstance(sample_prediction['predictions'], list)
        
        for i, pred in enumerate(sample_prediction['predictions']):
            print(f"\nValidating prediction {i+1}:")
            
            # Required fields
            required_fields = ['translation', 'size', 'rotation', 'velocity', 
                             'detection_name', 'detection_score', 'attribute_name']
            for field in required_fields:
                self.assertIn(field, pred)
                print(f"  ✅ {field}: {pred[field]}")
            
            # Validate data types and ranges
            self.assertIsInstance(pred['translation'], list)
            self.assertEqual(len(pred['translation']), 3)
            
            self.assertIsInstance(pred['size'], list)
            self.assertEqual(len(pred['size']), 3)
            self.assertTrue(all(s > 0 for s in pred['size']), "Size must be positive")
            
            self.assertIsInstance(pred['rotation'], list)
            self.assertEqual(len(pred['rotation']), 4)
            # Verify quaternion normalization
            quat_norm = np.sqrt(sum(q**2 for q in pred['rotation']))
            self.assertAlmostEqual(quat_norm, 1.0, places=3, msg="Quaternion must be normalized")
            
            self.assertIsInstance(pred['velocity'], list)
            self.assertEqual(len(pred['velocity']), 2)
            
            self.assertIsInstance(pred['detection_score'], (int, float))
            self.assertGreaterEqual(pred['detection_score'], 0.0)
            self.assertLessEqual(pred['detection_score'], 1.0)
            
            self.assertIn(pred['detection_name'], self.cfg['dataset']['class_names'])
            
            # Check for NaN/Inf values
            all_numeric_values = (pred['translation'] + pred['size'] + 
                                pred['rotation'] + pred['velocity'] + [pred['detection_score']])
            for val in all_numeric_values:
                self.assertFalse(np.isnan(val), f"NaN found: {val}")
                self.assertFalse(np.isinf(val), f"Inf found: {val}")
        
        print(f"✅ DevKit format validation successful")
        
        # Test JSON serialization (DevKit requirement)
        try:
            json_str = json.dumps(sample_prediction)
            reconstructed = json.loads(json_str)
            self.assertEqual(sample_prediction, reconstructed)
            print(f"✅ JSON serialization successful")
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")

    def test_step8_full_pipeline_integration(self):
        """Test Step 8: Full pipeline integration test."""
        print("\n🔍 Testing Step 8: Full Pipeline Integration")
        
        print("Testing complete pipeline: World -> Ego -> Augmentation -> Normalization -> Model -> Denormalization -> World")
        
        for name, vel_world_original in self.test_velocities_world.items():
            print(f"\n--- Full pipeline test for {name} ---")
            print(f"Step 1: Original world velocity: {vel_world_original} m/s")
            
            # Step 1: World to Ego transformation
            vel_ego_relative = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=vel_world_original,
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            print(f"Step 2: Ego relative velocity: {vel_ego_relative} m/s")
            
            # Step 2: Data augmentation (noise)
            sensor_cfg = self.cfg['dataset']['virtual_sensors'][0]
            _, vel_ego_noisy = add_noise_to_box(
                sensor_box_ego_phys_7d=self.test_box_ego,
                sensor_velocity_ego_phys_2d=vel_ego_relative,
                sensor_noise_cfg=sensor_cfg,
                sample_token=f"integration_test_{name}",
                global_seed=42
            )
            print(f"Step 3: Noisy ego velocity: {vel_ego_noisy} m/s")
            
            # Step 3: Normalization (dataset processing)
            vel_tensor = torch.from_numpy(vel_ego_noisy).double().unsqueeze(0)
            vel_normalized = self.normalizer.normalize_velocity_absolute(vel_tensor)
            print(f"Step 4: Normalized velocity: {vel_normalized.numpy()}")
            
            # Step 4: Simulate model processing (identity operation for testing)
            vel_model_output = vel_normalized.clone()
            print(f"Step 5: Model output (normalized): {vel_model_output.numpy()}")
            
            # Step 5: Denormalization (evaluation processing)
            vel_denormalized = self.normalizer.denormalize_velocity_absolute(vel_model_output)
            print(f"Step 6: Denormalized velocity: {vel_denormalized.numpy()} m/s")
            
            # Step 6: Ego to World transformation
            vel_world_final = transform_relative_velocity_ego_to_world_frame(
                relative_velocity_ego_2d=vel_denormalized.squeeze().numpy(),
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            print(f"Step 7: Final world velocity: {vel_world_final} m/s")
            
            # Analyze errors
            noise_error = np.linalg.norm(vel_ego_noisy - vel_ego_relative)
            normalization_error = np.linalg.norm(vel_denormalized.squeeze().numpy() - vel_ego_noisy)
            total_error = np.linalg.norm(vel_world_final - vel_world_original)
            
            print(f"\nError analysis:")
            print(f"  Noise error: {noise_error:.6f} m/s")
            print(f"  Normalization round-trip error: {normalization_error:.6f} m/s")
            print(f"  Total pipeline error: {total_error:.6f} m/s")
            
            # The total error should be dominated by the intentional noise
            # Normalization errors should be minimal
            self.assertLess(normalization_error, 1e-5, 
                          f"Normalization error too large: {normalization_error}")
            
            print(f"✅ Pipeline integration successful")

    def test_step9_radar_baseline_specific(self):
        """Test Step 9: Radar baseline specific checks."""
        print("\n🔍 Testing Step 9: Radar Baseline Specific Checks")
        
        # Radar should have excellent velocity estimation
        radar_cfg = {
            'name': 'virtual_radar',
            'velocity_noise_std': 0.078,  # Very low noise for radar
            'pos_noise_std': [0.85, 0.72, 1.40],  # High position noise
            'dim_noise_std': 0.58,  # High dimension noise
            'yaw_noise_std': 0.30,   # High yaw noise
            'dropout_rate': 0.18,
            'class_accuracy': 0.73,
            'attribute_accuracy': 0.65
        }
        
        print(f"Testing radar velocity noise (should be low): {radar_cfg['velocity_noise_std']} m/s")
        
        # Test multiple velocity scenarios for radar
        test_velocities = [
            np.array([0.0, 0.0], dtype=np.float64),    # Stationary
            np.array([10.0, 0.0], dtype=np.float64),   # Slow forward
            np.array([30.0, 0.0], dtype=np.float64),   # Fast forward
            np.array([20.0, 15.0], dtype=np.float64),  # Diagonal
            np.array([-10.0, 5.0], dtype=np.float64),  # Backward
        ]
        
        velocity_errors = []
        
        for i, vel_world in enumerate(test_velocities):
            print(f"\nTesting radar velocity #{i+1}: {vel_world} m/s")
            
            # Transform to ego frame
            vel_ego = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=vel_world,
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            
            # Apply radar noise multiple times to check consistency
            noise_samples = []
            for sample in range(10):
                _, vel_noisy = add_noise_to_box(
                    sensor_box_ego_phys_7d=self.test_box_ego,
                    sensor_velocity_ego_phys_2d=vel_ego,
                    sensor_noise_cfg=radar_cfg,
                    sample_token=f"radar_test_{i}_{sample}",
                    global_seed=42
                )
                noise_applied = vel_noisy - vel_ego
                noise_magnitude = np.linalg.norm(noise_applied)
                noise_samples.append(noise_magnitude)
            
            avg_noise = np.mean(noise_samples)
            std_noise = np.std(noise_samples)
            
            print(f"  Original ego velocity: {vel_ego} m/s")
            print(f"  Average noise magnitude: {avg_noise:.6f} m/s")
            print(f"  Noise std deviation: {std_noise:.6f} m/s")
            print(f"  Expected noise std: {radar_cfg['velocity_noise_std']} m/s")
            
            velocity_errors.append(avg_noise)
            
            # Radar should have low velocity noise
            self.assertLess(avg_noise, 0.5, 
                          f"Radar velocity noise too high: {avg_noise} > 0.5 m/s")
        
        print(f"\nRadar velocity error summary:")
        print(f"  Mean error: {np.mean(velocity_errors):.6f} m/s")
        print(f"  Max error: {np.max(velocity_errors):.6f} m/s")
        print(f"  Expected noise std: {radar_cfg['velocity_noise_std']} m/s")
        
        print(f"✅ Radar velocity characteristics verified")

    def test_step10_boundary_conditions(self):
        """Test Step 10: Boundary conditions and edge cases."""
        print("\n🔍 Testing Step 10: Boundary Conditions and Edge Cases")
        
        # Test extreme velocities
        extreme_velocities = [
            np.array([100.0, 0.0], dtype=np.float64),   # Very fast forward (360 km/h)
            np.array([0.0, 100.0], dtype=np.float64),   # Very fast sideways
            np.array([-50.0, -50.0], dtype=np.float64), # Fast diagonal backward
            np.array([0.001, 0.001], dtype=np.float64), # Very slow
            np.array([1e-10, 1e-10], dtype=np.float64), # Near zero
        ]
        
        for i, vel in enumerate(extreme_velocities):
            print(f"\nTesting extreme velocity #{i+1}: {vel} m/s")
            
            try:
                # Test transformations
                vel_ego = transform_velocity_world_to_ego_frame(
                    object_velocity_world_phys_2d=vel,
                    ego_rotation=self.ego_rotation,
                    ego_velocity_world_phys_2d=self.ego_velocity_world
                )
                
                # Test normalization
                vel_tensor = torch.from_numpy(vel_ego).float().unsqueeze(0)
                vel_norm = self.normalizer.normalize_velocity_absolute(vel_tensor)
                vel_denorm = self.normalizer.denormalize_velocity_absolute(vel_norm)
                
                # Check for numerical issues
                self.assertFalse(torch.any(torch.isnan(vel_norm)))
                self.assertFalse(torch.any(torch.isinf(vel_norm)))
                self.assertFalse(torch.any(torch.isnan(vel_denorm)))
                self.assertFalse(torch.any(torch.isinf(vel_denorm)))
                
                # Check round-trip accuracy
                error = torch.norm(vel_tensor - vel_denorm).item()
                print(f"  Normalization round-trip error: {error:.2e} m/s")
                
                self.assertLess(error, 1e-4, f"Round-trip error too large: {error}")
                
                print(f"  ✅ Extreme velocity handled correctly")
                
            except Exception as e:
                self.fail(f"Failed to handle extreme velocity {vel}: {e}")
        
        print(f"✅ All boundary conditions passed")


def main():
    """Run the comprehensive velocity pipeline test."""
    print("🚀 Starting Comprehensive Velocity Pipeline Test")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(VelocityPipelineTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED - Velocity pipeline is working correctly!")
    else:
        print("❌ TESTS FAILED - Issues found in velocity pipeline:")
        for failure in result.failures:
            print(f"  - {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"  - {error[0]}: {error[1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)