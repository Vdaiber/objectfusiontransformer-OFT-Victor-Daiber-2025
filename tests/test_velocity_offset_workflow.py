#!/usr/bin/env python3
"""
Test the REAL velocity offset workflow:
GT → Ego → Normalize → Anchor+Offset → Denormalize → Offset → Add to Anchor → Ego → World → Compare with GT

This tests the actual workflow used in training and evaluation.
"""

import unittest
import torch
import numpy as np
import yaml
import tempfile
import os
from typing import Dict, Any, List, Tuple
from pyquaternion import Quaternion as PyQuaternion
from pathlib import Path

# Import the modules we need to test
import sys
sys.path.append('/app/src')

from oft.transformer.datasets.truckscenes.transforms import (
    transform_velocity_world_to_ego_frame,
    transform_relative_velocity_ego_to_world_frame
)
from oft.transformer.utils.normalization_utils import CentralizedNormalizer
from oft.transformer.utils.geometry_utils import yaw_to_sin_cos, sin_cos_to_yaw


class VelocityOffsetWorkflowTest(unittest.TestCase):
    """Test the real velocity offset workflow used in training/evaluation."""

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
        
        # Test ego motion
        self.ego_velocity_world = np.array([20.0, 0.0], dtype=np.float64)  # Ego moving at 72 km/h
        self.ego_rotation = PyQuaternion(axis=[0, 0, 1], radians=0.0)  # No rotation initially
        self.ego_translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_complete_gt_to_offset_to_gt_workflow(self):
        """Test the complete GT → Offset → GT workflow."""
        print("\n🔍 Testing Complete GT → Offset → GT Workflow")
        
        # Test cases: realistic velocity scenarios
        test_cases = {
            'stationary_car': np.array([0.0, 0.0], dtype=np.float64),
            'moving_car_forward': np.array([25.0, 0.0], dtype=np.float64),  # 90 km/h forward
            'moving_car_diagonal': np.array([20.0, 10.0], dtype=np.float64),  # diagonal movement
            'fast_highway': np.array([35.0, -5.0], dtype=np.float64),  # 126 km/h with lane change
        }
        
        for name, gt_velocity_world in test_cases.items():
            print(f"\n--- Testing {name}: {gt_velocity_world} m/s (world) ---")
            
            # ==============================
            # STEP 1: GT TO EGO TRANSFORMATION
            # ==============================
            print("STEP 1: Transform GT velocity from World to Ego frame")
            gt_velocity_ego = transform_velocity_world_to_ego_frame(
                object_velocity_world_phys_2d=gt_velocity_world,
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            print(f"  GT World: {gt_velocity_world} m/s")
            print(f"  GT Ego:   {gt_velocity_ego} m/s")
            
            # ==============================
            # STEP 2: NORMALIZE GT VELOCITY (Dataset processing)
            # ==============================
            print("STEP 2: Normalize GT velocity for dataset")
            gt_velocity_ego_tensor = torch.from_numpy(gt_velocity_ego).double().unsqueeze(0)  # (1, 2)
            gt_velocity_normalized = self.normalizer.normalize_velocity_absolute(gt_velocity_ego_tensor)
            print(f"  GT Normalized: {gt_velocity_normalized.squeeze().numpy()}")
            
            # ==============================
            # STEP 3: CREATE ANCHOR BOX WITH NOISY VELOCITY (Sensor simulation)
            # ==============================
            print("STEP 3: Create anchor box with noisy velocity (sensor detection)")
            # Simulate sensor detection with some noise
            sensor_noise = np.random.normal(0, 0.1, 2)  # Small noise for anchor
            anchor_velocity_ego = gt_velocity_ego + sensor_noise
            anchor_velocity_ego_tensor = torch.from_numpy(anchor_velocity_ego).double().unsqueeze(0)
            anchor_velocity_normalized = self.normalizer.normalize_velocity_absolute(anchor_velocity_ego_tensor)
            print(f"  Anchor Ego:        {anchor_velocity_ego} m/s")
            print(f"  Anchor Normalized: {anchor_velocity_normalized.squeeze().numpy()}")
            
            # ==============================
            # STEP 4: CALCULATE VELOCITY OFFSET (Model target)
            # ==============================
            print("STEP 4: Calculate velocity offset (GT - Anchor)")
            velocity_offset_ego = gt_velocity_ego - anchor_velocity_ego
            velocity_offset_ego_tensor = torch.from_numpy(velocity_offset_ego).double().unsqueeze(0)
            print(f"  Velocity Offset Ego: {velocity_offset_ego} m/s")
            
            # ==============================
            # STEP 5: NORMALIZE VELOCITY OFFSET (Model output format)
            # ==============================
            print("STEP 5: Normalize velocity offset for model")
            velocity_offset_normalized = self.normalizer.normalize_velocity_offset(velocity_offset_ego_tensor)
            print(f"  Velocity Offset Normalized: {velocity_offset_normalized.squeeze().numpy()}")
            
            # ==============================
            # STEP 6: SIMULATE MODEL PREDICTION (Perfect prediction for testing)
            # ==============================
            print("STEP 6: Simulate model prediction (perfect prediction)")
            predicted_offset_normalized = velocity_offset_normalized.clone()  # Perfect prediction
            print(f"  Predicted Offset Normalized: {predicted_offset_normalized.squeeze().numpy()}")
            
            # ==============================
            # STEP 7: DENORMALIZE PREDICTED OFFSET (Evaluation processing)
            # ==============================
            print("STEP 7: Denormalize predicted offset")
            predicted_offset_ego_tensor = self.normalizer.denormalize_velocity_offset(predicted_offset_normalized)
            predicted_offset_ego = predicted_offset_ego_tensor.squeeze().numpy()
            print(f"  Predicted Offset Ego: {predicted_offset_ego} m/s")
            
            # ==============================
            # STEP 8: ADD OFFSET TO ANCHOR (Final velocity reconstruction)
            # ==============================
            print("STEP 8: Add offset to anchor to get final velocity")
            final_velocity_ego = anchor_velocity_ego + predicted_offset_ego
            print(f"  Anchor Ego:       {anchor_velocity_ego} m/s")
            print(f"  + Predicted Offset: {predicted_offset_ego} m/s")
            print(f"  = Final Ego:      {final_velocity_ego} m/s")
            
            # ==============================
            # STEP 9: TRANSFORM BACK TO WORLD (Evaluation format)
            # ==============================
            print("STEP 9: Transform final velocity back to world frame")
            final_velocity_world = transform_relative_velocity_ego_to_world_frame(
                relative_velocity_ego_2d=final_velocity_ego,
                ego_rotation=self.ego_rotation,
                ego_velocity_world_phys_2d=self.ego_velocity_world
            )
            print(f"  Final World: {final_velocity_world} m/s")
            
            # ==============================
            # STEP 10: COMPARE WITH ORIGINAL GT
            # ==============================
            print("STEP 10: Compare with original GT")
            total_error = np.linalg.norm(final_velocity_world - gt_velocity_world)
            offset_roundtrip_error = np.linalg.norm(predicted_offset_ego - velocity_offset_ego)
            anchor_noise_magnitude = np.linalg.norm(sensor_noise)
            
            print(f"  Original GT World:     {gt_velocity_world} m/s")
            print(f"  Final Reconstructed:   {final_velocity_world} m/s")
            print(f"  Total Error:           {total_error:.8f} m/s")
            print(f"  Offset Round-trip Error: {offset_roundtrip_error:.8f} m/s")
            print(f"  Anchor Noise Magnitude:  {anchor_noise_magnitude:.8f} m/s")
            
            # ==============================
            # ASSERTIONS: Check workflow accuracy
            # ==============================
            
            # The offset round-trip error should be minimal (normalization precision)
            self.assertLess(offset_roundtrip_error, 1e-6, 
                          f"Offset round-trip error too large: {offset_roundtrip_error}")
            print(f"  ✅ Offset round-trip accuracy verified")
            
            # The total error should be minimal - the offset system compensates for anchor noise!
            # This is the POINT of the offset system - to correct anchor inaccuracies
            self.assertLess(total_error, 1e-10,
                          f"Total error too large: {total_error} (should be ~0 due to offset correction)")
            print(f"  ✅ Pipeline perfectly corrects anchor noise (total error: {total_error:.2e} m/s)")
            
            print(f"  🎉 Complete workflow successful for {name}")

    def test_offset_normalization_edge_cases(self):
        """Test velocity offset normalization with edge cases."""
        print("\n🔍 Testing Velocity Offset Normalization Edge Cases")
        
        # Test various offset magnitudes
        test_offsets = [
            np.array([0.0, 0.0], dtype=np.float64),      # Zero offset
            np.array([0.1, -0.1], dtype=np.float64),     # Small offset
            np.array([1.0, 2.0], dtype=np.float64),      # Medium offset
            np.array([2.9, -2.9], dtype=np.float64),     # Maximum offset (at percentile)
            np.array([-2.9, 2.9], dtype=np.float64),     # Minimum offset (at percentile)
            np.array([3.5, -3.5], dtype=np.float64),     # Beyond percentile (clipping test)
        ]
        
        for i, offset in enumerate(test_offsets):
            print(f"\nTesting offset #{i+1}: {offset} m/s")
            
            # Convert to tensor
            offset_tensor = torch.from_numpy(offset).double().unsqueeze(0)
            
            # Normalize
            offset_normalized = self.normalizer.normalize_velocity_offset(offset_tensor)
            print(f"  Normalized: {offset_normalized.squeeze().numpy()}")
            
            # Denormalize
            offset_denormalized = self.normalizer.denormalize_velocity_offset(offset_normalized)
            print(f"  Denormalized: {offset_denormalized.squeeze().numpy()} m/s")
            
            # Check round-trip accuracy
            roundtrip_error = torch.norm(offset_tensor - offset_denormalized).item()
            print(f"  Round-trip error: {roundtrip_error:.2e} m/s")
            
            # For values within the percentile range, accuracy should be perfect
            if np.all(np.abs(offset) <= 2.9):
                self.assertLess(roundtrip_error, 1e-10, 
                              f"Round-trip error too large for in-range offset: {roundtrip_error}")
                print(f"  ✅ Perfect accuracy for in-range offset")
            else:
                # For values outside range, some error is expected due to clipping
                print(f"  ⚠️  Offset beyond training range - some error expected")

    def test_velocity_offset_vs_absolute_velocity(self):
        """Test difference between offset and absolute velocity normalization."""
        print("\n🔍 Testing Velocity Offset vs Absolute Velocity Normalization")
        
        # Test velocity values
        test_velocity = np.array([10.0, -5.0], dtype=np.float64)
        velocity_tensor = torch.from_numpy(test_velocity).double().unsqueeze(0)
        
        print(f"Test velocity: {test_velocity} m/s")
        
        # Normalize as absolute velocity (dataset processing)
        vel_norm_absolute = self.normalizer.normalize_velocity_absolute(velocity_tensor)
        vel_denorm_absolute = self.normalizer.denormalize_velocity_absolute(vel_norm_absolute)
        
        print(f"Absolute velocity normalization:")
        print(f"  Normalized: {vel_norm_absolute.squeeze().numpy()}")
        print(f"  Denormalized: {vel_denorm_absolute.squeeze().numpy()} m/s")
        
        # Normalize as velocity offset (model processing)
        vel_norm_offset = self.normalizer.normalize_velocity_offset(velocity_tensor)
        vel_denorm_offset = self.normalizer.denormalize_velocity_offset(vel_norm_offset)
        
        print(f"Velocity offset normalization:")
        print(f"  Normalized: {vel_norm_offset.squeeze().numpy()}")
        print(f"  Denormalized: {vel_denorm_offset.squeeze().numpy()} m/s")
        
        # Compare normalization ranges
        print(f"Normalization comparison:")
        print(f"  Absolute range difference: {(vel_norm_absolute - vel_norm_offset).squeeze().numpy()}")
        
        # Both should give perfect round-trip accuracy
        abs_error = torch.norm(velocity_tensor - vel_denorm_absolute).item()
        offset_error = torch.norm(velocity_tensor - vel_denorm_offset).item()
        
        print(f"Round-trip errors:")
        print(f"  Absolute velocity: {abs_error:.2e} m/s")
        print(f"  Velocity offset:   {offset_error:.2e} m/s")
        
        self.assertLess(abs_error, 1e-10, f"Absolute velocity round-trip error: {abs_error}")
        self.assertLess(offset_error, 1e-10, f"Offset velocity round-trip error: {offset_error}")
        
        print(f"✅ Both normalization methods have perfect accuracy")


def main():
    """Run the velocity offset workflow test."""
    print("🚀 Starting Velocity Offset Workflow Test")
    print("=" * 60)
    print("Testing the REAL GT → Offset → GT workflow used in training/evaluation")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(VelocityOffsetWorkflowTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL OFFSET WORKFLOW TESTS PASSED!")
        print("✅ The velocity offset pipeline is working correctly")
        print("✅ GT → Ego → Normalize → Offset → Denormalize → Ego → World workflow verified")
    else:
        print("❌ OFFSET WORKFLOW TESTS FAILED:")
        for failure in result.failures:
            print(f"  - {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"  - {error[0]}: {error[1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)