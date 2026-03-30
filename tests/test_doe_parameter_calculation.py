#!/usr/bin/env python3
"""
Unit tests for DoE parameter calculation functionality.

Tests the calculate_final_parameters function with various multiplier combinations
to ensure correct scaling of all 23 DOE parameters including odd scaling for
probability parameters.
"""

import unittest
import sys
from pathlib import Path
import copy

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.oft.transformer.scripts.preprocess_doe_plan import calculate_final_parameters


class TestDOEParameterCalculation(unittest.TestCase):
    """Test cases for DOE parameter calculation with all parameter types."""
    
    def setUp(self):
        """Set up base configuration for testing."""
        self.base_config = {
            'dataset': {
                'virtual_sensors': [
                    {
                        'name': 'virtual_lidar',
                        'pos_noise_std': [0.46, 0.46, 0.29],
                        'dim_noise_std': 0.27,
                        'yaw_noise_std': 0.22,
                        'velocity_noise_std': 0.16,
                        'dropout_rate': 0.24,
                        'class_accuracy': 0.78,
                        'attribute_accuracy': 0.68
                    },
                    {
                        'name': 'virtual_camera',
                        'pos_noise_std': [0.94, 0.83, 1.27],
                        'dim_noise_std': 0.53,
                        'yaw_noise_std': 0.29,
                        'velocity_noise_std': 0.08,
                        'dropout_rate': 0.38,
                        'class_accuracy': 0.56,
                        'attribute_accuracy': 0.60
                    },
                    {
                        'name': 'virtual_radar',
                        'pos_noise_std': [1.20, 1.10, 2.00],
                        'dim_noise_std': 0.75,
                        'yaw_noise_std': 0.40,
                        'velocity_noise_std': 0.012,
                        'dropout_rate': 0.25,
                        'class_accuracy': 0.64,
                        'attribute_accuracy': 0.70
                    }
                ],
                'simulation': {
                    'num_fps': 0
                }
            }
        }
    
    def test_baseline_unchanged(self):
        """Test that multiplier of 1.0 leaves parameters unchanged."""
        multipliers = {
            'lidar_pos_noise_multiplier': 1.0,
            'lidar_dim_noise_multiplier': 1.0,
            'lidar_dropout_odds_multiplier': 1.0,
            'false_positive_count': 1.0
        }
        
        result = calculate_final_parameters(self.base_config, multipliers)
        
        # Check that LiDAR parameters are unchanged
        lidar_sensor = result['dataset']['virtual_sensors'][0]
        self.assertEqual(lidar_sensor['pos_noise_std'], [0.46, 0.46, 0.29])
        self.assertEqual(lidar_sensor['dim_noise_std'], 0.27)
        self.assertAlmostEqual(lidar_sensor['dropout_rate'], 0.24, places=6)
        
        # Check that simulation parameters are unchanged
        self.assertEqual(result['dataset']['simulation']['num_fps'], 0)
    
    def test_additive_parameters_scaling(self):
        """Test scaling of additive parameters (pos, dim, yaw, velocity noise)."""
        multipliers = {
            'lidar_pos_noise_multiplier': 2.0,
            'camera_dim_noise_multiplier': 0.5,
            'radar_yaw_noise_multiplier': 3.0,
            'lidar_velocity_noise_multiplier': 1.5
        }
        
        result = calculate_final_parameters(self.base_config, multipliers)
        
        # Check LiDAR position noise doubled
        lidar_sensor = result['dataset']['virtual_sensors'][0]
        expected_pos = [x * 2.0 for x in self.base_config['dataset']['virtual_sensors'][0]['pos_noise_std']]
        self.assertEqual(lidar_sensor['pos_noise_std'], expected_pos)
        
        # Check camera dimension noise halved
        camera_sensor = result['dataset']['virtual_sensors'][1]
        expected_dim = self.base_config['dataset']['virtual_sensors'][1]['dim_noise_std'] * 0.5
        self.assertEqual(camera_sensor['dim_noise_std'], expected_dim)
        
        # Check radar yaw noise tripled
        radar_sensor = result['dataset']['virtual_sensors'][2]
        expected_yaw = self.base_config['dataset']['virtual_sensors'][2]['yaw_noise_std'] * 3.0
        self.assertEqual(radar_sensor['yaw_noise_std'], expected_yaw)
        
        # Check LiDAR velocity noise scaled by 1.5
        expected_vel = self.base_config['dataset']['virtual_sensors'][0]['velocity_noise_std'] * 1.5
        self.assertEqual(lidar_sensor['velocity_noise_std'], expected_vel)
    
    def test_odd_scaling_dropout_rate(self):
        """Test odd scaling for dropout rate parameters."""
        multipliers = {
            'lidar_dropout_odds_multiplier': 2.0,  # Should increase dropout rate
            'camera_dropout_odds_multiplier': 0.5  # Should decrease dropout rate
        }
        
        result = calculate_final_parameters(self.base_config, multipliers)
        
        # Test LiDAR dropout rate with multiplier 2.0
        lidar_sensor = result['dataset']['virtual_sensors'][0]
        baseline_lidar_dropout = 0.24
        baseline_odds = baseline_lidar_dropout / (1 - baseline_lidar_dropout)  # 0.316
        new_odds = baseline_odds * 2.0  # 0.632
        expected_dropout = new_odds / (1 + new_odds)  # 0.387
        self.assertAlmostEqual(lidar_sensor['dropout_rate'], expected_dropout, places=3)
        
        # Test camera dropout rate with multiplier 0.5
        camera_sensor = result['dataset']['virtual_sensors'][1]
        baseline_camera_dropout = 0.38
        baseline_odds = baseline_camera_dropout / (1 - baseline_camera_dropout)  # 0.613
        new_odds = baseline_odds * 0.5  # 0.306
        expected_dropout = new_odds / (1 + new_odds)  # 0.234
        self.assertAlmostEqual(camera_sensor['dropout_rate'], expected_dropout, places=3)
    
    def test_odd_scaling_accuracy_parameters(self):
        """Test odd scaling for accuracy parameters."""
        multipliers = {
            'radar_class_accuracy_odds_multiplier': 2.0,  # Should increase accuracy
            'lidar_attribute_accuracy_odds_multiplier': 0.5  # Should decrease accuracy
        }
        
        result = calculate_final_parameters(self.base_config, multipliers)
        
        # Test radar class accuracy with multiplier 2.0
        radar_sensor = result['dataset']['virtual_sensors'][2]
        baseline_radar_accuracy = 0.64
        baseline_odds = baseline_radar_accuracy / (1 - baseline_radar_accuracy)  # 1.778
        new_odds = baseline_odds * 2.0  # 3.556
        expected_accuracy = new_odds / (1 + new_odds)  # 0.781
        self.assertAlmostEqual(radar_sensor['class_accuracy'], expected_accuracy, places=3)
        
        # Test LiDAR attribute accuracy with multiplier 0.5
        lidar_sensor = result['dataset']['virtual_sensors'][0]
        baseline_lidar_attr = 0.68
        baseline_odds = baseline_lidar_attr / (1 - baseline_lidar_attr)  # 2.125
        new_odds = baseline_odds * 0.5  # 1.063
        expected_accuracy = new_odds / (1 + new_odds)  # 0.515
        self.assertAlmostEqual(lidar_sensor['attribute_accuracy'], expected_accuracy, places=3)
    
    def test_false_positive_count_scaling(self):
        """Test scaling of false positive count (discrete parameter)."""
        # Set base false positive count to 2
        config_with_fps = copy.deepcopy(self.base_config)
        config_with_fps['dataset']['simulation']['num_fps'] = 2
        
        multipliers = {
            'false_positive_count': 2.5  # Should give 2 * 2.5 = 5 (integer)
        }
        
        result = calculate_final_parameters(config_with_fps, multipliers)
        
        # Check false positive count is correctly scaled and converted to integer
        self.assertEqual(result['dataset']['simulation']['num_fps'], 5)
    
    def test_edge_cases_odd_scaling(self):
        """Test edge cases for odd scaling (extreme probabilities)."""
        # Create config with extreme probability values
        extreme_config = copy.deepcopy(self.base_config)
        extreme_config['dataset']['virtual_sensors'][0]['dropout_rate'] = 0.01  # Very low
        extreme_config['dataset']['virtual_sensors'][1]['class_accuracy'] = 0.99  # Very high
        
        multipliers = {
            'lidar_dropout_odds_multiplier': 10.0,  # Should still produce valid probability
            'camera_class_accuracy_odds_multiplier': 0.1  # Should still produce valid probability
        }
        
        result = calculate_final_parameters(extreme_config, multipliers)
        
        # Check that results are still valid probabilities [0, 1]
        lidar_dropout = result['dataset']['virtual_sensors'][0]['dropout_rate']
        camera_accuracy = result['dataset']['virtual_sensors'][1]['class_accuracy']
        
        self.assertGreaterEqual(lidar_dropout, 0.0)
        self.assertLessEqual(lidar_dropout, 1.0)
        self.assertGreaterEqual(camera_accuracy, 0.0)
        self.assertLessEqual(camera_accuracy, 1.0)
        
        # LiDAR dropout should increase significantly from 0.01
        self.assertGreater(lidar_dropout, 0.05)
        
        # Camera accuracy should decrease significantly from 0.99
        self.assertLess(camera_accuracy, 0.95)
    
    def test_multiple_parameters_simultaneously(self):
        """Test that multiple parameters can be scaled simultaneously without interference."""
        multipliers = {
            'lidar_pos_noise_multiplier': 1.5,
            'lidar_dim_noise_multiplier': 2.0,
            'lidar_dropout_odds_multiplier': 0.8,
            'camera_velocity_noise_multiplier': 3.0,
            'radar_class_accuracy_odds_multiplier': 1.2,
            'false_positive_count': 4.0
        }
        
        # Set base false positive count
        config_with_fps = copy.deepcopy(self.base_config)
        config_with_fps['dataset']['simulation']['num_fps'] = 1
        
        result = calculate_final_parameters(config_with_fps, multipliers)
        
        # Check that all parameters were correctly modified
        lidar_sensor = result['dataset']['virtual_sensors'][0]
        camera_sensor = result['dataset']['virtual_sensors'][1]
        radar_sensor = result['dataset']['virtual_sensors'][2]
        
        # LiDAR parameters
        expected_pos = [x * 1.5 for x in [0.46, 0.46, 0.29]]
        for i, expected in enumerate(expected_pos):
            self.assertAlmostEqual(lidar_sensor['pos_noise_std'][i], expected, places=6)
        self.assertEqual(lidar_sensor['dim_noise_std'], 0.54)  # 2.0x
        self.assertLess(lidar_sensor['dropout_rate'], 0.24)  # Should decrease with 0.8x odds
        
        # Camera velocity
        self.assertEqual(camera_sensor['velocity_noise_std'], 0.24)  # 3.0x
        
        # Radar accuracy should increase with 1.2x odds
        self.assertGreater(radar_sensor['class_accuracy'], 0.64)
        
        # False positive count
        self.assertEqual(result['dataset']['simulation']['num_fps'], 4)
    
    def test_config_immutability(self):
        """Test that original config is not modified (deep copy behavior)."""
        original_config = copy.deepcopy(self.base_config)
        
        multipliers = {
            'lidar_pos_noise_multiplier': 5.0,
            'camera_dropout_odds_multiplier': 10.0
        }
        
        result = calculate_final_parameters(self.base_config, multipliers)
        
        # Check that original config is unchanged
        self.assertEqual(self.base_config, original_config)
        
        # Check that result is different
        self.assertNotEqual(result['dataset']['virtual_sensors'][0]['pos_noise_std'],
                          original_config['dataset']['virtual_sensors'][0]['pos_noise_std'])


def run_tests():
    """Run all tests and return True if successful."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
