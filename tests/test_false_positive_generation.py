#!/usr/bin/env python3
"""
Unit tests for false-positive generation functionality.

This module tests the false-positive generation system that creates synthetic
false positive detections for training robustness. Includes integration tests
with the preprocessing pipeline and comprehensive validation of generated data.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import yaml
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the functions to test
from src.oft.transformer.datasets.preprocessing.data_augmentation import simulate_false_positives
from src.oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from src.oft.transformer.utils.reproducibility import set_seed


class TestFalsePositiveGeneration:
    """Test false-positive generation functionality."""
    
    @pytest.fixture
    def sample_fp_config(self):
        """Sample false positive simulation configuration."""
        return {
            'num_fps': 3,
            'fp_pos_range': {
                'x': [-50, 50],
                'y': [-30, 30], 
                'z': [-2, 2]
            },
            'fp_dim_range': {
                'w': [1.0, 3.0],
                'l': [2.0, 5.0],
                'h': [1.0, 3.0]
            },
            'fp_velocity_range': {
                'vx': [-8, 8],
                'vy': [-4, 4]
            }
        }
    
    @pytest.fixture
    def rng_fixture(self):
        """Fixed random number generator for reproducible tests."""
        return np.random.default_rng(42)
    
    def test_false_positive_generation_training_enabled(self, sample_fp_config, rng_fixture):
        """Test that false positives are generated during training mode."""
        is_training = True
        fps = simulate_false_positives(is_training, sample_fp_config, rng_fixture)
        
        # Should generate the specified number of false positives
        assert len(fps) == sample_fp_config['num_fps']
        
        # Each false positive should have the correct structure
        for fp in fps:
            assert 'box_7d_world' in fp
            assert 'velocity_world_2d' in fp
            assert 'class_idx' in fp
            assert 'attribute_idx' in fp
            
            # Check data types and shapes
            assert isinstance(fp['box_7d_world'], np.ndarray)
            assert isinstance(fp['velocity_world_2d'], np.ndarray)
            assert fp['box_7d_world'].shape == (7,)
            assert fp['velocity_world_2d'].shape == (2,)
            assert fp['box_7d_world'].dtype == np.float64
            assert fp['velocity_world_2d'].dtype == np.float64
            
            # Check false positive markers
            assert fp['class_idx'] == -1
            assert fp['attribute_idx'] == -1
    
    def test_false_positive_generation_training_disabled(self, sample_fp_config, rng_fixture):
        """Test that no false positives are generated when training is disabled."""
        is_training = False
        fps = simulate_false_positives(is_training, sample_fp_config, rng_fixture)
        
        # Should return empty list when not training
        assert fps == []
    
    def test_false_positive_parameter_ranges(self, sample_fp_config, rng_fixture):
        """Test that generated false positives respect parameter ranges."""
        is_training = True
        fps = simulate_false_positives(is_training, sample_fp_config, rng_fixture)
        
        for fp in fps:
            box = fp['box_7d_world']
            velocity = fp['velocity_world_2d']
            
            # Check position ranges
            assert sample_fp_config['fp_pos_range']['x'][0] <= box[0] <= sample_fp_config['fp_pos_range']['x'][1]
            assert sample_fp_config['fp_pos_range']['y'][0] <= box[1] <= sample_fp_config['fp_pos_range']['y'][1]
            assert sample_fp_config['fp_pos_range']['z'][0] <= box[2] <= sample_fp_config['fp_pos_range']['z'][1]
            
            # Check dimension ranges
            assert sample_fp_config['fp_dim_range']['w'][0] <= box[3] <= sample_fp_config['fp_dim_range']['w'][1]
            assert sample_fp_config['fp_dim_range']['l'][0] <= box[4] <= sample_fp_config['fp_dim_range']['l'][1]
            assert sample_fp_config['fp_dim_range']['h'][0] <= box[5] <= sample_fp_config['fp_dim_range']['h'][1]
            
            # Check yaw range (should be in [-π, π])
            assert -np.pi <= box[6] <= np.pi
            
            # Check velocity ranges
            assert sample_fp_config['fp_velocity_range']['vx'][0] <= velocity[0] <= sample_fp_config['fp_velocity_range']['vx'][1]
            assert sample_fp_config['fp_velocity_range']['vy'][0] <= velocity[1] <= sample_fp_config['fp_velocity_range']['vy'][1]
    
    def test_deterministic_behavior_with_seed(self, sample_fp_config):
        """Test that false positive generation is deterministic with the same seed."""
        is_training = True
        
        # Generate false positives with same seed twice
        rng1 = np.random.default_rng(123)
        fps1 = simulate_false_positives(is_training, sample_fp_config, rng1)
        
        rng2 = np.random.default_rng(123)
        fps2 = simulate_false_positives(is_training, sample_fp_config, rng2)
        
        # Results should be identical
        assert len(fps1) == len(fps2)
        for fp1, fp2 in zip(fps1, fps2):
            np.testing.assert_array_equal(fp1['box_7d_world'], fp2['box_7d_world'])
            np.testing.assert_array_equal(fp1['velocity_world_2d'], fp2['velocity_world_2d'])
            assert fp1['class_idx'] == fp2['class_idx']
            assert fp1['attribute_idx'] == fp2['attribute_idx']
    
    def test_different_seeds_produce_different_results(self, sample_fp_config):
        """Test that different seeds produce different false positives."""
        is_training = True
        
        # Generate false positives with different seeds
        rng1 = np.random.default_rng(111)
        fps1 = simulate_false_positives(is_training, sample_fp_config, rng1)
        
        rng2 = np.random.default_rng(222)
        fps2 = simulate_false_positives(is_training, sample_fp_config, rng2)
        
        # Results should be different (at least one difference expected)
        assert len(fps1) == len(fps2)
        differences_found = False
        for fp1, fp2 in zip(fps1, fps2):
            if not np.array_equal(fp1['box_7d_world'], fp2['box_7d_world']):
                differences_found = True
                break
        
        assert differences_found, "Different seeds should produce different results"
    
    def test_zero_false_positives_config(self, rng_fixture):
        """Test configuration with zero false positives."""
        config = {
            'num_fps': 0,
            'fp_pos_range': {'x': [-50, 50], 'y': [-30, 30], 'z': [-2, 2]},
            'fp_dim_range': {'w': [1.0, 3.0], 'l': [2.0, 5.0], 'h': [1.0, 3.0]},
            'fp_velocity_range': {'vx': [-8, 8], 'vy': [-4, 4]}
        }
        
        is_training = True
        fps = simulate_false_positives(is_training, config, rng_fixture)
        
        assert fps == []
    
    def test_large_number_false_positives(self, rng_fixture):
        """Test configuration with large number of false positives."""
        config = {
            'num_fps': 50,
            'fp_pos_range': {'x': [-100, 100], 'y': [-50, 50], 'z': [-3, 3]},
            'fp_dim_range': {'w': [0.5, 4.0], 'l': [1.0, 8.0], 'h': [0.5, 4.0]},
            'fp_velocity_range': {'vx': [-15, 15], 'vy': [-10, 10]}
        }
        
        is_training = True
        fps = simulate_false_positives(is_training, config, rng_fixture)
        
        assert len(fps) == 50
        # All should still have proper structure and ranges
        for fp in fps:
            assert fp['class_idx'] == -1
            assert fp['attribute_idx'] == -1
            assert fp['box_7d_world'].shape == (7,)
            assert fp['velocity_world_2d'].shape == (2,)


class TestFalsePositiveDataStructure:
    """Test the data structure and format of generated false positives."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for structure testing."""
        return {
            'num_fps': 1,
            'fp_pos_range': {'x': [0, 10], 'y': [0, 10], 'z': [0, 2]},
            'fp_dim_range': {'w': [1, 2], 'l': [2, 3], 'h': [1, 2]},
            'fp_velocity_range': {'vx': [-5, 5], 'vy': [-3, 3]}
        }
    
    def test_false_positive_data_types(self, basic_config):
        """Test that false positives have correct data types."""
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, basic_config, rng)
        
        fp = fps[0]
        
        # Test box_7d_world data type and values
        assert isinstance(fp['box_7d_world'], np.ndarray)
        assert fp['box_7d_world'].dtype == np.float64
        assert all(isinstance(x, (float, np.floating)) for x in fp['box_7d_world'])
        
        # Test velocity_world_2d data type and values
        assert isinstance(fp['velocity_world_2d'], np.ndarray)
        assert fp['velocity_world_2d'].dtype == np.float64
        assert all(isinstance(x, (float, np.floating)) for x in fp['velocity_world_2d'])
        
        # Test integer indices
        assert isinstance(fp['class_idx'], (int, np.integer))
        assert isinstance(fp['attribute_idx'], (int, np.integer))
    
    def test_false_positive_box_structure(self, basic_config):
        """Test that box_7d_world follows the expected [x, y, z, w, l, h, yaw] format."""
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, basic_config, rng)
        
        fp = fps[0]
        box = fp['box_7d_world']
        
        # Should have exactly 7 elements
        assert len(box) == 7
        
        # Elements should correspond to [x, y, z, w, l, h, yaw]
        x, y, z, w, l, h, yaw = box
        
        # Position (x, y, z) should be within specified ranges
        assert 0 <= x <= 10
        assert 0 <= y <= 10
        assert 0 <= z <= 2
        
        # Dimensions (w, l, h) should be positive and within ranges
        assert 1 <= w <= 2
        assert 2 <= l <= 3
        assert 1 <= h <= 2
        
        # Yaw should be in [-π, π]
        assert -np.pi <= yaw <= np.pi
    
    def test_false_positive_velocity_structure(self, basic_config):
        """Test that velocity_world_2d follows the expected [vx, vy] format."""
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, basic_config, rng)
        
        fp = fps[0]
        velocity = fp['velocity_world_2d']
        
        # Should have exactly 2 elements
        assert len(velocity) == 2
        
        # Elements should correspond to [vx, vy]
        vx, vy = velocity
        
        # Velocity components should be within specified ranges
        assert -5 <= vx <= 5
        assert -3 <= vy <= 3
    
    def test_false_positive_markers(self, basic_config):
        """Test that false positive markers are set correctly."""
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, basic_config, rng)
        
        for fp in fps:
            # False positives should have class_idx=-1 and attribute_idx=-1
            assert fp['class_idx'] == -1
            assert fp['attribute_idx'] == -1


class TestFalsePositiveConfigurationValidation:
    """Test various configuration scenarios and edge cases."""
    
    def test_minimal_ranges(self):
        """Test configuration with minimal ranges."""
        config = {
            'num_fps': 2,
            'fp_pos_range': {'x': [0, 0.1], 'y': [0, 0.1], 'z': [0, 0.1]},
            'fp_dim_range': {'w': [0.1, 0.2], 'l': [0.1, 0.2], 'h': [0.1, 0.2]},
            'fp_velocity_range': {'vx': [-0.1, 0.1], 'vy': [-0.1, 0.1]}
        }
        
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, config, rng)
        
        assert len(fps) == 2
        for fp in fps:
            # All values should be within the tight ranges
            box = fp['box_7d_world']
            velocity = fp['velocity_world_2d']
            
            assert 0 <= box[0] <= 0.1
            assert 0 <= box[1] <= 0.1
            assert 0 <= box[2] <= 0.1
            assert 0.1 <= box[3] <= 0.2
            assert 0.1 <= box[4] <= 0.2
            assert 0.1 <= box[5] <= 0.2
            assert -0.1 <= velocity[0] <= 0.1
            assert -0.1 <= velocity[1] <= 0.1
    
    def test_negative_position_ranges(self):
        """Test configuration with negative position ranges."""
        config = {
            'num_fps': 1,
            'fp_pos_range': {'x': [-100, -50], 'y': [-30, -10], 'z': [-5, -1]},
            'fp_dim_range': {'w': [1, 2], 'l': [2, 3], 'h': [1, 2]},
            'fp_velocity_range': {'vx': [-10, -5], 'vy': [5, 10]}
        }
        
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, config, rng)
        
        fp = fps[0]
        box = fp['box_7d_world']
        velocity = fp['velocity_world_2d']
        
        # All values should be within the negative ranges
        assert -100 <= box[0] <= -50
        assert -30 <= box[1] <= -10
        assert -5 <= box[2] <= -1
        assert -10 <= velocity[0] <= -5
        assert 5 <= velocity[1] <= 10


class TestFalsePositivePreprocessingIntegration:
    """Test integration with the preprocessing pipeline."""
    
    @pytest.fixture
    def sample_pipeline_config(self):
        """Sample pipeline configuration for integration testing."""
        return {
            'dataset': {
                'dataroot': '/tmp',  # Will be mocked
                'version': 'v1.0-mini',
                'class_names': ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 
                               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 
                               'barrier', 'animal', 'traffic_sign'],
                'simulation': {
                    'fn_rate': 0.0,
                    'num_fps': 5,
                    'fp_pos_range': {
                        'x': [-50, 50],
                        'y': [-25, 25], 
                        'z': [-2, 2]
                    },
                    'fp_dim_range': {
                        'w': [0.5, 3.5],
                        'l': [1.0, 6.0],
                        'h': [0.5, 3.5]
                    },
                    'fp_velocity_range': {
                        'vx': [-10, 10],
                        'vy': [-5, 5]
                    }
                }
            },
            'preprocessing': {
                'num_workers': 1,
                'cache_dir': '/tmp'
            }
        }
    
    def test_false_positive_integration_structure(self, sample_pipeline_config):
        """Test that false positives integrate properly with the pipeline structure."""
        simulation_cfg = sample_pipeline_config['dataset']['simulation']
        
        # Test that the configuration structure matches what the function expects
        required_keys = ['num_fps', 'fp_pos_range', 'fp_dim_range', 'fp_velocity_range']
        for key in required_keys:
            assert key in simulation_cfg
        
        # Test that sub-ranges have required keys
        assert all(k in simulation_cfg['fp_pos_range'] for k in ['x', 'y', 'z'])
        assert all(k in simulation_cfg['fp_dim_range'] for k in ['w', 'l', 'h'])
        assert all(k in simulation_cfg['fp_velocity_range'] for k in ['vx', 'vy'])
    
    def test_false_positive_generation_with_pipeline_config(self, sample_pipeline_config):
        """Test false positive generation using pipeline configuration format."""
        simulation_cfg = sample_pipeline_config['dataset']['simulation']
        rng = np.random.default_rng(42)
        
        fps = simulate_false_positives(True, simulation_cfg, rng)
        
        assert len(fps) == simulation_cfg['num_fps']
        
        # Verify generated false positives match pipeline expectations
        for fp in fps:
            # Should have the structure expected by the dataset
            assert 'box_7d_world' in fp
            assert 'velocity_world_2d' in fp
            assert 'class_idx' in fp
            assert 'attribute_idx' in fp
            
            # False positive markers should be set
            assert fp['class_idx'] == -1
            assert fp['attribute_idx'] == -1
    
    def test_false_positive_data_format_compatibility(self, sample_pipeline_config):
        """Test that false positive data format is compatible with GT data format."""
        simulation_cfg = sample_pipeline_config['dataset']['simulation']
        rng = np.random.default_rng(42)
        
        fps = simulate_false_positives(True, simulation_cfg, rng)
        
        # Each false positive should have the same structure as ground truth detections
        for fp in fps:
            # Test box format (7D: x, y, z, w, l, h, yaw)
            box = fp['box_7d_world']
            assert isinstance(box, np.ndarray)
            assert box.shape == (7,)
            assert box.dtype == np.float64
            
            # Test velocity format (2D: vx, vy)
            velocity = fp['velocity_world_2d']
            assert isinstance(velocity, np.ndarray)
            assert velocity.shape == (2,)
            assert velocity.dtype == np.float64
            
            # Test indices format
            assert isinstance(fp['class_idx'], (int, np.integer))
            assert isinstance(fp['attribute_idx'], (int, np.integer))


class TestFalsePositiveErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_config_structure(self):
        """Test handling of invalid configuration structure."""
        invalid_configs = [
            {},  # Empty config
            {'num_fps': 1},  # Missing ranges
            {'num_fps': 1, 'fp_pos_range': {}},  # Empty ranges
            {'num_fps': 1, 'fp_pos_range': {'x': [0, 1]}, 'fp_dim_range': {}, 'fp_velocity_range': {}}  # Incomplete ranges
        ]
        
        rng = np.random.default_rng(42)
        
        for config in invalid_configs:
            with pytest.raises(KeyError):
                simulate_false_positives(True, config, rng)
    
    def test_invalid_range_values(self):
        """Test handling of invalid range values."""
        # Config with inverted ranges (min > max)
        config = {
            'num_fps': 1,
            'fp_pos_range': {'x': [10, 0], 'y': [5, -5], 'z': [2, -2]},  # Invalid ranges
            'fp_dim_range': {'w': [3, 1], 'l': [5, 2], 'h': [3, 1]},     # Invalid ranges
            'fp_velocity_range': {'vx': [10, -10], 'vy': [5, -5]}         # Invalid ranges
        }
        
        rng = np.random.default_rng(42)
        
        # Function should raise an error for invalid ranges (min > max)
        with pytest.raises(ValueError):
            fps = simulate_false_positives(True, config, rng)


class TestFalsePositiveReproducibility:
    """Test reproducibility and deterministic behavior."""
    
    def test_reproducibility_across_calls(self):
        """Test that multiple calls with same seed produce identical results."""
        config = {
            'num_fps': 10,
            'fp_pos_range': {'x': [-100, 100], 'y': [-50, 50], 'z': [-3, 3]},
            'fp_dim_range': {'w': [0.5, 4.0], 'l': [1.0, 8.0], 'h': [0.5, 4.0]},
            'fp_velocity_range': {'vx': [-15, 15], 'vy': [-10, 10]}
        }
        
        results = []
        for _ in range(3):
            rng = np.random.default_rng(12345)
            fps = simulate_false_positives(True, config, rng)
            results.append(fps)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            for fp1, fp2 in zip(results[0], results[i]):
                np.testing.assert_array_equal(fp1['box_7d_world'], fp2['box_7d_world'])
                np.testing.assert_array_equal(fp1['velocity_world_2d'], fp2['velocity_world_2d'])
                assert fp1['class_idx'] == fp2['class_idx']
                assert fp1['attribute_idx'] == fp2['attribute_idx']
    
    def test_seed_independence(self):
        """Test that different seeds produce statistically different results."""
        config = {
            'num_fps': 100,  # Large number for statistical significance
            'fp_pos_range': {'x': [-50, 50], 'y': [-25, 25], 'z': [-2, 2]},
            'fp_dim_range': {'w': [1, 3], 'l': [2, 5], 'h': [1, 3]},
            'fp_velocity_range': {'vx': [-10, 10], 'vy': [-5, 5]}
        }
        
        # Generate false positives with different seeds
        rng1 = np.random.default_rng(111)
        fps1 = simulate_false_positives(True, config, rng1)
        
        rng2 = np.random.default_rng(999)
        fps2 = simulate_false_positives(True, config, rng2)
        
        # Extract positions for statistical comparison
        positions1 = np.array([fp['box_7d_world'][:3] for fp in fps1])
        positions2 = np.array([fp['box_7d_world'][:3] for fp in fps2])
        
        # Means should be different (with high probability)
        mean1 = np.mean(positions1, axis=0)
        mean2 = np.mean(positions2, axis=0)
        
        # At least one coordinate mean should be different
        assert not np.allclose(mean1, mean2, atol=1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
