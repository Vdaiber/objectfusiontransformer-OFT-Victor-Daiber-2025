#!/usr/bin/env python3
"""
Integration tests for false-positive generation with preprocessing pipeline.

This module tests the complete integration of false-positive generation within
the preprocessing pipeline, including data augmentation, virtual sensor processing,
and end-to-end preprocessing workflow.
"""

import pytest
import numpy as np
import sys
import tempfile
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the components to test
from src.oft.transformer.datasets.preprocessing.data_augmentation import simulate_false_positives
from src.oft.transformer.utils.reproducibility import set_seed


class TestFalsePositivePreprocessingPipeline:
    """Test false-positive generation within the preprocessing pipeline context."""
    
    @pytest.fixture
    def sample_scene_meta(self):
        """Sample scene metadata for preprocessing tests."""
        return {
            'weather': 'clear',
            'lighting': 'illuminated',
            'area': 'highway',
            'construction': 'unchanged',
            'structure': 'regular',
            'season': 'summer'
        }
    
    @pytest.fixture
    def sample_gt_detections(self):
        """Sample ground truth detections for integration testing."""
        return [
            {
                'box_7d_world': np.array([10.0, 5.0, 1.5, 4.0, 2.0, 1.8, 0.3], dtype=np.float64),
                'velocity_world_2d': np.array([8.0, -1.0], dtype=np.float64),
                'class_idx': 0,  # car
                'attribute_idx': 5  # vehicle.moving
            },
            {
                'box_7d_world': np.array([-15.0, -8.0, 1.2, 1.8, 0.8, 1.7, -0.5], dtype=np.float64),
                'velocity_world_2d': np.array([2.0, 0.5], dtype=np.float64),
                'class_idx': 5,  # pedestrian
                'attribute_idx': 0  # pedestrian.moving
            }
        ]
    
    @pytest.fixture
    def pipeline_config_with_fps(self):
        """Pipeline configuration with false-positive generation enabled."""
        return {
            'dataset': {
                'class_names': [
                    'car', 'truck', 'bus', 'trailer', 'other_vehicle',
                    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone',
                    'barrier', 'animal', 'traffic_sign'
                ],
                'simulation': {
                    'fn_rate': 0.1,  # 10% false negative rate
                    'num_fps': 3,    # 3 false positives per scene
                    'fp_pos_range': {
                        'x': [-70, 70],
                        'y': [-70, 70],
                        'z': [-2, 2]
                    },
                    'fp_dim_range': {
                        'w': [0.5, 3.5],
                        'l': [0.5, 6.0],
                        'h': [0.5, 3.5]
                    },
                    'fp_velocity_range': {
                        'vx': [-10, 10],
                        'vy': [-5, 5]
                    }
                },
                'virtual_sensors': [
                    {
                        'name': 'virtual_lidar',
                        'enabled': True,
                        'pos_noise_std': [0.15, 0.15, 0.10],
                        'dim_noise_std': 0.08,
                        'yaw_noise_std': 0.05,
                        'velocity_noise_std': 0.2,
                        'dropout_rate': 0.05,
                        'class_accuracy': 0.95,
                        'attribute_accuracy': 0.9
                    }
                ]
            }
        }
    
    def test_false_positive_generation_in_training_mode(self, pipeline_config_with_fps):
        """Test false-positive generation during training mode."""
        simulation_cfg = pipeline_config_with_fps['dataset']['simulation']
        rng = np.random.default_rng(42)
        
        # Test training mode (should generate false positives)
        is_training = True
        fps = simulate_false_positives(is_training, simulation_cfg, rng)
        
        assert len(fps) == simulation_cfg['num_fps']
        
        # Verify all false positives have correct markers and structure
        for fp in fps:
            assert fp['class_idx'] == -1
            assert fp['attribute_idx'] == -1
            assert fp['box_7d_world'].shape == (7,)
            assert fp['velocity_world_2d'].shape == (2,)
    
    def test_false_positive_generation_in_evaluation_mode(self, pipeline_config_with_fps):
        """Test that false positives are NOT generated during evaluation mode."""
        simulation_cfg = pipeline_config_with_fps['dataset']['simulation']
        rng = np.random.default_rng(42)
        
        # Test evaluation mode (should NOT generate false positives)
        is_training = False
        fps = simulate_false_positives(is_training, simulation_cfg, rng)
        
        assert len(fps) == 0
    
    def test_false_positive_augmentation_with_gt_data(self, sample_gt_detections, pipeline_config_with_fps):
        """Test integration of false positives with ground truth data."""
        simulation_cfg = pipeline_config_with_fps['dataset']['simulation']
        rng = np.random.default_rng(42)
        
        # Generate false positives
        fps = simulate_false_positives(True, simulation_cfg, rng)
        
        # Combine with ground truth (simulating dataset augmentation)
        augmented_detections = sample_gt_detections + fps
        
        # Verify the augmented dataset structure
        assert len(augmented_detections) == len(sample_gt_detections) + len(fps)
        
        # Count real detections vs false positives
        real_detections = [det for det in augmented_detections if det['class_idx'] != -1]
        false_positives = [det for det in augmented_detections if det['class_idx'] == -1]
        
        assert len(real_detections) == len(sample_gt_detections)
        assert len(false_positives) == simulation_cfg['num_fps']
        
        # Verify all detections have consistent structure
        for det in augmented_detections:
            assert 'box_7d_world' in det
            assert 'velocity_world_2d' in det
            assert 'class_idx' in det
            assert 'attribute_idx' in det
            assert det['box_7d_world'].shape == (7,)
            assert det['velocity_world_2d'].shape == (2,)
    
    def test_false_positive_position_distribution(self, pipeline_config_with_fps):
        """Test that false positives are distributed across the specified ranges."""
        simulation_cfg = pipeline_config_with_fps['dataset']['simulation']
        rng = np.random.default_rng(42)
        
        # Generate many false positives for statistical analysis
        config_many_fps = simulation_cfg.copy()
        config_many_fps['num_fps'] = 100
        
        fps = simulate_false_positives(True, config_many_fps, rng)
        
        # Extract positions
        positions = np.array([fp['box_7d_world'][:3] for fp in fps])
        
        # Check that positions span the expected ranges
        x_range = simulation_cfg['fp_pos_range']['x']
        y_range = simulation_cfg['fp_pos_range']['y']
        z_range = simulation_cfg['fp_pos_range']['z']
        
        assert np.min(positions[:, 0]) >= x_range[0]
        assert np.max(positions[:, 0]) <= x_range[1]
        assert np.min(positions[:, 1]) >= y_range[0]
        assert np.max(positions[:, 1]) <= y_range[1]
        assert np.min(positions[:, 2]) >= z_range[0]
        assert np.max(positions[:, 2]) <= z_range[1]
        
        # Check that positions are reasonably distributed (not all clustered)
        x_std = np.std(positions[:, 0])
        y_std = np.std(positions[:, 1])
        z_std = np.std(positions[:, 2])
        
        # Standard deviation should be significant fraction of range
        assert x_std > (x_range[1] - x_range[0]) * 0.1
        assert y_std > (y_range[1] - y_range[0]) * 0.1
        assert z_std > (z_range[1] - z_range[0]) * 0.1
    
    def test_false_positive_velocity_distribution(self, pipeline_config_with_fps):
        """Test that false positive velocities are distributed correctly."""
        simulation_cfg = pipeline_config_with_fps['dataset']['simulation']
        rng = np.random.default_rng(42)
        
        # Generate many false positives for statistical analysis
        config_many_fps = simulation_cfg.copy()
        config_many_fps['num_fps'] = 100
        
        fps = simulate_false_positives(True, config_many_fps, rng)
        
        # Extract velocities
        velocities = np.array([fp['velocity_world_2d'] for fp in fps])
        
        # Check that velocities span the expected ranges
        vx_range = simulation_cfg['fp_velocity_range']['vx']
        vy_range = simulation_cfg['fp_velocity_range']['vy']
        
        assert np.min(velocities[:, 0]) >= vx_range[0]
        assert np.max(velocities[:, 0]) <= vx_range[1]
        assert np.min(velocities[:, 1]) >= vy_range[0]
        assert np.max(velocities[:, 1]) <= vy_range[1]
        
        # Check velocity distribution
        vx_std = np.std(velocities[:, 0])
        vy_std = np.std(velocities[:, 1])
        
        assert vx_std > (vx_range[1] - vx_range[0]) * 0.1
        assert vy_std > (vy_range[1] - vy_range[0]) * 0.1


class TestFalsePositiveDataConsistency:
    """Test data consistency and format compatibility."""
    
    @pytest.fixture
    def realistic_config(self):
        """Realistic configuration based on actual pipeline settings."""
        return {
            'num_fps': 2,
            'fp_pos_range': {
                'x': [-70, 70],  # 140m range (highway scenario)
                'y': [-70, 70],  # 140m lateral range
                'z': [-2, 2]     # 4m vertical range
            },
            'fp_dim_range': {
                'w': [0.5, 3.5],  # 0.5m to 3.5m width (bike to truck)
                'l': [0.5, 6.0],  # 0.5m to 6m length (bike to truck)
                'h': [0.5, 3.5]   # 0.5m to 3.5m height (bike to truck)
            },
            'fp_velocity_range': {
                'vx': [-10, 10],  # ±10 m/s (±36 km/h)
                'vy': [-5, 5]     # ±5 m/s lateral velocity
            }
        }
    
    def test_realistic_false_positive_values(self, realistic_config):
        """Test that false positives have realistic physical values."""
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, realistic_config, rng)
        
        for fp in fps:
            box = fp['box_7d_world']
            velocity = fp['velocity_world_2d']
            
            # Check position realism (highway scenario)
            x, y, z = box[:3]
            assert -70 <= x <= 70    # Within sensor range
            assert -70 <= y <= 70    # Within lateral range
            assert -2 <= z <= 2      # Reasonable height range
            
            # Check dimension realism
            w, l, h = box[3:6]
            assert 0.5 <= w <= 3.5   # Bike to truck width
            assert 0.5 <= l <= 6.0   # Bike to truck length
            assert 0.5 <= h <= 3.5   # Bike to truck height
            
            # Check yaw realism
            yaw = box[6]
            assert -np.pi <= yaw <= np.pi  # Valid orientation range
            
            # Check velocity realism
            vx, vy = velocity
            assert -10 <= vx <= 10   # Reasonable longitudinal velocity
            assert -5 <= vy <= 5     # Reasonable lateral velocity
    
    def test_false_positive_data_types_consistency(self, realistic_config):
        """Test that false positives have consistent data types with GT data."""
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, realistic_config, rng)
        
        for fp in fps:
            # Box should be float64 array
            assert isinstance(fp['box_7d_world'], np.ndarray)
            assert fp['box_7d_world'].dtype == np.float64
            assert fp['box_7d_world'].shape == (7,)
            
            # Velocity should be float64 array
            assert isinstance(fp['velocity_world_2d'], np.ndarray)
            assert fp['velocity_world_2d'].dtype == np.float64
            assert fp['velocity_world_2d'].shape == (2,)
            
            # Indices should be integers
            assert isinstance(fp['class_idx'], (int, np.integer))
            assert isinstance(fp['attribute_idx'], (int, np.integer))
            
            # False positive markers
            assert fp['class_idx'] == -1
            assert fp['attribute_idx'] == -1
    
    def test_false_positive_serialization_compatibility(self, realistic_config):
        """Test that false positives can be serialized (for caching/preprocessing)."""
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, realistic_config, rng)
        
        # Test JSON serialization (common in preprocessing pipelines)
        serializable_fps = []
        for fp in fps:
            serializable_fp = {
                'box_7d_world': fp['box_7d_world'].tolist(),
                'velocity_world_2d': fp['velocity_world_2d'].tolist(),
                'class_idx': int(fp['class_idx']),
                'attribute_idx': int(fp['attribute_idx'])
            }
            serializable_fps.append(serializable_fp)
        
        # Should be able to serialize and deserialize
        json_str = json.dumps(serializable_fps)
        deserialized = json.loads(json_str)
        
        assert len(deserialized) == len(fps)
        for orig_fp, deser_fp in zip(fps, deserialized):
            np.testing.assert_array_almost_equal(
                orig_fp['box_7d_world'], 
                np.array(deser_fp['box_7d_world'])
            )
            np.testing.assert_array_almost_equal(
                orig_fp['velocity_world_2d'], 
                np.array(deser_fp['velocity_world_2d'])
            )
            assert orig_fp['class_idx'] == deser_fp['class_idx']
            assert orig_fp['attribute_idx'] == deser_fp['attribute_idx']


class TestFalsePositiveConfigurationScenarios:
    """Test various configuration scenarios for false-positive generation."""
    
    def test_disabled_false_positives(self):
        """Test configuration with false positives disabled."""
        config = {
            'num_fps': 0,  # Disabled
            'fp_pos_range': {'x': [-50, 50], 'y': [-25, 25], 'z': [-2, 2]},
            'fp_dim_range': {'w': [1, 3], 'l': [2, 5], 'h': [1, 3]},
            'fp_velocity_range': {'vx': [-10, 10], 'vy': [-5, 5]}
        }
        
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, config, rng)
        
        assert len(fps) == 0
    
    def test_high_false_positive_count(self):
        """Test configuration with high false positive count."""
        config = {
            'num_fps': 20,  # High count
            'fp_pos_range': {'x': [-100, 100], 'y': [-50, 50], 'z': [-3, 3]},
            'fp_dim_range': {'w': [0.5, 4], 'l': [1, 8], 'h': [0.5, 4]},
            'fp_velocity_range': {'vx': [-15, 15], 'vy': [-10, 10]}
        }
        
        rng = np.random.default_rng(42)
        fps = simulate_false_positives(True, config, rng)
        
        assert len(fps) == 20
        
        # All should have correct structure
        for fp in fps:
            assert fp['class_idx'] == -1
            assert fp['attribute_idx'] == -1
            assert fp['box_7d_world'].shape == (7,)
            assert fp['velocity_world_2d'].shape == (2,)
    
    def test_extreme_range_configurations(self):
        """Test configurations with extreme ranges."""
        # Very small ranges
        small_config = {
            'num_fps': 2,
            'fp_pos_range': {'x': [0, 1], 'y': [0, 1], 'z': [0, 0.1]},
            'fp_dim_range': {'w': [0.1, 0.2], 'l': [0.1, 0.2], 'h': [0.1, 0.2]},
            'fp_velocity_range': {'vx': [-0.1, 0.1], 'vy': [-0.1, 0.1]}
        }
        
        # Very large ranges
        large_config = {
            'num_fps': 2,
            'fp_pos_range': {'x': [-1000, 1000], 'y': [-500, 500], 'z': [-100, 100]},
            'fp_dim_range': {'w': [0.1, 50], 'l': [0.1, 100], 'h': [0.1, 50]},
            'fp_velocity_range': {'vx': [-100, 100], 'vy': [-50, 50]}
        }
        
        rng = np.random.default_rng(42)
        
        # Both should work without errors
        small_fps = simulate_false_positives(True, small_config, rng)
        large_fps = simulate_false_positives(True, large_config, rng)
        
        assert len(small_fps) == 2
        assert len(large_fps) == 2
        
        # Values should be within specified ranges
        for fp in small_fps:
            box = fp['box_7d_world']
            assert 0 <= box[0] <= 1
            assert 0 <= box[1] <= 1
            assert 0 <= box[2] <= 0.1
        
        for fp in large_fps:
            box = fp['box_7d_world']
            assert -1000 <= box[0] <= 1000
            assert -500 <= box[1] <= 500
            assert -100 <= box[2] <= 100


class TestFalsePositiveReproducibilityIntegration:
    """Test reproducibility in the context of preprocessing pipeline."""
    
    def test_deterministic_preprocessing_runs(self):
        """Test that preprocessing runs are deterministic with same seeds."""
        config = {
            'num_fps': 5,
            'fp_pos_range': {'x': [-50, 50], 'y': [-25, 25], 'z': [-2, 2]},
            'fp_dim_range': {'w': [1, 3], 'l': [2, 5], 'h': [1, 3]},
            'fp_velocity_range': {'vx': [-10, 10], 'vy': [-5, 5]}
        }
        
        # Simulate multiple preprocessing runs with same global seed
        results = []
        for _ in range(3):
            # Reset seed as would happen in actual preprocessing
            set_seed(12345)
            rng = np.random.default_rng(12345)
            fps = simulate_false_positives(True, config, rng)
            results.append(fps)
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            for fp1, fp2 in zip(results[0], results[i]):
                np.testing.assert_array_equal(fp1['box_7d_world'], fp2['box_7d_world'])
                np.testing.assert_array_equal(fp1['velocity_world_2d'], fp2['velocity_world_2d'])
                assert fp1['class_idx'] == fp2['class_idx']
                assert fp1['attribute_idx'] == fp2['attribute_idx']
    
    def test_sample_specific_determinism(self):
        """Test that false positives are deterministic per sample (as would occur in practice)."""
        config = {
            'num_fps': 3,
            'fp_pos_range': {'x': [-30, 30], 'y': [-15, 15], 'z': [-1, 1]},
            'fp_dim_range': {'w': [1, 2], 'l': [2, 4], 'h': [1, 2]},
            'fp_velocity_range': {'vx': [-5, 5], 'vy': [-3, 3]}
        }
        
        # Simulate processing same sample multiple times
        sample_token = "sample_12345"
        global_seed = 42
        
        results = []
        for _ in range(3):
            # Create sample-specific seed (as would happen in actual preprocessing)
            import hashlib
            fp_seed = global_seed + int(hashlib.md5((sample_token + "_fps").encode()).hexdigest()[:8], 16) % 1000000
            rng = np.random.default_rng(fp_seed)
            fps = simulate_false_positives(True, config, rng)
            results.append(fps)
        
        # All runs for same sample should produce identical results
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            for fp1, fp2 in zip(results[0], results[i]):
                np.testing.assert_array_equal(fp1['box_7d_world'], fp2['box_7d_world'])
                np.testing.assert_array_equal(fp1['velocity_world_2d'], fp2['velocity_world_2d'])
                assert fp1['class_idx'] == fp2['class_idx']
                assert fp1['attribute_idx'] == fp2['attribute_idx']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
