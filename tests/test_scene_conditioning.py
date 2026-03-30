#!/usr/bin/env python3
"""
Unit tests for scene conditioning functionality in data augmentation.

This module tests the scene-dependent noise conditioning system that adjusts
sensor noise parameters based on environmental conditions like weather, lighting,
area type, construction status, and other scene metadata.
"""

import pytest
import numpy as np
import tempfile
import yaml
import os
from pathlib import Path
from typing import Dict, Any

# Import the functions to test
from src.oft.transformer.datasets.preprocessing.data_augmentation import (
    load_scene_conditioning_config,
    calculate_scene_dependent_factor,
    enhance_noise_parameters,
    calculate_classification_factor,
    enhance_classification_accuracy,
    add_noise_to_box,
    add_class_noise,
    add_attribute_noise
)


class TestSceneConditioningConfig:
    """Test loading and parsing of scene conditioning configuration."""
    
    def test_load_valid_config(self):
        """Test loading a valid scene conditioning configuration."""
        config_data = {
            'scene_conditional_noise': {
                'virtual_camera': {
                    'weather_modifiers': {'clear': 1.0, 'rain': 1.6}
                }
            },
            'combination_rules': {
                'max_amplification': 3.0,
                'min_attenuation': 0.7
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loaded_config = load_scene_conditioning_config(temp_path)
            assert loaded_config == config_data
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_config(self):
        """Test handling of non-existent configuration file."""
        result = load_scene_conditioning_config("nonexistent_file.yaml")
        assert result == {}
    
    def test_load_invalid_yaml(self):
        """Test handling of invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # Invalid YAML
            temp_path = f.name
        
        try:
            result = load_scene_conditioning_config(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)


class TestSceneDependentFactor:
    """Test calculation of scene-dependent amplification factors."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample scene conditioning configuration for testing."""
        return {
            'scene_conditional_noise': {
                'virtual_camera': {
                    'weather_modifiers': {
                        'clear': 1.0,
                        'rain': 1.6,
                        'snow': 2.0
                    },
                    'lighting_modifiers': {
                        'illuminated': 1.0,
                        'dark': 2.5
                    },
                    'area_modifiers': {
                        'highway': 1.0,
                        'city': 1.3
                    }
                },
                'virtual_lidar': {
                    'weather_modifiers': {
                        'clear': 1.0,
                        'rain': 1.3
                    },
                    'lighting_modifiers': {
                        'illuminated': 1.0,
                        'dark': 1.0  # LiDAR unaffected by lighting
                    }
                }
            },
            'combination_rules': {
                'max_amplification': 3.0,
                'min_attenuation': 0.7
            }
        }
    
    def test_clear_conditions(self, sample_config):
        """Test factor calculation for clear conditions."""
        scene_meta = {'weather': 'clear', 'lighting': 'illuminated', 'area': 'highway'}
        factor = calculate_scene_dependent_factor('virtual_camera', scene_meta, sample_config)
        assert factor == 1.0  # All clear conditions should give 1.0
    
    def test_rain_conditions(self, sample_config):
        """Test factor calculation for rainy conditions."""
        scene_meta = {'weather': 'rain', 'lighting': 'illuminated', 'area': 'highway'}
        factor = calculate_scene_dependent_factor('virtual_camera', scene_meta, sample_config)
        assert factor == 1.6  # Only rain factor applies
    
    def test_combined_conditions(self, sample_config):
        """Test factor calculation for combined adverse conditions."""
        scene_meta = {'weather': 'rain', 'lighting': 'dark', 'area': 'city'}
        factor = calculate_scene_dependent_factor('virtual_camera', scene_meta, sample_config)
        expected = 1.6 * 2.5 * 1.3  # rain × dark × city = 5.2
        assert factor == 3.0  # Should be clipped to max_amplification
    
    def test_lidar_lighting_independence(self, sample_config):
        """Test that LiDAR is unaffected by lighting conditions."""
        scene_meta_day = {'weather': 'clear', 'lighting': 'illuminated'}
        scene_meta_night = {'weather': 'clear', 'lighting': 'dark'}
        
        factor_day = calculate_scene_dependent_factor('virtual_lidar', scene_meta_day, sample_config)
        factor_night = calculate_scene_dependent_factor('virtual_lidar', scene_meta_night, sample_config)
        
        assert factor_day == factor_night == 1.0
    
    def test_unknown_sensor(self, sample_config):
        """Test handling of unknown sensor names."""
        scene_meta = {'weather': 'rain'}
        factor = calculate_scene_dependent_factor('unknown_sensor', scene_meta, sample_config)
        assert factor == 1.0
    
    def test_empty_config(self):
        """Test handling of empty configuration."""
        scene_meta = {'weather': 'rain', 'lighting': 'dark'}
        factor = calculate_scene_dependent_factor('virtual_camera', scene_meta, {})
        assert factor == 1.0
    
    def test_unknown_scene_values(self, sample_config):
        """Test handling of unknown scene metadata values."""
        scene_meta = {'weather': 'unknown_weather', 'lighting': 'unknown_lighting'}
        factor = calculate_scene_dependent_factor('virtual_camera', scene_meta, sample_config)
        assert factor == 1.0  # Unknown values should not affect factor


class TestNoiseParameterEnhancement:
    """Test enhancement of noise parameters based on scene conditions."""
    
    @pytest.fixture
    def sample_sensor_cfg(self):
        """Sample sensor configuration for testing."""
        return {
            'name': 'virtual_camera',
            'pos_noise_std': [0.15, 0.10, 0.20],
            'dim_noise_std': 0.08,
            'yaw_noise_std': 0.05,
            'velocity_noise_std': 0.2
        }
    
    @pytest.fixture
    def sample_scene_config(self):
        """Sample scene conditioning configuration for testing."""
        return {
            'scene_conditional_noise': {
                'virtual_camera': {
                    'weather_modifiers': {'clear': 1.0, 'rain': 2.0}
                }
            },
            'component_sensitivity': {
                'virtual_camera': {
                    'pos_noise_std': 1.5,
                    'dim_noise_std': 2.0,
                    'yaw_noise_std': 1.2,
                    'velocity_noise_std': 1.8
                }
            },
            'combination_rules': {
                'max_amplification': 3.0,
                'min_attenuation': 0.7
            }
        }
    
    def test_clear_conditions_no_enhancement(self, sample_sensor_cfg, sample_scene_config):
        """Test that clear conditions don't enhance noise parameters."""
        scene_meta = {'weather': 'clear'}
        enhanced_cfg = enhance_noise_parameters(sample_sensor_cfg, scene_meta, sample_scene_config)
        
        # With clear weather (factor 1.0), parameters should be enhanced by component sensitivity only
        assert enhanced_cfg['pos_noise_std'] == [0.15 * 1.5, 0.10 * 1.5, 0.20 * 1.5]
        assert enhanced_cfg['dim_noise_std'] == 0.08 * 2.0
        assert enhanced_cfg['yaw_noise_std'] == 0.05 * 1.2
        assert enhanced_cfg['velocity_noise_std'] == 0.2 * 1.8
    
    def test_rain_conditions_enhancement(self, sample_sensor_cfg, sample_scene_config):
        """Test that rain conditions enhance noise parameters."""
        scene_meta = {'weather': 'rain'}
        enhanced_cfg = enhance_noise_parameters(sample_sensor_cfg, scene_meta, sample_scene_config)
        
        # With rain (factor 2.0), parameters should be enhanced by 2.0 × component sensitivity
        assert enhanced_cfg['pos_noise_std'] == [0.15 * 2.0 * 1.5, 0.10 * 2.0 * 1.5, 0.20 * 2.0 * 1.5]
        assert enhanced_cfg['dim_noise_std'] == 0.08 * 2.0 * 2.0
        assert enhanced_cfg['yaw_noise_std'] == 0.05 * 2.0 * 1.2
        assert enhanced_cfg['velocity_noise_std'] == 0.2 * 2.0 * 1.8
    
    def test_no_scene_conditioning(self, sample_sensor_cfg):
        """Test that missing scene conditioning returns original parameters."""
        scene_meta = {'weather': 'rain'}
        enhanced_cfg = enhance_noise_parameters(sample_sensor_cfg, scene_meta, None)
        assert enhanced_cfg == sample_sensor_cfg
    
    def test_no_component_sensitivity(self, sample_sensor_cfg):
        """Test fallback behavior when component sensitivity is missing."""
        scene_config = {
            'scene_conditional_noise': {
                'virtual_camera': {
                    'weather_modifiers': {'rain': 2.0}
                }
            },
            'combination_rules': {'max_amplification': 3.0, 'min_attenuation': 0.7}
        }
        
        scene_meta = {'weather': 'rain'}
        enhanced_cfg = enhance_noise_parameters(sample_sensor_cfg, scene_meta, scene_config)
        
        # Should apply scene factor (2.0) uniformly to all parameters
        assert enhanced_cfg['pos_noise_std'] == [0.15 * 2.0, 0.10 * 2.0, 0.20 * 2.0]
        assert enhanced_cfg['dim_noise_std'] == 0.08 * 2.0
        assert enhanced_cfg['yaw_noise_std'] == 0.05 * 2.0
        assert enhanced_cfg['velocity_noise_std'] == 0.2 * 2.0


class TestClassificationFactors:
    """Test calculation of classification accuracy factors."""
    
    @pytest.fixture
    def sample_classification_config(self):
        """Sample classification sensitivity configuration."""
        return {
            'classification_sensitivity': {
                'virtual_camera': {
                    'class_accuracy_modifiers': {
                        'weather': {'clear': 1.0, 'rain': 0.7},
                        'lighting': {'illuminated': 1.0, 'dark': 0.5}
                    },
                    'attribute_accuracy_modifiers': {
                        'weather': {'clear': 1.0, 'rain': 0.8},
                        'lighting': {'illuminated': 1.0, 'dark': 0.4}
                    }
                }
            }
        }
    
    def test_class_accuracy_clear_conditions(self, sample_classification_config):
        """Test class accuracy factor for clear conditions."""
        scene_meta = {'weather': 'clear', 'lighting': 'illuminated'}
        factor = calculate_classification_factor(
            'virtual_camera', scene_meta, sample_classification_config, 'class_accuracy_modifiers'
        )
        assert factor == 1.0
    
    def test_class_accuracy_adverse_conditions(self, sample_classification_config):
        """Test class accuracy factor for adverse conditions."""
        scene_meta = {'weather': 'rain', 'lighting': 'dark'}
        factor = calculate_classification_factor(
            'virtual_camera', scene_meta, sample_classification_config, 'class_accuracy_modifiers'
        )
        assert factor == 0.7 * 0.5  # rain × dark = 0.35
    
    def test_attribute_accuracy_adverse_conditions(self, sample_classification_config):
        """Test attribute accuracy factor for adverse conditions."""
        scene_meta = {'weather': 'rain', 'lighting': 'dark'}
        factor = calculate_classification_factor(
            'virtual_camera', scene_meta, sample_classification_config, 'attribute_accuracy_modifiers'
        )
        assert factor == 0.8 * 0.4  # rain × dark = 0.32


class TestClassificationEnhancement:
    """Test enhancement of classification accuracy based on scene conditions."""
    
    @pytest.fixture
    def sample_sensor_cfg(self):
        """Sample sensor configuration with accuracy values."""
        return {
            'name': 'virtual_camera',
            'class_accuracy': 0.94,
            'attribute_accuracy': 0.92
        }
    
    @pytest.fixture
    def sample_classification_config(self):
        """Sample classification sensitivity configuration."""
        return {
            'classification_sensitivity': {
                'virtual_camera': {
                    'class_accuracy_modifiers': {
                        'weather': {'clear': 1.0, 'rain': 0.7}
                    },
                    'attribute_accuracy_modifiers': {
                        'weather': {'clear': 1.0, 'rain': 0.8}
                    }
                }
            }
        }
    
    def test_clear_weather_no_degradation(self, sample_sensor_cfg, sample_classification_config):
        """Test that clear weather doesn't degrade accuracy."""
        scene_meta = {'weather': 'clear'}
        enhanced_cfg = enhance_classification_accuracy(sample_sensor_cfg, scene_meta, sample_classification_config)
        
        assert enhanced_cfg['class_accuracy'] == 0.94
        assert enhanced_cfg['attribute_accuracy'] == 0.92
    
    def test_rain_weather_degradation(self, sample_sensor_cfg, sample_classification_config):
        """Test that rain degrades accuracy."""
        scene_meta = {'weather': 'rain'}
        enhanced_cfg = enhance_classification_accuracy(sample_sensor_cfg, scene_meta, sample_classification_config)
        
        assert enhanced_cfg['class_accuracy'] == 0.94 * 0.7  # Degraded by rain
        assert enhanced_cfg['attribute_accuracy'] == 0.92 * 0.8  # Degraded by rain
    
    def test_accuracy_clamping(self, sample_classification_config):
        """Test that accuracy values are clamped to [0.0, 1.0]."""
        # Test with extreme degradation that would go below 0
        extreme_sensor_cfg = {'name': 'virtual_camera', 'class_accuracy': 0.1, 'attribute_accuracy': 0.1}
        extreme_config = {
            'classification_sensitivity': {
                'virtual_camera': {
                    'class_accuracy_modifiers': {'weather': {'rain': 0.01}},  # Extreme degradation
                    'attribute_accuracy_modifiers': {'weather': {'rain': 0.01}}
                }
            }
        }
        
        scene_meta = {'weather': 'rain'}
        enhanced_cfg = enhance_classification_accuracy(extreme_sensor_cfg, scene_meta, extreme_config)
        
        assert 0.0 <= enhanced_cfg['class_accuracy'] <= 1.0
        assert 0.0 <= enhanced_cfg['attribute_accuracy'] <= 1.0


class TestIntegratedNoiseApplication:
    """Test the integrated noise application with scene conditioning."""
    
    @pytest.fixture
    def sample_box_and_velocity(self):
        """Sample box and velocity for noise testing."""
        box_7d = np.array([10.0, 5.0, 2.0, 4.0, 2.0, 1.8, 0.5], dtype=np.float64)
        velocity_2d = np.array([8.0, -2.0], dtype=np.float64)
        return box_7d, velocity_2d
    
    @pytest.fixture
    def sample_sensor_cfg(self):
        """Sample sensor configuration for noise testing."""
        return {
            'name': 'virtual_camera',
            'pos_noise_std': [0.1, 0.1, 0.1],
            'dim_noise_std': 0.05,
            'yaw_noise_std': 0.02,
            'velocity_noise_std': 0.1,
            'class_accuracy': 0.95,
            'attribute_accuracy': 0.9
        }
    
    def test_noise_application_without_scene_conditioning(self, sample_box_and_velocity, sample_sensor_cfg):
        """Test that noise is applied correctly without scene conditioning."""
        box_7d, velocity_2d = sample_box_and_velocity
        
        # Set random seed for reproducible test
        np.random.seed(42)
        noisy_box_9d, noisy_velocity = add_noise_to_box(box_7d, velocity_2d, sample_sensor_cfg)
        
        # Check that noise was applied (output should differ from input)
        assert not np.array_equal(noisy_box_9d[:7], box_7d)
        assert not np.array_equal(noisy_velocity, velocity_2d)
        
        # Check output format
        assert noisy_box_9d.shape == (9,)  # 7D box + 2D velocity
        assert noisy_velocity.shape == (2,)
    
    def test_noise_application_with_scene_conditioning(self, sample_box_and_velocity, sample_sensor_cfg):
        """Test that scene conditioning amplifies noise appropriately."""
        box_7d, velocity_2d = sample_box_and_velocity
        
        scene_config = {
            'scene_conditional_noise': {
                'virtual_camera': {
                    'weather_modifiers': {'rain': 2.0}
                }
            },
            'component_sensitivity': {
                'virtual_camera': {
                    'pos_noise_std': 2.0,
                    'dim_noise_std': 2.0,
                    'yaw_noise_std': 2.0,
                    'velocity_noise_std': 2.0
                }
            },
            'combination_rules': {'max_amplification': 5.0, 'min_attenuation': 0.5}
        }
        
        scene_meta = {'weather': 'rain'}
        
        # Apply noise with same random seed for both cases
        np.random.seed(42)
        noisy_box_clear, _ = add_noise_to_box(box_7d, velocity_2d, sample_sensor_cfg)
        
        np.random.seed(42)
        noisy_box_rain, _ = add_noise_to_box(
            box_7d, velocity_2d, sample_sensor_cfg, scene_meta, scene_config
        )
        
        # Rain conditions should produce more noise (larger deviations from original)
        clear_deviation = np.linalg.norm(noisy_box_clear[:3] - box_7d[:3])
        rain_deviation = np.linalg.norm(noisy_box_rain[:3] - box_7d[:3])
        
        assert rain_deviation > clear_deviation


class TestClassificationNoise:
    """Test classification noise with scene conditioning."""
    
    @pytest.fixture
    def sample_sensor_cfg(self):
        """Sample sensor configuration for classification testing."""
        return {
            'name': 'virtual_camera',
            'class_accuracy': 0.9,
            'attribute_accuracy': 0.85
        }
    
    def test_class_noise_without_scene_conditioning(self, sample_sensor_cfg):
        """Test class noise application without scene conditioning."""
        class_names = ['car', 'truck', 'bus']
        
        # Test multiple times to check probabilistic behavior
        results = []
        for _ in range(100):
            result = add_class_noise(0, sample_sensor_cfg, class_names)  # GT class: car (index 0)
            results.append(result)
        
        # With 90% accuracy, most results should be correct (class 0)
        correct_count = sum(1 for r in results if r == 0)
        accuracy_rate = correct_count / len(results)
        
        # Allow some tolerance for randomness
        assert 0.8 <= accuracy_rate <= 1.0
    
    def test_class_noise_with_scene_conditioning(self, sample_sensor_cfg):
        """Test that scene conditioning reduces classification accuracy."""
        class_names = ['car', 'truck', 'bus']
        
        scene_config = {
            'classification_sensitivity': {
                'virtual_camera': {
                    'class_accuracy_modifiers': {
                        'weather': {'rain': 0.5}  # Reduce accuracy to 50%
                    }
                }
            }
        }
        scene_meta = {'weather': 'rain'}
        
        # Test multiple times
        results = []
        for _ in range(100):
            result = add_class_noise(
                0, sample_sensor_cfg, class_names, scene_meta, scene_config
            )
            results.append(result)
        
        # With 50% effective accuracy (0.9 * 0.5), more errors expected
        correct_count = sum(1 for r in results if r == 0)
        accuracy_rate = correct_count / len(results)
        
        # Should be around 45% (0.9 * 0.5 = 0.45) with some tolerance
        assert 0.3 <= accuracy_rate <= 0.6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])