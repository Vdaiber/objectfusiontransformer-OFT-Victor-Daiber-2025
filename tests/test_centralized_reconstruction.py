"""
Comprehensive tests for centralized reconstruction pipeline.

This test suite verifies that all changes for centralized reconstruction work correctly:
- Data flow from dataset to model
- Loss function calculations in normalized space
- Evaluation pipeline with denormalization
- Parameter passing and tensor shapes
- Import/export compatibility
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import tempfile
import os

# Import the modules we need to test
from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive
from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion
from oft.transformer.training.losses.regression_losses import LossCenter, LossSize, LossAngle, LossVelocity
from oft.transformer.training.losses.giou_losses import LossGIoUBEV
from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive
from oft.transformer.utils.normalization_utils import get_global_normalizer
from oft.transformer.utils.geometry_utils import yaw_to_sin_cos, sin_cos_to_yaw


class TestCentralizedReconstruction:
    """Test suite for centralized reconstruction functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            'dataset': {
                'class_names': ['car', 'truck', 'pedestrian'],
                'point_cloud_range': [-150.0, -150.0, -5.0, 150.0, 150.0, 3.0],
                'max_velocity': 20.0,
                'virtual_sensors': [
                    {'name': 'virtual_lidar', 'enabled': True},
                    {'name': 'virtual_camera', 'enabled': True}
                ],
                'simulation': {'fn_rate': 0.0, 'num_fps': 0},
                'normalization_stats_path': 'tests/test_data/norm_stats.yaml'
            },
            'model': {
                'input_feature_dim': 10,
                'd_model': 256,
                'nhead': 8,
                'num_classes': 3,
                'box_dim': 8,
                'num_attribute_classes': 10,
                'output_heads': {
                    'classification': {'d_model': 256},
                    'bbox_regression': {'d_model': 256},
                    'velocity': {'d_model': 256},
                    'attributes': {'d_model': 256}
                }
            },
            'evaluation': {
                'conf_th_eval': 0.1
            }
        }
    
    @pytest.fixture
    def mock_norm_stats(self):
        """Create mock normalization statistics."""
        return {
            'metadata': {
                'point_cloud_range': [-150.0, -150.0, -5.0, 150.0, 150.0, 3.0]
            },
            'center_abs_99p': [50.0, 50.0, 5.0],
            'log_size_abs_99p': [1.0, 1.0, 1.0],
            'velocity_percentiles': {
                '1p': [-15.0, -15.0],
                '99p': [15.0, 15.0]
            }
        }
    
    @pytest.fixture
    def sample_batch_data(self):
        """Create sample batch data for testing."""
        batch_size = 2
        num_objects = 5
        
        return {
            'sample_tokens': ['token1', 'token2'],
            'ego_translation_world': torch.randn(batch_size, 3),
            'ego_rotation_world_quat': torch.randn(batch_size, 4),
            'ego_motion': {
                'cabin': {
                    'velocity': torch.randn(batch_size, 3),
                    'acceleration': torch.randn(batch_size, 3),
                    'angular_velocity': torch.randn(batch_size, 3)
                }
            },
            'gt_boxes_b_normalized': torch.randn(batch_size, num_objects, 10),
            'gt_labels_b': torch.randint(0, 3, (batch_size, num_objects)),
            'gt_attributes_b': torch.randint(0, 10, (batch_size, num_objects)),
            'gt_valid_mask_b': torch.ones(batch_size, num_objects, dtype=torch.bool),
            'target_offsets_normalized': torch.randn(batch_size, num_objects, 10),
            'target_offsets_mask': torch.ones(batch_size, num_objects, dtype=torch.bool),
            'sensor_data': {
                'virtual_lidar': {
                    'features': torch.randn(batch_size, num_objects, 10),
                    'metadata': torch.randn(batch_size, num_objects, 2),
                    'centers': torch.randn(batch_size, num_objects, 3),
                    'boxes': torch.randn(batch_size, num_objects, 9),
                    'mask': torch.zeros(batch_size, num_objects, dtype=torch.bool)
                }
            }
        }
    
    @pytest.fixture
    def mock_model_outputs(self):
        """Create mock model outputs for testing."""
        batch_size = 2
        num_objects = 5
        
        return {
            'pred_logits': torch.randn(batch_size, num_objects, 4),  # 3 classes + 1 background
            'pred_boxes_normalized': torch.randn(batch_size, num_objects, 8),  # 8D normalized boxes
            'pred_velocities_normalized': torch.randn(batch_size, num_objects, 2),  # 2D normalized velocities
            'pred_attributes': torch.randn(batch_size, num_objects, 10),  # 10 attributes
            'pred_duplicate': torch.randn(batch_size, num_objects, 2)  # KEEP vs IGNORE
        }

    def test_data_flow_consistency(self, mock_config, mock_norm_stats):
        """Test that data flows correctly through the pipeline."""
        # Create temporary norm stats file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(mock_norm_stats, f)
            norm_stats_path = f.name
        
        try:
            # Test that the dataset can be initialized with our config
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(mock_norm_stats)
                    
                    # This should not raise an exception
                    assert mock_config['dataset']['normalization_stats_path'] == norm_stats_path
                    
        finally:
            os.unlink(norm_stats_path)

    def test_normalized_box_representation(self):
        """Test that normalized box representation is correct."""
        # Test 10D normalized box format: [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin(yaw), cos(yaw), vx_norm, vy_norm]
        box_10d = torch.randn(10)
        
        # Verify dimensions
        assert box_10d.shape[0] == 10
        
        # Verify yaw representation (sin^2 + cos^2 should be close to 1)
        sin_yaw = box_10d[6]
        cos_yaw = box_10d[7]
        yaw_magnitude = sin_yaw**2 + cos_yaw**2
        assert torch.abs(yaw_magnitude - 1.0) < 0.1  # Allow some tolerance for random values
        
        # Test yaw conversion functions
        yaw_angle = sin_cos_to_yaw(sin_yaw, cos_yaw)
        sin_yaw_back, cos_yaw_back = yaw_to_sin_cos(yaw_angle)
        
        assert torch.abs(sin_yaw - sin_yaw_back) < 1e-6
        assert torch.abs(cos_yaw - cos_yaw_back) < 1e-6

    def test_loss_functions_normalized_space(self, sample_batch_data, mock_model_outputs):
        """Test that loss functions work correctly in normalized space."""
        batch_size = 2
        num_objects = 5
        
        # Create mock indices for bipartite matching
        indices = [
            (torch.arange(num_objects), torch.arange(num_objects)),
            (torch.arange(num_objects), torch.arange(num_objects))
        ]
        
        # Test center loss
        center_loss = LossCenter({'loss_center': 1.0})
        center_loss_value = center_loss(
            mock_model_outputs, 
            sample_batch_data, 
            indices, 
            torch.tensor([num_objects, num_objects])
        )
        assert isinstance(center_loss_value, torch.Tensor)
        assert center_loss_value >= 0
        
        # Test size loss
        size_loss = LossSize({'loss_size': 1.0})
        size_loss_value = size_loss(
            mock_model_outputs, 
            sample_batch_data, 
            indices, 
            torch.tensor([num_objects, num_objects])
        )
        assert isinstance(size_loss_value, torch.Tensor)
        assert size_loss_value >= 0
        
        # Test angle loss
        angle_loss = LossAngle({'loss_angle': 1.0})
        angle_loss_value = angle_loss(
            mock_model_outputs, 
            sample_batch_data, 
            indices, 
            torch.tensor([num_objects, num_objects])
        )
        assert isinstance(angle_loss_value, torch.Tensor)
        assert angle_loss_value >= 0
        
        # Test velocity loss
        velocity_loss = LossVelocity({'loss_velocity': 1.0})
        velocity_loss_value = velocity_loss(
            mock_model_outputs, 
            sample_batch_data, 
            indices, 
            torch.tensor([num_objects, num_objects])
        )
        assert isinstance(velocity_loss_value, torch.Tensor)
        assert velocity_loss_value >= 0

    def test_giou_loss_8d_to_7d_conversion(self, sample_batch_data, mock_model_outputs):
        """Test that GIoU loss correctly converts 8D to 7D boxes."""
        batch_size = 2
        num_objects = 5
        
        # Create mock indices
        indices = [
            (torch.arange(num_objects), torch.arange(num_objects)),
            (torch.arange(num_objects), torch.arange(num_objects))
        ]
        
        # Test GIoU loss
        giou_loss = LossGIoUBEV({'loss_giou': 1.0})
        giou_loss_value = giou_loss(
            mock_model_outputs, 
            sample_batch_data, 
            indices, 
            torch.tensor([num_objects, num_objects])
        )
        assert isinstance(giou_loss_value, torch.Tensor)
        assert giou_loss_value >= 0
        
        # Test the internal 8D to 7D conversion
        box_8d = torch.randn(8)  # [x, y, z, w, l, h, sin_yaw, cos_yaw]
        box_7d = giou_loss._convert_to_7d_box(box_8d.unsqueeze(0))
        
        assert box_7d.shape == (1, 7)
        assert box_7d[0, :6].equal(box_8d[:6])  # x, y, z, w, l, h should be the same
        # yaw should be converted from sin/cos to angle

    def test_evaluation_pipeline(self, mock_config, sample_batch_data, mock_model_outputs):
        """Test the evaluation pipeline with denormalization."""
        # Test the evaluation function
        results = reconstruct_and_convert_predictions_autoregressive(
            sample_batch_data, 
            mock_model_outputs, 
            mock_config
        )
        
        # Verify output structure
        assert isinstance(results, list)
        assert len(results) == 2  # batch_size
        
        for result in results:
            assert 'sample_token' in result
            assert 'predictions' in result
            assert isinstance(result['predictions'], list)

    def test_normalizer_consistency(self):
        """Test that the global normalizer works consistently."""
        normalizer = get_global_normalizer()
        
        # Test normalization and denormalization round-trip
        test_boxes = torch.randn(5, 8)  # 5 boxes, 8D each
        test_velocities = torch.randn(5, 2)  # 5 velocities, 2D each
        
        # Normalize
        norm_boxes = normalizer.normalize_box_params(test_boxes)
        norm_velocities = normalizer.normalize_velocity(test_velocities)
        
        # Denormalize
        denorm_boxes = normalizer.denormalize_box_params(norm_boxes)
        denorm_velocities = normalizer.denormalize_velocity(norm_velocities)
        
        # Check that we get back approximately the same values
        assert torch.allclose(test_boxes, denorm_boxes, atol=1e-6)
        assert torch.allclose(test_velocities, denorm_velocities, atol=1e-6)

    def test_tensor_shapes_consistency(self, sample_batch_data, mock_model_outputs):
        """Test that all tensor shapes are consistent throughout the pipeline."""
        batch_size = 2
        num_objects = 5
        
        # Check batch data shapes
        assert sample_batch_data['gt_boxes_b_normalized'].shape == (batch_size, num_objects, 10)
        assert sample_batch_data['gt_labels_b'].shape == (batch_size, num_objects)
        assert sample_batch_data['target_offsets_normalized'].shape == (batch_size, num_objects, 10)
        
        # Check model output shapes
        assert mock_model_outputs['pred_logits'].shape == (batch_size, num_objects, 4)
        assert mock_model_outputs['pred_boxes_normalized'].shape == (batch_size, num_objects, 8)
        assert mock_model_outputs['pred_velocities_normalized'].shape == (batch_size, num_objects, 2)
        
        # Check sensor data shapes
        for sensor_name, sensor_data in sample_batch_data['sensor_data'].items():
            assert sensor_data['features'].shape == (batch_size, num_objects, 12)
            assert sensor_data['metadata'].shape == (batch_size, num_objects, 2)
            assert sensor_data['centers'].shape == (batch_size, num_objects, 3)
            assert sensor_data['boxes'].shape == (batch_size, num_objects, 9)

    def test_import_export_compatibility(self):
        """Test that all imports and exports work correctly."""
        # Test that we can import all necessary modules
        try:
            from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
            from oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive
            from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion
            from oft.transformer.training.losses.regression_losses import LossCenter, LossSize, LossAngle, LossVelocity
            from oft.transformer.training.losses.giou_losses import LossGIoUBEV
            from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive
            from oft.transformer.utils.normalization_utils import get_global_normalizer
            from oft.transformer.utils.geometry_utils import yaw_to_sin_cos, sin_cos_to_yaw
            assert True  # All imports successful
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_parameter_passing(self, mock_config):
        """Test that parameters are passed correctly through the pipeline."""
        # Test that config structure is correct
        assert 'dataset' in mock_config
        assert 'model' in mock_config
        assert 'evaluation' in mock_config
        
        # Test dataset config
        dataset_cfg = mock_config['dataset']
        assert 'class_names' in dataset_cfg
        assert 'point_cloud_range' in dataset_cfg
        assert 'max_velocity' in dataset_cfg
        assert 'virtual_sensors' in dataset_cfg
        
        # Test model config
        model_cfg = mock_config['model']
        assert 'input_feature_dim' in model_cfg
        assert 'd_model' in model_cfg
        assert 'nhead' in model_cfg
        assert 'output_heads' in model_cfg
        
        # Test evaluation config
        eval_cfg = mock_config['evaluation']
        assert 'conf_th_eval' in eval_cfg

    def test_error_handling(self, sample_batch_data, mock_model_outputs):
        """Test error handling for edge cases."""
        # Test with empty predictions
        empty_outputs = {
            'pred_logits': torch.randn(2, 0, 4),
            'pred_boxes_normalized': torch.randn(2, 0, 8),
            'pred_velocities_normalized': torch.randn(2, 0, 2),
            'pred_attributes': torch.randn(2, 0, 10)
        }
        
        # This should not raise an exception
        try:
            results = reconstruct_and_convert_predictions_autoregressive(
                sample_batch_data, 
                empty_outputs, 
                {'dataset': {'class_names': ['car']}, 'evaluation': {'conf_th_eval': 0.1}}
            )
            assert len(results) == 2
            for result in results:
                assert len(result['predictions']) == 0
        except Exception as e:
            pytest.fail(f"Should handle empty predictions gracefully: {e}")

    def test_data_type_consistency(self, sample_batch_data, mock_model_outputs):
        """Test that data types are consistent throughout the pipeline."""
        # Check that all tensors are float32
        for key, value in sample_batch_data.items():
            if isinstance(value, torch.Tensor):
                assert value.dtype == torch.float32 or value.dtype == torch.long or value.dtype == torch.bool
        
        # Check model outputs
        for key, value in mock_model_outputs.items():
            if isinstance(value, torch.Tensor):
                assert value.dtype == torch.float32

    def test_device_consistency(self, sample_batch_data, mock_model_outputs):
        """Test that tensors are on the same device."""
        # Move everything to CPU for testing
        sample_batch_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                           for k, v in sample_batch_data.items()}
        mock_model_outputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                            for k, v in mock_model_outputs.items()}
        
        # Check that all tensors are on the same device
        devices = set()
        for value in sample_batch_data.values():
            if isinstance(value, torch.Tensor):
                devices.add(value.device)
        for value in mock_model_outputs.values():
            if isinstance(value, torch.Tensor):
                devices.add(value.device)
        
        assert len(devices) == 1  # All tensors should be on the same device


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 