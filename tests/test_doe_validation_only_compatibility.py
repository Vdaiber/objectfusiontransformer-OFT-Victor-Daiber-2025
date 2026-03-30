#!/usr/bin/env python3
"""
Unit test to verify that test_model_on_doe.py can handle validation-only preprocessed datasets.

Tests whether the DOE testing script correctly works with datasets that only have validation
splits (no training data), as created by preprocess_doe_plan.py.
"""

import unittest
import sys
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.oft.transformer.scripts.test_model_on_doe import run_model_evaluation
from src.oft.transformer.datasets.loaders.autoregressive_loader import build_autoregressive_dataloaders


class TestDOEValidationOnlyCompatibility(unittest.TestCase):
    """Test that DOE testing works with validation-only datasets."""
    
    def setUp(self):
        """Set up test configuration for validation-only datasets."""
        self.test_config = {
            'dataset': {
                'dataroot': '/data',
                'version': 'v1.0-mini',
                'virtual_sensors': [
                    {
                        'name': 'virtual_lidar',
                        'enabled': True,
                        'pos_noise_std': [0.46, 0.46, 0.29],
                        'dim_noise_std': 0.27,
                        'yaw_noise_std': 0.22,
                        'velocity_noise_std': 0.16,
                        'dropout_rate': 0.24,
                        'class_accuracy': 0.78,
                        'attribute_accuracy': 0.68
                    }
                ],
                'simulation': {'fn_rate': 0.0, 'num_fps': 0},
                'class_names': ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 
                              'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 
                              'barrier', 'animal', 'traffic_sign'],
                'cutoff_dist': 70.0,
                'normalization_stats_path': '/app/config/normalization_stats.yaml'
            },
            'training': {
                'train_split_name': 'mini_train',
                'val_split_name': 'mini_val',
                'batch_size': 2,
                'num_workers': 1,
                'pin_memory': False,
                'dataset_verbose': False,
                'preprocessed_cache_dir': '/tmp/test_cache'
            },
            'model': {
                'input_feature_dim': 12,
                'd_model': 256,
                'nhead': 8,
                'dropout': 0.1
            },
            'loss': {
                'loss_weight_dict': {'loss_ce': 1.0, 'loss_bbox': 1.0},
                'losses_to_compute': ['labels', 'boxes'],
                'class_eos_coefficient': 0.1
            }
        }
    
    def test_build_dataloaders_validation_only(self):
        """Test that build_autoregressive_dataloaders handles missing training data gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock cache directory with only validation data
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir()
            
            # Create fake validation dataset file (but no training file)
            config_hash = "test_hash_123"
            val_dataset_path = cache_dir / f"config_{config_hash}" / "complete_dataset_val.json"
            val_dataset_path.parent.mkdir(parents=True)
            
            # Create minimal validation dataset
            val_data = {
                "sample_001": {
                    "sample_token": "sample_001",
                    "sensor_data": {"virtual_lidar": {"features": [], "metadata": [], "centers": [], "boxes": []}},
                    "ground_truth": {"normalized": [], "physical": [], "labels": []},
                    "ego_translation_world": [0, 0, 0],
                    "ego_rotation_world_quat": [1, 0, 0, 0],
                    "scene_meta": {},
                    "ego_motion": {"cabin": {"velocity": [0, 0, 0], "acceleration": [0, 0, 0], "angular_velocity": [0, 0, 0]},
                                  "chassis": {"velocity": [0, 0, 0], "acceleration": [0, 0, 0], "angular_velocity": [0, 0, 0]}},
                    "temporal_info": {"next_sample_token": "", "prev_sample_token": "", "scene_token": ""}
                }
            }
            
            with open(val_dataset_path, 'w') as f:
                json.dump(val_data, f)
            
            # Update config to point to test cache
            test_config = self.test_config.copy()
            test_config['training']['preprocessed_cache_dir'] = str(cache_dir)
            
            # Mock the config hash generation to return our test hash
            with patch('oft.transformer.utils.config_hash_utils.generate_unified_config_hash', return_value=config_hash):
                with patch('oft.transformer.datasets.loaders.autoregressive_loader._check_preprocessed_data_exists', return_value=True):
                    with patch('oft.transformer.datasets.loaders.autoregressive_loader.load_complete_dataset_to_ram', return_value=val_data):
                        # Test building dataloaders - should only build validation dataloader
                        import logging
                        logger = logging.getLogger('test')
                        
                        try:
                            # Mock the actual dataset loading to avoid file system dependencies
                            with patch('oft.transformer.datasets.truckscenes.dataset.ObjectFusionGTDatasetStaged') as mock_dataset_class:
                                mock_dataset = MagicMock()
                                mock_dataset.__len__.return_value = 1
                                mock_dataset_class.return_value = mock_dataset
                                
                                # This should work and only return validation dataloader
                                dataloaders = build_autoregressive_dataloaders(
                                    test_config, 
                                    logger, 
                                    splits_to_build=['val']  # Only build validation split
                                )
                            
                            # Should have validation dataloader
                            self.assertIn('val', dataloaders)
                            self.assertIsNotNone(dataloaders['val'])
                            
                            # Should NOT have training dataloader (we didn't request it)
                            self.assertNotIn('train', dataloaders)
                            
                        except Exception as e:
                            self.fail(f"build_autoregressive_dataloaders failed with validation-only data: {e}")
    
    def test_doe_testing_script_compatibility(self):
        """Test that the DOE testing script can handle validation-only datasets."""
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
            yaml.dump(self.test_config, temp_config)
            temp_config_path = temp_config.name
        
        try:
            # Create mock output directory
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir)
                
                # Mock the dataloader building to return only validation dataloader
                mock_val_dataloader = MagicMock()
                mock_val_dataloader.__len__.return_value = 1
                mock_val_dataloader.__iter__.return_value = iter([{
                    'sample_tokens': ['test_token'],
                    'sensor_data': {'virtual_lidar': {'features': torch.zeros(1, 0, 12), 'mask': torch.ones(1, 0)}},
                    'gt_boxes_b_normalized': torch.zeros(1, 0, 10),
                    'gt_labels_b': torch.zeros(1, 0, dtype=torch.long),
                    'gt_valid_mask_b': torch.zeros(1, 0, dtype=torch.bool)
                }])
                
                mock_dataloaders = {'val': mock_val_dataloader}
                
                with patch('src.oft.transformer.scripts.test_model_on_doe.build_autoregressive_dataloaders', return_value=mock_dataloaders):
                    with patch('src.oft.transformer.scripts.test_model_on_doe.run_devkit_evaluation', return_value=0.25):
                        
                        import logging
                        logger = logging.getLogger('test')
                        device = torch.device('cpu')
                        
                        # This should work without errors
                        try:
                            nds_score = run_model_evaluation(
                                mock_model, 
                                self.test_config, 
                                temp_config_path, 
                                output_path, 
                                device, 
                                logger
                            )
                            
                            # Should return a valid NDS score
                            self.assertIsNotNone(nds_score)
                            self.assertIsInstance(nds_score, (int, float))
                            
                        except Exception as e:
                            self.fail(f"run_model_evaluation failed with validation-only data: {e}")
        
        finally:
            # Clean up temp file
            Path(temp_config_path).unlink()
    
    def test_dataloader_builder_splits_parameter(self):
        """Test that build_autoregressive_dataloaders respects splits_to_build parameter."""
        
        # Mock the smart loading to return validation data
        mock_val_data = {"sample_001": {"sample_token": "sample_001"}}
        
        with patch('oft.transformer.datasets.loaders.autoregressive_loader._smart_load_preprocessed_data', return_value=mock_val_data):
            with patch('oft.transformer.datasets.truckscenes.dataset.ObjectFusionGTDatasetStaged') as mock_dataset:
                mock_dataset.return_value.__len__.return_value = 1
                
                import logging
                logger = logging.getLogger('test')
                
                # Test building only validation split
                dataloaders = build_autoregressive_dataloaders(
                    self.test_config,
                    logger,
                    splits_to_build=['val']  # Only validation
                )
                
                # Should only contain validation dataloader
                self.assertEqual(len(dataloaders), 1)
                self.assertIn('val', dataloaders)
                self.assertNotIn('train', dataloaders)
    
    def test_missing_training_data_no_error(self):
        """Test that missing training data doesn't cause errors when only validation is needed."""
        
        # Configure for validation-only mode
        test_config = self.test_config.copy()
        
        # Mock that only validation data exists
        with patch('oft.transformer.datasets.loaders.autoregressive_loader._smart_load_preprocessed_data') as mock_load:
            # Return None for training (missing), mock data for validation
            def side_effect(cfg, split_name, logger):
                if 'train' in split_name:
                    return None  # Training data missing
                else:
                    return {"sample_001": {"sample_token": "sample_001"}}  # Validation data exists
            
            mock_load.side_effect = side_effect
            
            with patch('oft.transformer.datasets.truckscenes.dataset.ObjectFusionGTDatasetStaged') as mock_dataset:
                mock_dataset.return_value.__len__.return_value = 1
                
                import logging
                logger = logging.getLogger('test')
                
                # This should work fine - only building validation
                try:
                    dataloaders = build_autoregressive_dataloaders(
                        test_config,
                        logger,
                        splits_to_build=['val']
                    )
                    
                    self.assertIn('val', dataloaders)
                    
                except Exception as e:
                    self.fail(f"Validation-only dataloader building failed: {e}")


def run_tests():
    """Run all compatibility tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
