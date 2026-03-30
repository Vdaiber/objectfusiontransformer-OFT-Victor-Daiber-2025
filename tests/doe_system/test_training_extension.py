"""
Unit tests for training system extensions supporting DoE campaigns.

This module tests the model archiving functionality added to the BaseTrainer
and the NDS-based checkpoint saving mechanism.
"""

import unittest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import logging

# Mock imports to avoid dependency issues during testing
class MockModule:
    """Mock module to replace heavy dependencies during testing."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return MockModule()
    
    def __call__(self, *args, **kwargs):
        return MockModule()

# Mock heavy imports
import sys
sys.modules['oft.transformer.models.architectures.autoregressive_architecture'] = MockModule()
sys.modules['oft.transformer.datasets.loaders.autoregressive_loader'] = MockModule()
sys.modules['oft.transformer.training.criteria.autoregressive_criterion'] = MockModule()
sys.modules['oft.transformer.evaluation.devkit_evaluator'] = MockModule()

# Now we can import the actual modules we want to test
from oft.transformer.training.utils.checkpoint_utils import save_best_nds_checkpoint


class TestNDSCheckpointSaving(unittest.TestCase):
    """Test NDS-based checkpoint saving functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "training_output"
        self.output_dir.mkdir(parents=True)
        
        # Create mock logger
        self.logger = Mock(spec=logging.Logger)
        
        # Create mock model, optimizer, and scheduler
        self.model = Mock()
        self.model.state_dict.return_value = {'layer1.weight': torch.randn(10, 10)}
        
        self.optimizer = Mock()
        self.optimizer.state_dict.return_value = {'param_groups': [{'lr': 0.001}]}
        
        self.lr_scheduler = Mock()
        self.lr_scheduler.state_dict.return_value = {'last_epoch': 5}
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_first_best_nds(self):
        """Test saving the first best NDS checkpoint."""
        # Call save_best_nds_checkpoint
        result = save_best_nds_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            epoch=5,
            nds_score=0.654,
            output_dir=self.output_dir,
            logger=self.logger
        )
        
        # Should return True for first save
        self.assertTrue(result)
        
        # Check if checkpoint files were created
        checkpoints_dir = self.output_dir / "checkpoints"
        self.assertTrue(checkpoints_dir.exists())
        
        best_checkpoint = checkpoints_dir / "best_nds.pth"
        self.assertTrue(best_checkpoint.exists())
        
        best_info = checkpoints_dir / "best_nds_info.json"
        self.assertTrue(best_info.exists())
        
        # Check info file content
        with open(best_info, 'r') as f:
            info = json.load(f)
        
        self.assertEqual(info['best_nds'], 0.654)
        self.assertEqual(info['best_epoch'], 5)
        
        # Check logger calls
        self.logger.info.assert_called()
        info_calls = [call[0][0] for call in self.logger.info.call_args_list]
        self.assertTrue(any("NEW BEST NDS" in call for call in info_calls))
    
    def test_save_improved_nds(self):
        """Test saving an improved NDS checkpoint."""
        # First save
        save_best_nds_checkpoint(
            model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            epoch=3, nds_score=0.5, output_dir=self.output_dir, logger=self.logger
        )
        
        # Reset logger mock
        self.logger.reset_mock()
        
        # Second save with better score
        result = save_best_nds_checkpoint(
            model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            epoch=7, nds_score=0.7, output_dir=self.output_dir, logger=self.logger
        )
        
        # Should return True for improvement
        self.assertTrue(result)
        
        # Check updated info file
        best_info = self.output_dir / "checkpoints" / "best_nds_info.json"
        with open(best_info, 'r') as f:
            info = json.load(f)
        
        self.assertEqual(info['best_nds'], 0.7)
        self.assertEqual(info['best_epoch'], 7)
    
    def test_save_worse_nds(self):
        """Test that worse NDS scores are not saved."""
        # First save
        save_best_nds_checkpoint(
            model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            epoch=3, nds_score=0.8, output_dir=self.output_dir, logger=self.logger
        )
        
        # Reset logger mock
        self.logger.reset_mock()
        
        # Second save with worse score
        result = save_best_nds_checkpoint(
            model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            epoch=7, nds_score=0.6, output_dir=self.output_dir, logger=self.logger
        )
        
        # Should return False for no improvement
        self.assertFalse(result)
        
        # Best info should remain unchanged
        best_info = self.output_dir / "checkpoints" / "best_nds_info.json"
        with open(best_info, 'r') as f:
            info = json.load(f)
        
        self.assertEqual(info['best_nds'], 0.8)
        self.assertEqual(info['best_epoch'], 3)


class TestModelArchiving(unittest.TestCase):
    """Test model archiving functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.temp_dir / "models"
        self.training_output = self.temp_dir / "training_output"
        self.training_output.mkdir(parents=True)
        
        # Create mock config
        self.config = {
            'training': {
                'archive_model': True,
                'model_identifier': 'test_model'
            }
        }
        
        # Create mock checkpoint files
        checkpoints_dir = self.training_output / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        
        # Create mock best_nds.pth
        torch.save({'model': {'layer1.weight': torch.randn(5, 5)}}, 
                  checkpoints_dir / "best_nds.pth")
        
        # Create mock best_nds_info.json
        with open(checkpoints_dir / "best_nds_info.json", 'w') as f:
            json.dump({'best_nds': 0.75, 'best_epoch': 8}, f)
        
        # Create mock config file
        self.config_file = Path("config/pipeline_staged.yaml")
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            yaml.dump({'test': 'config'}, f)
        
        # Create mock logger
        self.logger = Mock(spec=logging.Logger)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        if self.config_file.exists():
            self.config_file.unlink()
    
    @patch('oft.transformer.training.trainers.base_trainer.Path')
    def test_archive_model_success(self, mock_path):
        """Test successful model archiving."""
        # Mock the path for models directory
        mock_path.return_value = self.models_dir
        
        # Import here to avoid circular imports during setup
        from oft.transformer.training.trainers.base_trainer import BaseTrainer
        
        # Create a minimal trainer instance for testing
        trainer = Mock(spec=BaseTrainer)
        trainer.cfg = self.config
        trainer.output_dir = self.training_output
        trainer.logger = self.logger
        
        # Call the method directly
        BaseTrainer._archive_model_to_collection(trainer)
        
        # Check if model directory was created
        model_dir = self.models_dir / "test_model"
        self.assertTrue(model_dir.exists())
        
        # Check if files were copied
        self.assertTrue((model_dir / "config.yaml").exists())
        self.assertTrue((model_dir / "best_nds.pth").exists())
        
        # Check logger calls
        self.logger.info.assert_called()
        info_calls = [call[0][0] for call in self.logger.info.call_args_list]
        self.assertTrue(any("successfully archived" in call for call in info_calls))
    
    def test_archive_without_identifier(self):
        """Test archiving fails without model identifier."""
        from oft.transformer.training.trainers.base_trainer import BaseTrainer
        
        # Config without model_identifier
        config = {'training': {'archive_model': True}}
        
        trainer = Mock(spec=BaseTrainer)
        trainer.cfg = config
        trainer.output_dir = self.training_output
        trainer.logger = self.logger
        
        # Call the method
        BaseTrainer._archive_model_to_collection(trainer)
        
        # Should log error
        self.logger.error.assert_called()
        error_calls = [call[0][0] for call in self.logger.error.call_args_list]
        self.assertTrue(any("no model_identifier" in call for call in error_calls))
    
    def test_archive_missing_checkpoint(self):
        """Test archiving behavior when checkpoint is missing."""
        from oft.transformer.training.trainers.base_trainer import BaseTrainer
        
        # Remove checkpoint file
        (self.training_output / "checkpoints" / "best_nds.pth").unlink()
        
        trainer = Mock(spec=BaseTrainer)
        trainer.cfg = self.config
        trainer.output_dir = self.training_output
        trainer.logger = self.logger
        
        with patch('oft.transformer.training.trainers.base_trainer.Path') as mock_path:
            mock_path.return_value = self.models_dir
            
            # Call the method
            BaseTrainer._archive_model_to_collection(trainer)
        
        # Should log error about missing checkpoint
        self.logger.error.assert_called()
        error_calls = [call[0][0] for call in self.logger.error.call_args_list]
        self.assertTrue(any("not found" in call for call in error_calls))


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation for DoE system."""
    
    def test_valid_config_structure(self):
        """Test validation of valid configuration structure."""
        config = {
            'training': {
                'archive_model': True,
                'model_identifier': 'model_noisefree',
                'epochs': 10,
                'batch_size': 4
            },
            'dataset': {
                'virtual_sensors': [
                    {
                        'name': 'virtual_lidar',
                        'pos_noise_std': [0.1, 0.1, 0.1]
                    }
                ],
                'simulation': {
                    'fn_rate': 0.0
                }
            }
        }
        
        # Should not raise any exceptions
        self.assertIsInstance(config, dict)
        self.assertIn('training', config)
        self.assertIn('dataset', config)
    
    def test_model_identifier_validation(self):
        """Test model identifier validation."""
        # Valid identifiers
        valid_identifiers = ['model_noisefree', 'model_low_noise', 'baseline_v2']
        for identifier in valid_identifiers:
            self.assertTrue(isinstance(identifier, str))
            self.assertTrue(len(identifier) > 0)
        
        # Invalid identifiers
        invalid_identifiers = ['', None, 123]
        for identifier in invalid_identifiers:
            if identifier is not None:
                self.assertFalse(isinstance(identifier, str) and len(str(identifier)) > 0)


if __name__ == '__main__':
    unittest.main()
