"""
Unit tests for DoE model testing functionality.

This module tests the model testing script including model loading,
evaluation execution, and results generation with JSON references.


"""

import unittest
import tempfile
import shutil
import json
import yaml
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Mock heavy dependencies
class MockModule:
    """Mock module to replace heavy dependencies during testing."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return MockModule()
    
    def __call__(self, *args, **kwargs):
        return MockModule()

sys.modules['torch'] = MockModule()
sys.modules['oft.transformer.models.architectures.autoregressive_architecture'] = MockModule()
sys.modules['oft.transformer.datasets.loaders.autoregressive_loader'] = MockModule()
sys.modules['oft.transformer.training.criteria.autoregressive_criterion'] = MockModule()
sys.modules['oft.transformer.evaluation.devkit_evaluator'] = MockModule()
sys.modules['oft.transformer.training.utils.checkpoint_utils'] = MockModule()


class TestModelArchiveLoading(unittest.TestCase):
    """Test model archive loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.models_root = self.temp_dir / "models"
        self.models_root.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_existing_model(self):
        """Test loading an existing model from archive."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        # Create mock model archive
        model_name = "test_model"
        model_dir = self.models_root / model_name
        model_dir.mkdir()
        
        # Create mock config
        config_data = {
            'model': {'d_model': 512},
            'dataset': {'virtual_sensors': []},
            'preprocessing': {'cache_dir': '/test/cache'}
        }
        with open(model_dir / "config.yaml", 'w') as f:
            yaml.dump(config_data, f)
        
        # Create mock checkpoint (we'll mock the actual loading)
        (model_dir / "best_nds.pth").touch()
        
        logger = Mock()
        
        # Mock device and torch operations
        with patch('src.oft.transformer.scripts.test_model_on_doe.torch') as mock_torch:
            mock_device = Mock()
            mock_torch.device.return_value = mock_device
            
            # Mock model loading
            mock_model = Mock()
            mock_checkpoint = {
                'model': {'layer1.weight': 'mock_weights'},
                'nds_score': 0.75,
                'epoch': 10
            }
            mock_torch.load.return_value = mock_checkpoint
            
            with patch('src.oft.transformer.scripts.test_model_on_doe.ObjectFusionTransformerAutoregressive') as mock_model_class:
                mock_model_class.return_value = mock_model
                
                # Test model loading
                with patch('src.oft.transformer.scripts.test_model_on_doe.Path') as mock_path:
                    # Mock the models root path
                    mock_path.return_value = self.models_root
                    
                    config, model = testing_script.load_model_from_archive(model_name, mock_device, logger)
        
        # Verify results
        self.assertEqual(config['model']['d_model'], 512)
        self.assertIsNotNone(model)
        mock_model.load_state_dict.assert_called_once()
        mock_model.eval.assert_called_once()
    
    def test_load_nonexistent_model(self):
        """Test loading a non-existent model."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        logger = Mock()
        mock_device = Mock()
        
        with patch('src.oft.transformer.scripts.test_model_on_doe.Path') as mock_path:
            mock_path.return_value = self.models_root
            
            with self.assertRaises(FileNotFoundError):
                testing_script.load_model_from_archive("nonexistent_model", mock_device, logger)
    
    def test_load_model_missing_config(self):
        """Test loading a model with missing config file."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        # Create model directory without config
        model_name = "incomplete_model"
        model_dir = self.models_root / model_name
        model_dir.mkdir()
        (model_dir / "best_nds.pth").touch()
        
        logger = Mock()
        mock_device = Mock()
        
        with patch('src.oft.transformer.scripts.test_model_on_doe.Path') as mock_path:
            mock_path.return_value = self.models_root
            
            with self.assertRaises(FileNotFoundError):
                testing_script.load_model_from_archive(model_name, mock_device, logger)


class TestDataReferencing(unittest.TestCase):
    """Test data referencing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_root = self.temp_dir / "cache"
        self.cache_root.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_check_preprocessed_data_exists(self):
        """Test checking for existing preprocessed data."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        config_hash = "abc123def456"
        cache_dir = self.cache_root / f"config_{config_hash}"
        cache_dir.mkdir()
        
        # Create required files
        (cache_dir / "complete_dataset_val.json").touch()
        baseline_dir = cache_dir / "baseline_evaluation_results"
        baseline_dir.mkdir()
        
        result = testing_script.check_preprocessed_data_exists(config_hash, str(self.cache_root))
        self.assertTrue(result)
    
    def test_check_preprocessed_data_missing(self):
        """Test checking for missing preprocessed data."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        config_hash = "nonexistent"
        
        result = testing_script.check_preprocessed_data_exists(config_hash, str(self.cache_root))
        self.assertFalse(result)
    
    def test_create_data_reference(self):
        """Test creating data reference JSON."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        config_hash = "abc123def456"
        
        reference = testing_script.create_data_reference(config_hash, str(self.cache_root))
        
        self.assertEqual(reference['config_hash'], config_hash)
        self.assertTrue(reference['preprocessed_data_path'].endswith('complete_dataset_val.json'))
        self.assertTrue(reference['baseline_results_path'].endswith('baseline_evaluation_results'))


class TestParameterCalculation(unittest.TestCase):
    """Test parameter calculation for consistency with preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            'dataset': {
                'virtual_sensors': [
                    {
                        'name': 'virtual_lidar',
                        'pos_noise_std': [0.5, 0.5, 0.3]
                    },
                    {
                        'name': 'virtual_radar',
                        'pos_noise_std': [1.0, 1.0, 2.0]
                    }
                ],
                'simulation': {
                    'fn_rate': 0.1
                }
            }
        }
    
    def test_parameter_calculation_consistency(self):
        """Test that parameter calculation is consistent with preprocessing."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        multipliers = {
            'virtual_lidar_pos_noise_std_multiplier': 1.5,
            'virtual_radar_pos_noise_std_multiplier': 2.0,
            'fn_rate_multiplier': 0.8
        }
        
        result_config = testing_script.calculate_final_parameters(self.base_config, multipliers)
        
        # Check LiDAR parameters
        lidar_sensor = next(s for s in result_config['dataset']['virtual_sensors'] if 'lidar' in s['name'])
        expected_lidar = [0.5 * 1.5, 0.5 * 1.5, 0.3 * 1.5]
        self.assertEqual(lidar_sensor['pos_noise_std'], expected_lidar)
        
        # Check radar parameters
        radar_sensor = next(s for s in result_config['dataset']['virtual_sensors'] if 'radar' in s['name'])
        expected_radar = [1.0 * 2.0, 1.0 * 2.0, 2.0 * 2.0]
        self.assertEqual(radar_sensor['pos_noise_std'], expected_radar)
        
        # Check FN rate
        expected_fn_rate = 0.1 * 0.8
        self.assertEqual(result_config['dataset']['simulation']['fn_rate'], expected_fn_rate)


class TestResultsGeneration(unittest.TestCase):
    """Test results generation and summary functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results_dir = self.temp_dir / "evaluation_campaign_results"
        self.results_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_experiment_directory_creation(self):
        """Test creation of experiment directory structure."""
        model_name = "test_model"
        run_name = "test_run"
        
        campaign_dir = self.results_dir / model_name / run_name
        campaign_dir.mkdir(parents=True)
        
        # Create experiment directories
        for i in range(1, 4):
            experiment_dir = campaign_dir / f"experiment_{i:03d}"
            experiment_dir.mkdir()
            
            # Create required files
            with open(experiment_dir / "1_config.yaml", 'w') as f:
                yaml.dump({'test': f'config_{i}'}, f)
            
            with open(experiment_dir / "2_data_reference.json", 'w') as f:
                json.dump({'config_hash': f'hash_{i}'}, f)
        
        # Verify structure
        self.assertTrue(campaign_dir.exists())
        self.assertTrue((campaign_dir / "experiment_001").exists())
        self.assertTrue((campaign_dir / "experiment_002").exists())
        self.assertTrue((campaign_dir / "experiment_003").exists())
    
    def test_summary_results_generation(self):
        """Test generation of summary results CSV."""
        # Create mock summary data
        summary_data = [
            {'experiment_name': 'experiment_001', 'nds_score': 0.65},
            {'experiment_name': 'experiment_002', 'nds_score': 0.72},
            {'experiment_name': 'experiment_003', 'nds_score': 0.68}
        ]
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.temp_dir / "_summary_results.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Verify file creation and content
        self.assertTrue(summary_path.exists())
        
        # Read back and verify
        loaded_df = pd.read_csv(summary_path)
        self.assertEqual(len(loaded_df), 3)
        self.assertTrue('experiment_name' in loaded_df.columns)
        self.assertTrue('nds_score' in loaded_df.columns)
        self.assertEqual(loaded_df['nds_score'].max(), 0.72)


class TestModelEvaluationMocking(unittest.TestCase):
    """Test model evaluation with mocked dependencies."""
    
    @patch('src.oft.transformer.scripts.test_model_on_doe.run_devkit_evaluation')
    @patch('src.oft.transformer.scripts.test_model_on_doe.build_autoregressive_dataloaders')
    @patch('src.oft.transformer.scripts.test_model_on_doe.SetCriterion')
    def test_model_evaluation_success(self, mock_criterion, mock_dataloaders, mock_devkit):
        """Test successful model evaluation."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        # Setup mocks
        mock_model = Mock()
        mock_model.eval.return_value = None
        
        mock_dataloader = Mock()
        mock_batch = {
            'input_data': Mock(),
            'targets': Mock()
        }
        mock_dataloader.__iter__.return_value = iter([mock_batch])
        mock_dataloaders.return_value = {'val': mock_dataloader}
        
        mock_eval_result = Mock()
        mock_eval_result.metrics.nd_score = 0.75
        mock_devkit.return_value = mock_eval_result
        
        # Test data
        config_dict = {
            'loss': {
                'loss_weight_dict': {'loss_class': 1.0},
                'losses_to_compute': ['class'],
                'class_eos_coefficient': 0.1
            }
        }
        
        temp_config_path = "/tmp/test_config.yaml"
        output_dir = Path(tempfile.mkdtemp())
        device = Mock()
        logger = Mock()
        
        try:
            # Execute evaluation
            nds_score = testing_script.run_model_evaluation(
                mock_model, config_dict, temp_config_path, output_dir, device, logger
            )
            
            # Verify results
            self.assertEqual(nds_score, 0.75)
            mock_model.eval.assert_called_once()
            mock_devkit.assert_called_once()
            
            # Check if results file was created
            results_dir = output_dir / "4_fusion_model_evaluation_results"
            final_metrics_file = results_dir / "final_metrics.json"
            self.assertTrue(final_metrics_file.exists())
            
            # Verify metrics content
            with open(final_metrics_file, 'r') as f:
                metrics = json.load(f)
            self.assertEqual(metrics['nds_score'], 0.75)
            
        finally:
            shutil.rmtree(output_dir)
    
    @patch('src.oft.transformer.scripts.test_model_on_doe.run_devkit_evaluation')
    def test_model_evaluation_failure(self, mock_devkit):
        """Test model evaluation failure handling."""
        import src.oft.transformer.scripts.test_model_on_doe as testing_script
        
        # Mock failure
        mock_devkit.return_value = None
        
        mock_model = Mock()
        config_dict = {'loss': {}}
        temp_config_path = "/tmp/test_config.yaml"
        output_dir = Path(tempfile.mkdtemp())
        device = Mock()
        logger = Mock()
        
        try:
            # Execute evaluation (should fail gracefully)
            nds_score = testing_script.run_model_evaluation(
                mock_model, config_dict, temp_config_path, output_dir, device, logger
            )
            
            # Should return None on failure
            self.assertIsNone(nds_score)
            logger.error.assert_called()
            
        finally:
            shutil.rmtree(output_dir)


if __name__ == '__main__':
    unittest.main()
