"""
Unit tests for DoE preprocessing functionality.

This module tests the DoE plan preprocessing script including parameter calculation,
config hash generation, and cache management.
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

sys.modules['oft.transformer.utils.config_hash_utils'] = MockModule()


class TestDoeParameterCalculation(unittest.TestCase):
    """Test DoE parameter calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            'dataset': {
                'virtual_sensors': [
                    {
                        'name': 'virtual_lidar',
                        'pos_noise_std': [0.5, 0.5, 0.3],
                        'enabled': True
                    },
                    {
                        'name': 'virtual_radar',
                        'pos_noise_std': [1.0, 1.0, 2.0],
                        'enabled': True
                    },
                    {
                        'name': 'virtual_camera',
                        'pos_noise_std': [0.8, 0.8, 1.2],
                        'enabled': True
                    }
                ],
                'simulation': {
                    'fn_rate': 0.1,
                    'num_fps': 5
                }
            }
        }
    
    def test_parameter_calculation_basic(self):
        """Test basic parameter calculation with multipliers."""
        # Import the function we want to test
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        multipliers = {
            'virtual_lidar_pos_noise_std_multiplier': 1.5,
            'virtual_radar_pos_noise_std_multiplier': 2.0,
            'fn_rate_multiplier': 0.5
        }
        
        result_config = doe_script.calculate_final_parameters(self.base_config, multipliers)
        
        # Check LiDAR noise was multiplied
        lidar_sensor = next(s for s in result_config['dataset']['virtual_sensors'] if 'lidar' in s['name'])
        expected_lidar_noise = [0.5 * 1.5, 0.5 * 1.5, 0.3 * 1.5]
        self.assertEqual(lidar_sensor['pos_noise_std'], expected_lidar_noise)
        
        # Check radar noise was multiplied
        radar_sensor = next(s for s in result_config['dataset']['virtual_sensors'] if 'radar' in s['name'])
        expected_radar_noise = [1.0 * 2.0, 1.0 * 2.0, 2.0 * 2.0]
        self.assertEqual(radar_sensor['pos_noise_std'], expected_radar_noise)
        
        # Check FN rate was multiplied
        expected_fn_rate = 0.1 * 0.5
        self.assertEqual(result_config['dataset']['simulation']['fn_rate'], expected_fn_rate)
    
    def test_parameter_calculation_selective(self):
        """Test parameter calculation with only some multipliers."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        multipliers = {
            'virtual_lidar_pos_noise_std_multiplier': 2.0
            # No other multipliers
        }
        
        result_config = doe_script.calculate_final_parameters(self.base_config, multipliers)
        
        # Check only LiDAR was modified
        lidar_sensor = next(s for s in result_config['dataset']['virtual_sensors'] if 'lidar' in s['name'])
        expected_lidar_noise = [0.5 * 2.0, 0.5 * 2.0, 0.3 * 2.0]
        self.assertEqual(lidar_sensor['pos_noise_std'], expected_lidar_noise)
        
        # Check radar was not modified
        radar_sensor = next(s for s in result_config['dataset']['virtual_sensors'] if 'radar' in s['name'])
        self.assertEqual(radar_sensor['pos_noise_std'], [1.0, 1.0, 2.0])
        
        # Check FN rate was not modified
        self.assertEqual(result_config['dataset']['simulation']['fn_rate'], 0.1)
    
    def test_parameter_calculation_no_multipliers(self):
        """Test parameter calculation with no multipliers."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        multipliers = {}
        
        result_config = doe_script.calculate_final_parameters(self.base_config, multipliers)
        
        # Config should be unchanged
        self.assertEqual(result_config['dataset']['virtual_sensors'], self.base_config['dataset']['virtual_sensors'])
        self.assertEqual(result_config['dataset']['simulation'], self.base_config['dataset']['simulation'])


class TestCacheManagement(unittest.TestCase):
    """Test cache management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_root = self.temp_dir / "cache"
        self.cache_root.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_exists_complete(self):
        """Test cache existence check when all files exist."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        config_hash = "abc123def456"
        cache_dir = self.cache_root / f"config_{config_hash}"
        cache_dir.mkdir()
        
        # Create required files
        (cache_dir / "complete_dataset_val.json").touch()
        baseline_dir = cache_dir / "baseline_evaluation_results"
        baseline_dir.mkdir()
        
        result = doe_script.check_cache_exists(config_hash, str(self.cache_root))
        self.assertTrue(result)
    
    def test_cache_exists_missing_data(self):
        """Test cache existence check when data file is missing."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        config_hash = "abc123def456"
        cache_dir = self.cache_root / f"config_{config_hash}"
        cache_dir.mkdir()
        
        # Only create baseline directory, not data file
        baseline_dir = cache_dir / "baseline_evaluation_results"
        baseline_dir.mkdir()
        
        result = doe_script.check_cache_exists(config_hash, str(self.cache_root))
        self.assertFalse(result)
    
    def test_cache_exists_missing_baseline(self):
        """Test cache existence check when baseline directory is missing."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        config_hash = "abc123def456"
        cache_dir = self.cache_root / f"config_{config_hash}"
        cache_dir.mkdir()
        
        # Only create data file, not baseline directory
        (cache_dir / "complete_dataset_val.json").touch()
        
        result = doe_script.check_cache_exists(config_hash, str(self.cache_root))
        self.assertFalse(result)
    
    def test_cache_exists_no_directory(self):
        """Test cache existence check when cache directory doesn't exist."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        config_hash = "nonexistent"
        
        result = doe_script.check_cache_exists(config_hash, str(self.cache_root))
        self.assertFalse(result)


class TestDoeDataLoading(unittest.TestCase):
    """Test DoE data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_doe_plan(self):
        """Test loading a valid DoE plan CSV."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        # Create test CSV
        csv_data = {
            'experiment_id': [1, 2, 3],
            'virtual_lidar_pos_noise_std_multiplier': [1.0, 1.5, 2.0],
            'virtual_radar_pos_noise_std_multiplier': [1.0, 1.0, 1.2],
            'fn_rate_multiplier': [1.0, 0.8, 1.2]
        }
        
        df = pd.DataFrame(csv_data)
        csv_path = self.temp_dir / "test_plan.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and test
        result_df = doe_script.load_doe_plan(str(csv_path))
        
        self.assertEqual(len(result_df), 3)
        self.assertTrue('experiment_id' in result_df.columns)
        self.assertTrue('virtual_lidar_pos_noise_std_multiplier' in result_df.columns)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent DoE plan file."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        with self.assertRaises(FileNotFoundError):
            doe_script.load_doe_plan(str(self.temp_dir / "nonexistent.csv"))
    
    def test_load_empty_csv(self):
        """Test loading an empty CSV file."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        # Create empty CSV
        csv_path = self.temp_dir / "empty.csv"
        pd.DataFrame().to_csv(csv_path, index=False)
        
        with self.assertRaises(ValueError):
            doe_script.load_doe_plan(str(csv_path))


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        # Create test config
        config_data = {
            'dataset': {
                'virtual_sensors': [{'name': 'test'}],
                'simulation': {'fn_rate': 0.1}
            },
            'preprocessing': {
                'cache_dir': '/test/cache'
            }
        }
        
        config_path = self.temp_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and test
        result_config = doe_script.load_base_config(str(config_path))
        
        self.assertEqual(result_config['dataset']['simulation']['fn_rate'], 0.1)
        self.assertEqual(result_config['preprocessing']['cache_dir'], '/test/cache')
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent config file."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        with self.assertRaises(FileNotFoundError):
            doe_script.load_base_config(str(self.temp_dir / "nonexistent.yaml"))
    
    def test_load_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        # Create invalid YAML
        config_path = self.temp_dir / "invalid.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        with self.assertRaises(yaml.YAMLError):
            doe_script.load_base_config(str(config_path))


class TestIntegrationMocks(unittest.TestCase):
    """Test integration with mocked external dependencies."""
    
    @patch('subprocess.run')
    def test_preprocessing_execution_success(self, mock_subprocess):
        """Test successful preprocessing execution."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        logger = Mock()
        config_dict = {'test': 'config'}
        temp_path = "/tmp/test_config.yaml"
        
        result = doe_script.run_preprocessing(config_dict, temp_path, logger)
        
        self.assertTrue(result)
        mock_subprocess.assert_called_once()
        logger.info.assert_called()
    
    @patch('subprocess.run')
    def test_preprocessing_execution_failure(self, mock_subprocess):
        """Test failed preprocessing execution."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        # Mock failed subprocess execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_subprocess.return_value = mock_result
        
        logger = Mock()
        config_dict = {'test': 'config'}
        temp_path = "/tmp/test_config.yaml"
        
        result = doe_script.run_preprocessing(config_dict, temp_path, logger)
        
        self.assertFalse(result)
        logger.error.assert_called()
    
    @patch('subprocess.run')
    def test_baseline_evaluation_success(self, mock_subprocess):
        """Test successful baseline evaluation execution."""
        import src.oft.transformer.scripts.preprocess_doe_plan as doe_script
        
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Evaluation complete"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        logger = Mock()
        config_dict = {'test': 'config'}
        temp_path = "/tmp/test_config.yaml"
        
        result = doe_script.run_baseline_evaluation(config_dict, temp_path, logger)
        
        self.assertTrue(result)
        mock_subprocess.assert_called_once()
        logger.info.assert_called()


if __name__ == '__main__':
    unittest.main()
