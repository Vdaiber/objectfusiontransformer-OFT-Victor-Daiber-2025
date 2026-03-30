import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import yaml
import json

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.oft.transformer.scripts import test_model_on_doe
# Import the correct function for hash calculation
from src.oft.transformer.scripts.preprocess_doe_plan import calculate_final_parameters as calculate_final_parameters_for_hash
from src.oft.transformer.scripts.preprocess_doe_plan import convert_csv_columns_to_parameter_names


class TestDoeResultStructure(unittest.TestCase):
    """
    Unit test for the DoE model testing script.
    Focuses on verifying the final directory and file structure without
    running the actual model evaluation.
    """

    def setUp(self):
        """Set up a temporary directory structure for testing."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create temporary directories for inputs and outputs
        self.cache_dir = Path(self.test_dir) / "virtual_sensor_cache"
        self.models_dir = Path(self.test_dir) / "models"
        self.results_dir = Path(self.test_dir) / "evaluation_campaign_results"
        
        self.cache_dir.mkdir()
        self.models_dir.mkdir()
        self.results_dir.mkdir()
        
        self.model_name = "test_model"
        self.run_name = "test_run"
        self.doe_plan_path = Path(self.test_dir) / "test_doe_plan.csv"

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def _create_mock_inputs(self):
        """Create mock input files needed for the script to run."""
        
        # 1. Mock DoE Plan CSV
        doe_plan_data = {
            'Nr': [1, 2],
            'lid_pos': [1.0, 0.5],
            'cam_pos': [1.0, 1.5]
        }
        pd.DataFrame(doe_plan_data).to_csv(self.doe_plan_path, index=False)

        # 2. Mock Model Archive
        model_archive_dir = self.models_dir / self.model_name
        model_archive_dir.mkdir()
        
        mock_model_config = {
            'preprocessing': {'cache_dir': str(self.cache_dir)},
            'dataset': {
                'version': 'v1.0-mini',
                'simulation': {},
                'normalization_stats_path': '',
                'virtual_sensors': [
                    {
                        'name': 'virtual_lidar',
                        'pos_noise_std': [0.1, 0.1, 0.1]
                    },
                    {
                        'name': 'virtual_camera',
                        'pos_noise_std': [0.2, 0.2, 0.2]
                    }
                ]
            }
        }
        with open(model_archive_dir / "config.yaml", 'w') as f:
            yaml.dump(mock_model_config, f)
        
        # Create a dummy checkpoint file
        (model_archive_dir / "best_nds.pth").touch()

        # 3. Mock Preprocessed Data in Cache
        # We need to manually calculate the hashes that the script will generate
        from src.oft.transformer.utils.config_hash_utils import generate_unified_config_hash
        
        # Hash for experiment 1
        multipliers1 = convert_csv_columns_to_parameter_names(pd.Series({'lid_pos': 1.0, 'cam_pos': 1.0}))
        config1 = calculate_final_parameters_for_hash(mock_model_config, multipliers1)
        hash1 = generate_unified_config_hash(config1)
        cache_dir1 = self.cache_dir / f"config_{hash1}"
        cache_dir1.mkdir()
        (cache_dir1 / "complete_dataset_val.json").touch()
        (cache_dir1 / "baseline_evaluation_results").mkdir()

        # Hash for experiment 2
        multipliers2 = convert_csv_columns_to_parameter_names(pd.Series({'lid_pos': 0.5, 'cam_pos': 1.5}))
        config2 = calculate_final_parameters_for_hash(mock_model_config, multipliers2)
        hash2 = generate_unified_config_hash(config2)
        cache_dir2 = self.cache_dir / f"config_{hash2}"
        cache_dir2.mkdir()
        (cache_dir2 / "complete_dataset_val.json").touch()
        (cache_dir2 / "baseline_evaluation_results").mkdir()
        
        return hash1, hash2

    @patch('src.oft.transformer.scripts.test_model_on_doe.load_model_from_archive')
    @patch('src.oft.transformer.scripts.test_model_on_doe.run_model_evaluation')
    def test_directory_structure_creation(self, mock_run_evaluation, mock_load_model):
        """
        Test if the script creates the expected directory and file structure.
        """
        # --- Setup Mocks ---
        hash1, hash2 = self._create_mock_inputs()
        
        # Mock model loading to return the config and a dummy model object
        mock_model_config_loaded = {
            'preprocessing': {'cache_dir': str(self.cache_dir)},
            'dataset': {
                'version': 'v1.0-mini',
                'simulation': {},
                'normalization_stats_path': '',
                'virtual_sensors': [
                    {
                        'name': 'virtual_lidar',
                        'pos_noise_std': [0.1, 0.1, 0.1]
                    },
                    {
                        'name': 'virtual_camera',
                        'pos_noise_std': [0.2, 0.2, 0.2]
                    }
                ]
            }
        }
        mock_load_model.return_value = (mock_model_config_loaded, MagicMock())

        # Mock model evaluation to simulate success and return a dummy NDS score
        mock_run_evaluation.side_effect = lambda model, config, path, out_dir, dev, log: (out_dir / "4_fusion_model_evaluation_results").mkdir(exist_ok=True) or 0.75

        # --- Run the Script ---
        # Use sys.argv patching to simulate command line arguments
        test_args = [
            "test_model_on_doe.py",
            "--model-name", self.model_name,
            "--doe-plan-csv", str(self.doe_plan_path),
            "--run-name", self.run_name,
            "--models-root", str(self.models_dir),
            "--results-root", str(self.results_dir)
        ]
        
        with patch.object(sys, 'argv', test_args):
            test_model_on_doe.main()


        # --- Assertions ---
        campaign_dir = self.results_dir / self.model_name / self.run_name
        self.assertTrue(campaign_dir.exists())

        # 1. Check summary file
        summary_csv = campaign_dir / "_summary_results.csv"
        self.assertTrue(summary_csv.exists())
        df = pd.read_csv(summary_csv)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['experiment_id', 'experiment_name', 'nds_score'])
        self.assertEqual(df.iloc[0]['experiment_name'], 'experiment_001')
        self.assertAlmostEqual(df.iloc[0]['nds_score'], 0.75)

        # 2. Check experiment directories
        exp1_dir = campaign_dir / "experiment_001"
        exp2_dir = campaign_dir / "experiment_002"
        self.assertTrue(exp1_dir.exists())
        self.assertTrue(exp2_dir.exists())

        # 3. Check contents of experiment 1
        self.assertTrue((exp1_dir / "1_config.yaml").exists())
        ref_json_path = exp1_dir / "2_data_reference.json"
        self.assertTrue(ref_json_path.exists())
        with open(ref_json_path, 'r') as f:
            ref_data = json.load(f)
            self.assertEqual(ref_data['config_hash'], hash1)
            self.assertEqual(ref_data['preprocessed_data_path'], str(self.cache_dir / f"config_{hash1}" / "complete_dataset_val.json"))
        
        self.assertTrue((exp1_dir / "4_fusion_model_evaluation_results").exists())

        # 4. Check contents of experiment 2
        ref_json_path_2 = exp2_dir / "2_data_reference.json"
        with open(ref_json_path_2, 'r') as f:
            ref_data = json.load(f)
            self.assertEqual(ref_data['config_hash'], hash2)

if __name__ == '__main__':
    unittest.main()
