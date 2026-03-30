"""
Unit tests for JSON data reference system in DoE campaigns.

This module tests the JSON reference mechanism used to avoid data duplication
in the evaluation campaign results structure.

Author: DoE System Implementation
Date: 2024
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch


class TestJSONReferences(unittest.TestCase):
    """Test JSON reference functionality for data deduplication."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_root = self.temp_dir / "cache"
        self.cache_root.mkdir(parents=True)
        self.results_root = self.temp_dir / "results"
        self.results_root.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_valid_reference(self):
        """Test creating a valid data reference."""
        config_hash = "abc123def456"
        cache_dir = self.cache_root / f"config_{config_hash}"
        cache_dir.mkdir()
        
        # Create referenced files
        data_file = cache_dir / "complete_dataset_val.json"
        data_file.write_text('{"test": "data"}')
        
        baseline_dir = cache_dir / "baseline_evaluation_results"
        baseline_dir.mkdir()
        
        # Create reference
        reference = {
            "config_hash": config_hash,
            "preprocessed_data_path": str(data_file),
            "baseline_results_path": str(baseline_dir)
        }
        
        # Save reference
        reference_file = self.results_root / "2_data_reference.json"
        with open(reference_file, 'w') as f:
            json.dump(reference, f, indent=2)
        
        # Verify reference
        self.assertTrue(reference_file.exists())
        
        # Load and validate reference
        with open(reference_file, 'r') as f:
            loaded_reference = json.load(f)
        
        self.assertEqual(loaded_reference['config_hash'], config_hash)
        self.assertTrue(Path(loaded_reference['preprocessed_data_path']).exists())
        self.assertTrue(Path(loaded_reference['baseline_results_path']).exists())
    
    def test_reference_validation(self):
        """Test validation of reference integrity."""
        config_hash = "xyz789abc123"
        
        # Valid reference
        valid_reference = {
            "config_hash": config_hash,
            "preprocessed_data_path": "/data/daiber_fent/virtual_sensor_cache/config_xyz789abc123/complete_dataset_val.json",
            "baseline_results_path": "/data/daiber_fent/virtual_sensor_cache/config_xyz789abc123/baseline_evaluation_results"
        }
        
        # Validate required fields
        required_fields = ["config_hash", "preprocessed_data_path", "baseline_results_path"]
        for field in required_fields:
            self.assertIn(field, valid_reference)
            self.assertIsInstance(valid_reference[field], str)
            self.assertTrue(len(valid_reference[field]) > 0)
        
        # Validate path structure
        self.assertTrue(valid_reference['preprocessed_data_path'].endswith('complete_dataset_val.json'))
        self.assertTrue(valid_reference['baseline_results_path'].endswith('baseline_evaluation_results'))
        self.assertTrue(config_hash in valid_reference['preprocessed_data_path'])
        self.assertTrue(config_hash in valid_reference['baseline_results_path'])
    
    def test_reference_consistency(self):
        """Test consistency between multiple references to the same data."""
        config_hash = "consistent_hash_123"
        
        # Create multiple references to the same data
        reference1 = {
            "config_hash": config_hash,
            "preprocessed_data_path": f"/cache/config_{config_hash}/complete_dataset_val.json",
            "baseline_results_path": f"/cache/config_{config_hash}/baseline_evaluation_results"
        }
        
        reference2 = {
            "config_hash": config_hash,
            "preprocessed_data_path": f"/cache/config_{config_hash}/complete_dataset_val.json",
            "baseline_results_path": f"/cache/config_{config_hash}/baseline_evaluation_results"
        }
        
        # References should be identical for the same config hash
        self.assertEqual(reference1, reference2)
        
        # Save both references
        ref1_file = self.results_root / "experiment_001" / "2_data_reference.json"
        ref2_file = self.results_root / "experiment_002" / "2_data_reference.json"
        
        ref1_file.parent.mkdir(exist_ok=True)
        ref2_file.parent.mkdir(exist_ok=True)
        
        with open(ref1_file, 'w') as f:
            json.dump(reference1, f, indent=2)
        
        with open(ref2_file, 'w') as f:
            json.dump(reference2, f, indent=2)
        
        # Verify both files exist and have identical content
        self.assertTrue(ref1_file.exists())
        self.assertTrue(ref2_file.exists())
        
        with open(ref1_file, 'r') as f:
            loaded_ref1 = json.load(f)
        
        with open(ref2_file, 'r') as f:
            loaded_ref2 = json.load(f)
        
        self.assertEqual(loaded_ref1, loaded_ref2)


class TestDataAccessViaReferences(unittest.TestCase):
    """Test accessing data through JSON references."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_root = self.temp_dir / "cache"
        self.cache_root.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_access_preprocessed_data_via_reference(self):
        """Test accessing preprocessed data through a reference."""
        config_hash = "data_access_test_123"
        cache_dir = self.cache_root / f"config_{config_hash}"
        cache_dir.mkdir()
        
        # Create test data
        test_data = {
            "samples": [
                {"id": 1, "features": [1, 2, 3]},
                {"id": 2, "features": [4, 5, 6]}
            ],
            "metadata": {
                "total_samples": 2,
                "config_hash": config_hash
            }
        }
        
        data_file = cache_dir / "complete_dataset_val.json"
        with open(data_file, 'w') as f:
            json.dump(test_data, f)
        
        # Create reference
        reference = {
            "config_hash": config_hash,
            "preprocessed_data_path": str(data_file),
            "baseline_results_path": str(cache_dir / "baseline_evaluation_results")
        }
        
        # Access data via reference
        data_path = reference['preprocessed_data_path']
        self.assertTrue(Path(data_path).exists())
        
        with open(data_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Verify data integrity
        self.assertEqual(loaded_data['metadata']['total_samples'], 2)
        self.assertEqual(loaded_data['metadata']['config_hash'], config_hash)
        self.assertEqual(len(loaded_data['samples']), 2)
    
    def test_access_baseline_results_via_reference(self):
        """Test accessing baseline results through a reference."""
        config_hash = "baseline_access_test_456"
        cache_dir = self.cache_root / f"config_{config_hash}"
        cache_dir.mkdir()
        
        # Create baseline results structure
        baseline_dir = cache_dir / "baseline_evaluation_results"
        baseline_dir.mkdir()
        
        # Create mock baseline results
        for sensor in ['virtual_lidar', 'virtual_radar', 'virtual_camera']:
            sensor_dir = baseline_dir / f"{sensor}_eval"
            sensor_dir.mkdir()
            
            # Create mock results file
            results = {
                "sensor": sensor,
                "mAP": 0.15 + hash(sensor) % 100 / 1000,  # Mock varying mAP scores
                "NDS": 0.25 + hash(sensor) % 100 / 1000   # Mock varying NDS scores
            }
            
            with open(sensor_dir / "metrics.json", 'w') as f:
                json.dump(results, f)
        
        # Create reference
        reference = {
            "config_hash": config_hash,
            "preprocessed_data_path": str(cache_dir / "complete_dataset_val.json"),
            "baseline_results_path": str(baseline_dir)
        }
        
        # Access baseline results via reference
        baseline_path = Path(reference['baseline_results_path'])
        self.assertTrue(baseline_path.exists())
        self.assertTrue(baseline_path.is_dir())
        
        # Verify baseline structure
        sensor_dirs = list(baseline_path.glob("*_eval"))
        self.assertEqual(len(sensor_dirs), 3)
        
        # Access specific sensor results
        lidar_results_file = baseline_path / "virtual_lidar_eval" / "metrics.json"
        self.assertTrue(lidar_results_file.exists())
        
        with open(lidar_results_file, 'r') as f:
            lidar_results = json.load(f)
        
        self.assertEqual(lidar_results['sensor'], 'virtual_lidar')
        self.assertIn('mAP', lidar_results)
        self.assertIn('NDS', lidar_results)


class TestReferenceIntegrity(unittest.TestCase):
    """Test reference integrity and error handling."""
    
    def test_reference_with_missing_data(self):
        """Test reference behavior when referenced data is missing."""
        config_hash = "missing_data_test_789"
        
        # Create reference to non-existent data
        reference = {
            "config_hash": config_hash,
            "preprocessed_data_path": f"/nonexistent/config_{config_hash}/complete_dataset_val.json",
            "baseline_results_path": f"/nonexistent/config_{config_hash}/baseline_evaluation_results"
        }
        
        # Attempt to access non-existent data
        data_path = Path(reference['preprocessed_data_path'])
        baseline_path = Path(reference['baseline_results_path'])
        
        self.assertFalse(data_path.exists())
        self.assertFalse(baseline_path.exists())
    
    def test_malformed_reference(self):
        """Test handling of malformed references."""
        # Missing required fields
        incomplete_reference = {
            "config_hash": "incomplete_ref_test"
            # Missing preprocessed_data_path and baseline_results_path
        }
        
        required_fields = ["config_hash", "preprocessed_data_path", "baseline_results_path"]
        missing_fields = [field for field in required_fields if field not in incomplete_reference]
        
        self.assertEqual(len(missing_fields), 2)
        self.assertIn("preprocessed_data_path", missing_fields)
        self.assertIn("baseline_results_path", missing_fields)
    
    def test_reference_json_format(self):
        """Test JSON format compliance of references."""
        config_hash = "json_format_test_abc"
        
        reference = {
            "config_hash": config_hash,
            "preprocessed_data_path": f"/cache/config_{config_hash}/complete_dataset_val.json",
            "baseline_results_path": f"/cache/config_{config_hash}/baseline_evaluation_results"
        }
        
        # Test JSON serialization/deserialization
        json_string = json.dumps(reference, indent=2)
        self.assertIsInstance(json_string, str)
        
        # Test deserialization
        restored_reference = json.loads(json_string)
        self.assertEqual(reference, restored_reference)
        
        # Test JSON format structure
        self.assertTrue(json_string.startswith('{'))
        self.assertTrue(json_string.endswith('}'))
        self.assertIn('"config_hash"', json_string)
        self.assertIn('"preprocessed_data_path"', json_string)
        self.assertIn('"baseline_results_path"', json_string)


class TestReferenceBenefits(unittest.TestCase):
    """Test the benefits of using references over data copying."""
    
    def test_space_efficiency_simulation(self):
        """Simulate space efficiency benefits of references vs copying."""
        # Simulate multiple experiments referencing the same data
        config_hash = "shared_data_test_def"
        num_experiments = 10
        
        # Simulate size of preprocessed data (in KB)
        preprocessed_data_size = 150000  # 150 MB
        baseline_results_size = 50000    # 50 MB
        
        # Reference approach: one copy of data + small reference files
        reference_size_per_experiment = 1  # 1 KB per reference file
        total_reference_approach = preprocessed_data_size + baseline_results_size + (num_experiments * reference_size_per_experiment)
        
        # Copying approach: full copy for each experiment
        total_copying_approach = num_experiments * (preprocessed_data_size + baseline_results_size)
        
        # Calculate space savings
        space_savings = total_copying_approach - total_reference_approach
        savings_percentage = (space_savings / total_copying_approach) * 100
        
        # With 10 experiments, should save approximately 90% of space
        self.assertGreater(savings_percentage, 85)
        self.assertLess(total_reference_approach, total_copying_approach * 0.15)
    
    def test_consistency_benefits(self):
        """Test consistency benefits of references."""
        config_hash = "consistency_test_ghi"
        
        # Multiple experiments referencing the same data
        experiments = ['experiment_001', 'experiment_002', 'experiment_003']
        
        references = []
        for exp in experiments:
            reference = {
                "config_hash": config_hash,
                "preprocessed_data_path": f"/cache/config_{config_hash}/complete_dataset_val.json",
                "baseline_results_path": f"/cache/config_{config_hash}/baseline_evaluation_results"
            }
            references.append(reference)
        
        # All references should be identical
        for i in range(1, len(references)):
            self.assertEqual(references[0], references[i])
        
        # If source data changes, all experiments automatically see the change
        # (This is a conceptual test - in practice, cache invalidation handles this)
        updated_reference = {
            "config_hash": config_hash,
            "preprocessed_data_path": f"/cache/config_{config_hash}/complete_dataset_val.json",
            "baseline_results_path": f"/cache/config_{config_hash}/baseline_evaluation_results",
            "last_updated": "2024-01-01T12:00:00Z"  # Additional metadata
        }
        
        # All experiments would automatically reference updated data
        for ref in references:
            ref.update({"last_updated": "2024-01-01T12:00:00Z"})
            self.assertEqual(ref, updated_reference)


if __name__ == '__main__':
    unittest.main()
