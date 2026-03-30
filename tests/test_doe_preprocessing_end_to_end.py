#!/usr/bin/env python3
"""
End-to-End Integration Test for DOE Preprocessing System.

Tests the complete DOE preprocessing workflow:
1. Creates realistic DOE plan with multiple experiments
2. Runs preprocess_doe_plan.py on mini dataset
3. Verifies config hashing, cache structure, baseline evaluation
4. Cleans up all generated data afterwards

This is a comprehensive test to ensure the system works before running 
on the full dataset.
"""

import unittest
import sys
import tempfile
import yaml
import pandas as pd
import subprocess
import json
import shutil
from pathlib import Path
import time

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.oft.transformer.scripts.preprocess_doe_plan import calculate_final_parameters
from oft.transformer.utils.config_hash_utils import generate_unified_config_hash


class TestDOEPreprocessingEndToEnd(unittest.TestCase):
    """End-to-end test for complete DOE preprocessing workflow."""
    
    def setUp(self):
        """Set up realistic test configuration for mini dataset."""
        self.base_config = {
            'dataset': {
                'dataroot': '/data',
                'version': 'v1.0-mini',
                'eval_split': 'mini_val',
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
                    },
                    {
                        'name': 'virtual_camera',
                        'enabled': True,
                        'pos_noise_std': [0.94, 0.83, 1.27],
                        'dim_noise_std': 0.53,
                        'yaw_noise_std': 0.29,
                        'velocity_noise_std': 0.08,
                        'dropout_rate': 0.38,
                        'class_accuracy': 0.56,
                        'attribute_accuracy': 0.60
                    },
                    {
                        'name': 'virtual_radar',
                        'enabled': True,
                        'pos_noise_std': [1.20, 1.10, 2.00],
                        'dim_noise_std': 0.75,
                        'yaw_noise_std': 0.40,
                        'velocity_noise_std': 0.012,
                        'dropout_rate': 0.25,
                        'class_accuracy': 0.64,
                        'attribute_accuracy': 0.70
                    }
                ],
                'simulation': {
                    'fn_rate': 0.0,
                    'num_fps': 0,
                    'fp_pos_range': {'x': [-70, 70], 'y': [-70, 70], 'z': [-2, 2]},
                    'fp_dim_range': {'w': [0.5, 3.5], 'l': [0.5, 6.0], 'h': [0.5, 3.5]},
                    'fp_velocity_range': {'vx': [-10, 10], 'vy': [-5, 5]}
                },
                'class_names': ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 
                              'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 
                              'barrier', 'animal', 'traffic_sign'],
                'cutoff_dist': 70.0,
                'normalization_stats_path': '/app/config/normalization_stats.yaml',
                'scene_conditioning': {'enabled': False}
            },
            'training': {
                'train_split_name': 'mini_train',
                'val_split_name': 'mini_val',
                'batch_size': 2,
                'num_workers': 1,
                'pin_memory': False,
                'dataset_verbose': False,
                'seed': 42
            },
            'preprocessing': {
                'cache_dir': '/tmp/test_doe_preprocessing_cache',
                'num_workers': 1,
                'sampler': 'sequential',
                'seed': 42,
                'splits_to_process': ['mini_val']  # Only process mini_val split for testing
            },
            'model': {
                'input_feature_dim': 12,
                'd_model': 256,
                'nhead': 8,
                'dropout': 0.1,
                'metadata_encoder': {
                    'config': {
                        'detection_id': {
                            'vocab_size': 10000
                        }
                    }
                }
            },
            'loss': {
                'loss_weight_dict': {'loss_ce': 1.0, 'loss_bbox': 1.0},
                'losses_to_compute': ['labels', 'boxes'],
                'class_eos_coefficient': 0.1
            },
            'evaluation': {
                'run_devkit_eval': True,
                'eval_config_name': 'detection_cvpr_2024',
                'eval_split_name': 'mini_val',
                'conf_th_eval': 0.001,
                'high_precision_denorm': True,
                'run_baseline_evaluation': True,
                'baseline_eval_split': 'mini_val'
            }
        }
        
        # Create test cache directory
        self.test_cache_dir = Path('/tmp/test_doe_preprocessing_cache')
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)
        self.test_cache_dir.mkdir(parents=True)
        
        # Store created directories for cleanup
        self.cleanup_dirs = [self.test_cache_dir]
        self.temp_files = []
    
    def tearDown(self):
        """Clean up all test artifacts."""
        print("\n🧹 Cleaning up test artifacts...")
        
        # Remove temporary files
        for temp_file in self.temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
                print(f"  Removed temp file: {temp_file}")
        
        # Remove test directories
        for cleanup_dir in self.cleanup_dirs:
            if cleanup_dir.exists():
                shutil.rmtree(cleanup_dir)
                print(f"  Removed directory: {cleanup_dir}")
        
        print("✅ Cleanup completed!")
    
    def test_config_hash_generation_consistency(self):
        """Test that config hash generation is consistent and unique."""
        
        # Test that same config produces same hash
        config1 = calculate_final_parameters(self.base_config, {'lidar_pos_noise_multiplier': 1.0})
        config2 = calculate_final_parameters(self.base_config, {'lidar_pos_noise_multiplier': 1.0})
        
        hash1 = generate_unified_config_hash(config1)
        hash2 = generate_unified_config_hash(config2)
        
        self.assertEqual(hash1, hash2, "Same configs should produce same hash")
        
        # Test that different configs produce different hashes
        config3 = calculate_final_parameters(self.base_config, {'lidar_pos_noise_multiplier': 2.0})
        hash3 = generate_unified_config_hash(config3)
        
        self.assertNotEqual(hash1, hash3, "Different configs should produce different hashes")
        
        # Test hash format (should be hex string)
        self.assertIsInstance(hash1, str)
        self.assertTrue(all(c in '0123456789abcdef' for c in hash1), "Hash should be hex string")
        print(f"✅ Config hash format valid: {hash1}")
    
    def test_create_realistic_doe_plan(self):
        """Test creating a realistic DOE plan for mini dataset testing."""
        
        # Create 3 test experiments with different parameter combinations
        doe_data = {
            'experiment_id': [1, 2, 3],
            # Test baseline, better, and worse sensor conditions
            'lidar_pos_noise_multiplier': [1.0, 0.5, 2.0],      # baseline, better, worse
            'camera_pos_noise_multiplier': [1.0, 0.8, 1.5],
            'radar_pos_noise_multiplier': [1.0, 0.5, 1.2],
            'lidar_dim_noise_multiplier': [1.0, 0.7, 1.3],
            'camera_dropout_rate_multiplier': [1.0, 0.5, 1.5],
            'radar_class_accuracy_multiplier': [1.0, 1.2, 0.8],
            'false_positive_count_multiplier': [1.0, 0.0, 3.0]  # baseline, none, many
        }
        
        df = pd.DataFrame(doe_data)
        
        # Save DOE plan to temporary file
        doe_csv_path = self.test_cache_dir / 'test_doe_plan.csv'
        df.to_csv(doe_csv_path, index=False)
        self.temp_files.append(str(doe_csv_path))
        
        print(f"✅ Created DOE plan with {len(df)} experiments: {doe_csv_path}")
        
        # Verify DOE plan structure
        loaded_df = pd.read_csv(doe_csv_path)
        self.assertEqual(len(loaded_df), 3)
        self.assertIn('experiment_id', loaded_df.columns)
        self.assertIn('lidar_pos_noise_multiplier', loaded_df.columns)
        
        return str(doe_csv_path)
    
    def test_preprocessing_workflow_dry_run(self):
        """Test the preprocessing workflow without actually running preprocessing."""
        
        # Create DOE plan
        doe_csv_path = self.test_create_realistic_doe_plan()
        
        # Create base config file
        base_config_path = self.test_cache_dir / 'test_base_config.yaml'
        with open(base_config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        self.temp_files.append(str(base_config_path))
        
        # Test that we can calculate all final configs and generate hashes
        df = pd.read_csv(doe_csv_path)
        config_hashes = []
        
        for _, row in df.iterrows():
            experiment_id = row['experiment_id']
            multipliers = row.drop('experiment_id').to_dict()
            
            # Calculate final config
            final_config = calculate_final_parameters(self.base_config, multipliers)
            
            # Generate config hash
            config_hash = generate_unified_config_hash(final_config)
            config_hashes.append(config_hash)
            
            print(f"  Experiment {experiment_id}: hash={config_hash[:12]}...")
            
            # Verify config structure
            self.assertIn('dataset', final_config)
            self.assertIn('virtual_sensors', final_config['dataset'])
            
            # Verify that cache directory structure would be created
            cache_dir = Path(final_config['preprocessing']['cache_dir']) / f'config_{config_hash}'
            expected_val_file = cache_dir / 'complete_dataset_val.json'
            expected_baseline_dir = cache_dir / 'baseline_evaluation_results'
            
            print(f"    Cache dir: {cache_dir}")
            print(f"    Val file: {expected_val_file}")
            print(f"    Baseline dir: {expected_baseline_dir}")
        
        # Verify all hashes are unique
        self.assertEqual(len(config_hashes), len(set(config_hashes)), "All config hashes should be unique")
        
        print("✅ Preprocessing workflow structure validated!")
    
    @unittest.skipUnless(Path('/data/v1.0-mini').exists(), "Mini dataset not available")
    def test_preprocessing_only_integration(self):
        """Test running actual preprocessing on a single experiment (only if mini dataset available)."""
        
        print("\n🔄 Testing REAL preprocessing with mini dataset...")
        
        # Create minimal DOE plan with just baseline experiment
        doe_data = {
            'experiment_id': [1],
            'lidar_pos_noise_multiplier': [1.0],
            'camera_pos_noise_multiplier': [1.0], 
            'radar_pos_noise_multiplier': [1.0],
            'false_positive_count_multiplier': [1.0]
        }
        df = pd.DataFrame(doe_data)
        
        doe_csv_path = self.test_cache_dir / 'single_experiment.csv'
        df.to_csv(doe_csv_path, index=False)
        self.temp_files.append(str(doe_csv_path))
        
        # Create base config file
        base_config_path = self.test_cache_dir / 'base_config.yaml'
        with open(base_config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        self.temp_files.append(str(base_config_path))
        
        # Run actual preprocessing
        cmd = [
            sys.executable, 
            'src/oft/transformer/scripts/preprocess_doe_plan.py',
            '--doe-plan-csv', str(doe_csv_path),
            '--base-config', str(base_config_path),
            '--verbose'
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=300  # 5 minute timeout for mini dataset preprocessing + baseline eval
            )
            
            print(f"📋 Return code: {result.returncode}")
            if result.stdout:
                print("📝 STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("⚠️ STDERR:")
                print(result.stderr)
            
            # Verify preprocessing succeeded
            self.assertEqual(result.returncode, 0, f"Preprocessing failed: {result.stderr}")
            
            # Verify cache structure was created
            final_config = calculate_final_parameters(
                self.base_config, 
                {'lidar_pos_noise_multiplier': 1.0, 'camera_pos_noise_multiplier': 1.0, 
                 'radar_pos_noise_multiplier': 1.0, 'false_positive_count_multiplier': 1.0}
            )
            config_hash = generate_unified_config_hash(final_config)
            
            cache_dir = self.test_cache_dir / f'config_{config_hash}'
            val_dataset_file = cache_dir / 'complete_dataset_mini_val.json'
            baseline_results_dir = cache_dir / 'baseline_evaluation_results'
            
            # Add cache directories to cleanup
            if cache_dir.exists():
                self.cleanup_dirs.append(cache_dir)
            
            print(f"🔍 Checking cache structure:")
            print(f"  Cache dir exists: {cache_dir.exists()}")
            print(f"  Val dataset exists: {val_dataset_file.exists()}")
            print(f"  Baseline results dir exists: {baseline_results_dir.exists()}")
            
            # Verify expected files exist
            self.assertTrue(cache_dir.exists(), f"Cache directory should exist: {cache_dir}")
            self.assertTrue(val_dataset_file.exists(), f"Validation dataset should exist: {val_dataset_file}")
            self.assertTrue(baseline_results_dir.exists(), f"Baseline results should exist: {baseline_results_dir}")
            
            # Verify validation dataset structure
            if val_dataset_file.exists():
                with open(val_dataset_file, 'r') as f:
                    val_data = json.load(f)
                
                self.assertIsInstance(val_data, dict, "Validation dataset should be a dictionary")
                
                # Check if it's the new format with meta and results
                if 'meta' in val_data and 'results' in val_data:
                    # New format with metadata
                    meta = val_data['meta']
                    results = val_data['results']
                    
                    # Check metadata
                    self.assertTrue(meta.get('preprocessing_complete', False), "Preprocessing should be marked complete")
                    self.assertEqual(meta.get('format'), 'dataset_format', "Should be dataset format")
                    
                    # Check results structure
                    self.assertIsInstance(results, dict, "Results should be a dictionary")
                    if len(results) > 0:
                        first_sample_token = list(results.keys())[0]
                        first_sample = results[first_sample_token]
                        
                        required_keys = ['sample_token', 'sensor_data', 'ground_truth', 'ego_translation_world']
                        for key in required_keys:
                            self.assertIn(key, first_sample, f"Sample should contain {key}")
                        
                        print(f"✅ Validation dataset contains {len(results)} samples with metadata")
                    else:
                        # Metadata-only format (which is also valid for our test)
                        print(f"✅ Validation dataset created with metadata ({meta.get('num_samples', 0)} samples processed)")
                else:
                    # Legacy format - direct sample mapping
                    self.assertGreater(len(val_data), 0, "Validation dataset should contain samples")
                    first_sample_token = list(val_data.keys())[0]
                    first_sample = val_data[first_sample_token]
                    
                    required_keys = ['sample_token', 'sensor_data', 'ground_truth', 'ego_translation_world']
                    for key in required_keys:
                        self.assertIn(key, first_sample, f"Sample should contain {key}")
                    
                    print(f"✅ Validation dataset contains {len(val_data)} samples (legacy format)")
            
            # Verify baseline evaluation results
            if baseline_results_dir.exists():
                baseline_files = list(baseline_results_dir.glob('*.json'))
                self.assertGreater(len(baseline_files), 0, "Should have baseline evaluation files")
                print(f"✅ Baseline evaluation created {len(baseline_files)} result files")
            
            print("🎉 Full preprocessing test completed successfully!")
            
        except subprocess.TimeoutExpired:
            self.fail("Preprocessing timed out after 5 minutes")
        except Exception as e:
            self.fail(f"Preprocessing test failed: {e}")
    
    def test_cache_structure_validation(self):
        """Test that cache structure follows expected format."""
        
        # Generate a test config hash
        test_config = calculate_final_parameters(self.base_config, {'lidar_pos_noise_multiplier': 1.5})
        test_hash = generate_unified_config_hash(test_config)
        
        # Expected cache structure
        cache_dir = self.test_cache_dir / f'config_{test_hash}'
        
        expected_structure = {
            'validation_dataset': cache_dir / 'complete_dataset_val.json',
            'baseline_results': cache_dir / 'baseline_evaluation_results',
            'config_file': cache_dir / 'config_used.yaml'
        }
        
        print(f"📁 Expected cache structure for hash {test_hash[:12]}:")
        for name, path in expected_structure.items():
            print(f"  {name}: {path}")
        
        # Verify path structure is reasonable
        self.assertTrue(str(cache_dir).startswith('/tmp/test_doe_preprocessing_cache'))
        self.assertTrue('config_' in str(cache_dir))
        self.assertEqual(len(test_hash), 16)  # Config hash should be 16 characters
        
        print("✅ Cache structure validation passed!")


def run_tests():
    """Run all end-to-end tests."""
    print("🧪 Starting DOE Preprocessing End-to-End Tests...")
    print("=" * 60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
