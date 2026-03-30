#!/usr/bin/env python3
"""
Comprehensive Unit Test for DOE Preprocessing Integration.

Tests the complete DOE preprocessing pipeline with real DOE CSV data, 
validates parameter calculation, config generation, and runs actual 
preprocessing on mini-dataset to verify end-to-end functionality.
"""

import unittest
import sys
import pandas as pd
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.oft.transformer.scripts.preprocess_doe_plan import (
    calculate_final_parameters, 
    load_base_config,
    generate_unified_config_hash
)


class TestDOEPreprocessingComprehensive(unittest.TestCase):
    """Comprehensive test suite for DOE preprocessing with real CSV data."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with real configuration."""
        # Load real base configuration
        cls.base_config = load_base_config("config/pipeline_staged.yaml")
        
        # Load real DOE CSV data
        cls.doe_csv_path = "DOE_CSV/DOE_D_Optimal.csv"
        if not Path(cls.doe_csv_path).exists():
            raise FileNotFoundError(f"DOE CSV not found: {cls.doe_csv_path}")
        
        cls.doe_plan = pd.read_csv(cls.doe_csv_path)
        
        # Map short CSV column names to full parameter names
        cls.column_mapping = {
            'lid_pos': 'lidar_pos_noise_multiplier',
            'cam_pos': 'camera_pos_noise_multiplier', 
            'ra_pos': 'radar_pos_noise_multiplier',
            'lid_dim': 'lidar_dim_noise_multiplier',
            'cam_dim': 'camera_dim_noise_multiplier',
            'ra_dim': 'radar_dim_noise_multiplier',
            'lid_yaw': 'lidar_yaw_noise_multiplier',
            'cam_yaw': 'camera_yaw_noise_multiplier',
            'ra_yaw': 'radar_yaw_noise_multiplier',
            'lid_vel': 'lidar_velocity_noise_multiplier',
            'cam_vel': 'camera_velocity_noise_multiplier',
            'ra_vel': 'radar_velocity_noise_multiplier',
            'lid_drop': 'lidar_dropout_odds_multiplier',
            'cam_drop': 'camera_dropout_odds_multiplier',
            'ra_drop': 'radar_dropout_odds_multiplier',
            'lid_clas': 'lidar_class_accuracy_multiplier',
            'cam_clas': 'camera_class_accuracy_multiplier',
            'ra_clas': 'radar_class_accuracy_multiplier',
            'lid_at': 'lidar_attribute_accuracy_multiplier',
            'cam_at': 'camera_attribute_accuracy_multiplier',
            'ra_at': 'radar_attribute_accuracy_multiplier',
            'FP': 'false_positive_count_multiplier'
        }
        
        # Extract baseline values from real config for validation
        cls.baseline_values = {}
        for sensor in cls.base_config['dataset']['virtual_sensors']:
            name = sensor['name']
            if 'lidar' in name:
                prefix = 'lidar'
            elif 'camera' in name:
                prefix = 'camera'
            elif 'radar' in name:
                prefix = 'radar'
            else:
                continue
                
            cls.baseline_values[f'{prefix}_pos_noise_std'] = sensor.get('pos_noise_std')
            cls.baseline_values[f'{prefix}_dim_noise_std'] = sensor.get('dim_noise_std') 
            cls.baseline_values[f'{prefix}_yaw_noise_std'] = sensor.get('yaw_noise_std')
            cls.baseline_values[f'{prefix}_velocity_noise_std'] = sensor.get('velocity_noise_std')
            cls.baseline_values[f'{prefix}_dropout_rate'] = sensor.get('dropout_rate')
            cls.baseline_values[f'{prefix}_class_accuracy'] = sensor.get('class_accuracy')
            cls.baseline_values[f'{prefix}_attribute_accuracy'] = sensor.get('attribute_accuracy')
        
        cls.baseline_values['false_positive_count'] = cls.base_config.get('dataset', {}).get('simulation', {}).get('num_fps', 1)
        
        print(f"✅ Loaded DOE CSV with {len(cls.doe_plan)} experiments")
        print(f"✅ Loaded baseline config with {len(cls.base_config['dataset']['virtual_sensors'])} sensors")
    
    def setUp(self):
        """Set up for each test."""
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up after each test."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def convert_csv_row_to_multipliers(self, row: pd.Series) -> Dict[str, float]:
        """Convert DOE CSV row to multiplier dictionary."""
        multipliers = {}
        for csv_col, param_name in self.column_mapping.items():
            if csv_col in row:
                multipliers[param_name] = float(row[csv_col])
        return multipliers
    
    def test_baseline_values_extraction(self):
        """Test that baseline values are correctly extracted from config."""
        print("\n🔍 Testing baseline value extraction...")
        
        # Verify LiDAR baseline values
        self.assertEqual(self.baseline_values['lidar_pos_noise_std'], [0.46, 0.46, 0.29])
        self.assertEqual(self.baseline_values['lidar_dim_noise_std'], 0.27)
        self.assertEqual(self.baseline_values['lidar_yaw_noise_std'], 0.22)
        self.assertEqual(self.baseline_values['lidar_velocity_noise_std'], 0.16)
        self.assertEqual(self.baseline_values['lidar_dropout_rate'], 0.24)
        self.assertEqual(self.baseline_values['lidar_class_accuracy'], 0.78)
        self.assertEqual(self.baseline_values['lidar_attribute_accuracy'], 0.68)
        
        # Verify Camera baseline values
        self.assertEqual(self.baseline_values['camera_pos_noise_std'], [0.94, 0.83, 1.27])
        self.assertEqual(self.baseline_values['camera_class_accuracy'], 0.56)
        self.assertEqual(self.baseline_values['camera_attribute_accuracy'], 0.60)
        
        # Verify Radar baseline values
        self.assertEqual(self.baseline_values['radar_class_accuracy'], 0.64)
        self.assertEqual(self.baseline_values['radar_attribute_accuracy'], 0.70)
        
        print("✅ All baseline values correctly extracted")
    
    def test_doe_csv_structure(self):
        """Test that DOE CSV has correct structure and parameter ranges."""
        print("\n🔍 Testing DOE CSV structure...")
        
        # Check required columns exist
        required_columns = list(self.column_mapping.keys()) + ['Nr']
        for col in required_columns:
            self.assertIn(col, self.doe_plan.columns, f"Missing column: {col}")
        
        # Check parameter ranges match actual DOE CSV ranges
        test_ranges = {
            'lid_pos': (0.0, 3.5), 'cam_pos': (0.0, 2.5), 'ra_pos': (0.0, 2.2),
            'lid_dim': (0.0, 3.8), 'cam_dim': (0.0, 2.3), 'ra_dim': (0.0, 1.9),
            'lid_yaw': (0.0, 9.0), 'cam_yaw': (0.0, 6.5), 'ra_yaw': (0.0, 4.5),
            'lid_vel': (0.0, 22.0), 'cam_vel': (0.0, 35.0), 'ra_vel': (0.0, 85.0),
            'lid_drop': (0.0, 12.0), 'cam_drop': (0.0, 8.0), 'ra_drop': (0.0, 10.0),
            'lid_clas': (0.0, 2.0), 'cam_clas': (2.0, 10.0), 'ra_clas': (0.0, 2.0),  # Camera has different range!
            'lid_at': (0.0, 2.0), 'cam_at': (0.0, 2.0), 'ra_at': (0.0, 2.0),
            'FP': (0.0, 12.0)
        }
        
        for col, (min_val, max_val) in test_ranges.items():
            if col in self.doe_plan.columns:
                actual_min = self.doe_plan[col].min()
                actual_max = self.doe_plan[col].max()
                self.assertGreaterEqual(actual_min, min_val, f"{col} min value {actual_min} < expected {min_val}")
                self.assertLessEqual(actual_max, max_val, f"{col} max value {actual_max} > expected {max_val}")
                print(f"  ✅ {col}: [{actual_min:.1f}, {actual_max:.1f}] within [{min_val}, {max_val}]")
        
        print(f"✅ DOE CSV structure validated with {len(self.doe_plan)} experiments")
    
    def test_parameter_calculation_accuracy(self):
        """Test parameter calculation with specific DOE rows."""
        print("\n🔍 Testing parameter calculation accuracy...")
        
        # Test several representative rows from DOE plan
        test_rows = [1, 2, 3, 37]  # Including centerpoint (row 37)
        
        for row_idx in test_rows:
            with self.subTest(row=row_idx):
                row = self.doe_plan.iloc[row_idx]
                multipliers = self.convert_csv_row_to_multipliers(row)
                
                print(f"\n--- Testing Row {row_idx + 1} (Nr={row['Nr']}) ---")
                
                # Calculate final parameters
                final_config = calculate_final_parameters(self.base_config, multipliers)
                
                # Validate LiDAR sensor modifications
                lidar_sensor = None
                for sensor in final_config['dataset']['virtual_sensors']:
                    if 'lidar' in sensor['name']:
                        lidar_sensor = sensor
                        break
                
                self.assertIsNotNone(lidar_sensor, "LiDAR sensor not found")
                
                # Test position noise calculation
                pos_mult = multipliers.get('lidar_pos_noise_multiplier', 1.0)
                baseline_pos = self.baseline_values['lidar_pos_noise_std']
                expected_pos = [x * pos_mult for x in baseline_pos]
                actual_pos = lidar_sensor['pos_noise_std']
                
                for i in range(3):
                    self.assertAlmostEqual(actual_pos[i], expected_pos[i], places=6,
                                         msg=f"Row {row_idx}: LiDAR pos_noise[{i}] mismatch")
                
                # Test dimension noise calculation
                dim_mult = multipliers.get('lidar_dim_noise_multiplier', 1.0)
                baseline_dim = self.baseline_values['lidar_dim_noise_std']
                expected_dim = baseline_dim * dim_mult
                actual_dim = lidar_sensor['dim_noise_std']
                self.assertAlmostEqual(actual_dim, expected_dim, places=6,
                                     msg=f"Row {row_idx}: LiDAR dim_noise mismatch")
                
                # Test class accuracy calculation (bidirectional scaling)
                class_mult = multipliers.get('lidar_class_accuracy_multiplier', 1.0)
                baseline_class = self.baseline_values['lidar_class_accuracy']
                
                if class_mult <= 0.0:
                    expected_class = 1.0  # Perfect
                elif class_mult <= 1.0:
                    expected_class = 1.0 - (1.0 - baseline_class) * class_mult
                else:
                    # LiDAR uses 0.0-2.0 range
                    max_multiplier = 2.0
                    degradation = (class_mult - 1.0) / (max_multiplier - 1.0)
                    degradation = min(1.0, degradation)
                    expected_class = baseline_class - (baseline_class - 0.05) * degradation
                
                actual_class = lidar_sensor['class_accuracy']
                self.assertAlmostEqual(actual_class, expected_class, places=6,
                                     msg=f"Row {row_idx}: LiDAR class_accuracy mismatch")
                
                # Test Camera class accuracy if present in multipliers
                camera_class_mult = multipliers.get('camera_class_accuracy_multiplier')
                if camera_class_mult is not None:
                    camera_sensor = None
                    for sensor in final_config['dataset']['virtual_sensors']:
                        if 'camera' in sensor['name']:
                            camera_sensor = sensor
                            break
                    
                    if camera_sensor:
                        baseline_camera_class = self.baseline_values['camera_class_accuracy']
                        
                        if camera_class_mult <= 0.0:
                            expected_camera_class = 1.0
                        elif camera_class_mult <= 1.0:
                            expected_camera_class = 1.0 - (1.0 - baseline_camera_class) * camera_class_mult
                        else:
                            # Camera uses 2.0-10.0 range in DOE CSV
                            max_multiplier = 10.0
                            degradation = (camera_class_mult - 1.0) / (max_multiplier - 1.0)
                            degradation = min(1.0, degradation)
                            expected_camera_class = baseline_camera_class - (baseline_camera_class - 0.05) * degradation
                        
                        actual_camera_class = camera_sensor['class_accuracy']
                        self.assertAlmostEqual(actual_camera_class, expected_camera_class, places=6,
                                             msg=f"Row {row_idx}: Camera class_accuracy mismatch")
                
                # Test dropout rate calculation (odd scaling)
                dropout_mult = multipliers.get('lidar_dropout_odds_multiplier', 1.0)
                baseline_dropout = self.baseline_values['lidar_dropout_rate']
                
                if baseline_dropout > 0 and baseline_dropout < 1:
                    baseline_odds = baseline_dropout / (1 - baseline_dropout)
                    new_odds = baseline_odds * dropout_mult
                    expected_dropout = new_odds / (1 + new_odds)
                    expected_dropout = max(0.0, min(1.0, expected_dropout))
                else:
                    expected_dropout = baseline_dropout
                
                actual_dropout = lidar_sensor['dropout_rate']
                self.assertAlmostEqual(actual_dropout, expected_dropout, places=6,
                                     msg=f"Row {row_idx}: LiDAR dropout_rate mismatch")
                
                print(f"  ✅ Row {row_idx + 1}: Parameter calculations verified")
        
        print("✅ Parameter calculation accuracy validated")
    
    def test_config_hash_consistency(self):
        """Test that config hash generation is consistent."""
        print("\n🔍 Testing config hash consistency...")
        
        # Test same parameters produce same hash
        row = self.doe_plan.iloc[0]
        multipliers = self.convert_csv_row_to_multipliers(row)
        
        config1 = calculate_final_parameters(self.base_config, multipliers)
        config2 = calculate_final_parameters(self.base_config, multipliers)
        
        hash1 = generate_unified_config_hash(config1)
        hash2 = generate_unified_config_hash(config2)
        
        self.assertEqual(hash1, hash2, "Same config should produce same hash")
        
        # Test different parameters produce different hashes
        row2 = self.doe_plan.iloc[1]
        multipliers2 = self.convert_csv_row_to_multipliers(row2)
        config3 = calculate_final_parameters(self.base_config, multipliers2)
        hash3 = generate_unified_config_hash(config3)
        
        self.assertNotEqual(hash1, hash3, "Different configs should produce different hashes")
        
        print(f"✅ Config hash consistency validated")
        print(f"  Hash 1: {hash1}")
        print(f"  Hash 2: {hash2} (same as 1)")
        print(f"  Hash 3: {hash3} (different)")
    
    @unittest.skipUnless(Path('/data/v1.0-mini').exists(), "Mini dataset not available")
    def test_end_to_end_preprocessing_integration(self):
        """Test complete end-to-end preprocessing with real mini dataset."""
        print("\n🔍 Testing end-to-end preprocessing integration...")
        
        # Create temporary directory for this test
        self.temp_dir = tempfile.mkdtemp(prefix='test_doe_preprocessing_')
        temp_cache_dir = Path(self.temp_dir) / 'cache'
        temp_cache_dir.mkdir()
        
        # Select a representative experiment (centerpoint)
        centerpoint_row = self.doe_plan[self.doe_plan['Nr'] == self.doe_plan['Nr'].iloc[36]]  # Row 37 is centerpoint
        if centerpoint_row.empty:
            centerpoint_row = self.doe_plan.iloc[0]  # Fallback to first row
        else:
            centerpoint_row = centerpoint_row.iloc[0]
        
        multipliers = self.convert_csv_row_to_multipliers(centerpoint_row)
        
        print(f"  Testing with experiment Nr={centerpoint_row['Nr']}")
        print(f"  Multipliers: {multipliers}")
        
        # Calculate final config for mini dataset
        final_config = calculate_final_parameters(self.base_config, multipliers)
        
        # Modify config for mini dataset testing
        final_config['dataset']['version'] = 'v1.0-mini'
        final_config['dataset']['eval_split'] = 'mini_val'
        final_config['training']['train_split_name'] = 'mini_train'
        final_config['training']['val_split_name'] = 'mini_val'
        final_config['preprocessing']['cache_dir'] = str(temp_cache_dir)
        final_config['preprocessing']['splits_to_process'] = ['mini_val']
        final_config['preprocessing']['num_workers'] = 1
        final_config['preprocessing']['sampler'] = 'sequential'
        
        # Add evaluation section for baseline evaluation
        final_config['evaluation'] = {
            'run_devkit_eval': True,
            'eval_config_name': 'detection_cvpr_2024',
            'eval_split_name': 'mini_val',
            'conf_th_eval': 0.001,
            'high_precision_denorm': True,
            'run_baseline_evaluation': True,
            'baseline_eval_split': 'mini_val'
        }
        
        # Generate config hash
        config_hash = generate_unified_config_hash(final_config)
        expected_cache_dir = temp_cache_dir / f'config_{config_hash}'
        
        print(f"  Config hash: {config_hash}")
        print(f"  Expected cache dir: {expected_cache_dir}")
        
        # Create temporary config file
        temp_config_file = Path(self.temp_dir) / 'test_config.yaml'
        with open(temp_config_file, 'w') as f:
            import yaml
            yaml.dump(final_config, f, default_flow_style=False)
        
        # Test preprocessing using the actual DOE preprocessing script
        cmd = [
            sys.executable, 
            "src/oft/transformer/scripts/preprocess_doe_plan.py",
            "--doe-plan-csv", str(temp_config_file).replace('.yaml', '.csv'),
            "--base-config", str(temp_config_file)
        ]
        
        # Create temporary DOE CSV with just our test experiment
        temp_csv = Path(self.temp_dir) / 'test_doe.csv'
        test_df = pd.DataFrame([centerpoint_row])
        test_df.to_csv(temp_csv, index=False)
        
        # Update command with correct arguments
        cmd = [
            sys.executable, 
            "src/oft/transformer/scripts/preprocess_doe_plan.py",
            "--doe-plan-csv", str(temp_csv),
            "--base-config", str(temp_config_file)
        ]
        
        print(f"  Running command: {' '.join(cmd)}")
        
        # Run preprocessing with timeout
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minutes timeout
                cwd=Path.cwd()
            )
            
            print(f"  Return code: {result.returncode}")
            if result.stdout:
                print(f"  STDOUT: {result.stdout[-500:]}")  # Last 500 chars
            if result.stderr:
                print(f"  STDERR: {result.stderr[-500:]}")  # Last 500 chars
            
            # Check if preprocessing was successful
            self.assertEqual(result.returncode, 0, f"Preprocessing failed: {result.stderr}")
            
            # Verify cache structure was created
            self.assertTrue(expected_cache_dir.exists(), "Cache directory not created")
            
            # Check for preprocessed data
            expected_data_file = expected_cache_dir / "complete_dataset_mini_val.json"
            self.assertTrue(expected_data_file.exists(), "Preprocessed data file not created")
            
            # Check for baseline evaluation results
            baseline_dir = expected_cache_dir / "baseline_evaluation_results"
            self.assertTrue(baseline_dir.exists(), "Baseline evaluation directory not created")
            
            # Validate JSON structure
            with open(expected_data_file, 'r') as f:
                data = json.load(f)
            
            self.assertIsInstance(data, dict, "Data should be a dictionary")
            
            if 'meta' in data and 'results' in data:
                meta = data['meta']
                self.assertTrue(meta.get('preprocessing_complete', False), 
                              "Preprocessing should be marked complete")
                self.assertEqual(meta.get('format'), 'dataset_format', 
                              "Should be dataset format")
                print(f"  ✅ Preprocessed {meta.get('num_samples', 0)} samples")
            else:
                # Legacy format
                self.assertGreater(len(data), 0, "Should contain samples")
                print(f"  ✅ Preprocessed {len(data)} samples (legacy format)")
            
            print("✅ End-to-end preprocessing integration successful")
            
        except subprocess.TimeoutExpired:
            self.fail("Preprocessing timed out after 5 minutes")
        except Exception as e:
            self.fail(f"Preprocessing failed with exception: {e}")
    
    def test_parameter_scientific_ranges(self):
        """Test that calculated parameters stay within scientifically meaningful ranges."""
        print("\n🔍 Testing parameter scientific ranges...")
        
        # Test extreme values from DOE plan
        extreme_rows = [
            self.doe_plan.iloc[self.doe_plan['lid_pos'].idxmax()],  # Max LiDAR position noise
            self.doe_plan.iloc[self.doe_plan['cam_clas'].idxmin()],  # Min Camera class accuracy
            self.doe_plan.iloc[self.doe_plan['ra_drop'].idxmax()],   # Max Radar dropout
        ]
        
        for i, row in enumerate(extreme_rows):
            with self.subTest(extreme_case=i):
                multipliers = self.convert_csv_row_to_multipliers(row)
                final_config = calculate_final_parameters(self.base_config, multipliers)
                
                for sensor in final_config['dataset']['virtual_sensors']:
                    # Position noise should be positive
                    if 'pos_noise_std' in sensor:
                        for pos_val in sensor['pos_noise_std']:
                            self.assertGreaterEqual(pos_val, 0.0, "Position noise should be non-negative")
                            self.assertLess(pos_val, 10.0, "Position noise should be reasonable")
                    
                    # Dimension noise should be positive
                    if 'dim_noise_std' in sensor:
                        self.assertGreaterEqual(sensor['dim_noise_std'], 0.0, "Dimension noise should be non-negative")
                        self.assertLess(sensor['dim_noise_std'], 5.0, "Dimension noise should be reasonable")
                    
                    # Dropout rate should be [0, 1]
                    if 'dropout_rate' in sensor:
                        self.assertGreaterEqual(sensor['dropout_rate'], 0.0, "Dropout rate should be >= 0")
                        self.assertLessEqual(sensor['dropout_rate'], 1.0, "Dropout rate should be <= 1")
                    
                    # Accuracy should be [0, 1]
                    for acc_key in ['class_accuracy', 'attribute_accuracy']:
                        if acc_key in sensor:
                            self.assertGreaterEqual(sensor[acc_key], 0.0, f"{acc_key} should be >= 0")
                            self.assertLessEqual(sensor[acc_key], 1.0, f"{acc_key} should be <= 1")
                
                # False positives should be non-negative integer
                if 'simulation' in final_config.get('dataset', {}):
                    fps = final_config['dataset']['simulation'].get('num_fps', 0)
                    self.assertGreaterEqual(fps, 0, "False positives should be non-negative")
                    self.assertIsInstance(fps, int, "False positives should be integer")
        
        print("✅ Parameter scientific ranges validated")
    
    def test_doe_coverage_analysis(self):
        """Analyze DOE plan coverage and parameter distributions."""
        print("\n🔍 Analyzing DOE plan coverage...")
        
        # Analyze parameter distributions
        key_params = ['lid_clas', 'cam_clas', 'ra_clas', 'lid_drop', 'cam_drop', 'ra_drop']
        
        coverage_stats = {}
        for param in key_params:
            if param in self.doe_plan.columns:
                values = self.doe_plan[param]
                coverage_stats[param] = {
                    'min': values.min(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'std': values.std(),
                    'unique_values': len(values.unique())
                }
        
        # Print coverage analysis
        for param, stats in coverage_stats.items():
            print(f"  {param}: min={stats['min']:.2f}, max={stats['max']:.2f}, "
                  f"mean={stats['mean']:.2f}, unique={stats['unique_values']}")
        
        # Verify good coverage
        for param in key_params:
            if param in coverage_stats:
                stats = coverage_stats[param]
                self.assertGreater(stats['unique_values'], 3, 
                                 f"{param} should have good value diversity")
                
                # Check that extreme values are covered
                if param.endswith('_clas') or param.endswith('_at'):  # Accuracy parameters
                    if param == 'cam_clas':  # Camera has different range in DOE CSV
                        self.assertGreaterEqual(stats['min'], 2.0, f"{param} should cover camera-specific range")
                        self.assertGreaterEqual(stats['max'], 8.0, f"{param} should cover degraded range")
                    else:
                        self.assertLessEqual(stats['min'], 0.1, f"{param} should cover near-perfect range")
                        self.assertGreaterEqual(stats['max'], 1.5, f"{param} should cover degraded range")
                elif param.endswith('_drop'):  # Dropout parameters
                    self.assertLessEqual(stats['min'], 1.0, f"{param} should cover low dropout")
                    self.assertGreaterEqual(stats['max'], 8.0, f"{param} should cover high dropout")
        
        print("✅ DOE coverage analysis completed")


def run_comprehensive_tests():
    """Run all comprehensive DOE preprocessing tests."""
    print("🧪 COMPREHENSIVE DOE PREPROCESSING TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDOEPreprocessingComprehensive)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED!")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
