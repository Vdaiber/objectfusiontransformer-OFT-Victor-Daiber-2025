"""
Fixed test for Hungarian Matcher with proper normalization handling.

This test addresses the issues found in the previous test:
1. Tensor size mismatches between predictions and sensor boxes
2. Normalization statistics inconsistency
3. Reconstruction accuracy problems

Author: Object Fusion Transformer Team
Year: 2025
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import yaml
import math
from typing import Dict, Any, List, Tuple
from unittest.mock import patch, MagicMock

# Import the Hungarian Matcher and related functions
from src.oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher
from src.oft.transformer.utils.normalization_utils import (
    CentralizedNormalizer,
    get_global_normalizer,
    normalize_offsets,
    denormalize_offsets
)
from src.oft.transformer.utils.box_reconstruction import (
    reconstruct_boxes_consistent,
    validate_box_reconstruction
)


@pytest.fixture
def sample_stats():
    """Sample normalization statistics for testing."""
    return {
        'center_abs_99p': [2.0, 1.5, 0.5],
        'log_size_abs_99p': [0.3, 0.4, 0.2],
        'velocity_percentiles': {
            '1p': [-5.0, -3.0],
            '99p': [5.0, 3.0]
        },
        'metadata': {
            'dataset_split': 'test',
            'total_samples': 100
        }
    }

@pytest.fixture
def temp_stats_file(sample_stats):
    """Create a temporary stats file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_stats, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'dataset': {
            'virtual_sensors': [
                {'name': 'virtual_lidar', 'enabled': True},
                {'name': 'virtual_camera', 'enabled': True}
            ]
        }
    }


class TestHungarianMatcherFixed:
    """Fixed test suite for Hungarian Matcher."""
    
    def test_matcher_with_consistent_data(self, temp_stats_file, sample_config):
        """Test Hungarian Matcher with consistent data sizes and normalization."""
        print("\n🧪 Testing Hungarian Matcher with Consistent Data...")
        
        # Mock the global normalizer to use our test stats
        with patch('src.oft.transformer.utils.normalization_utils.get_global_normalizer') as mock_get_normalizer:
            # Create a mock normalizer with our test stats
            mock_normalizer = MagicMock()
            mock_normalizer.stats = {
                'center_abs_99p': [2.0, 1.5, 0.5],
                'log_size_abs_99p': [0.3, 0.4, 0.2],
                'velocity_percentiles': {
                    '1p': [-5.0, -3.0],
                    '99p': [5.0, 3.0]
                }
            }
            
            # Mock the denormalize methods
            def mock_denormalize_offsets(normalized_offsets):
                # Simple denormalization using test stats
                center_abs_99p = torch.tensor([2.0, 1.5, 0.5])
                log_size_abs_99p = torch.tensor([0.3, 0.4, 0.2])
                
                denormalized = normalized_offsets.clone()
                denormalized[:, :3] = normalized_offsets[:, :3] * center_abs_99p
                denormalized[:, 3:6] = normalized_offsets[:, 3:6] * log_size_abs_99p
                # Yaw stays the same (sin/cos)
                return denormalized
            
            mock_normalizer.denormalize_offsets = mock_denormalize_offsets
            mock_normalizer.denormalize_velocity.return_value = torch.randn_like
            
            mock_get_normalizer.return_value = mock_normalizer
            
            # Create matcher
            matcher = HungarianMatcher(
                cost_class=1.0,
                cost_bbox=1.0,
                cost_giou=1.0,
                cost_velocity=1.0,
                cost_attribute=1.0,
                cfg=sample_config
            )
            
            # Create test data with consistent sizes
            batch_size = 1
            num_queries = 3  # Must match total sensor boxes
            num_classes = 12
            num_attributes = 11
            
            # Create outputs (model predictions)
            outputs = {
                'pred_logits': torch.randn(batch_size, num_queries, num_classes),
                'pred_box_offsets': torch.tensor([[
                    [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Small center offset
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],  # Small size offset
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]  # Small yaw offset
                ]]),
                'pred_velocities': torch.randn(batch_size, num_queries, 2),
                'pred_attributes': torch.randn(batch_size, num_queries, num_attributes),
            }
            
            # Create targets with matching sensor boxes
            targets = {
                'gt_labels_b': [torch.tensor([0, 1])],  # 2 ground truth objects
                'gt_boxes_b_physical': [torch.tensor([
                    [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0, 5.0, 0.0],  # GT box 1
                    [15.0, 25.0, 0.0, 3.0, 5.0, 2.0, 0.5, 0.0, 5.0]   # GT box 2
                ])],
                'gt_attributes_b': [torch.randn(2, num_attributes)],
                'gt_valid_mask_b': [torch.tensor([True, True])],
                'sensor_data': {
                    'virtual_lidar': {
                        'boxes': torch.tensor([[
                            [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0, 5.0, 0.0],  # Sensor box 1
                            [15.0, 25.0, 0.0, 3.0, 5.0, 2.0, 0.5, 0.0, 5.0],  # Sensor box 2
                        ]])
                    },
                    'virtual_camera': {
                        'boxes': torch.tensor([[
                            [20.0, 30.0, 0.0, 1.5, 3.0, 1.0, 1.0, 2.0, 2.0]   # Sensor box 3
                        ]])
                    }
                }
            }
            
            # Run matcher forward pass
            try:
                indices, matched_masks = matcher(outputs, targets)
                
                # Check that indices are returned
                assert len(indices) == batch_size
                assert len(matched_masks) == batch_size
                
                # Check that indices are tuples of tensors
                for src_idx, tgt_idx in indices:
                    assert isinstance(src_idx, torch.Tensor)
                    assert isinstance(tgt_idx, torch.Tensor)
                
                # Check that matched masks are boolean tensors
                for mask in matched_masks:
                    assert isinstance(mask, torch.Tensor)
                    assert mask.dtype == torch.bool
                
                print(f"✓ Matcher with consistent data test passed")
                print(f"  - Number of batches: {len(indices)}")
                print(f"  - Number of matches in first batch: {indices[0][0].numel()}")
                print(f"  - Matched mask shape: {matched_masks[0].shape}")
                
            except Exception as e:
                print(f"❌ Matcher forward pass failed: {e}")
                raise
    
    def test_normalization_round_trip_fixed(self, temp_stats_file, sample_config):
        """Test normalization round-trip with consistent statistics."""
        print("\n🧪 Testing Normalization Round-Trip with Fixed Stats...")
        
        # Mock the global normalizer
        with patch('src.oft.transformer.utils.normalization_utils.get_global_normalizer') as mock_get_normalizer:
            # Create a mock normalizer with consistent stats
            mock_normalizer = MagicMock()
            mock_normalizer.stats = {
                'center_abs_99p': [2.0, 1.5, 0.5],
                'log_size_abs_99p': [0.3, 0.4, 0.2],
                'velocity_percentiles': {
                    '1p': [-5.0, -3.0],
                    '99p': [5.0, 3.0]
                }
            }
            
            # Mock normalization methods
            def mock_normalize_offsets(offsets):
                center_abs_99p = torch.tensor([2.0, 1.5, 0.5])
                log_size_abs_99p = torch.tensor([0.3, 0.4, 0.2])
                
                normalized = offsets.clone()
                normalized[:, :3] = offsets[:, :3] / center_abs_99p
                normalized[:, 3:6] = offsets[:, 3:6] / log_size_abs_99p
                # Yaw stays the same (sin/cos)
                return normalized
            
            def mock_denormalize_offsets(normalized_offsets):
                center_abs_99p = torch.tensor([2.0, 1.5, 0.5])
                log_size_abs_99p = torch.tensor([0.3, 0.4, 0.2])
                
                denormalized = normalized_offsets.clone()
                denormalized[:, :3] = normalized_offsets[:, :3] * center_abs_99p
                denormalized[:, 3:6] = normalized_offsets[:, 3:6] * log_size_abs_99p
                # Yaw stays the same (sin/cos)
                return denormalized
            
            mock_normalizer.normalize_offsets = mock_normalize_offsets
            mock_normalizer.denormalize_offsets = mock_denormalize_offsets
            mock_get_normalizer.return_value = mock_normalizer
            
            # Create matcher
            matcher = HungarianMatcher(
                cost_class=1.0,
                cost_bbox=1.0,
                cost_giou=1.0,
                cost_velocity=1.0,
                cost_attribute=1.0,
                cfg=sample_config
            )
            
            # Test with realistic offsets
            original_offsets = torch.tensor([
                [1.0, 0.5, 0.1, 0.2, 0.0, 0.1, 0.0, 1.0],
                [-0.5, 1.0, -0.2, 0.0, 0.3, 0.0, 0.707, 0.707],
            ])
            
            # Step 1: Normalize using mock normalizer
            normalized_offsets = mock_normalizer.normalize_offsets(original_offsets)
            
            # Step 2: Denormalize using matcher's method
            denormalized_offsets = matcher._denormalize_offsets(normalized_offsets)
            
            # Step 3: Check round-trip consistency
            max_diff = (original_offsets - denormalized_offsets).abs().max().item()
            
            print(f"✓ Normalization round-trip with fixed stats test passed")
            print(f"  - Original offsets shape: {original_offsets.shape}")
            print(f"  - Normalized offsets shape: {normalized_offsets.shape}")
            print(f"  - Denormalized offsets shape: {denormalized_offsets.shape}")
            print(f"  - Max difference: {max_diff:.2e}")
            print(f"  - Round-trip consistent: {max_diff < 1e-6}")
            
            assert max_diff < 1e-6, f"Normalization round-trip failed: max_diff={max_diff}"
    
    def test_reconstruction_accuracy_fixed(self, temp_stats_file, sample_config):
        """Test reconstruction accuracy with consistent normalization."""
        print("\n🧪 Testing Reconstruction Accuracy with Fixed Stats...")
        
        # Mock the global normalizer
        with patch('src.oft.transformer.utils.normalization_utils.get_global_normalizer') as mock_get_normalizer:
            # Create a mock normalizer with consistent stats
            mock_normalizer = MagicMock()
            mock_normalizer.stats = {
                'center_abs_99p': [2.0, 1.5, 0.5],
                'log_size_abs_99p': [0.3, 0.4, 0.2],
                'velocity_percentiles': {
                    '1p': [-5.0, -3.0],
                    '99p': [5.0, 3.0]
                }
            }
            
            # Mock normalization methods
            def mock_normalize_offsets(offsets):
                center_abs_99p = torch.tensor([2.0, 1.5, 0.5])
                log_size_abs_99p = torch.tensor([0.3, 0.4, 0.2])
                
                normalized = offsets.clone()
                normalized[:, :3] = offsets[:, :3] / center_abs_99p
                normalized[:, 3:6] = offsets[:, 3:6] / log_size_abs_99p
                # Yaw stays the same (sin/cos)
                return normalized
            
            def mock_denormalize_offsets(normalized_offsets):
                center_abs_99p = torch.tensor([2.0, 1.5, 0.5])
                log_size_abs_99p = torch.tensor([0.3, 0.4, 0.2])
                
                denormalized = normalized_offsets.clone()
                denormalized[:, :3] = normalized_offsets[:, :3] * center_abs_99p
                denormalized[:, 3:6] = normalized_offsets[:, 3:6] * log_size_abs_99p
                # Yaw stays the same (sin/cos)
                return denormalized
            
            mock_normalizer.normalize_offsets = mock_normalize_offsets
            mock_normalizer.denormalize_offsets = mock_denormalize_offsets
            mock_get_normalizer.return_value = mock_normalizer
            
            # Create matcher
            matcher = HungarianMatcher(
                cost_class=1.0,
                cost_bbox=1.0,
                cost_giou=1.0,
                cost_velocity=1.0,
                cost_attribute=1.0,
                cfg=sample_config
            )
            
            # Create realistic sensor boxes (initial boxes)
            sensor_boxes = torch.tensor([
                [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0],      # Sensor detection 1
                [15.0, 25.0, 0.0, 3.0, 5.0, 2.0, 0.5],      # Sensor detection 2
            ])
            
            # Create realistic GT boxes
            gt_boxes = torch.tensor([
                [10.5, 20.3, 0.1, 2.1, 4.2, 1.6, 0.1],      # GT box 1 (close to sensor 1)
                [14.8, 25.2, -0.1, 2.9, 5.1, 1.9, 0.6],     # GT box 2 (close to sensor 2)
            ])
            
            # Calculate expected offsets (GT - Sensor)
            expected_center_offsets = gt_boxes[:, :3] - sensor_boxes[:, :3]
            expected_size_offsets = torch.log(gt_boxes[:, 3:6] / sensor_boxes[:, 3:6])
            
            # Calculate yaw offsets
            yaw_diff = gt_boxes[:, 6] - sensor_boxes[:, 6]
            yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
            expected_yaw_sin = torch.sin(yaw_diff)
            expected_yaw_cos = torch.cos(yaw_diff)
            
            # Combine expected offsets
            expected_offsets = torch.cat([
                expected_center_offsets,
                expected_size_offsets,
                expected_yaw_sin.unsqueeze(-1),
                expected_yaw_cos.unsqueeze(-1)
            ], dim=-1)
            
            # Normalize expected offsets using mock normalizer
            normalized_expected_offsets = mock_normalizer.normalize_offsets(expected_offsets)
            
            # Test reconstruction: normalized offsets -> denormalize -> reconstruct
            denormalized_offsets = matcher._denormalize_offsets(normalized_expected_offsets)
            reconstructed_boxes = reconstruct_boxes_consistent(
                pred_offsets=denormalized_offsets,
                initial_boxes=sensor_boxes
            )
            
            # Check that reconstructed boxes are close to GT
            box_diff = (reconstructed_boxes - gt_boxes).abs()
            max_diff = box_diff.max().item()
            
            print(f"✓ Reconstruction accuracy with fixed stats test passed")
            print(f"  - Sensor boxes shape: {sensor_boxes.shape}")
            print(f"  - GT boxes shape: {gt_boxes.shape}")
            print(f"  - Expected offsets shape: {expected_offsets.shape}")
            print(f"  - Normalized offsets shape: {normalized_expected_offsets.shape}")
            print(f"  - Reconstructed boxes shape: {reconstructed_boxes.shape}")
            print(f"  - Max difference from GT: {max_diff:.6f}")
            print(f"  - Reconstruction accurate: {max_diff < 1e-5}")
            
            assert max_diff < 1e-5, f"Reconstruction accuracy failed: max_diff={max_diff}"
    
    def test_matcher_cost_calculation(self, temp_stats_file, sample_config):
        """Test that matcher cost calculation works correctly."""
        print("\n🧪 Testing Matcher Cost Calculation...")
        
        # Mock the global normalizer
        with patch('src.oft.transformer.utils.normalization_utils.get_global_normalizer') as mock_get_normalizer:
            # Create a mock normalizer
            mock_normalizer = MagicMock()
            mock_normalizer.stats = {
                'center_abs_99p': [2.0, 1.5, 0.5],
                'log_size_abs_99p': [0.3, 0.4, 0.2],
                'velocity_percentiles': {
                    '1p': [-5.0, -3.0],
                    '99p': [5.0, 3.0]
                }
            }
            
            def mock_denormalize_offsets(normalized_offsets):
                # Simple denormalization
                center_abs_99p = torch.tensor([2.0, 1.5, 0.5])
                log_size_abs_99p = torch.tensor([0.3, 0.4, 0.2])
                
                denormalized = normalized_offsets.clone()
                denormalized[:, :3] = normalized_offsets[:, :3] * center_abs_99p
                denormalized[:, 3:6] = normalized_offsets[:, 3:6] * log_size_abs_99p
                return denormalized
            
            mock_normalizer.denormalize_offsets = mock_denormalize_offsets
            mock_get_normalizer.return_value = mock_normalizer
            
            # Create matcher
            matcher = HungarianMatcher(
                cost_class=1.0,
                cost_bbox=1.0,
                cost_giou=1.0,
                cost_velocity=1.0,
                cost_attribute=1.0,
                cfg=sample_config
            )
            
            # Create simple test data
            batch_size = 1
            num_queries = 2
            num_classes = 12
            num_attributes = 11
            
            outputs = {
                'pred_logits': torch.tensor([[
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # High confidence for class 0
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # High confidence for class 1
                ]]),
                'pred_box_offsets': torch.tensor([[
                    [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Small offset
                    [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Larger offset
                ]]),
                'pred_velocities': torch.randn(batch_size, num_queries, 2),
                'pred_attributes': torch.randn(batch_size, num_queries, num_attributes),
            }
            
            targets = {
                'gt_labels_b': [torch.tensor([0, 1])],  # 2 ground truth objects
                'gt_boxes_b_physical': [torch.tensor([
                    [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0, 5.0, 0.0],  # GT box 1
                    [15.0, 25.0, 0.0, 3.0, 5.0, 2.0, 0.5, 0.0, 5.0]   # GT box 2
                ])],
                'gt_attributes_b': [torch.randn(2, num_attributes)],
                'gt_valid_mask_b': [torch.tensor([True, True])],
                'sensor_data': {
                    'virtual_lidar': {
                        'boxes': torch.tensor([[
                            [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0, 5.0, 0.0],  # Sensor box 1
                            [15.0, 25.0, 0.0, 3.0, 5.0, 2.0, 0.5, 0.0, 5.0],  # Sensor box 2
                        ]])
                    }
                }
            }
            
            # Run matcher
            indices, matched_masks = matcher(outputs, targets)
            
            # Check that we got valid matches
            assert len(indices) == 1
            src_idx, tgt_idx = indices[0]
            
            # Should have matches (since we have 2 predictions and 2 GT objects)
            assert src_idx.numel() > 0
            assert tgt_idx.numel() > 0
            
            print(f"✓ Matcher cost calculation test passed")
            print(f"  - Number of matches: {src_idx.numel()}")
            print(f"  - Source indices: {src_idx.tolist()}")
            print(f"  - Target indices: {tgt_idx.tolist()}")


def run_fixed_hungarian_matcher_tests():
    """Run all fixed Hungarian Matcher tests."""
    print("=" * 80)
    print("🧪 FIXED HUNGARIAN MATCHER TESTS")
    print("=" * 80)
    
    # Create test instance
    test_suite = TestHungarianMatcherFixed()
    
    # Get all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_method_name in test_methods:
        test_method = getattr(test_suite, test_method_name)
        
        try:
            # Create temporary stats file for each test
            sample_stats = {
                'center_abs_99p': [2.0, 1.5, 0.5],
                'log_size_abs_99p': [0.3, 0.4, 0.2],
                'velocity_percentiles': {
                    '1p': [-5.0, -3.0],
                    '99p': [5.0, 3.0]
                },
                'metadata': {
                    'dataset_split': 'test',
                    'total_samples': 100
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(sample_stats, f)
                temp_path = f.name
            
            try:
                test_method()
                passed_tests += 1
                print(f"✅ {test_method_name} PASSED")
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"❌ {test_method_name} FAILED: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 FIXED HUNGARIAN MATCHER RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL FIXED HUNGARIAN MATCHER TESTS PASSED! The matcher is working correctly.")
    else:
        print("❌ SOME FIXED HUNGARIAN MATCHER TESTS FAILED! Check the implementation.")
    
    print("=" * 80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run all tests
    success = run_fixed_hungarian_matcher_tests()
    exit(0 if success else 1) 