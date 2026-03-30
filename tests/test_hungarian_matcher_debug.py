"""
Debug test for Hungarian Matcher to identify normalization and reconstruction issues.

This test specifically focuses on the Hungarian Matcher's use of normalization
and reconstruction functions to ensure consistency across the pipeline.

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


class TestHungarianMatcherDebug:
    """Debug test suite for Hungarian Matcher."""
    
    def test_matcher_initialization(self, temp_stats_file, sample_config):
        """Test Hungarian Matcher initialization with normalization stats."""
        print("\n🧪 Testing Hungarian Matcher Initialization...")
        
        # Create matcher
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            cost_velocity=1.0,
            cost_attribute=1.0,
            cfg=sample_config
        )
        
        # Check that normalizer is loaded
        assert hasattr(matcher, 'normalizer')
        assert matcher.normalizer is not None
        
        # Check that normalization stats are available
        assert hasattr(matcher, 'norm_stats')
        
        print(f"✓ Matcher initialization test passed")
        print(f"  - Normalizer loaded: {matcher.normalizer is not None}")
        print(f"  - Stats available: {matcher.norm_stats is not None}")
    
    def test_denormalization_consistency(self, temp_stats_file, sample_config):
        """Test that matcher's denormalization is consistent with global normalizer."""
        print("\n🧪 Testing Denormalization Consistency...")
        
        # Create matcher
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            cost_velocity=1.0,
            cost_attribute=1.0,
            cfg=sample_config
        )
        
        # Test offsets
        test_offsets = torch.tensor([
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0],
        ])
        
        # Test matcher's denormalization
        matcher_denorm = matcher._denormalize_offsets(test_offsets)
        
        # Test global normalizer denormalization
        global_denorm = denormalize_offsets(test_offsets)
        
        # Check consistency
        assert torch.allclose(matcher_denorm, global_denorm, atol=1e-6)
        
        print(f"✓ Denormalization consistency test passed")
        print(f"  - Matcher denorm shape: {matcher_denorm.shape}")
        print(f"  - Global denorm shape: {global_denorm.shape}")
        print(f"  - Max difference: {(matcher_denorm - global_denorm).abs().max():.2e}")
    
    def test_reconstruction_consistency(self, temp_stats_file, sample_config):
        """Test that matcher's reconstruction is consistent."""
        print("\n🧪 Testing Reconstruction Consistency...")
        
        # Create matcher
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            cost_velocity=1.0,
            cost_attribute=1.0,
            cfg=sample_config
        )
        
        # Test data
        initial_boxes = torch.tensor([
            [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [15.0, 25.0, 0.0, 3.0, 5.0, 2.0, 0.5],
        ])
        
        normalized_offsets = torch.tensor([
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0],
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0, 1.0],
        ])
        
        # Step 1: Denormalize offsets using matcher's method
        denormalized_offsets = matcher._denormalize_offsets(normalized_offsets)
        
        # Step 2: Reconstruct boxes using consistent function
        reconstructed_boxes = reconstruct_boxes_consistent(
            pred_offsets=denormalized_offsets,
            initial_boxes=initial_boxes
        )
        
        # Step 3: Validate reconstruction
        is_valid = validate_box_reconstruction(reconstructed_boxes, initial_boxes)
        
        assert is_valid
        assert not torch.isnan(reconstructed_boxes).any()
        assert not torch.isinf(reconstructed_boxes).any()
        
        print(f"✓ Reconstruction consistency test passed")
        print(f"  - Initial boxes shape: {initial_boxes.shape}")
        print(f"  - Denormalized offsets shape: {denormalized_offsets.shape}")
        print(f"  - Reconstructed boxes shape: {reconstructed_boxes.shape}")
        print(f"  - Validation passed: {is_valid}")
    
    def test_matcher_forward_consistency(self, temp_stats_file, sample_config):
        """Test that matcher's forward pass is consistent with expected behavior."""
        print("\n🧪 Testing Matcher Forward Consistency...")
        
        # Create matcher
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            cost_velocity=1.0,
            cost_attribute=1.0,
            cfg=sample_config
        )
        
        # Create test outputs (model predictions)
        batch_size = 1
        num_queries = 3
        num_classes = 12
        num_attributes = 11
        
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
        
        # Create test targets (ground truth)
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
                        [20.0, 30.0, 0.0, 1.5, 3.0, 1.0, 1.0, 2.0, 2.0]   # Sensor box 3
                    ]])
                },
                'virtual_camera': {
                    'boxes': torch.tensor([[
                        [5.0, 10.0, 0.0, 1.0, 2.0, 0.8, -0.3, 1.0, 1.0],  # Camera box 1
                        [25.0, 35.0, 0.0, 2.5, 4.5, 1.8, 0.8, 3.0, 3.0]   # Camera box 2
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
            
            print(f"✓ Matcher forward consistency test passed")
            print(f"  - Number of batches: {len(indices)}")
            print(f"  - Number of matches in first batch: {indices[0][0].numel()}")
            print(f"  - Matched mask shape: {matched_masks[0].shape}")
            
        except Exception as e:
            print(f"❌ Matcher forward pass failed: {e}")
            raise
    
    def test_normalization_round_trip_in_matcher(self, temp_stats_file, sample_config):
        """Test that normalization round-trip works correctly in matcher context."""
        print("\n🧪 Testing Normalization Round-Trip in Matcher Context...")
        
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
        
        # Step 1: Normalize using global normalizer
        normalized_offsets = normalize_offsets(original_offsets)
        
        # Step 2: Denormalize using matcher's method
        denormalized_offsets = matcher._denormalize_offsets(normalized_offsets)
        
        # Step 3: Check round-trip consistency
        max_diff = (original_offsets - denormalized_offsets).abs().max().item()
        
        print(f"✓ Normalization round-trip in matcher test passed")
        print(f"  - Original offsets shape: {original_offsets.shape}")
        print(f"  - Normalized offsets shape: {normalized_offsets.shape}")
        print(f"  - Denormalized offsets shape: {denormalized_offsets.shape}")
        print(f"  - Max difference: {max_diff:.2e}")
        print(f"  - Round-trip consistent: {max_diff < 1e-6}")
        
        assert max_diff < 1e-6, f"Normalization round-trip failed: max_diff={max_diff}"
    
    def test_reconstruction_with_realistic_data(self, temp_stats_file, sample_config):
        """Test reconstruction with realistic sensor and GT data."""
        print("\n🧪 Testing Reconstruction with Realistic Data...")
        
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
            [20.0, 30.0, 0.0, 1.5, 3.0, 1.0, 1.0],      # Sensor detection 3
        ])
        
        # Create realistic GT boxes
        gt_boxes = torch.tensor([
            [10.5, 20.3, 0.1, 2.1, 4.2, 1.6, 0.1],      # GT box 1 (close to sensor 1)
            [14.8, 25.2, -0.1, 2.9, 5.1, 1.9, 0.6],     # GT box 2 (close to sensor 2)
        ])
        
        # Calculate expected offsets (GT - Sensor)
        expected_center_offsets = gt_boxes[:, :3] - sensor_boxes[:2, :3]
        expected_size_offsets = torch.log(gt_boxes[:, 3:6] / sensor_boxes[:2, 3:6])
        
        # Calculate yaw offsets
        yaw_diff = gt_boxes[:, 6] - sensor_boxes[:2, 6]
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
        
        # Normalize expected offsets
        normalized_expected_offsets = normalize_offsets(expected_offsets)
        
        # Test reconstruction: normalized offsets -> denormalize -> reconstruct
        denormalized_offsets = matcher._denormalize_offsets(normalized_expected_offsets)
        reconstructed_boxes = reconstruct_boxes_consistent(
            pred_offsets=denormalized_offsets,
            initial_boxes=sensor_boxes[:2]
        )
        
        # Check that reconstructed boxes are close to GT
        box_diff = (reconstructed_boxes - gt_boxes).abs()
        max_diff = box_diff.max().item()
        
        print(f"✓ Reconstruction with realistic data test passed")
        print(f"  - Sensor boxes shape: {sensor_boxes.shape}")
        print(f"  - GT boxes shape: {gt_boxes.shape}")
        print(f"  - Expected offsets shape: {expected_offsets.shape}")
        print(f"  - Normalized offsets shape: {normalized_expected_offsets.shape}")
        print(f"  - Reconstructed boxes shape: {reconstructed_boxes.shape}")
        print(f"  - Max difference from GT: {max_diff:.6f}")
        print(f"  - Reconstruction accurate: {max_diff < 1e-5}")
        
        assert max_diff < 1e-5, f"Reconstruction accuracy failed: max_diff={max_diff}"


def run_hungarian_matcher_debug_tests():
    """Run all Hungarian Matcher debug tests."""
    print("=" * 80)
    print("🧪 HUNGARIAN MATCHER DEBUG TESTS")
    print("=" * 80)
    
    # Create test instance
    test_suite = TestHungarianMatcherDebug()
    
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
    print(f"📊 HUNGARIAN MATCHER DEBUG RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL HUNGARIAN MATCHER TESTS PASSED! The matcher is working correctly.")
    else:
        print("❌ SOME HUNGARIAN MATCHER TESTS FAILED! Check the implementation.")
    
    print("=" * 80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run all tests
    success = run_hungarian_matcher_debug_tests()
    exit(0 if success else 1) 