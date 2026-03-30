"""
Simple test for Hungarian Matcher to identify the core issue.

This test focuses on the fundamental problem: the matcher loads real normalization
statistics that don't match our test data, causing inconsistencies.

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


def test_normalization_inconsistency():
    """Test to identify the normalization inconsistency issue."""
    print("\n🧪 Testing Normalization Inconsistency Issue...")
    
    # Create a simple matcher
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=1.0,
        cost_giou=1.0,
        cost_velocity=1.0,
        cost_attribute=1.0,
        cfg={'dataset': {'virtual_sensors': []}}
    )
    
    # Check what normalization stats the matcher loaded
    print(f"Matcher loaded stats: {matcher.norm_stats}")
    
    if matcher.norm_stats:
        print(f"  - Center 99p: {matcher.norm_stats['center_abs_99p']}")
        print(f"  - Size 99p: {matcher.norm_stats['log_size_abs_99p']}")
        print(f"  - Velocity 1p/99p: {matcher.norm_stats['velocity_percentiles']}")
    
    # Test with simple offsets
    test_offsets = torch.tensor([
        [1.0, 0.5, 0.1, 0.2, 0.0, 0.1, 0.0, 1.0],
        [-0.5, 1.0, -0.2, 0.0, 0.3, 0.0, 0.707, 0.707],
    ])
    
    # Test normalization round-trip
    normalized = matcher.normalizer.normalize_offsets(test_offsets)
    denormalized = matcher.normalizer.denormalize_offsets(normalized)
    
    max_diff = (test_offsets - denormalized).abs().max().item()
    
    print(f"✓ Normalization round-trip test:")
    print(f"  - Original offsets: {test_offsets}")
    print(f"  - Normalized offsets: {normalized}")
    print(f"  - Denormalized offsets: {denormalized}")
    print(f"  - Max difference: {max_diff:.6f}")
    print(f"  - Round-trip consistent: {max_diff < 1e-6}")
    
    # The issue is that the matcher loads real stats, but we're testing with different values
    # This is expected to fail because the real stats don't match our test data
    print(f"⚠️ Expected failure: Real normalization stats don't match test data")
    
    return max_diff


def test_reconstruction_with_real_stats():
    """Test reconstruction using the actual normalization stats loaded by the matcher."""
    print("\n🧪 Testing Reconstruction with Real Stats...")
    
    # Create a simple matcher
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=1.0,
        cost_giou=1.0,
        cost_velocity=1.0,
        cost_attribute=1.0,
        cfg={'dataset': {'virtual_sensors': []}}
    )
    
    # Get the actual stats used by the matcher
    if not matcher.norm_stats:
        print("⚠️ No normalization stats available - skipping test")
        return
    
    center_abs_99p = torch.tensor(matcher.norm_stats['center_abs_99p'])
    log_size_abs_99p = torch.tensor(matcher.norm_stats['log_size_abs_99p'])
    
    print(f"Using real stats:")
    print(f"  - Center 99p: {center_abs_99p}")
    print(f"  - Size 99p: {log_size_abs_99p}")
    
    # Create test data using the actual stats
    sensor_boxes = torch.tensor([
        [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0],
        [15.0, 25.0, 0.0, 3.0, 5.0, 2.0, 0.5],
    ])
    
    # Create offsets that are within the normalization range
    small_center_offsets = center_abs_99p * 0.1  # 10% of normalization range
    small_size_offsets = log_size_abs_99p * 0.1  # 10% of normalization range
    
    test_offsets = torch.tensor([
        [small_center_offsets[0], small_center_offsets[1], small_center_offsets[2], 
         small_size_offsets[0], small_size_offsets[1], small_size_offsets[2], 0.0, 1.0],
        [-small_center_offsets[0], -small_center_offsets[1], -small_center_offsets[2], 
         -small_size_offsets[0], -small_size_offsets[1], -small_size_offsets[2], 0.0, 1.0],
    ])
    
    # Test normalization round-trip with real stats
    normalized = matcher.normalizer.normalize_offsets(test_offsets)
    denormalized = matcher.normalizer.denormalize_offsets(normalized)
    
    max_diff = (test_offsets - denormalized).abs().max().item()
    
    print(f"✓ Normalization round-trip with real stats:")
    print(f"  - Max difference: {max_diff:.6f}")
    print(f"  - Round-trip consistent: {max_diff < 1e-6}")
    
    # Test reconstruction
    reconstructed_boxes = reconstruct_boxes_consistent(
        pred_offsets=denormalized,
        initial_boxes=sensor_boxes
    )
    
    # Calculate expected boxes
    expected_centers = sensor_boxes[:, :3] + denormalized[:, :3]
    expected_sizes = sensor_boxes[:, 3:6] * torch.exp(denormalized[:, 3:6])
    expected_yaws = sensor_boxes[:, 6:7] + torch.atan2(denormalized[:, 6:7], denormalized[:, 7:8])
    
    expected_boxes = torch.cat([expected_centers, expected_sizes, expected_yaws], dim=-1)
    
    reconstruction_diff = (reconstructed_boxes - expected_boxes).abs().max().item()
    
    print(f"✓ Reconstruction test:")
    print(f"  - Reconstruction difference: {reconstruction_diff:.6f}")
    print(f"  - Reconstruction accurate: {reconstruction_diff < 1e-6}")
    
    # These should work with the real stats
    assert max_diff < 1e-6, f"Normalization round-trip failed: max_diff={max_diff}"
    assert reconstruction_diff < 1e-6, f"Reconstruction failed: diff={reconstruction_diff}"


def test_matcher_forward_simple():
    """Test matcher forward pass with simple, consistent data."""
    print("\n🧪 Testing Matcher Forward Pass with Simple Data...")
    
    # Create matcher
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=1.0,
        cost_giou=1.0,
        cost_velocity=1.0,
        cost_attribute=1.0,
        cfg={'dataset': {'virtual_sensors': [{'name': 'virtual_lidar', 'enabled': True}]}}
    )
    
    # Create simple test data with matching sizes
    batch_size = 1
    num_queries = 2
    num_classes = 12
    num_attributes = 11
    
    # Create outputs with small, normalized offsets
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
            }
        }
    }
    
    # Run matcher
    try:
        indices, matched_masks = matcher(outputs, targets)
        
        print(f"✓ Matcher forward pass successful:")
        print(f"  - Number of batches: {len(indices)}")
        print(f"  - Number of matches: {indices[0][0].numel()}")
        print(f"  - Source indices: {indices[0][0].tolist()}")
        print(f"  - Target indices: {indices[0][1].tolist()}")
        
        # Should have matches
        assert len(indices) == 1
        src_idx, tgt_idx = indices[0]
        assert src_idx.numel() > 0
        assert tgt_idx.numel() > 0
        
    except Exception as e:
        print(f"❌ Matcher forward pass failed: {e}")
        raise


def run_simple_tests():
    """Run all simple tests."""
    print("=" * 80)
    print("🧪 SIMPLE HUNGARIAN MATCHER TESTS")
    print("=" * 80)
    
    tests = [
        test_normalization_inconsistency,
        test_reconstruction_with_real_stats,
        test_matcher_forward_simple,
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        try:
            test()
            passed_tests += 1
            print(f"✅ {test.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 SIMPLE TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL SIMPLE TESTS PASSED!")
    else:
        print("❌ SOME SIMPLE TESTS FAILED!")
    
    print("=" * 80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run all tests
    success = run_simple_tests()
    exit(0 if success else 1) 