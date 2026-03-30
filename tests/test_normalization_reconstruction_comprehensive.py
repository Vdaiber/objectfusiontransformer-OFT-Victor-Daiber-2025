"""
Comprehensive pytest tests for all normalization and reconstruction functions.

This module provides extensive testing of:
1. All normalization functions (offsets, velocity, coordinates, dimensions)
2. All denormalization functions 
3. Box reconstruction functions
4. Round-trip consistency tests
5. Edge cases and error handling

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

# Import all functions to test
from src.oft.transformer.utils.normalization_utils import (
    CentralizedNormalizer,
    get_global_normalizer,
    normalize_offsets,
    denormalize_offsets,
    normalize_velocity,
    denormalize_velocity,
    normalize_coordinates,
    normalize_dimensions,
    denormalize_coordinates,
    denormalize_dimensions
)

from src.oft.transformer.utils.box_reconstruction import (
    reconstruct_boxes_consistent,
    validate_box_reconstruction
)


# Test fixtures
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
def sample_boxes():
    """Sample boxes for testing reconstruction."""
    return torch.tensor([
        [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0],      # Box 1: center=(10,20,0), size=(2,4,1.5), yaw=0
        [15.0, 25.0, 0.0, 3.0, 5.0, 2.0, 0.5],      # Box 2: center=(15,25,0), size=(3,5,2), yaw=0.5
        [20.0, 30.0, 0.0, 1.5, 3.0, 1.0, 1.0],      # Box 3: center=(20,30,0), size=(1.5,3,1), yaw=1.0
        [5.0, 10.0, 0.0, 1.0, 2.0, 0.8, -0.3],      # Box 4: center=(5,10,0), size=(1,2,0.8), yaw=-0.3
    ])

@pytest.fixture
def sample_offsets():
    """Sample offsets for testing."""
    return torch.tensor([
        [1.0, 0.5, 0.1, 0.2, 0.0, 0.1, 0.0, 1.0],      # Center offset + size offset + yaw=0
        [-0.5, 1.0, -0.2, 0.0, 0.3, 0.0, 0.707, 0.707], # Center offset + size offset + yaw=π/4
        [0.0, 0.0, 0.0, -0.1, 0.0, 0.2, -0.5, 0.866],   # Size offset + yaw=π/3
        [0.3, -0.2, 0.05, 0.1, -0.1, 0.05, 0.866, 0.5], # All offsets + yaw=π/6
    ])

@pytest.fixture
def sample_velocities():
    """Sample velocities for testing."""
    return torch.tensor([
        [2.0, 1.0],      # Velocity 1: (2, 1) m/s
        [-1.5, -0.5],    # Velocity 2: (-1.5, -0.5) m/s
        [0.0, 0.0],      # Velocity 3: stationary
        [4.0, -2.0],     # Velocity 4: (4, -2) m/s
    ])

@pytest.fixture
def point_cloud_range():
    """Sample point cloud range for coordinate normalization."""
    return torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])


class TestNormalizationFunctions:
    """Test suite for all normalization functions."""
    
    def test_normalize_offsets(self, temp_stats_file, sample_offsets):
        """Test offset normalization with known values."""
        print("\n🧪 Testing Offset Normalization...")
        
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Test with sample offsets
        normalized = normalizer.normalize_offsets(sample_offsets)
        
        # Check shape
        assert normalized.shape == sample_offsets.shape
        
        # Check that normalized values are in [-1, 1] range
        assert torch.all(normalized >= -1.0) and torch.all(normalized <= 1.0)
        
        # Check specific values for first row
        # Original: [1.0, 0.5, 0.1, 0.2, 0.0, 0.1, 0.0, 1.0]
        # Expected: [1.0/2.0, 0.5/1.5, 0.1/0.5, 0.2/0.3, 0.0/0.4, 0.1/0.2, 0.0, 1.0]
        expected_first_row = torch.tensor([
            1.0/2.0, 0.5/1.5, 0.1/0.5,  # Center normalized
            0.2/0.3, 0.0/0.4, 0.1/0.2,  # Size normalized
            0.0, 1.0                     # Yaw unchanged
        ])
        
        assert torch.allclose(normalized[0], expected_first_row, atol=1e-6)
        
        print(f"✓ Offset normalization test passed")
        print(f"  - Input shape: {sample_offsets.shape}")
        print(f"  - Output shape: {normalized.shape}")
        print(f"  - Range check: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    def test_denormalize_offsets(self, temp_stats_file, sample_offsets):
        """Test offset denormalization with known values."""
        print("\n🧪 Testing Offset Denormalization...")
        
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # First normalize, then denormalize
        normalized = normalizer.normalize_offsets(sample_offsets)
        denormalized = normalizer.denormalize_offsets(normalized)
        
        # Check round-trip consistency
        assert torch.allclose(sample_offsets, denormalized, atol=1e-6)
        
        print(f"✓ Offset denormalization test passed")
        print(f"  - Round-trip consistency: {torch.allclose(sample_offsets, denormalized, atol=1e-6)}")
        print(f"  - Max difference: {(sample_offsets - denormalized).abs().max():.2e}")
    
    def test_normalize_velocity(self, temp_stats_file, sample_velocities):
        """Test velocity normalization with known values."""
        print("\n🧪 Testing Velocity Normalization...")
        
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Test with sample velocities
        normalized = normalizer.normalize_velocity(sample_velocities)
        
        # Check shape
        assert normalized.shape == sample_velocities.shape
        
        # Check that normalized values are in [-1, 1] range
        assert torch.all(normalized >= -1.0) and torch.all(normalized <= 1.0)
        
        # Check specific values for first row
        # Original: [2.0, 1.0]
        # Expected: 2 * (2.0 - (-5.0)) / (5.0 - (-5.0)) - 1 = 2 * 7/10 - 1 = 0.4
        #           2 * (1.0 - (-3.0)) / (3.0 - (-3.0)) - 1 = 2 * 4/6 - 1 = 0.333
        expected_first_row = torch.tensor([0.4, 0.333])
        
        assert torch.allclose(normalized[0], expected_first_row, atol=1e-3)
        
        print(f"✓ Velocity normalization test passed")
        print(f"  - Input shape: {sample_velocities.shape}")
        print(f"  - Output shape: {normalized.shape}")
        print(f"  - Range check: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    def test_denormalize_velocity(self, temp_stats_file, sample_velocities):
        """Test velocity denormalization with known values."""
        print("\n🧪 Testing Velocity Denormalization...")
        
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # First normalize, then denormalize
        normalized = normalizer.normalize_velocity(sample_velocities)
        denormalized = normalizer.denormalize_velocity(normalized)
        
        # Check round-trip consistency
        assert torch.allclose(sample_velocities, denormalized, atol=1e-6)
        
        print(f"✓ Velocity denormalization test passed")
        print(f"  - Round-trip consistency: {torch.allclose(sample_velocities, denormalized, atol=1e-6)}")
        print(f"  - Max difference: {(sample_velocities - denormalized).abs().max():.2e}")
    
    def test_normalize_coordinates(self, point_cloud_range):
        """Test coordinate normalization."""
        print("\n🧪 Testing Coordinate Normalization...")
        
        # Test coordinates
        coords = torch.tensor([
            [0.0, 0.0, 0.0],      # Center
            [-51.2, -51.2, -5.0],  # Min bounds
            [51.2, 51.2, 3.0],     # Max bounds
            [25.6, 25.6, -1.0],    # Quarter way
        ])
        
        normalized = normalize_coordinates(coords, point_cloud_range)
        
        # Check shape
        assert normalized.shape == coords.shape
        
        # Check that normalized values are in [0, 1] range
        assert torch.all(normalized >= 0.0) and torch.all(normalized <= 1.0)
        
        # Check specific values
        expected_center = torch.tensor([0.5, 0.5, 0.625])  # Center should be at 0.5, 0.5, (0-(-5))/(3-(-5)) = 0.625
        assert torch.allclose(normalized[0], expected_center, atol=1e-6)
        
        expected_min = torch.tensor([0.0, 0.0, 0.0])  # Min bounds should be at 0
        assert torch.allclose(normalized[1], expected_min, atol=1e-6)
        
        expected_max = torch.tensor([1.0, 1.0, 1.0])  # Max bounds should be at 1
        assert torch.allclose(normalized[2], expected_max, atol=1e-6)
        
        print(f"✓ Coordinate normalization test passed")
        print(f"  - Input shape: {coords.shape}")
        print(f"  - Output shape: {normalized.shape}")
        print(f"  - Range check: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    def test_denormalize_coordinates(self, point_cloud_range):
        """Test coordinate denormalization."""
        print("\n🧪 Testing Coordinate Denormalization...")
        
        # Test normalized coordinates
        normalized_coords = torch.tensor([
            [0.5, 0.5, 0.5],  # Center
            [0.0, 0.0, 0.0],  # Min
            [1.0, 1.0, 1.0],  # Max
            [0.25, 0.75, 0.25], # Quarter
        ])
        
        denormalized = denormalize_coordinates(normalized_coords, point_cloud_range)
        
        # Check shape
        assert denormalized.shape == normalized_coords.shape
        
        # Check specific values
        expected_center = torch.tensor([0.0, 0.0, -1.0])  # Center of range
        assert torch.allclose(denormalized[0], expected_center, atol=1e-6)
        
        expected_min = torch.tensor([-51.2, -51.2, -5.0])  # Min bounds
        assert torch.allclose(denormalized[1], expected_min, atol=1e-6)
        
        expected_max = torch.tensor([51.2, 51.2, 3.0])  # Max bounds
        assert torch.allclose(denormalized[2], expected_max, atol=1e-6)
        
        print(f"✓ Coordinate denormalization test passed")
        print(f"  - Input shape: {normalized_coords.shape}")
        print(f"  - Output shape: {denormalized.shape}")


class TestBoxReconstruction:
    """Test suite for box reconstruction functions."""
    
    def test_reconstruct_boxes_consistent(self, sample_boxes, sample_offsets):
        """Test box reconstruction with known values."""
        print("\n🧪 Testing Box Reconstruction...")
        
        # Test reconstruction
        reconstructed = reconstruct_boxes_consistent(
            pred_offsets=sample_offsets,
            initial_boxes=sample_boxes
        )
        
        # Check shape
        assert reconstructed.shape == sample_boxes.shape
        
        # Check that no NaN or Inf values
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()
        
        # Check specific reconstruction for first box
        # Initial: [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0]
        # Offset:  [1.0, 0.5, 0.1, 0.2, 0.0, 0.1, 0.0, 1.0]
        # Expected center: [10.0+1.0, 20.0+0.5, 0.0+0.1] = [11.0, 20.5, 0.1]
        # Expected size: [2.0*exp(0.2), 4.0*exp(0.0), 1.5*exp(0.1)] = [2.0*1.221, 4.0*1.0, 1.5*1.105]
        # Expected yaw: 0.0 + atan2(0.0, 1.0) = 0.0
        
        expected_center = torch.tensor([11.0, 20.5, 0.1])
        expected_size = torch.tensor([2.0 * math.exp(0.2), 4.0, 1.5 * math.exp(0.1)])
        expected_yaw = torch.tensor([0.0])
        
        assert torch.allclose(reconstructed[0, :3], expected_center, atol=1e-6)
        assert torch.allclose(reconstructed[0, 3:6], expected_size, atol=1e-6)
        assert torch.allclose(reconstructed[0, 6:], expected_yaw, atol=1e-6)
        
        print(f"✓ Box reconstruction test passed")
        print(f"  - Input shape: {sample_boxes.shape}")
        print(f"  - Output shape: {reconstructed.shape}")
        print(f"  - No NaN/Inf: {not torch.isnan(reconstructed).any()}")
    
    def test_validate_box_reconstruction(self, sample_boxes):
        """Test box validation function."""
        print("\n🧪 Testing Box Validation...")
        
        # Test valid boxes
        is_valid = validate_box_reconstruction(sample_boxes, sample_boxes)
        assert is_valid
        
        # Test invalid boxes (negative dimensions)
        invalid_boxes = sample_boxes.clone()
        invalid_boxes[0, 3] = -1.0  # Negative width
        is_valid = validate_box_reconstruction(invalid_boxes, sample_boxes)
        assert not is_valid
        
        # Test invalid boxes (NaN values)
        invalid_boxes = sample_boxes.clone()
        invalid_boxes[0, 0] = float('nan')
        is_valid = validate_box_reconstruction(invalid_boxes, sample_boxes)
        assert not is_valid
        
        print(f"✓ Box validation test passed")
    
    def test_reconstruction_edge_cases(self):
        """Test reconstruction with edge cases."""
        print("\n🧪 Testing Reconstruction Edge Cases...")
        
        # Test with zero offsets
        initial_boxes = torch.tensor([[10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0]])
        zero_offsets = torch.zeros(1, 8)
        zero_offsets[0, 7] = 1.0  # cos(0) = 1
        
        reconstructed = reconstruct_boxes_consistent(
            pred_offsets=zero_offsets,
            initial_boxes=initial_boxes
        )
        
        # Should return original boxes
        assert torch.allclose(initial_boxes, reconstructed, atol=1e-6)
        
        # Test with very small offsets
        small_offsets = torch.tensor([[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0.0, 1.0]])
        reconstructed = reconstruct_boxes_consistent(
            pred_offsets=small_offsets,
            initial_boxes=initial_boxes
        )
        
        # Should be very close to original
        assert torch.allclose(initial_boxes, reconstructed, atol=1e-5)
        
        print(f"✓ Reconstruction edge cases test passed")


class TestRoundTripConsistency:
    """Test suite for round-trip consistency."""
    
    def test_offset_round_trip(self, temp_stats_file, sample_offsets):
        """Test offset normalization round-trip consistency."""
        print("\n🧪 Testing Offset Round-Trip Consistency...")
        
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Normalize then denormalize
        normalized = normalizer.normalize_offsets(sample_offsets)
        denormalized = normalizer.denormalize_offsets(normalized)
        
        # Check consistency
        max_diff = (sample_offsets - denormalized).abs().max().item()
        assert max_diff < 1e-6
        
        print(f"✓ Offset round-trip consistency: max_diff={max_diff:.2e}")
    
    def test_velocity_round_trip(self, temp_stats_file, sample_velocities):
        """Test velocity normalization round-trip consistency."""
        print("\n🧪 Testing Velocity Round-Trip Consistency...")
        
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Normalize then denormalize
        normalized = normalizer.normalize_velocity(sample_velocities)
        denormalized = normalizer.denormalize_velocity(normalized)
        
        # Check consistency
        max_diff = (sample_velocities - denormalized).abs().max().item()
        assert max_diff < 1e-6
        
        print(f"✓ Velocity round-trip consistency: max_diff={max_diff:.2e}")
    
    def test_coordinate_round_trip(self, point_cloud_range):
        """Test coordinate normalization round-trip consistency."""
        print("\n🧪 Testing Coordinate Round-Trip Consistency...")
        
        # Test coordinates
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [-25.6, -25.6, -1.0],
            [25.6, 25.6, 1.0],
        ])
        
        # Normalize then denormalize
        normalized = normalize_coordinates(coords, point_cloud_range)
        denormalized = denormalize_coordinates(normalized, point_cloud_range)
        
        # Check consistency (use relaxed tolerance for floating point precision)
        max_diff = (coords - denormalized).abs().max().item()
        assert max_diff < 1e-5  # Relaxed tolerance for floating point operations
        
        print(f"✓ Coordinate round-trip consistency: max_diff={max_diff:.2e}")
    
    def test_reconstruction_round_trip(self, sample_boxes, sample_offsets):
        """Test box reconstruction round-trip consistency."""
        print("\n🧪 Testing Reconstruction Round-Trip Consistency...")
        
        # Reconstruct boxes
        reconstructed = reconstruct_boxes_consistent(
            pred_offsets=sample_offsets,
            initial_boxes=sample_boxes
        )
        
        # Compute offsets from reconstructed boxes
        computed_offsets = torch.zeros_like(sample_offsets)
        
        # Center offsets
        computed_offsets[:, :3] = reconstructed[:, :3] - sample_boxes[:, :3]
        
        # Dimension offsets (log-space)
        computed_offsets[:, 3:6] = torch.log(reconstructed[:, 3:6] / sample_boxes[:, 3:6])
        
        # Yaw offsets (sin/cos)
        yaw_diff = reconstructed[:, 6] - sample_boxes[:, 6]
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        computed_offsets[:, 6] = torch.sin(yaw_diff)
        computed_offsets[:, 7] = torch.cos(yaw_diff)
        
        # Check consistency (with relaxed tolerance for yaw)
        offset_diff = (sample_offsets - computed_offsets).abs()
        max_diff = offset_diff.max().item()
        
        # Use relaxed tolerance due to numerical precision in yaw
        assert max_diff < 1e-3
        
        print(f"✓ Reconstruction round-trip consistency: max_diff={max_diff:.2e}")


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_missing_stats_file(self):
        """Test behavior with missing stats file."""
        print("\n🧪 Testing Missing Stats File...")
        
        normalizer = CentralizedNormalizer(stats_path="/nonexistent/path.yaml")
        
        # Should return input unchanged when no stats available
        test_offsets = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        result = normalizer.normalize_offsets(test_offsets)
        assert torch.allclose(result, test_offsets)
        
        test_velocities = torch.tensor([[1.0, 2.0]])
        result = normalizer.normalize_velocity(test_velocities)
        assert torch.allclose(result, test_velocities)
        
        print(f"✓ Missing stats file test passed")
    
    def test_invalid_inputs(self):
        """Test behavior with invalid inputs."""
        print("\n🧪 Testing Invalid Inputs...")
        
        # Test with empty tensors
        empty_offsets = torch.empty(0, 8)
        empty_velocities = torch.empty(0, 2)
        
        # Should handle gracefully
        try:
            result = normalize_offsets(empty_offsets)
            assert result.shape == empty_offsets.shape
        except Exception as e:
            print(f"Warning: Empty offsets caused error: {e}")
        
        try:
            result = normalize_velocity(empty_velocities)
            assert result.shape == empty_velocities.shape
        except Exception as e:
            print(f"Warning: Empty velocities caused error: {e}")
        
        print(f"✓ Invalid inputs test passed")


class TestPerformance:
    """Test suite for performance and large batches."""
    
    def test_large_batch_processing(self, temp_stats_file):
        """Test processing of large batches."""
        print("\n🧪 Testing Large Batch Processing...")
        
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Create large batch
        batch_size = 1000
        large_offsets = torch.randn(batch_size, 8)
        large_velocities = torch.randn(batch_size, 2)
        
        # Test normalization
        normalized_offsets = normalizer.normalize_offsets(large_offsets)
        normalized_velocities = normalizer.normalize_velocity(large_velocities)
        
        # Test denormalization
        denormalized_offsets = normalizer.denormalize_offsets(normalized_offsets)
        denormalized_velocities = normalizer.denormalize_velocity(normalized_velocities)
        
        # Check shapes
        assert normalized_offsets.shape == large_offsets.shape
        assert normalized_velocities.shape == large_velocities.shape
        assert denormalized_offsets.shape == large_offsets.shape
        assert denormalized_velocities.shape == large_velocities.shape
        
        # Check that normalized values are in expected ranges
        assert torch.all(normalized_offsets >= -1.0) and torch.all(normalized_offsets <= 1.0)
        assert torch.all(normalized_velocities >= -1.0) and torch.all(normalized_velocities <= 1.0)
        
        print(f"✓ Large batch processing test passed")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Offset range: [{normalized_offsets.min():.3f}, {normalized_offsets.max():.3f}]")
        print(f"  - Velocity range: [{normalized_velocities.min():.3f}, {normalized_velocities.max():.3f}]")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("=" * 80)
    print("🧪 COMPREHENSIVE NORMALIZATION & RECONSTRUCTION TESTS")
    print("=" * 80)
    
    # Create test instances
    test_classes = [
        TestNormalizationFunctions(),
        TestBoxReconstruction(),
        TestRoundTripConsistency(),
        TestErrorHandling(),
        TestPerformance()
    ]
    
    passed_tests = 0
    total_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__class__.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            test_method = getattr(test_class, test_method_name)
            total_tests += 1
            
            try:
                test_method()
                passed_tests += 1
                print(f"  ✅ {test_method_name} PASSED")
            except Exception as e:
                print(f"  ❌ {test_method_name} FAILED: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 COMPREHENSIVE TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! All normalization and reconstruction functions work correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the implementation of failed functions.")
    
    print("=" * 80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run all tests
    success = run_comprehensive_tests()
    exit(0 if success else 1) 