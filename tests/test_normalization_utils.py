"""
Comprehensive pytest tests for centralized normalization utilities.

This module tests all normalization and denormalization functions to ensure
consistency and correctness across the pipeline.

Author: Object Fusion Transformer Team
Year: 2025
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import yaml
from unittest.mock import patch, MagicMock

# Import the functions to test
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


# Global fixtures for all test classes
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
    os.unlink(temp_path)


class TestCentralizedNormalizer:
    """Test suite for CentralizedNormalizer class."""
    
    def test_initialization_with_valid_stats(self, temp_stats_file, sample_stats):
        """Test normalizer initialization with valid statistics."""
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        assert normalizer.stats is not None
        assert normalizer.stats == sample_stats
        assert torch.allclose(normalizer.center_abs_99p, torch.tensor(sample_stats['center_abs_99p']))
        assert torch.allclose(normalizer.log_size_abs_99p, torch.tensor(sample_stats['log_size_abs_99p']))
        assert torch.allclose(normalizer.velo_1p, torch.tensor(sample_stats['velocity_percentiles']['1p']))
        assert torch.allclose(normalizer.velo_99p, torch.tensor(sample_stats['velocity_percentiles']['99p']))
    
    def test_initialization_without_stats(self):
        """Test normalizer initialization without stats file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
            # Create empty file
            pass
        
        normalizer = CentralizedNormalizer(stats_path="/nonexistent/path.yaml")
        
        assert normalizer.stats is None
        assert torch.allclose(normalizer.center_abs_99p, torch.ones(3))
        assert torch.allclose(normalizer.log_size_abs_99p, torch.ones(3))
        assert torch.allclose(normalizer.velo_1p, torch.tensor([-1.0, -1.0]))
        assert torch.allclose(normalizer.velo_99p, torch.tensor([1.0, 1.0]))
    
    def test_validation_missing_fields(self):
        """Test validation with missing required fields."""
        invalid_stats = {
            'center_abs_99p': [2.0, 1.5, 0.5],
            # Missing log_size_abs_99p and velocity_percentiles
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_stats, f)
            temp_path = f.name
        
        try:
            normalizer = CentralizedNormalizer(stats_path=temp_path)
            assert normalizer.stats is None  # Should be None due to validation failure
        finally:
            os.unlink(temp_path)
    
    def test_normalize_offsets(self, temp_stats_file):
        """Test offset normalization."""
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Test input: raw offsets [N, 8]
        raw_offsets = torch.tensor([
            [1.0, 0.75, 0.25, 0.15, 0.2, 0.1, 0.5, 0.8],  # Within range
            [4.0, 3.0, 1.0, 0.6, 0.8, 0.4, 1.0, 1.0],    # Outside range (should be clamped)
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # Zero offsets
        ])
        
        normalized = normalizer.normalize_offsets(raw_offsets)
        
        # Check shape
        assert normalized.shape == raw_offsets.shape
        
        # Check range: all values should be in [-1, 1]
        assert torch.all(normalized >= -1.0)
        assert torch.all(normalized <= 1.0)
        
        # Check specific values
        # First row: should be normalized by stats
        expected_center = torch.tensor([1.0/2.0, 0.75/1.5, 0.25/0.5])
        assert torch.allclose(normalized[0, :3], expected_center, atol=1e-6)
        
        # Second row: should be clamped to [-1, 1]
        assert torch.all(normalized[1, :3] <= 1.0)
        assert torch.all(normalized[1, 3:6] <= 1.0)
        
        # Third row: should remain zero
        assert torch.allclose(normalized[2], torch.zeros(8), atol=1e-6)
    
    def test_denormalize_offsets(self, temp_stats_file):
        """Test offset denormalization."""
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Test input: normalized offsets [N, 8] in [-1, 1] range
        normalized_offsets = torch.tensor([
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        denormalized = normalizer.denormalize_offsets(normalized_offsets)
        
        # Check shape
        assert denormalized.shape == normalized_offsets.shape
        
        # Check specific values
        # First row: should be denormalized by stats
        expected_center = torch.tensor([0.5 * 2.0, 0.5 * 1.5, 0.5 * 0.5])
        assert torch.allclose(denormalized[0, :3], expected_center, atol=1e-6)
        
        # Second row: should be denormalized with negative values
        expected_center_neg = torch.tensor([-1.0 * 2.0, -1.0 * 1.5, -1.0 * 0.5])
        assert torch.allclose(denormalized[1, :3], expected_center_neg, atol=1e-6)
        
        # Third row: should remain zero
        assert torch.allclose(denormalized[2], torch.zeros(8), atol=1e-6)
    
    def test_normalize_velocity(self, temp_stats_file):
        """Test velocity normalization."""
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Test input: raw velocities [N, 2] in m/s
        raw_velocities = torch.tensor([
            [0.0, 0.0],      # Zero velocity
            [2.5, 1.5],      # Positive velocity
            [-2.5, -1.5],    # Negative velocity
            [10.0, 6.0],     # Outside range (should be clamped)
        ])
        
        normalized = normalizer.normalize_velocity(raw_velocities)
        
        # Check shape
        assert normalized.shape == raw_velocities.shape
        
        # Check range: all values should be in [-1, 1]
        assert torch.all(normalized >= -1.0)
        assert torch.all(normalized <= 1.0)
        
        # Check specific values
        # Zero velocity should map to middle of range (not -1)
        expected_zero = torch.tensor([0.0, 0.0])  # Middle of [-5, 5] and [-3, 3]
        assert torch.allclose(normalized[0], expected_zero, atol=1e-6)
        
        # Positive velocity should be normalized
        expected_pos = torch.tensor([(2.5 - (-5.0)) / (5.0 - (-5.0)) * 2 - 1, 
                                   (1.5 - (-3.0)) / (3.0 - (-3.0)) * 2 - 1])
        assert torch.allclose(normalized[1], expected_pos, atol=1e-6)
        
        # Negative velocity should be normalized
        expected_neg = torch.tensor([(-2.5 - (-5.0)) / (5.0 - (-5.0)) * 2 - 1, 
                                   (-1.5 - (-3.0)) / (3.0 - (-3.0)) * 2 - 1])
        assert torch.allclose(normalized[2], expected_neg, atol=1e-6)
    
    def test_denormalize_velocity(self, temp_stats_file):
        """Test velocity denormalization."""
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Test input: normalized velocities [N, 2] in [-1, 1] range
        normalized_velocities = torch.tensor([
            [-1.0, -1.0],    # Min values
            [0.0, 0.0],      # Zero (middle)
            [1.0, 1.0],      # Max values
        ])
        
        denormalized = normalizer.denormalize_velocity(normalized_velocities)
        
        # Check shape
        assert denormalized.shape == normalized_velocities.shape
        
        # Check specific values
        # Min values should map to 1p
        assert torch.allclose(denormalized[0], torch.tensor([-5.0, -3.0]), atol=1e-6)
        
        # Zero should map to middle of range
        expected_middle = torch.tensor([0.0, 0.0])  # (1p + 99p) / 2
        assert torch.allclose(denormalized[1], expected_middle, atol=1e-6)
        
        # Max values should map to 99p
        assert torch.allclose(denormalized[2], torch.tensor([5.0, 3.0]), atol=1e-6)
    
    def test_round_trip_consistency(self, temp_stats_file):
        """Test that normalize -> denormalize preserves original values."""
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Test offsets
        original_offsets = torch.tensor([
            [1.0, 0.75, 0.25, 0.15, 0.2, 0.1, 0.5, 0.8],
            [-0.5, -0.3, -0.1, -0.05, -0.1, -0.05, -0.2, -0.3]
        ])
        
        normalized = normalizer.normalize_offsets(original_offsets)
        denormalized = normalizer.denormalize_offsets(normalized)
        
        # Should be close to original (within numerical precision)
        assert torch.allclose(original_offsets, denormalized, atol=1e-6)
        
        # Test velocity
        original_velocities = torch.tensor([
            [2.0, 1.0],
            [-1.5, -0.5]
        ])
        
        normalized_vel = normalizer.normalize_velocity(original_velocities)
        denormalized_vel = normalizer.denormalize_velocity(normalized_vel)
        
        # Should be close to original (within numerical precision)
        assert torch.allclose(original_velocities, denormalized_vel, atol=1e-6)
    
    def test_get_stats_summary(self, temp_stats_file, sample_stats):
        """Test getting statistics summary."""
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        summary = normalizer.get_stats_summary()
        
        assert summary['status'] == 'loaded'
        assert summary['center_abs_99p'] == sample_stats['center_abs_99p']
        # Use numpy for float comparison to handle precision differences
        assert np.allclose(summary['log_size_abs_99p'], sample_stats['log_size_abs_99p'], atol=1e-6)
        assert summary['velocity_1p'] == sample_stats['velocity_percentiles']['1p']
        assert summary['velocity_99p'] == sample_stats['velocity_percentiles']['99p']
        assert summary['metadata'] == sample_stats['metadata']


class TestGlobalFunctions:
    """Test suite for global convenience functions."""
    
    def test_get_global_normalizer(self, temp_stats_file):
        """Test global normalizer singleton."""
        # Clear any existing global normalizer
        import src.oft.transformer.utils.normalization_utils as norm_utils
        norm_utils._global_normalizer = None
        
        # Test with stats file
        with patch('src.oft.transformer.utils.normalization_utils.CentralizedNormalizer') as mock_normalizer:
            mock_instance = MagicMock()
            mock_normalizer.return_value = mock_instance
            
            normalizer = get_global_normalizer()
            
            assert normalizer == mock_instance
            mock_normalizer.assert_called_once()
    
    def test_normalize_offsets_function(self, temp_stats_file):
        """Test global normalize_offsets function."""
        # Create test data
        raw_offsets = torch.tensor([[1.0, 0.75, 0.25, 0.15, 0.2, 0.1, 0.5, 0.8]])
        
        # Mock the global normalizer
        with patch('src.oft.transformer.utils.normalization_utils.get_global_normalizer') as mock_get:
            mock_normalizer = MagicMock()
            mock_normalizer.normalize_offsets.return_value = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8]])
            mock_get.return_value = mock_normalizer
            
            result = normalize_offsets(raw_offsets)
            
            mock_normalizer.normalize_offsets.assert_called_once_with(raw_offsets)
            assert result.shape == raw_offsets.shape
    
    def test_denormalize_offsets_function(self, temp_stats_file):
        """Test global denormalize_offsets function."""
        # Create test data
        normalized_offsets = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8]])
        
        # Mock the global normalizer
        with patch('src.oft.transformer.utils.normalization_utils.get_global_normalizer') as mock_get:
            mock_normalizer = MagicMock()
            mock_normalizer.denormalize_offsets.return_value = torch.tensor([[1.0, 0.75, 0.25, 0.15, 0.2, 0.1, 0.5, 0.8]])
            mock_get.return_value = mock_normalizer
            
            result = denormalize_offsets(normalized_offsets)
            
            mock_normalizer.denormalize_offsets.assert_called_once_with(normalized_offsets)
            assert result.shape == normalized_offsets.shape
    
    def test_normalize_velocity_function(self, temp_stats_file):
        """Test global normalize_velocity function."""
        # Create test data
        raw_velocities = torch.tensor([[2.0, 1.0]])
        
        # Mock the global normalizer
        with patch('src.oft.transformer.utils.normalization_utils.get_global_normalizer') as mock_get:
            mock_normalizer = MagicMock()
            mock_normalizer.normalize_velocity.return_value = torch.tensor([[0.5, 0.5]])
            mock_get.return_value = mock_normalizer
            
            result = normalize_velocity(raw_velocities)
            
            mock_normalizer.normalize_velocity.assert_called_once_with(raw_velocities)
            assert result.shape == raw_velocities.shape
    
    def test_denormalize_velocity_function(self, temp_stats_file):
        """Test global denormalize_velocity function."""
        # Create test data
        normalized_velocities = torch.tensor([[0.5, 0.5]])
        
        # Mock the global normalizer
        with patch('src.oft.transformer.utils.normalization_utils.get_global_normalizer') as mock_get:
            mock_normalizer = MagicMock()
            mock_normalizer.denormalize_velocity.return_value = torch.tensor([[2.0, 1.0]])
            mock_get.return_value = mock_normalizer
            
            result = denormalize_velocity(normalized_velocities)
            
            mock_normalizer.denormalize_velocity.assert_called_once_with(normalized_velocities)
            assert result.shape == normalized_velocities.shape


# Legacy compatibility tests removed - deprecated functions have been deleted
# The active pipeline uses the centralized normalization utilities instead.


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_invalid_stats_file(self):
        """Test handling of invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            normalizer = CentralizedNormalizer(stats_path=temp_path)
            assert normalizer.stats is None  # Should handle YAML parsing error
        finally:
            os.unlink(temp_path)
    
    def test_empty_stats_file(self):
        """Test handling of empty stats file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            normalizer = CentralizedNormalizer(stats_path=temp_path)
            assert normalizer.stats is None  # Should handle empty file
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_stats_file(self):
        """Test handling of nonexistent stats file."""
        normalizer = CentralizedNormalizer(stats_path="/nonexistent/path.yaml")
        assert normalizer.stats is None
    
    def test_normalize_without_stats(self):
        """Test normalization behavior without stats."""
        normalizer = CentralizedNormalizer(stats_path="/nonexistent/path.yaml")
        
        # Should return input unchanged when no stats available
        test_offsets = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        result = normalizer.normalize_offsets(test_offsets)
        assert torch.allclose(result, test_offsets)
        
        test_velocities = torch.tensor([[1.0, 2.0]])
        result = normalizer.normalize_velocity(test_velocities)
        assert torch.allclose(result, test_velocities)
    
    def test_denormalize_without_stats(self):
        """Test denormalization behavior without stats."""
        normalizer = CentralizedNormalizer(stats_path="/nonexistent/path.yaml")
        
        # Should return input unchanged when no stats available
        test_offsets = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        result = normalizer.denormalize_offsets(test_offsets)
        assert torch.allclose(result, test_offsets)
        
        test_velocities = torch.tensor([[1.0, 2.0]])
        result = normalizer.denormalize_velocity(test_velocities)
        assert torch.allclose(result, test_velocities)


class TestPerformance:
    """Test suite for performance characteristics."""
    
    def test_cache_functionality(self, sample_stats):
        """Test that stats are cached to avoid repeated file loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_stats, f)
            temp_path = f.name
        
        try:
            # First initialization should load from file
            normalizer1 = CentralizedNormalizer(stats_path=temp_path)
            
            # Second initialization should use cache
            normalizer2 = CentralizedNormalizer(stats_path=temp_path)
            
            # Both should have the same stats
            assert normalizer1.stats == normalizer2.stats
            assert normalizer1.stats == sample_stats
            
        finally:
            os.unlink(temp_path)
    
    def test_batch_processing(self, temp_stats_file):
        """Test processing of large batches."""
        normalizer = CentralizedNormalizer(stats_path=temp_stats_file)
        
        # Create large batch
        batch_size = 1000
        raw_offsets = torch.randn(batch_size, 8)
        raw_velocities = torch.randn(batch_size, 2)
        
        # Test normalization
        normalized_offsets = normalizer.normalize_offsets(raw_offsets)
        normalized_velocities = normalizer.normalize_velocity(raw_velocities)
        
        # Test denormalization
        denormalized_offsets = normalizer.denormalize_offsets(normalized_offsets)
        denormalized_velocities = normalizer.denormalize_velocity(normalized_velocities)
        
        # Check shapes
        assert normalized_offsets.shape == raw_offsets.shape
        assert normalized_velocities.shape == raw_velocities.shape
        assert denormalized_offsets.shape == raw_offsets.shape
        assert denormalized_velocities.shape == raw_velocities.shape
        
        # Check round-trip consistency (with clipping, some values may be clipped)
        # For clipped values, we expect the denormalized result to be within the clipping range
        
        # For offsets: Check that denormalized values are within clipping range
        # Center offsets should be within [-center_abs_99p, center_abs_99p]
        # Size offsets should be within [-log_size_abs_99p, log_size_abs_99p]
        center_denorm = denormalized_offsets[:, :3]
        size_denorm = denormalized_offsets[:, 3:6]
        
        assert torch.all(center_denorm >= -normalizer.center_abs_99p)
        assert torch.all(center_denorm <= normalizer.center_abs_99p)
        assert torch.all(size_denorm >= -normalizer.log_size_abs_99p)
        assert torch.all(size_denorm <= normalizer.log_size_abs_99p)
        
        # For velocities: Check that denormalized values are within clipping range
        # Velocities should be within [velo_1p, velo_99p]
        assert torch.all(denormalized_velocities >= normalizer.velo_1p)
        assert torch.all(denormalized_velocities <= normalizer.velo_99p)
        
        # Check that unclipped values maintain round-trip consistency
        # Find indices where values are within the clipping range
        center_in_range = torch.all(torch.abs(raw_offsets[:, :3]) <= normalizer.center_abs_99p, dim=1)
        size_in_range = torch.all(torch.abs(raw_offsets[:, 3:6]) <= normalizer.log_size_abs_99p, dim=1)
        velocity_in_range = torch.all((raw_velocities >= normalizer.velo_1p) & (raw_velocities <= normalizer.velo_99p), dim=1)
        
        # For values within range, check round-trip consistency
        if center_in_range.any():
            center_consistent = torch.allclose(
                raw_offsets[center_in_range, :3], 
                denormalized_offsets[center_in_range, :3], 
                atol=1e-6
            )
            assert center_consistent, "Center offsets within range should maintain round-trip consistency"
        
        if size_in_range.any():
            size_consistent = torch.allclose(
                raw_offsets[size_in_range, 3:6], 
                denormalized_offsets[size_in_range, 3:6], 
                atol=1e-6
            )
            assert size_consistent, "Size offsets within range should maintain round-trip consistency"
        
        if velocity_in_range.any():
            velocity_consistent = torch.allclose(
                raw_velocities[velocity_in_range], 
                denormalized_velocities[velocity_in_range], 
                atol=1e-6
            )
            assert velocity_consistent, "Velocities within range should maintain round-trip consistency"


class TestCoordinateNormalization:
    """Test coordinate normalization functions."""
    
    def test_normalize_coordinates_basic(self):
        """Test basic coordinate normalization."""
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        point_cloud_range = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        
        normalized = normalize_coordinates(coords, point_cloud_range)
        
        # Center should be at [0.5, 0.5, 0.5]
        expected = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        assert torch.allclose(normalized, expected, atol=1e-6)
    
    def test_normalize_coordinates_clipping(self):
        """Test coordinate normalization with clipping."""
        coords = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        point_cloud_range = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        
        normalized = normalize_coordinates(coords, point_cloud_range)
        
        # Should be clipped to [0, 1]
        expected = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert torch.allclose(normalized, expected, atol=1e-6)
    
    def test_denormalize_coordinates_basic(self):
        """Test basic coordinate denormalization."""
        coords_normalized = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        point_cloud_range = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        
        denormalized = denormalize_coordinates(coords_normalized, point_cloud_range)
        
        # Should map back to original range
        expected = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        assert torch.allclose(denormalized, expected, atol=1e-6)
    
    def test_coordinate_roundtrip(self):
        """Test that coordinate normalization is reversible."""
        original_coords = torch.tensor([[0.5, -0.3, 0.8], [-0.2, 0.7, -0.1]])
        point_cloud_range = torch.tensor([-2.0, -2.0, -2.0, 2.0, 2.0, 2.0])
        
        normalized = normalize_coordinates(original_coords, point_cloud_range)
        denormalized = denormalize_coordinates(normalized, point_cloud_range)
        
        # Should be very close to original (within clipping tolerance)
        assert torch.allclose(original_coords, denormalized, atol=1e-6)
    
    def test_coordinate_normalization_realistic_range(self):
        """Test with realistic point cloud range."""
        coords = torch.tensor([[0.0, 0.0, 0.0], [50.0, 50.0, 2.0]])
        point_cloud_range = torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        
        normalized = normalize_coordinates(coords, point_cloud_range)
        
        # Center should be at [0.5, 0.5, 0.625] and [0.988, 0.988, 0.875]
        expected = torch.tensor([[0.5, 0.5, 0.625], [0.988, 0.988, 0.875]])
        assert torch.allclose(normalized, expected, atol=1e-3)


class TestDimensionNormalization:
    """Test dimension normalization functions."""
    
    def test_normalize_dimensions_basic(self):
        """Test basic dimension normalization."""
        dims = torch.tensor([[1.0, 2.0, 1.5], [4.0, 8.0, 2.0]])
        
        normalized = normalize_dimensions(dims)
        
        # Should be log-transformed and clipped
        expected = torch.tensor([
            [0.0, 0.693, 0.405],  # log(1), log(2), log(1.5)
            [1.386, 2.079, 0.693]  # log(4), log(8), log(2)
        ])
        assert torch.allclose(normalized, expected, atol=1e-3)
    
    def test_normalize_dimensions_small_values(self):
        """Test dimension normalization with small values."""
        dims = torch.tensor([[0.1, 0.05, 0.2], [0.01, 0.001, 0.5]])
        
        normalized = normalize_dimensions(dims)
        
        # Should handle small values gracefully (minimum 0.1)
        expected = torch.tensor([
            [-2.303, -2.303, -1.609],  # log(0.1) for all
            [-2.303, -2.303, -0.693]   # log(0.1), log(0.1), log(0.5)
        ])
        assert torch.allclose(normalized, expected, atol=1e-3)
    
    def test_normalize_dimensions_clipping(self):
        """Test dimension normalization with clipping."""
        dims = torch.tensor([[0.01, 0.001, 0.0001], [100.0, 200.0, 50.0]])
        
        normalized = normalize_dimensions(dims)
        
        # Should be clipped to [-2.0, 2.0]
        assert torch.all(normalized >= -2.0)
        assert torch.all(normalized <= 2.0)
    
    def test_denormalize_dimensions_basic(self):
        """Test basic dimension denormalization."""
        dims_normalized = torch.tensor([[0.0, 0.693, 0.405], [1.386, 2.079, 0.693]])
        
        denormalized = denormalize_dimensions(dims_normalized)
        
        # Should be exp-transformed
        expected = torch.tensor([
            [1.0, 2.0, 1.5],  # exp(0), exp(0.693), exp(0.405)
            [4.0, 8.0, 2.0]   # exp(1.386), exp(2.079), exp(0.693)
        ])
        assert torch.allclose(denormalized, expected, atol=1e-3)
    
    def test_dimension_roundtrip(self):
        """Test that dimension normalization is reversible."""
        original_dims = torch.tensor([[1.5, 3.0, 2.5], [5.0, 10.0, 1.8]])
        
        normalized = normalize_dimensions(original_dims)
        denormalized = denormalize_dimensions(normalized)
        
        # Should be very close to original (within clipping tolerance)
        assert torch.allclose(original_dims, denormalized, atol=1e-3)
    
    def test_dimension_normalization_realistic_values(self):
        """Test with realistic vehicle dimensions."""
        # Typical car dimensions: [width, length, height]
        car_dims = torch.tensor([[1.8, 4.5, 1.5], [2.5, 8.0, 3.0]])  # car, truck
        
        normalized = normalize_dimensions(car_dims)
        
        # Should be reasonable log values
        expected = torch.tensor([
            [0.588, 1.504, 0.405],  # log(1.8), log(4.5), log(1.5)
            [0.916, 2.079, 1.099]   # log(2.5), log(8.0), log(3.0)
        ])
        assert torch.allclose(normalized, expected, atol=1e-3)


class TestNormalizationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_tensors(self):
        """Test normalization with empty tensors."""
        empty_coords = torch.empty(0, 3)
        empty_dims = torch.empty(0, 3)
        point_cloud_range = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        
        # Should handle empty tensors gracefully
        normalized_coords = normalize_coordinates(empty_coords, point_cloud_range)
        normalized_dims = normalize_dimensions(empty_dims)
        
        assert normalized_coords.shape == (0, 3)
        assert normalized_dims.shape == (0, 3)
    
    def test_single_element_tensors(self):
        """Test normalization with single element tensors."""
        coords = torch.tensor([[0.5, 0.3, 0.7]])
        dims = torch.tensor([[2.0, 4.0, 1.5]])
        point_cloud_range = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        
        normalized_coords = normalize_coordinates(coords, point_cloud_range)
        normalized_dims = normalize_dimensions(dims)
        
        assert normalized_coords.shape == (1, 3)
        assert normalized_dims.shape == (1, 3)
    
    def test_device_consistency(self):
        """Test that functions work on different devices."""
        if torch.cuda.is_available():
            coords = torch.tensor([[0.0, 0.0, 0.0]], device='cuda')
            dims = torch.tensor([[1.0, 2.0, 1.5]], device='cuda')
            point_cloud_range = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda')
            
            normalized_coords = normalize_coordinates(coords, point_cloud_range)
            normalized_dims = normalize_dimensions(dims)
            
            assert normalized_coords.device == coords.device
            assert normalized_dims.device == dims.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 