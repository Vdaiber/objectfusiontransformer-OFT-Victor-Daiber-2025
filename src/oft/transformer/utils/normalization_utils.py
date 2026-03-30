"""
Centralized Normalization Utilities for Object Fusion Transformer Pipeline.

This module provides a single source of truth for all normalization and denormalization
operations, ensuring consistency across the entire pipeline.


"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import numpy as np
import yaml
import os
import math
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global cache for normalization stats to avoid repeated file loading
_NORMALIZATION_STATS_CACHE = None


class CentralizedNormalizer(nn.Module):
    """
    Centralized normalizer that provides consistent normalization/denormalization
    across all pipeline components.
    
    This class ensures that all normalization operations use the same statistics
    from /app/config/normalization_stats.yaml and follow the same protocols.
    """
    
    def __init__(self, stats_path: str = "/app/config/normalization_stats.yaml"):
        """Initialize the centralized normalizer.
        
        Args:
            stats_path: Path to normalization statistics file
        """
        super().__init__()
        self.stats_path = stats_path
        self.stats = self._load_normalization_stats()
        self._validate_stats()
        
        # Register normalization tensors as buffers for GPU compatibility
        if self.stats:
            self._register_normalization_tensors()
        else:
            logger.warning("⚠️ CentralizedNormalizer: No stats found - using identity normalization")
            self._register_identity_tensors()
    
    def _load_normalization_stats(self) -> Optional[Dict[str, Any]]:
        """Load normalization statistics from the specified path.
        
        Returns:
            Dictionary containing normalization statistics or None if not found
        """
        # Try to load from specified path first
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r') as f:
                    stats = yaml.safe_load(f)
                return stats
            except Exception as e:
                logger.error(f"❌ Error loading {self.stats_path}: {e}")
                return None
        
        # Only use fallback paths for default stats path
        if self.stats_path == "/app/config/normalization_stats.yaml":
            fallback_paths = [
                "config/normalization_stats.yaml",
                "app/config/normalization_stats.yaml",
                "src/oft/transformer/utils/normalization_stats.yaml"
            ]
            
            for path in fallback_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            stats = yaml.safe_load(f)
                        return stats
                    except Exception as e:
                        logger.warning(f"⚠️ Error loading fallback {path}: {e}")
                        continue
        
        logger.warning(f"❌ No normalization stats found at: {self.stats_path}")
        return None
    
    def _validate_stats(self):
        """Validate that loaded statistics contain required fields."""
        if not self.stats:
            return
        
        required_fields = [
            'center_abs_99p',
            'log_size_abs_99p', 
            'velocity_percentiles'  # Legacy field
        ]
        
        missing_fields = [field for field in required_fields if field not in self.stats]
        if missing_fields:
            logger.error(f"❌ Missing required fields in normalization stats: {missing_fields}")
            self.stats = None
            return
        
        # Validate velocity percentiles structure (legacy)
        if 'velocity_percentiles' in self.stats:
            velo_stats = self.stats['velocity_percentiles']
            if '1p' not in velo_stats or '99p' not in velo_stats:
                logger.error("❌ Invalid velocity_percentiles structure")
                self.stats = None
                return
        
        # Validate new velocity statistics (optional for backwards compatibility)
        if 'velocity_offset_percentiles' in self.stats:
            offset_stats = self.stats['velocity_offset_percentiles']
            if '1p' not in offset_stats or '99p' not in offset_stats:
                logger.warning("⚠️ Invalid velocity_offset_percentiles structure")
        
        if 'absolute_velocity_percentiles' in self.stats:
            abs_stats = self.stats['absolute_velocity_percentiles']
            if '1p' not in abs_stats or '99p' not in abs_stats:
                logger.warning("⚠️ Invalid absolute_velocity_percentiles structure")
        
    
    def _register_normalization_tensors(self):
        """Store normalization statistics as CPU-only tensors to avoid CUDA worker issues."""
        if not self.stats:
            return
        
        # Store as regular attributes (NOT buffers) so they never move to GPU
        # This ensures DataLoader workers can always access them on CPU
        self.center_abs_99p = torch.tensor(self.stats['center_abs_99p'], dtype=torch.float64)
        self.log_size_abs_99p = torch.tensor(self.stats['log_size_abs_99p'], dtype=torch.float64)
        
        # Legacy velocity percentiles (for backwards compatibility)
        self.velo_1p = torch.tensor(self.stats['velocity_percentiles']['1p'], dtype=torch.float64)
        self.velo_99p = torch.tensor(self.stats['velocity_percentiles']['99p'], dtype=torch.float64)
        
        # NEW: Velocity offset percentiles for Object Decoder
        if 'velocity_offset_percentiles' in self.stats:
            self.velo_offset_1p = torch.tensor(self.stats['velocity_offset_percentiles']['1p'], dtype=torch.float64)
            self.velo_offset_99p = torch.tensor(self.stats['velocity_offset_percentiles']['99p'], dtype=torch.float64)
        else:
            # Fallback to legacy values
            self.velo_offset_1p = self.velo_1p.clone()
            self.velo_offset_99p = self.velo_99p.clone()
        
        # NEW: Absolute velocity percentiles for Dataset  
        if 'absolute_velocity_percentiles' in self.stats:
            self.velo_absolute_1p = torch.tensor(self.stats['absolute_velocity_percentiles']['1p'], dtype=torch.float64)
            self.velo_absolute_99p = torch.tensor(self.stats['absolute_velocity_percentiles']['99p'], dtype=torch.float64)
        else:
            # Fallback with larger range for absolute velocities
            self.velo_absolute_1p = torch.tensor([-15.0, -15.0], dtype=torch.float64)
            self.velo_absolute_99p = torch.tensor([15.0, 15.0], dtype=torch.float64)
    
    def _register_identity_tensors(self):
        """Store identity tensors for fallback normalization as CPU-only attributes."""
        # Store as regular attributes (NOT buffers) to avoid GPU issues
        self.center_abs_99p = torch.ones(3, dtype=torch.float64)
        self.log_size_abs_99p = torch.ones(3, dtype=torch.float64)
        
        # Legacy velocity tensors
        self.velo_1p = torch.tensor([-1.0, -1.0], dtype=torch.float64)
        self.velo_99p = torch.tensor([1.0, 1.0], dtype=torch.float64)
        
        # NEW: Separate velocity tensors
        self.velo_offset_1p = torch.tensor([-3.0, -3.0], dtype=torch.float64)
        self.velo_offset_99p = torch.tensor([3.0, 3.0], dtype=torch.float64)
        self.velo_absolute_1p = torch.tensor([-15.0, -15.0], dtype=torch.float64)
        self.velo_absolute_99p = torch.tensor([15.0, 15.0], dtype=torch.float64)
    
    def normalize_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        """Normalize offsets to [-1, 1] range using loaded statistics.
        
        Strategy: First clip to 99th percentile range, then normalize to [-1, 1].
        This prevents extreme outliers from affecting the normalization.
        
        Args:
            offsets: Raw offsets [N, 8] (center[3], size[3], yaw[2])
            
        Returns:
            Normalized offsets [N, 8] in [-1, 1] range
        """
        if not self.stats:
            return offsets
        
        # Extract components
        center_offsets = offsets[:, :3]  # [N, 3]
        size_offsets = offsets[:, 3:6]   # [N, 3]
        yaw_offsets = offsets[:, 6:8]    # [N, 2]
        
        # Step 1: Clip center offsets to 99th percentile range
        center_clipped = center_offsets.clamp(-self.center_abs_99p, self.center_abs_99p)
        # Step 2: Normalize clipped center offsets to [-1, 1]
        center_normalized = center_clipped / self.center_abs_99p
        
        # Step 1: Clip size offsets to 99th percentile range
        size_clipped = size_offsets.clamp(-self.log_size_abs_99p, self.log_size_abs_99p)
        # Step 2: Normalize clipped size offsets to [-1, 1]
        size_normalized = size_clipped / self.log_size_abs_99p
        
        # Yaw offsets: Already in sin/cos format, clip to [-1, 1] for numerical stability
        yaw_normalized = yaw_offsets.clamp(-1.0, 1.0)
        
        # Combine normalized offsets
        normalized_offsets = torch.cat([
            center_normalized,
            size_normalized,
            yaw_normalized
        ], dim=-1)
        
        return normalized_offsets
    
    def denormalize_offsets(self, normalized_offsets: torch.Tensor) -> torch.Tensor:
        """Denormalize offsets from [-1, 1] range back to physical units.
        
        Args:
            normalized_offsets: Normalized offsets [N, 8] in [-1, 1] range
            
        Returns:
            Denormalized offsets [N, 8] in physical units
        """
        if not self.stats:
            return normalized_offsets
        
        # Extract components
        center_offsets_norm = normalized_offsets[:, :3]  # [N, 3]
        size_offsets_norm = normalized_offsets[:, 3:6]   # [N, 3]
        yaw_offsets_norm = normalized_offsets[:, 6:8]    # [N, 2]
        
        # Denormalize center offsets: [-1, 1] → physical meters
        center_offsets_denorm = center_offsets_norm * self.center_abs_99p
        
        # Denormalize size offsets: [-1, 1] → log-space
        size_offsets_denorm = size_offsets_norm * self.log_size_abs_99p
        
        # Yaw offsets: Already in correct format (sin/cos)
        yaw_offsets_denorm = yaw_offsets_norm
        
        # Combine denormalized offsets
        denormalized_offsets = torch.cat([
            center_offsets_denorm,
            size_offsets_denorm,
            yaw_offsets_denorm
        ], dim=-1)
        
        return denormalized_offsets
    
    def normalize_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """Normalize velocity to [-1, 1] range using loaded statistics.
        
        Strategy: First clip to 1%-99% percentile range, then normalize to [-1, 1].
        This prevents extreme outliers from affecting the normalization.
        
        Args:
            velocity: Raw velocity [N, 2] in m/s
            
        Returns:
            Normalized velocity [N, 2] in [-1, 1] range
        """
        if not self.stats:
            return velocity
        
        # REMOVED: Clipping corrupts the transformation and makes it non-invertible.
        # velocity_clipped = velocity.clamp(self.velo_1p, self.velo_99p)
        
        # Step 2: Normalize clipped velocity to [-1, 1] range
        # All normalization stats are CPU-only, so ensure input is also on CPU
        velocity_cpu = velocity.cpu()
        
        velo_range = self.velo_99p - self.velo_1p
        # Avoid division by zero
        velo_range = torch.where(velo_range == 0, torch.ones_like(velo_range), velo_range)
        velocity_normalized = 2.0 * (velocity_cpu - self.velo_1p) / velo_range - 1.0
        
        return velocity_normalized
    
    def denormalize_velocity(self, velocity_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize velocity from [-1, 1] range back to m/s.
        
        Args:
            velocity_normalized: Normalized velocity [N, 2] in [-1, 1] range
            
        Returns:
            Denormalized velocity [N, 2] in m/s
        """
        if not self.stats:
            return velocity_normalized
        
        # Two-sided scaling: [-1, 1] → [velo_1p, velo_99p]
        # All normalization stats are CPU-only, so ensure input is also on CPU
        velocity_normalized_cpu = velocity_normalized.cpu()
        
        velo_range = self.velo_99p - self.velo_1p
        velocity_denorm = self.velo_1p + (velocity_normalized_cpu + 1.0) * velo_range / 2.0
        
        return velocity_denorm
    
    # ====================================================================
    # NEW: VELOCITY OFFSET NORMALIZATION (für Object Decoder)
    # ====================================================================
    
    def normalize_velocity_offset(self, velocity_offset: torch.Tensor) -> torch.Tensor:
        """Normalize velocity OFFSETS to [-1, 1] range using offset-specific statistics.
        
        Used by Object Decoder to normalize velocity offset predictions. These offsets
        represent the difference between predicted and anchor velocities and have
        a smaller value range than absolute velocities.
        
        Args:
            velocity_offset: Raw velocity offsets [N, 2] in m/s (Object Decoder predictions)
            
        Returns:
            Normalized velocity offsets [N, 2] in [-1, 1] range
        """
        if not self.stats:
            return velocity_offset
        
        # Two-sided scaling: [velo_offset_1p, velo_offset_99p] → [-1, 1]
        # Device-agnostic: move stats to input device for Model usage
        input_device = velocity_offset.device
        velo_offset_1p = self.velo_offset_1p.to(input_device)
        velo_offset_99p = self.velo_offset_99p.to(input_device)
        
        offset_range = velo_offset_99p - velo_offset_1p
        # Avoid division by zero
        offset_range = torch.where(offset_range == 0, torch.ones_like(offset_range), offset_range)
        velocity_offset_normalized = 2.0 * (velocity_offset - velo_offset_1p) / offset_range - 1.0
        
        return velocity_offset_normalized
    
    def denormalize_velocity_offset(self, velocity_offset_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize velocity OFFSETS from [-1, 1] range back to m/s.
        
        Used by Autoregressive Architecture to denormalize Object Decoder velocity offset
        predictions before adding them to anchor velocities.
        
        Args:
            velocity_offset_normalized: Normalized velocity offsets [N, 2] in [-1, 1] range
            
        Returns:
            Denormalized velocity offsets [N, 2] in m/s
        """
        if not self.stats:
            return velocity_offset_normalized
        
        # Two-sided scaling: [-1, 1] → [velo_offset_1p, velo_offset_99p]
        # Device-agnostic: move stats to input device for Model usage
        input_device = velocity_offset_normalized.device
        velo_offset_1p = self.velo_offset_1p.to(input_device)
        velo_offset_99p = self.velo_offset_99p.to(input_device)
        
        offset_range = velo_offset_99p - velo_offset_1p
        velocity_offset_denorm = velo_offset_1p + (velocity_offset_normalized + 1.0) * offset_range / 2.0
        
        return velocity_offset_denorm
    

    # ====================================================================
    # NEW: ABSOLUTE VELOCITY NORMALIZATION (für Dataset)
    # ====================================================================
    
    def normalize_velocity_absolute(self, velocity_absolute: torch.Tensor) -> torch.Tensor:
        """Normalize ABSOLUTE relative velocities to [-1, 1] range using absolute-specific statistics.
        
        Used by Dataset to normalize absolute relative velocities (object velocity relative to ego).
        These velocities have a larger value range than velocity offsets and require separate
        normalization statistics for proper scaling.
        
        Args:
            velocity_absolute: Raw absolute relative velocities [N, 2] in m/s (Dataset GT targets)
            
        Returns:
            Normalized absolute velocities [N, 2] in [-1, 1] range
        """
        if not self.stats:
            return velocity_absolute
        
        # Two-sided scaling: [velo_absolute_1p, velo_absolute_99p] → [-1, 1]
        # Device-agnostic: move stats to input device for Model usage
        input_device = velocity_absolute.device
        velo_absolute_1p = self.velo_absolute_1p.to(input_device)
        velo_absolute_99p = self.velo_absolute_99p.to(input_device)
        
        absolute_range = velo_absolute_99p - velo_absolute_1p
        # Avoid division by zero
        absolute_range = torch.where(absolute_range == 0, torch.ones_like(absolute_range), absolute_range)
        velocity_absolute_normalized = 2.0 * (velocity_absolute - velo_absolute_1p) / absolute_range - 1.0
        
        return velocity_absolute_normalized
    
    def denormalize_velocity_absolute(self, velocity_absolute_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize ABSOLUTE relative velocities from [-1, 1] range back to m/s.
        
        Used by Evaluation utilities to denormalize model predictions of absolute relative
        velocities back to physical units for evaluation metrics computation.
        
        Args:
            velocity_absolute_normalized: Normalized absolute velocities [N, 2] in [-1, 1] range
            
        Returns:
            Denormalized absolute velocities [N, 2] in m/s
        """
        if not self.stats:
            return velocity_absolute_normalized
        
        # Two-sided scaling: [-1, 1] → [velo_absolute_1p, velo_absolute_99p]
        # Device-agnostic: move stats to input device for Model usage
        input_device = velocity_absolute_normalized.device
        velo_absolute_1p = self.velo_absolute_1p.to(input_device)
        velo_absolute_99p = self.velo_absolute_99p.to(input_device)
        
        absolute_range = velo_absolute_99p - velo_absolute_1p
        velocity_absolute_denorm = velo_absolute_1p + (velocity_absolute_normalized + 1.0) * absolute_range / 2.0
        
        return velocity_absolute_denorm
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of loaded normalization statistics.
        
        Returns:
            Dictionary containing statistics summary
        """
        if not self.stats:
            return {"status": "no_stats_loaded"}
        
        return {
            "status": "loaded",
            "center_abs_99p": self.center_abs_99p.tolist(),
            "log_size_abs_99p": self.log_size_abs_99p.tolist(),
            "velocity_1p": self.velo_1p.tolist(),
            "velocity_99p": self.velo_99p.tolist(),
            "metadata": self.stats.get('metadata', {})
        }


# Global instance for easy access
_global_normalizer = None


def get_global_normalizer() -> CentralizedNormalizer:
    """Get the global normalizer instance.
    
    Returns:
        CentralizedNormalizer instance
    """
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = CentralizedNormalizer()
    return _global_normalizer





def normalize_offsets(offsets: torch.Tensor) -> torch.Tensor:
    """Convenience function to normalize offsets using global normalizer.
    
    Args:
        offsets: Raw offsets [N, 8]
        
    Returns:
        Normalized offsets [N, 8] in [-1, 1] range
    """
    return get_global_normalizer().normalize_offsets(offsets)


def denormalize_offsets(normalized_offsets: torch.Tensor) -> torch.Tensor:
    """Convenience function to denormalize offsets using global normalizer.
    
    Args:
        normalized_offsets: Normalized offsets [N, 8] in [-1, 1] range
        
    Returns:
        Denormalized offsets [N, 8] in physical units
    """
    return get_global_normalizer().denormalize_offsets(normalized_offsets)


def normalize_velocity(velocity: torch.Tensor) -> torch.Tensor:
    """Convenience function to normalize velocity using global normalizer.
    
    Args:
        velocity: Raw velocity [N, 2] in m/s
        
    Returns:
        Normalized velocity [N, 2] in [-1, 1] range
    """
    return get_global_normalizer().normalize_velocity(velocity)


def denormalize_velocity(velocity_normalized: torch.Tensor) -> torch.Tensor:
    """Convenience function to denormalize velocity using global normalizer.
    
    Args:
        velocity_normalized: Normalized velocity [N, 2] in [-1, 1] range
        
    Returns:
        Denormalized velocity [N, 2] in m/s
    """
    return get_global_normalizer().denormalize_velocity(velocity_normalized)


# ====================================================================
# NEW: VELOCITY OFFSET CONVENIENCE FUNCTIONS
# ====================================================================

def normalize_velocity_offset(velocity_offset: torch.Tensor) -> torch.Tensor:
    """Convenience function to normalize velocity offsets using global normalizer.
    
    Args:
        velocity_offset: Raw velocity offsets [N, 2] in m/s
        
    Returns:
        Normalized velocity offsets [N, 2] in [-1, 1] range
    """
    return get_global_normalizer().normalize_velocity_offset(velocity_offset)


def denormalize_velocity_offset(velocity_offset_normalized: torch.Tensor) -> torch.Tensor:
    """Convenience function to denormalize velocity offsets using global normalizer.
    
    Args:
        velocity_offset_normalized: Normalized velocity offsets [N, 2] in [-1, 1] range
        
    Returns:
        Denormalized velocity offsets [N, 2] in m/s
    """
    return get_global_normalizer().denormalize_velocity_offset(velocity_offset_normalized)


# ====================================================================
# NEW: ABSOLUTE VELOCITY CONVENIENCE FUNCTIONS  
# ====================================================================

def normalize_velocity_absolute(velocity_absolute: torch.Tensor) -> torch.Tensor:
    """Convenience function to normalize absolute velocities using global normalizer.
    
    Args:
        velocity_absolute: Raw absolute relative velocities [N, 2] in m/s
        
    Returns:
        Normalized absolute velocities [N, 2] in [-1, 1] range
    """
    return get_global_normalizer().normalize_velocity_absolute(velocity_absolute)



def denormalize_velocity_absolute(velocity_absolute_normalized: torch.Tensor) -> torch.Tensor:
    """Convenience function to denormalize absolute velocities using global normalizer.
    
    Args:
        velocity_absolute_normalized: Normalized absolute velocities [N, 2] in [-1, 1] range
        
    Returns:
        Denormalized absolute velocities [N, 2] in m/s
    """
    return get_global_normalizer().denormalize_velocity_absolute(velocity_absolute_normalized)


# Legacy functions removed - these were only used in deprecated uncertainty_weighted_regression_losses.py
# The active pipeline uses LossCenter, LossSize, LossAngle, LossVelocity from regression_losses.py
# which use the centralized normalization utilities instead.
#
# Removed functions:
# - load_normalization_stats(): Legacy function (deprecated)
# - denormalize_offsets_with_stats(): Legacy function (deprecated)
# - denormalize_velocity_offsets_with_stats(): Legacy function (deprecated)
# - get_normalization_tensors_from_stats(): Legacy function (deprecated)
# - DynamicOffsetNormalizer: Dynamic batch-based normalization (deprecated)
# - LogBasedOffsetNormalizer: Log-based normalization (deprecated)  
# - create_offset_normalizer: Factory function for deprecated normalizers
# - normalize_coordinates_absolute: Absolute coordinate normalization (unused)
# - denormalize_coordinates_absolute: Absolute coordinate denormalization (unused)


# NEU: Direkte Normalisierungsfunktionen für Koordinaten und Dimensionen
def normalize_coordinates(coords: torch.Tensor, point_cloud_range: torch.Tensor) -> torch.Tensor:
    """Normalize coordinates using point_cloud_range to [0, 1] range.
    
    Normalizes 3D coordinates to the range [0, 1] based on the point_cloud_range
    configuration parameter. Ensures consistent scaling across the pipeline.
    Device-agnostic for Dataset (CPU) and Model (GPU) usage.
    
    Args:
        coords: 3D coordinates (x, y, z) in ego frame [N, 3].
        point_cloud_range: Point cloud range [6] (x_min, y_min, z_min, x_max, y_max, z_max).
        
    Returns:
        Normalized coordinates in range [0, 1] [N, 3] (same device as input).
    """
    # Device-agnostic: preserve input device for Model usage
    input_device = coords.device
    coords_work = coords
    point_cloud_range_work = point_cloud_range.to(input_device)
    
    x_min, y_min, z_min = point_cloud_range_work[:3]
    x_max, y_max, z_max = point_cloud_range_work[3:]
    
    min_vals = torch.tensor([x_min, y_min, z_min], device=input_device)
    range_vals = torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min], device=input_device)
    
    normalized = (coords_work - min_vals) / range_vals
    
    return torch.clamp(normalized, 0.0, 1.0)


def normalize_dimensions(dims: torch.Tensor) -> torch.Tensor:
    """Normalize dimensions using log transformation.
    
    Applies log transformation to box dimensions to handle the wide range
    of object sizes in the dataset. Device-agnostic for Dataset (CPU) and Model (GPU) usage.
    
    Args:
        dims: Box dimensions (w, l, h) in meters [N, 3].
        
    Returns:
        Log-normalized dimensions [N, 3] (same device as input).
    """
    # Device-agnostic: preserve input device for Model usage
    input_device = dims.device
    
    # Apply log transformation with minimum bound for numerical stability
    min_bound = torch.tensor(0.1, device=input_device)
    log_dims = torch.log(torch.maximum(dims, min_bound))
    return log_dims  # No clamp() - preserves round-trip accuracy!


def denormalize_coordinates(coords_normalized: torch.Tensor, point_cloud_range: torch.Tensor) -> torch.Tensor:
    """Denormalize coordinates from [0, 1] range back to physical units.
    
    Args:
        coords_normalized: Normalized coordinates [N, 3] in [0, 1] range.
        point_cloud_range: Point cloud range [6] (x_min, y_min, z_min, x_max, y_max, z_max).
        
    Returns:
        Denormalized coordinates in physical units [N, 3] (same device as input).
    """
    # Device-agnostic: preserve input device for Model usage
    input_device = coords_normalized.device
    coords_work = coords_normalized
    point_cloud_range_work = point_cloud_range.to(input_device)
    
    x_min, y_min, z_min = point_cloud_range_work[:3]
    x_max, y_max, z_max = point_cloud_range_work[3:]
    
    range_vals = torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min], device=input_device)
    min_vals = torch.tensor([x_min, y_min, z_min], device=input_device)
    
    return coords_work * range_vals + min_vals


def denormalize_dimensions(dims_normalized: torch.Tensor) -> torch.Tensor:
    """Denormalize dimensions from log-space back to physical units.
    
    Args:
        dims_normalized: Log-normalized dimensions [N, 3].
        
    Returns:
        Denormalized dimensions in meters [N, 3] (same device as input).
    """
    # Device-agnostic: preserve input device for Model usage
    return torch.exp(dims_normalized) 