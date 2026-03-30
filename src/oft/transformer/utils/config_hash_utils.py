# File: src/oft/transformer/utils/config_hash_utils.py
"""
Unified configuration hashing utilities for consistent cache invalidation.

This module provides a single, canonical implementation for generating 
configuration hashes across all components (standalone preprocessing, 
autoregressive loader, and virtual sensor cache).
"""

import hashlib
import json
from typing import Dict, Any


def generate_unified_config_hash(config_dict: Dict[str, Any]) -> str:
    """Generate unified configuration hash for cache invalidation.
    
    This is the SINGLE source of truth for config hashing across:
    - standalone_preprocessing.py  
    - autoregressive_loader.py
    - virtual_sensor_cache.py
    
    Args:
        config_dict: Complete configuration dictionary
        
    Returns:
        16-character deterministic hash string
    """
    # Extract ONLY relevant config components for hashing
    # These are the parameters that actually affect the generated data content
    config_components = {
        'dataset_version': config_dict['dataset']['version'],
        'virtual_sensors': config_dict['dataset']['virtual_sensors'],
        'simulation': config_dict['dataset']['simulation'],
        'scene_conditioning': config_dict['dataset'].get('scene_conditioning', {}),
        'normalization_stats_path': config_dict['dataset']['normalization_stats_path']
        # NOTE: preprocessing_settings completely excluded - proven to NOT affect final data content
    }
    
    # Normalize floating point values for deterministic hashing
    def normalize_floats(obj):
        """Normalize floating point values for consistent hashing."""
        if isinstance(obj, dict):
            return {k: normalize_floats(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [normalize_floats(item) for item in obj]
        elif isinstance(obj, float):
            return round(obj, 10)  # 10 decimal places for high precision
        else:
            return obj
    
    # Generate deterministic hash
    normalized_config = normalize_floats(config_components)
    config_str = json.dumps(normalized_config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    return config_hash


def get_cache_directory_path(base_cache_dir: str, config_hash: str) -> str:
    """Get standardized cache directory path.
    
    Args:
        base_cache_dir: Base cache directory path
        config_hash: Configuration hash from generate_unified_config_hash()
        
    Returns:
        Complete cache directory path: base_cache_dir/config_HASH/
    """
    import os
    return os.path.join(base_cache_dir, f"config_{config_hash}")


def get_complete_dataset_path(base_cache_dir: str, config_hash: str, split_name: str) -> str:
    """Get standardized complete dataset file path.
    
    Args:
        base_cache_dir: Base cache directory path  
        config_hash: Configuration hash from generate_unified_config_hash()
        split_name: Dataset split name (e.g., 'mini_train', 'mini_val')
        
    Returns:
        Complete dataset file path: base_cache_dir/config_HASH/complete_dataset_SPLIT.json
    """
    import os
    cache_dir = get_cache_directory_path(base_cache_dir, config_hash)
    return os.path.join(cache_dir, f"complete_dataset_{split_name}.json")