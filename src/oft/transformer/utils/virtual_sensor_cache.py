# File: src/oft/transformer/utils/virtual_sensor_cache.py
"""
Simplified Virtual Sensor Cache System for complete dataset preprocessing.

This module provides simplified caching functionality for complete dataset files.
Focuses ONLY on complete dataset collection functions, removing all legacy 
sample-level caching complexity.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from .config_hash_utils import get_cache_directory_path, get_complete_dataset_path


class VirtualSensorCache:
    """
    Simplified cache manager for complete dataset files.
    
    Features:
    - Save/load complete dataset collections as single JSON files per split
    - Unified config hashing via config_hash_utils
    - Thread-safe file operations with atomic writes
    """
    
    def __init__(self, cache_dir: str = "/cache", logger: Optional[logging.Logger] = None):
        """Initialize virtual sensor cache manager.
        
        Args:
            cache_dir: Directory path for cache storage on HDD
            logger: Optional logger instance for cache operations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"VirtualSensorCache initialized: {self.cache_dir}")
    
    def save_complete_dataset_collection(self, config_hash: str, split_name: str,
                                       complete_dataset: Dict[str, Dict[str, Any]]) -> bool:
        """Save complete dataset collection as single JSON file.
        
        Args:
            config_hash: Configuration hash from config_hash_utils
            split_name: Dataset split name (e.g., 'mini_train', 'mini_val')
            complete_dataset: Dict mapping sample_token -> complete_sample_data
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Use unified path utilities
            output_path = get_complete_dataset_path(str(self.cache_dir), config_hash, split_name)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Prepare data for JSON serialization
            serializable_dataset = {
                sample_token: self._make_json_serializable(sample_data)
                for sample_token, sample_data in complete_dataset.items()
            }
            
            # Create complete file structure
            complete_file_data = {
                "meta": {
                    "preprocessing_complete": True,
                    "format": "dataset_format",
                    "split_name": split_name,
                    "config_hash": config_hash,
                    "num_samples": len(complete_dataset)
                },
                "results": serializable_dataset
            }
            
            # Atomic write operation
            temp_path = output_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(complete_file_data, f, indent=2)
            
            # Atomic rename (safe for concurrent access)
            os.rename(temp_path, output_path)
            
            self.logger.info(f"✅ Saved complete_dataset_{split_name}.json: {len(complete_dataset)} complete samples")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save complete dataset collection: {e}")
            return False
    
    def load_complete_dataset_collection(self, config_hash: str, split_name: str) -> Optional[Dict[str, Dict]]:
        """Load complete dataset collection from JSON file.
        
        Args:
            config_hash: Configuration hash from config_hash_utils
            split_name: Dataset split name (e.g., 'mini_train', 'mini_val')
            
        Returns:
            Dict mapping sample_token -> complete_sample_data if found, None otherwise
        """
        try:
            # Use unified path utilities
            dataset_path = get_complete_dataset_path(str(self.cache_dir), config_hash, split_name)
            
            if not os.path.exists(dataset_path):
                self.logger.warning(f"⚠️ Complete dataset file not found: {dataset_path}")
                return None
            
            # Load complete dataset
            with open(dataset_path, 'r') as f:
                file_data = json.load(f)
            
            # Verify file structure and extract results
            if "results" not in file_data:
                self.logger.error(f"❌ Invalid file format: Missing 'results' key in {dataset_path}")
                return None
            
            results = file_data["results"]
            
            # Verify metadata
            meta = file_data.get("meta", {})
            expected_samples = meta.get("num_samples", len(results))
            if len(results) != expected_samples:
                self.logger.warning(f"⚠️ Sample count mismatch: Expected {expected_samples}, found {len(results)}")
            
            self.logger.info(f"✅ Loaded complete_dataset_{split_name}.json: {len(results)} samples")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load complete dataset collection: {e}")
            return None
    
    def _make_json_serializable(self, obj):
        """Convert complex objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'tolist'):  # Fallback for other NumPy types
            return obj.tolist()
        else:
            return obj


# ===============================================================================
# FACTORY FUNCTION AND STANDALONE UTILITIES
# ===============================================================================

def get_virtual_sensor_cache(cache_dir: str = "/cache",
                           logger: Optional[logging.Logger] = None) -> VirtualSensorCache:
    """Factory function to create VirtualSensorCache instance.
    
    Args:
        cache_dir: Directory path for cache storage
        logger: Optional logger instance
        
    Returns:
        VirtualSensorCache instance
    """
    return VirtualSensorCache(cache_dir=cache_dir, logger=logger)


def load_complete_devkit_to_ram(dataroot: str, version: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Load complete TruckScenes DevKit metadata to RAM for fast access during preprocessing and training.
    
    This eliminates the major disk I/O bottleneck by preloading all TruckScenes JSON metadata
    files into RAM with fast lookup dictionaries.
    
    Args:
        dataroot: TruckScenes dataset root directory
        version: Dataset version (e.g., 'v1.0-mini')
        logger: Optional logger instance
        
    Returns:
        Dictionary containing all DevKit metadata + lookup tables + original TruckScenes object
    """
    import time
    from truckscenes import TruckScenes
    
    if logger:
        logger.info(f"🔄 Loading complete TruckScenes DevKit to RAM...")
    start_time = time.time()
    
    # Initialize DevKit (this loads all JSON metadata files from disk)
    ts = TruckScenes(version=version, dataroot=dataroot, verbose=False)
    
    # Extract all metadata tables into a dictionary for RAM storage
    # Use getattr to safely handle missing attributes in different TruckScenes versions
    devkit_ram = {
        'sample': getattr(ts, 'sample', []),
        'sample_data': getattr(ts, 'sample_data', []),
        'sample_annotation': getattr(ts, 'sample_annotation', []),
        'instance': getattr(ts, 'instance', []),
        'category': getattr(ts, 'category', []),
        'attribute': getattr(ts, 'attribute', []),
        'visibility': getattr(ts, 'visibility', []),
        'scene': getattr(ts, 'scene', []),
        'log': getattr(ts, 'log', []),
        'ego_pose': getattr(ts, 'ego_pose', []),
        'calibrated_sensor': getattr(ts, 'calibrated_sensor', []),
        'sensor': getattr(ts, 'sensor', []),
        'map': getattr(ts, 'map', [])
    }
    
    # Create O(1) lookup dictionaries by token for fast access
    devkit_ram['lookup'] = {}
    for table_name, table_data in devkit_ram.items():
        if isinstance(table_data, list) and len(table_data) > 0:
            # Only create lookup if table has 'token' field
            if isinstance(table_data[0], dict) and 'token' in table_data[0]:
                devkit_ram['lookup'][table_name] = {item['token']: item for item in table_data}
    
    # Add the original TruckScenes object for compatibility with existing code
    devkit_ram['ts_object'] = ts
    
    # Calculate memory usage and performance metrics
    total_records = sum(len(table) for table in devkit_ram.values() if isinstance(table, list))
    load_time = time.time() - start_time
    
    if logger:
        logger.info(f"✅ DevKit loaded to RAM: {total_records} records + lookup tables in {load_time:.2f}s")
        logger.info(f"🔧 Available lookups: {list(devkit_ram['lookup'].keys())}")
    
    return devkit_ram


def load_complete_dataset_to_ram(cache_dir: str, config_hash: str, split_name: str,
                                logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Dict]]:
    """Standalone function to load complete dataset to RAM.
    
    This is the main function used by autoregressive_loader.py to load
    preprocessed data into RAM for fast training.
    
    Args:
        cache_dir: Cache directory path
        config_hash: Configuration hash from config_hash_utils
        split_name: Dataset split name (e.g., 'mini_train', 'mini_val')
        logger: Optional logger instance
        
    Returns:
        Dict mapping sample_token -> complete_sample_data if found, None otherwise
    """
    cache = get_virtual_sensor_cache(cache_dir=cache_dir, logger=logger)
    return cache.load_complete_dataset_collection(config_hash=config_hash, split_name=split_name)