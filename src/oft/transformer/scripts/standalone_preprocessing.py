#!/usr/bin/env python3
"""
Standalone Virtual Sensor Preprocessing Pipeline
=============================================

Generates complete preprocessed dataset in current Dataset format for fast training.

This script performs the PREPROCESSING PHASE of the two-phase architecture:
1. Uses SEQUENTIAL sampling for accurate velocity calculation 
2. Generates virtual sensor data with scene conditioning and normalization
3. Caches complete Dataset-format data (sensor_data + ground_truth + metadata)
4. Outputs single JSON file per split with all preprocessing completed

Usage:
    python standalone_preprocessing.py  # Processes ALL splits automatically from config

Output Format (Dataset format as JSON):
    complete_dataset_{split}.json:
    {
        "meta": {"preprocessing_complete": true, "format": "dataset_format"},
        "results": {
            "sample_token": {
                "sensor_data": {
                    "virtual_lidar": {
                        "features": [[x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin_yaw, cos_yaw, vx_norm, vy_norm, class_idx, attr_idx]],
                        "metadata": [[confidence, sensor_id, detection_id]],
                        "centers": [[x, y, z]],
                        "boxes": [[x, y, z, w, l, h, yaw, vx, vy]]
                    }
                },
                "ground_truth": {...},
                "ego_translation_world": [...],
                "ego_rotation_world_quat": [...],
                "scene_meta": {...},
                "ego_motion": {...}
            }
        }
    }
"""

import os
import sys
import argparse
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.utils.virtual_sensor_cache import get_virtual_sensor_cache, load_complete_devkit_to_ram
from oft.transformer.utils.reproducibility import set_seed
from oft.transformer.utils.config_hash_utils import generate_unified_config_hash, get_complete_dataset_path

class StandalonePreprocessor:
    """Standalone preprocessing for virtual sensor data in Dataset format."""
    
    def __init__(self, config: DictConfig, output_dir: str = None):
        """Initialize preprocessor.
        
        Args:
            config: Hydra configuration
            output_dir: Output directory (defaults to config.preprocessing.cache_dir)
        """
        self.config = config
        self.output_dir = output_dir or config.preprocessing.cache_dir
        
        # Determine splits from config - support DOE-specific splits
        self.train_split = config.training.train_split_name
        self.val_split = config.training.val_split_name
        
        # Check if specific splits are requested (for DOE: only validation)
        if hasattr(config, 'preprocessing') and hasattr(config.preprocessing, 'splits_to_process'):
            self.splits_to_process = config.preprocessing.splits_to_process
        else:
            # Default: process both train and val
            self.splits_to_process = [self.train_split, self.val_split]
        
        self.logger = self._setup_logger()
        
        # Log splits after logger is initialized
        if hasattr(config, 'preprocessing') and hasattr(config.preprocessing, 'splits_to_process'):
            self.logger.info(f"🎯 Using config-specified splits: {self.splits_to_process}")
        else:
            self.logger.info(f"📊 Processing default splits: {self.splits_to_process}")
        self.cache = get_virtual_sensor_cache(cache_dir=self.output_dir, logger=self.logger)
        
        # Generate config hash for cache directory
        self.config_hash = self._generate_config_hash()
        
        self.logger.info(f"🚀 Standalone Preprocessing für {self.splits_to_process}")
        self.logger.info(f"📁 Cache-Dir: {self.output_dir}")
        self.logger.info(f"🔑 Config-Hash: {self.config_hash}")
        
        # PERFORMANCE BOOST: Pre-load DevKit to RAM to avoid disk I/O per sample
        self.logger.info(f"⚡ Pre-loading TruckScenes DevKit to RAM...")
        self.devkit_ram = load_complete_devkit_to_ram(
            dataroot=config.dataset.dataroot,
            version=config.dataset.version,
            logger=self.logger
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for preprocessing."""
        logger = logging.getLogger("StandalonePreprocessing")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _generate_config_hash(self) -> str:
        """Generate deterministic hash from config for cache invalidation."""
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        return generate_unified_config_hash(config_dict)
    
    def check_preprocessing_completed(self) -> Dict[str, bool]:
        """Check if preprocessing is already completed for all splits.
        
        Returns:
            Dict mapping split_name -> is_completed
        """
        completion_status = {}
        
        for split_name in self.splits_to_process:
            complete_data = self.cache.load_complete_dataset_collection(self.config_hash, split_name)
            completion_status[split_name] = bool(complete_data)
            
            if complete_data:
                self.logger.info(f"✅ {split_name}: Bereits abgeschlossen ({len(complete_data)} samples)")
            else:
                self.logger.info(f"🔄 {split_name}: Preprocessing erforderlich")
        
        return completion_status
    
    def preprocess(self, force: bool = False) -> bool:
        """Run complete preprocessing pipeline for ALL splits.
        
        Args:
            force: If True, force preprocessing even if already completed
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if force:
                # Force processing all splits
                splits_to_process = self.splits_to_process
                self.logger.info(f"🔄 FORCE: Processing all splits: {splits_to_process}")
            else:
                # Check if already completed
                completion_status = self.check_preprocessing_completed()
                splits_to_process = [split for split, completed in completion_status.items() if not completed]
                
                if not splits_to_process:
                    self.logger.info("✅ Alle Splits bereits abgeschlossen!")
                    return True
                
                self.logger.info(f"🔄 Processing splits: {splits_to_process}")
            
            # Set seed for reproducible preprocessing
            set_seed(self.config.preprocessing.get('seed', 42), self.logger)
            
            # Process each split
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            
            # Note: Dataset reads from 'training' config, so we copy preprocessing values there
            config_dict['training'].update({
                'batch_size': 1,  # FIXED: Always 1 for preprocessing (hardcoded)
                'num_workers': self.config.preprocessing.num_workers,  # From preprocessing config
                'sampler': self.config.preprocessing.sampler  # From preprocessing config
            })
            
            # CRITICAL: Force training mode for ALL splits to enable data augmentation
            config_dict['_force_training_mode'] = True
            
            for split_name in splits_to_process:
                self.logger.info(f"📊 Processing split: {split_name}")
                
                dataset = ObjectFusionGTDatasetStaged(
                    dataroot=config_dict['dataset']['dataroot'],
                    version=config_dict['dataset']['version'],
                    split_name=split_name,
                    pipeline_config=config_dict,
                    verbose=False,  # Reduce noise during preprocessing
                    devkit_ram=self.devkit_ram  # PERFORMANCE BOOST: Use pre-loaded DevKit
                )
                
                self.logger.info(f"🔄 Preprocessing {len(dataset)} samples for {split_name}...")
                
                # Create preprocessing DataLoader using unified configuration
                complete_dataset_data = {}
                dataloader = self._create_preprocessing_dataloader(dataset, config_dict['preprocessing']['num_workers'])
                
                self.logger.info(f"📦 Using DataLoader with {config_dict['preprocessing']['num_workers']} workers for parallel preprocessing")
                
                # Process all samples with DataLoader parallelization
                processed_count = 0
                for sample_data in tqdm(dataloader, desc=f"Processing {split_name}", leave=True):
                    try:
                        sample_token = sample_data['sample_token']
                        
                        # Store in collection
                        complete_dataset_data[sample_token] = sample_data
                        
                        processed_count += 1
                        if processed_count % 50 == 0:
                            self.logger.info(f"{split_name}: Processed {processed_count}/{len(dataset)} samples")
                    
                    except Exception as e:
                        self.logger.error(f"❌ {split_name}: Failed to process sample: {e}")
                        continue
                
                # Save complete dataset collection
                self.logger.info(f"💾 Speichere {len(complete_dataset_data)} preprocessed samples für {split_name}...")
                self.cache.save_complete_dataset_collection(
                    config_hash=self.config_hash,
                    split_name=split_name,
                    complete_dataset=complete_dataset_data
                )
                
                # Write metadata
                self._write_preprocessing_metadata(len(complete_dataset_data), split_name)
                
                self.logger.info(f"✅ {split_name}: Preprocessing abgeschlossen ({len(complete_dataset_data)} samples)")
            
            self.logger.info(f"🎉 ALLE SPLITS ABGESCHLOSSEN: {splits_to_process}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {e}")
            return False
    
    def _create_preprocessing_dataloader(self, dataset, num_workers: int) -> DataLoader:
        """Creates optimized DataLoader for preprocessing with fixed configuration.
        
        Preprocessing DataLoader uses batch_size=1 since each sample is processed independently
        and stored as individual JSON files. Only num_workers affects performance.
        
        Args:
            dataset: ObjectFusionGTDatasetStaged instance for preprocessing
            num_workers: Number of parallel worker processes for data loading
            
        Returns:
            Configured DataLoader optimized for preprocessing workflow
        """
        return DataLoader(
            dataset, 
            batch_size=1,  # Fixed: Each sample processed independently
            num_workers=num_workers,
            sampler=None,  # Sequential processing for deterministic results
            collate_fn=lambda x: x[0],  # Simple collate: return single sample
            pin_memory=False,  # CPU-only preprocessing
            persistent_workers=False  # Avoid memory buildup during long preprocessing
        )
    
    def _write_preprocessing_metadata(self, num_samples: int, split_name: str):
        """Write metadata file with preprocessing information."""
        metadata = {
            'config_hash': self.config_hash,
            'split_name': split_name,
            'num_samples': num_samples,
            'timestamp': self._get_timestamp(),
            'dataset_version': self.config.dataset.version,
            'preprocessing_settings': OmegaConf.to_container(self.config.preprocessing, resolve=True)
        }
        
        cache_dir = os.path.join(self.output_dir, f"config_{self.config_hash}")
        metadata_file = os.path.join(cache_dir, f"preprocessing_metadata_{split_name}.json")
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"📋 Metadata saved: {metadata_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save metadata: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Main entry point for standalone preprocessing."""
    parser = argparse.ArgumentParser(
        description="Standalone preprocessing for virtual sensor data (processes ALL splits from config)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help="Output directory (defaults to config.preprocessing.cache_dir)"
    )
    parser.add_argument(
        '--config',
        type=str,
        help="Path to config YAML file (defaults to config/pipeline_staged.yaml)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force preprocessing even if already completed"
    )
    
    args = parser.parse_args()
    
    # Load config with Hydra (support custom config path)
    if hasattr(args, 'config') and args.config:
        config_path = Path(args.config)
    else:
        config_path = Path("config/pipeline_staged.yaml")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    # Initialize Hydra
    with hydra.initialize_config_dir(config_dir=str(config_path.parent.absolute()), version_base=None):
        cfg = hydra.compose(config_name=config_path.stem)
        
        # Create preprocessor
        preprocessor = StandalonePreprocessor(
            config=cfg,
            output_dir=args.output_dir
        )
        
        # Check if already completed (unless forced)
        if not args.force:
            completion_status = preprocessor.check_preprocessing_completed()
            if all(completion_status.values()):
                print(f"✅ Preprocessing bereits abgeschlossen für alle Splits: {list(completion_status.keys())}")
                print(f"💡 Verwende --force um erneut zu preprocessen")
                sys.exit(0)
        
        # Run preprocessing
        success = preprocessor.preprocess(force=args.force)
        
        if success:
            print(f"✅ Preprocessing erfolgreich abgeschlossen für alle Splits")
            sys.exit(0)
        else:
            print(f"❌ Preprocessing fehlgeschlagen")
            sys.exit(1)


if __name__ == "__main__":
    main()