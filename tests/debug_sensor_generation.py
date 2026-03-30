"""
Debug: Warum haben LIVE und RAM Mode unterschiedliche Sensor-Counts?
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from typing import Dict, Any
import numpy as np

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 DEBUG: Sensor Generation Unterschiede")
    print("="*60)
    
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    
    # ================================================================
    # SEED CONTROL
    # ================================================================
    print(f"\n🎲 SEED CONTROL:")
    print("-" * 20)
    
    # Set identical seeds
    import random
    import torch
    
    def set_seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # ================================================================
    # LIVE MODE 1
    # ================================================================
    print(f"\n📦 LIVE MODE (Run 1):")
    print("-" * 30)
    
    set_seeds(42)
    live_dataset1 = ObjectFusionGTDatasetStaged(
        dataroot=cfg_dict["dataset"]["dataroot"],
        version=cfg_dict["dataset"]["version"],
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None
    )
    
    live_frame1 = live_dataset1[0]
    print(f"Sample: {live_frame1['sample_token']}")
    
    for sensor_name, sensor_info in live_frame1['sensor_data'].items():
        count = len(sensor_info['features'])
        print(f"{sensor_name}: {count} objects")
    
    # ================================================================
    # LIVE MODE 2 (identisch?)
    # ================================================================
    print(f"\n📦 LIVE MODE (Run 2):")
    print("-" * 30)
    
    set_seeds(42)  # Same seed
    live_dataset2 = ObjectFusionGTDatasetStaged(
        dataroot=cfg_dict["dataset"]["dataroot"],
        version=cfg_dict["dataset"]["version"],
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None
    )
    
    live_frame2 = live_dataset2[0]
    print(f"Sample: {live_frame2['sample_token']}")
    
    for sensor_name, sensor_info in live_frame2['sensor_data'].items():
        count = len(sensor_info['features'])
        print(f"{sensor_name}: {count} objects")
    
    # ================================================================
    # VERGLEICH LIVE RUNS
    # ================================================================
    print(f"\n⚖️ LIVE RUN VERGLEICH:")
    print("-" * 30)
    
    for sensor_name in live_frame1['sensor_data'].keys():
        count1 = len(live_frame1['sensor_data'][sensor_name]['features'])
        count2 = len(live_frame2['sensor_data'][sensor_name]['features'])
        match = "✅" if count1 == count2 else "❌"
        print(f"{sensor_name}: Run1={count1}, Run2={count2} {match}")
    
    # ================================================================
    # RAM MODE
    # ================================================================
    print(f"\n🚀 RAM MODE:")
    print("-" * 30)
    
    try:
        from oft.transformer.utils.virtual_sensor_cache import load_complete_dataset_to_ram
        from oft.transformer.utils.config_hash_utils import generate_unified_config_hash
        
        config_hash = generate_unified_config_hash(cfg_dict)
        cache_dir = cfg_dict.get('training', {}).get('preprocessed_cache_dir', '/data/daiber_fent/virtual_sensor_cache')
        
        ram_cache = load_complete_dataset_to_ram(
            cache_dir=cache_dir,
            config_hash=config_hash,
            split_name=val_split,
            logger=None
        )
        
        ram_dataset = ObjectFusionGTDatasetStaged(
            dataroot=cfg_dict["dataset"]["dataroot"],
            version=cfg_dict["dataset"]["version"],
            split_name=val_split,
            pipeline_config=cfg_dict,
            verbose=False,
            ram_cache=ram_cache
        )
        
        ram_frame = ram_dataset[0]
        print(f"Sample: {ram_frame['sample_token']}")
        
        for sensor_name, sensor_info in ram_frame['sensor_data'].items():
            count = len(sensor_info['features'])
            print(f"{sensor_name}: {count} objects")
        
        # ================================================================
        # FINAL VERGLEICH
        # ================================================================
        print(f"\n🎯 LIVE vs RAM VERGLEICH:")
        print("-" * 30)
        
        for sensor_name in live_frame1['sensor_data'].keys():
            live_count = len(live_frame1['sensor_data'][sensor_name]['features'])
            ram_count = len(ram_frame['sensor_data'][sensor_name]['features'])
            match = "✅" if live_count == ram_count else "❌"
            diff = abs(live_count - ram_count)
            print(f"{sensor_name}: Live={live_count}, RAM={ram_count}, Diff={diff} {match}")
        
        # ================================================================
        # CONFIGURATION VERGLEICH
        # ================================================================
        print(f"\n🔧 CONFIGURATION ANALYSIS:")
        print("-" * 30)
        
        # Check virtual sensors config
        virtual_sensors = cfg_dict.get('dataset', {}).get('virtual_sensors', {})
        print(f"Virtual sensors enabled:")
        for sensor_cfg in virtual_sensors:
            if sensor_cfg.get('enabled', False):
                name = sensor_cfg.get('name', 'Unknown')
                detection_rate = sensor_cfg.get('detection_rate', 'N/A')
                false_positive_rate = sensor_cfg.get('false_positive_rate', 'N/A')
                print(f"  {name}: detection_rate={detection_rate}, fp_rate={false_positive_rate}")
        
        # Check for any randomness sources
        preprocessing_cfg = cfg_dict.get('preprocessing', {})
        training_cfg = cfg_dict.get('training', {})
        
        print(f"\nRandomness sources:")
        print(f"  preprocessing seed: {preprocessing_cfg.get('seed', 'NOT SET')}")
        print(f"  training seed: {training_cfg.get('seed', 'NOT SET')}")
        
        if live_count != ram_count:
            print(f"\n🚨 URSACHE:")
            print(f"  RAM Cache wurde mit anderer Config/Seed erstellt!")
            print(f"  Pre-Processing verwendet möglicherweise andere Parameter!")
    
    except Exception as e:
        print(f"❌ RAM Mode failed: {e}")


if __name__ == "__main__":
    main()
