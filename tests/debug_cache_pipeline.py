"""
Debug script: Cache Pipeline Analysis
Analysiert die komplette Caching-Pipeline um zu prüfen ob fehlerhaftes GT-Caching 
die Ursache für das Problem ist.
"""

from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig
from typing import Dict, Any
import torch

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.utils.config_hash_utils import generate_unified_config_hash, get_complete_dataset_path
from oft.transformer.utils.virtual_sensor_cache import load_complete_dataset_to_ram


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 DEBUG: Cache Pipeline Analysis")
    print("="*60)
    
    # 1. Config Hash generieren
    config_hash = generate_unified_config_hash(cfg_dict)
    print(f"\n1. CONFIG HASH:")
    print(f"   Generated hash: {config_hash}")
    
    # 2. Cache Verzeichnis prüfen
    cache_dir = cfg_dict.get('training', {}).get('preprocessed_cache_dir', 
                             cfg_dict.get('preprocessing', {}).get('cache_dir', 
                                        '/data/daiber_fent/virtual_sensor_cache'))
    print(f"\n2. CACHE DIRECTORY:")
    print(f"   Cache dir: {cache_dir}")
    print(f"   Exists: {os.path.exists(cache_dir)}")
    
    # 3. Split Pfade prüfen
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    train_split = cfg_dict["training"].get("train_split_name", "mini_train")
    
    print(f"\n3. SPLIT PATHS:")
    for split_name in [train_split, val_split]:
        dataset_path = get_complete_dataset_path(cache_dir, config_hash, split_name)
        exists = os.path.exists(dataset_path)
        size = os.path.getsize(dataset_path) if exists else 0
        
        print(f"   {split_name}:")
        print(f"     Path: {dataset_path}")
        print(f"     Exists: {exists}")
        print(f"     Size: {size / (1024*1024):.2f} MB" if exists else "     Size: N/A")
    
    # 4. RAM Cache laden und analysieren
    print(f"\n4. RAM CACHE ANALYSIS:")
    try:
        ram_cache = load_complete_dataset_to_ram(
            cache_dir=cache_dir,
            config_hash=config_hash,
            split_name=val_split,
            logger=None
        )
        
        if ram_cache:
            print(f"   RAM Cache loaded: ✅")
            print(f"   Samples in cache: {len(ram_cache)}")
            
            # Ersten Sample analysieren
            first_token = next(iter(ram_cache.keys()))
            first_sample = ram_cache[first_token]
            
            print(f"\n   FIRST SAMPLE ANALYSIS ({first_token}):")
            print(f"     Keys: {list(first_sample.keys())}")
            
            # Ground Truth analysieren
            gt_data = first_sample.get("ground_truth", {})
            print(f"     GT Keys: {list(gt_data.keys())}")
            
            gt_normalized = gt_data.get("normalized", [])
            gt_labels = gt_data.get("labels", [])
            
            print(f"     GT normalized count: {len(gt_normalized)}")
            print(f"     GT labels count: {len(gt_labels) if hasattr(gt_labels, '__len__') else 'scalar'}")
            print(f"     GT labels type: {type(gt_labels)}")
            
            if len(gt_normalized) > 0:
                print(f"     First GT normalized: {gt_normalized[0]}")
            
            if hasattr(gt_labels, '__len__') and len(gt_labels) > 0:
                print(f"     GT labels sample: {gt_labels[:5]}...")
                print(f"     GT labels unique: {set(gt_labels) if len(gt_labels) > 0 else 'LEER'}")
            else:
                print(f"     GT labels value: {gt_labels}")
                
        else:
            print(f"   RAM Cache loading: ❌ FAILED")
            
    except Exception as e:
        print(f"   RAM Cache error: ❌ {e}")
    
    # 5. LIVE Dataset zum Vergleich
    print(f"\n5. LIVE DATASET COMPARISON:")
    try:
        live_dataset = ObjectFusionGTDatasetStaged(
            dataroot=cfg_dict["dataset"]["dataroot"],
            version=cfg_dict["dataset"]["version"],
            split_name=val_split,
            pipeline_config=cfg_dict,
            verbose=False,
            ram_cache=None  # Force LIVE mode
        )
        
        print(f"   LIVE Dataset mode: {live_dataset.mode}")
        print(f"   LIVE Dataset samples: {len(live_dataset)}")
        
        # Ersten Sample laden
        live_sample = live_dataset[0]
        live_gt = live_sample.get("ground_truth", {})
        live_gt_normalized = live_gt.get("normalized", [])
        live_gt_labels = live_gt.get("labels", [])
        
        print(f"   LIVE GT normalized count: {len(live_gt_normalized)}")
        print(f"   LIVE GT labels count: {len(live_gt_labels) if hasattr(live_gt_labels, '__len__') else 'scalar'}")
        
        if hasattr(live_gt_labels, '__len__') and len(live_gt_labels) > 0:
            print(f"   LIVE GT labels sample: {live_gt_labels[:5]}...")
            print(f"   LIVE GT labels unique: {set(live_gt_labels)}")
        
    except Exception as e:
        print(f"   LIVE Dataset error: ❌ {e}")
    
    # 6. RAM vs LIVE Vergleich
    print(f"\n6. RAM vs LIVE COMPARISON:")
    try:
        # RAM Dataset
        ram_dataset = ObjectFusionGTDatasetStaged(
            dataroot=cfg_dict["dataset"]["dataroot"],
            version=cfg_dict["dataset"]["version"],
            split_name=val_split,
            pipeline_config=cfg_dict,
            verbose=False,
            ram_cache=ram_cache  # Use RAM cache
        )
        
        print(f"   RAM Dataset mode: {ram_dataset.mode}")
        print(f"   RAM Dataset samples: {len(ram_dataset)}")
        
        # Vergleiche ersten Sample
        ram_sample = ram_dataset[0]
        ram_gt = ram_sample.get("ground_truth", {})
        ram_gt_normalized = ram_gt.get("normalized", [])
        ram_gt_labels = ram_gt.get("labels", [])
        
        print(f"   RAM GT normalized count: {len(ram_gt_normalized)}")
        print(f"   RAM GT labels: {ram_gt_labels[:5] if hasattr(ram_gt_labels, '__len__') and len(ram_gt_labels) > 0 else ram_gt_labels}...")
        
        # Sind sie identisch?
        if 'live_gt_labels' in locals() and 'ram_gt_labels' in locals():
            if hasattr(live_gt_labels, '__len__') and hasattr(ram_gt_labels, '__len__'):
                labels_match = (live_gt_labels == ram_gt_labels).all() if hasattr(live_gt_labels, '__len__') else live_gt_labels == ram_gt_labels
                print(f"   GT Labels match: {labels_match}")
            
    except Exception as e:
        print(f"   RAM Dataset error: ❌ {e}")
    
    print("\n✅ Cache Pipeline Analysis completed!")
    
    print(f"\n💡 SUMMARY:")
    print(f"   • Config hash: {config_hash}")
    print(f"   • Cache functioning properly if all components show correct GT data")
    print(f"   • If RAM cache has empty GT but LIVE has data → Caching bug")
    print(f"   • If both have empty GT → Dataset loading bug (fixed)")


if __name__ == "__main__":
    main()
