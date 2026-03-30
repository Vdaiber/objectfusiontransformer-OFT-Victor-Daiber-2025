"""
Debug: RAM vs LIVE Mode Unterschiede
Findet heraus warum RAM Mode weniger Objekte hat als LIVE Mode
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from typing import Dict, Any
import json

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.utils.config_hash_utils import generate_unified_config_hash


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 DEBUG: RAM vs LIVE Mode Unterschiede")
    print("="*60)
    
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    test_sample_token = "32d2bcf46e734dffb14fe2e0a823d059"
    
    # ================================================================
    # 1. LIVE MODE
    # ================================================================
    print(f"\n📦 LIVE MODE:")
    print("-" * 30)
    
    live_dataset = ObjectFusionGTDatasetStaged(
        dataroot=cfg_dict["dataset"]["dataroot"],
        version=cfg_dict["dataset"]["version"],
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None  # ← LIVE MODE
    )
    
    live_frame = live_dataset[0]
    live_sample_token = live_frame['sample_token']
    
    print(f"Sample Token: {live_sample_token}")
    
    # GT Analysis
    live_gt = live_frame.get('ground_truth', {})
    live_gt_normalized = live_gt.get('normalized', [])
    live_gt_labels = live_gt.get('labels', [])
    
    print(f"GT normalized: {len(live_gt_normalized)} objects")
    print(f"GT labels: {len(live_gt_labels) if hasattr(live_gt_labels, '__len__') else 'Not a list'}")
    
    # Sensor Analysis
    live_sensor_data = live_frame.get('sensor_data', {})
    for sensor_name, sensor_info in live_sensor_data.items():
        features_count = len(sensor_info.get('features', []))
        print(f"{sensor_name}: {features_count} objects")
    
    # ================================================================
    # 2. CACHE FILE DIREKT
    # ================================================================
    print(f"\n💾 CACHE FILE DIRECT:")
    print("-" * 30)
    
    config_hash = generate_unified_config_hash(cfg_dict)
    cache_path = f"/data/daiber_fent/virtual_sensor_cache/config_{config_hash}/complete_dataset_mini_val.json"
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        results = cache_data.get('results', {})
        
        if live_sample_token in results:
            cached_sample = results[live_sample_token]
            
            # GT Analysis
            cached_gt = cached_sample.get('ground_truth', {})
            cached_gt_normalized = cached_gt.get('normalized', [])
            cached_gt_labels = cached_gt.get('labels', [])
            
            print(f"GT normalized: {len(cached_gt_normalized)} objects")
            print(f"GT labels: {len(cached_gt_labels) if hasattr(cached_gt_labels, '__len__') else 'Not a list'}")
            
            # Sensor Analysis
            cached_sensor_data = cached_sample.get('sensor_data', {})
            for sensor_name, sensor_info in cached_sensor_data.items():
                features_count = len(sensor_info.get('features', []))
                print(f"{sensor_name}: {features_count} objects")
            
            # Detaillierter Vergleich GT
            print(f"\n🔍 GT DETAILVERGLEICH:")
            if len(live_gt_normalized) == len(cached_gt_normalized):
                print(f"✅ GT Count Match: {len(live_gt_normalized)}")
                
                # Vergleiche ersten GT-Eintrag
                if len(live_gt_normalized) > 0 and len(cached_gt_normalized) > 0:
                    live_first = live_gt_normalized[0]
                    cached_first = cached_gt_normalized[0]
                    
                    print(f"Live First GT keys: {list(live_first.keys())}")
                    print(f"Cached First GT keys: {list(cached_first.keys())}")
                    
                    if isinstance(live_first.get('class_idx'), int) and isinstance(cached_first.get('class_idx'), int):
                        print(f"Live class_idx: {live_first['class_idx']}")
                        print(f"Cached class_idx: {cached_first['class_idx']}")
                        print(f"Class match: {'✅' if live_first['class_idx'] == cached_first['class_idx'] else '❌'}")
            else:
                print(f"❌ GT Count Mismatch: Live={len(live_gt_normalized)}, Cached={len(cached_gt_normalized)}")
        
        else:
            print(f"❌ Sample {live_sample_token} not found in cache!")
            print(f"Available tokens: {list(results.keys())[:5]}...")
    
    except Exception as e:
        print(f"❌ Cache analysis failed: {e}")
    
    # ================================================================
    # 3. RAM MODE (Dataset mit Cache)
    # ================================================================
    print(f"\n🚀 RAM MODE (Dataset mit Cache):")
    print("-" * 30)
    
    try:
        # Lade Cache in RAM
        from oft.transformer.utils.virtual_sensor_cache import load_complete_dataset_to_ram
        
        cache_dir = cfg_dict.get('training', {}).get('preprocessed_cache_dir', '/data/daiber_fent/virtual_sensor_cache')
        ram_cache = load_complete_dataset_to_ram(
            cache_dir=cache_dir,
            config_hash=config_hash,
            split_name=val_split,
            logger=None
        )
        
        print(f"RAM Cache loaded: {len(ram_cache) if ram_cache else 0} samples")
        
        # Dataset mit RAM Cache
        ram_dataset = ObjectFusionGTDatasetStaged(
            dataroot=cfg_dict["dataset"]["dataroot"],
            version=cfg_dict["dataset"]["version"],
            split_name=val_split,
            pipeline_config=cfg_dict,
            verbose=False,
            ram_cache=ram_cache  # ← RAM MODE
        )
        
        ram_frame = ram_dataset[0]
        ram_sample_token = ram_frame['sample_token']
        
        print(f"Sample Token: {ram_sample_token}")
        
        # GT Analysis
        ram_gt = ram_frame.get('ground_truth', {})
        ram_gt_normalized = ram_gt.get('normalized', [])
        ram_gt_labels = ram_gt.get('labels', [])
        
        print(f"GT normalized: {len(ram_gt_normalized) if hasattr(ram_gt_normalized, '__len__') else 'Not a list'}")
        print(f"GT labels: {len(ram_gt_labels) if hasattr(ram_gt_labels, '__len__') else 'Not a list'}")
        
        # Sensor Analysis
        ram_sensor_data = ram_frame.get('sensor_data', {})
        for sensor_name, sensor_info in ram_sensor_data.items():
            features_count = len(sensor_info.get('features', []))
            print(f"{sensor_name}: {features_count} objects")
        
        # ================================================================
        # 4. VERGLEICH ALLER DREI
        # ================================================================
        print(f"\n⚖️ VOLLVERGLEICH:")
        print("-" * 30)
        
        print(f"Sample Tokens:")
        print(f"  Live: {live_sample_token}")
        print(f"  RAM:  {ram_sample_token}")
        print(f"  Match: {'✅' if live_sample_token == ram_sample_token else '❌'}")
        
        print(f"\nGT Object Counts:")
        live_gt_count = len(live_gt_normalized)
        ram_gt_count = len(ram_gt_normalized) if hasattr(ram_gt_normalized, '__len__') else 0
        print(f"  Live: {live_gt_count}")
        print(f"  RAM:  {ram_gt_count}")
        print(f"  Match: {'✅' if live_gt_count == ram_gt_count else '❌'}")
        
        print(f"\nSensor Object Counts:")
        for sensor_name in live_sensor_data.keys():
            live_count = len(live_sensor_data[sensor_name].get('features', []))
            ram_count = len(ram_sensor_data.get(sensor_name, {}).get('features', []))
            print(f"  {sensor_name}: Live={live_count}, RAM={ram_count}, Match={'✅' if live_count == ram_count else '❌'}")
        
        if live_gt_count != ram_gt_count:
            print(f"\n🚨 PROBLEM GEFUNDEN: GT Object Count unterschiedlich!")
            print(f"Grund muss in _ensure_tensor_format oder Cache Loading liegen!")
        
    except Exception as e:
        print(f"❌ RAM Mode Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
