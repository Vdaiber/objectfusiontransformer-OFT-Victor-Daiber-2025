"""
Debug: Analyse warum Pre-Processing leere GT produziert
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
    
    print("🔍 DEBUG: Pre-Processing GT Analysis")
    print("="*50)
    
    # 1. Config Hash
    config_hash = generate_unified_config_hash(cfg_dict)
    print(f"Config Hash: {config_hash}")
    
    # 2. Cache Pfad
    cache_path = f"/data/daiber_fent/virtual_sensor_cache/config_{config_hash}/complete_dataset_mini_val.json"
    print(f"Cache Path: {cache_path}")
    
    # 3. Lade Cache-Datei direkt
    print(f"\n📦 CACHE FILE ANALYSIS:")
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        results = cache_data.get('results', {})
        print(f"Cache has {len(results)} samples")
        
        # Analysiere ersten Sample aus Cache
        if results:
            first_token = list(results.keys())[0]
            first_sample = results[first_token]
            
            print(f"\nFirst sample token: {first_token}")
            print(f"First sample keys: {list(first_sample.keys())}")
            
            # GT Analysis
            gt_data = first_sample.get('ground_truth', {})
            print(f"GT keys: {list(gt_data.keys())}")
            
            gt_normalized = gt_data.get('normalized', [])
            gt_labels = gt_data.get('labels', [])
            
            print(f"GT normalized: {len(gt_normalized)} objects")
            print(f"GT labels: {len(gt_labels) if hasattr(gt_labels, '__len__') else 'Not a list'}")
            
            if len(gt_normalized) > 0:
                print(f"First GT object: {gt_normalized[0]}")
            else:
                print("❌ NO GT OBJECTS IN CACHE!")
    
    except Exception as e:
        print(f"❌ Cache analysis failed: {e}")
    
    # 4. Teste LIVE Dataset direkt (ohne Cache)
    print(f"\n🔴 LIVE DATASET TEST:")
    try:
        live_dataset = ObjectFusionGTDatasetStaged(
            dataroot=cfg_dict["dataset"]["dataroot"],
            version=cfg_dict["dataset"]["version"],
            split_name="mini_val",
            pipeline_config=cfg_dict,
            verbose=False,
            ram_cache=None  # ← Force LIVE mode
        )
        
        # Ersten Sample direkt laden
        frame = live_dataset[0]
        sample_token = frame['sample_token']
        
        print(f"Sample token: {sample_token}")
        
        # GT Analysis
        gt_data = frame.get('ground_truth', {})
        gt_normalized = gt_data.get('normalized', [])
        gt_labels = gt_data.get('labels', [])
        
        print(f"LIVE GT normalized: {len(gt_normalized)} objects")
        print(f"LIVE GT labels: {len(gt_labels) if hasattr(gt_labels, '__len__') else type(gt_labels)}")
        
        if len(gt_normalized) > 0:
            print(f"LIVE First GT: {gt_normalized[0]}")
            print(f"✅ LIVE mode has GT data!")
        else:
            print(f"❌ LIVE mode also has no GT!")
    
    except Exception as e:
        print(f"❌ LIVE dataset test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Prüfe Pre-Processing Mode
    print(f"\n🔧 PRE-PROCESSING MODE CHECK:")
    preprocessing_config = cfg_dict.get('preprocessing', {})
    training_config = cfg_dict.get('training', {})
    
    print(f"Preprocessing config keys: {list(preprocessing_config.keys())}")
    print(f"Training config keys: {list(training_config.keys())}")
    
    # Prüfe _force_training_mode
    force_training = cfg_dict.get('_force_training_mode', False)
    print(f"_force_training_mode: {force_training}")
    
    # Prüfe Batch Size
    batch_size = training_config.get('batch_size', 'NOT_SET')
    print(f"batch_size: {batch_size}")


if __name__ == "__main__":
    main()
