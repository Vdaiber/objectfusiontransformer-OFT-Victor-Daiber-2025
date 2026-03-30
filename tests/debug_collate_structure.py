"""
Debug script: Collate Function Output Structure Analysis
Zeigt die komplette Struktur des Collate Function Outputs und aktuellen Config Hash.
"""

from __future__ import annotations

import os
import hydra
import pprint
from omegaconf import DictConfig
from typing import Dict, Any

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive
from oft.transformer.utils.config_hash_utils import generate_unified_config_hash, get_complete_dataset_path


def analyze_structure(obj, prefix="", max_depth=3, current_depth=0):
    """Rekursive Struktur-Analyse mit Tiefenbegrenzung."""
    if current_depth >= max_depth:
        return f"{prefix}... (max depth reached)"
    
    result = []
    
    if isinstance(obj, dict):
        result.append(f"{prefix}Dict with {len(obj)} keys:")
        for key, value in obj.items():
            if hasattr(value, 'shape'):  # Tensor/Array
                result.append(f"{prefix}  {key}: {type(value).__name__} {value.shape} {value.dtype}")
            elif hasattr(value, '__len__') and not isinstance(value, str):
                result.append(f"{prefix}  {key}: {type(value).__name__} length={len(value)}")
                if current_depth < max_depth - 1 and len(value) > 0:
                    # Zeige Struktur des ersten Elements
                    first_elem = value[0] if isinstance(value, (list, tuple)) else next(iter(value.values()))
                    result.append(analyze_structure(first_elem, f"{prefix}    [0]: ", max_depth, current_depth + 1))
            else:
                result.append(f"{prefix}  {key}: {type(value).__name__} = {value}")
    elif isinstance(obj, (list, tuple)):
        result.append(f"{prefix}{type(obj).__name__} with {len(obj)} elements:")
        if len(obj) > 0 and current_depth < max_depth - 1:
            result.append(analyze_structure(obj[0], f"{prefix}  [0]: ", max_depth, current_depth + 1))
    elif hasattr(obj, 'shape'):  # Tensor/Array
        result.append(f"{prefix}{type(obj).__name__} {obj.shape} {obj.dtype}")
    else:
        result.append(f"{prefix}{type(obj).__name__} = {obj}")
    
    return "\n".join(result)


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 DEBUG: Collate Function Structure & Config Hash")
    print("="*70)
    
    # 1. Aktueller Config Hash
    config_hash = generate_unified_config_hash(cfg_dict)
    cache_dir = cfg_dict.get('training', {}).get('preprocessed_cache_dir', 
                             cfg_dict.get('preprocessing', {}).get('cache_dir', 
                                        '/data/daiber_fent/virtual_sensor_cache'))
    
    print(f"\n1. AKTUELLER CONFIG HASH:")
    print(f"   Hash: {config_hash}")
    print(f"   Cache dir: {cache_dir}")
    
    # Cache-Pfade prüfen
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    train_split = cfg_dict["training"].get("train_split_name", "mini_train")
    
    print(f"\n2. CACHE-PFADE:")
    for split_name in [train_split, val_split]:
        dataset_path = get_complete_dataset_path(cache_dir, config_hash, split_name)
        exists = os.path.exists(dataset_path)
        
        print(f"   {split_name}:")
        print(f"     Path: {dataset_path}")
        print(f"     Exists: {exists}")
        if exists:
            size_mb = os.path.getsize(dataset_path) / (1024*1024)
            print(f"     Size: {size_mb:.2f} MB")
    
    # 3. LIVE Dataset Sample laden
    print(f"\n3. LIVE DATASET SAMPLE STRUCTURE:")
    try:
        live_dataset = ObjectFusionGTDatasetStaged(
            dataroot=cfg_dict["dataset"]["dataroot"],
            version=cfg_dict["dataset"]["version"],
            split_name=val_split,
            pipeline_config=cfg_dict,
            verbose=False,
            ram_cache=None  # Force LIVE mode
        )
        
        # Ersten Sample laden
        frame = live_dataset[0]
        sample_token = frame['sample_token']
        
        print(f"   Sample token: {sample_token}")
        print(f"   Frame structure:")
        print(analyze_structure(frame, "     ", max_depth=4))
        
    except Exception as e:
        print(f"   ❌ LIVE Dataset error: {e}")
    
    # 4. COLLATE FUNCTION OUTPUT STRUCTURE
    print(f"\n4. COLLATE FUNCTION OUTPUT STRUCTURE:")
    try:
        # Batch mit einem Sample
        batch = object_fusion_gt_collate_fn_autoregressive([frame])
        
        print(f"   Batch structure (AFTER COLLATE):")
        print(analyze_structure(batch, "     ", max_depth=4))
        
        # Spezifische GT-Analyse
        print(f"\n   GROUND TRUTH DETAILED ANALYSIS:")
        gt_boxes_norm = batch['gt_boxes_b_normalized']
        gt_labels = batch['gt_labels_b']
        gt_mask = batch['gt_valid_mask_b']
        
        print(f"     gt_boxes_b_normalized: {gt_boxes_norm.shape} {gt_boxes_norm.dtype}")
        print(f"     gt_labels_b: {gt_labels.shape} {gt_labels.dtype}")
        print(f"     gt_valid_mask_b: {gt_mask.shape} {gt_mask.dtype}")
        
        # Erste Werte anzeigen
        if gt_boxes_norm.shape[1] > 0:
            print(f"     First GT box: {gt_boxes_norm[0, 0]}")
            print(f"     First GT label: {gt_labels[0, 0].item()}")
            print(f"     First GT valid: {gt_mask[0, 0].item()}")
            
        # Statistiken
        valid_count = gt_mask[0].sum().item()
        unique_labels = set(gt_labels[0][gt_mask[0]].tolist())
        
        print(f"     Valid GT count: {valid_count}")
        print(f"     Unique labels: {unique_labels}")
        
    except Exception as e:
        print(f"   ❌ Collate Function error: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. EMPFEHLUNG
    print(f"\n5. EMPFEHLUNG:")
    cache_exists = any(
        os.path.exists(get_complete_dataset_path(cache_dir, config_hash, split))
        for split in [train_split, val_split]
    )
    
    if cache_exists:
        print(f"   ⚠️  ALTE CACHE MIT FEHLERHAFTER COLLATE FUNCTION GEFUNDEN!")
        print(f"   📝 Lösche Cache-Verzeichnis: {cache_dir}/config_{config_hash}/")
        print(f"   🔄 Danach neues Training/Preprocessing starten")
    else:
        print(f"   ✅ Keine alte Cache gefunden - kann direkt starten")
    
    print(f"\n✅ Struktur-Analyse abgeschlossen!")


if __name__ == "__main__":
    main()
