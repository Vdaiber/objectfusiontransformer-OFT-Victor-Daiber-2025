"""
KRITISCHER TEST: Collate Function Konsistenz
Vergleicht Pre-Processing Pfad vs Training Pfad für 100% Übereinstimmung
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from typing import Dict, Any
import torch
import numpy as np

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive
from oft.transformer.datasets.loaders.autoregressive_loader import build_autoregressive_dataloaders


def compare_tensors(tensor1, tensor2, name: str, tolerance: float = 1e-6) -> bool:
    """Vergleicht zwei Tensoren auf Gleichheit."""
    if tensor1.shape != tensor2.shape:
        print(f"    ❌ {name}: Shape mismatch - {tensor1.shape} vs {tensor2.shape}")
        return False
    
    if torch.allclose(tensor1, tensor2, atol=tolerance):
        print(f"    ✅ {name}: Identical (max diff: {torch.max(torch.abs(tensor1 - tensor2)).item():.2e})")
        return True
    else:
        max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        print(f"    ❌ {name}: Different (max diff: {max_diff:.2e})")
        return False


def compare_batches(batch1: Dict[str, Any], batch2: Dict[str, Any], prefix: str = "") -> Dict[str, bool]:
    """Vergleicht zwei Batches vollständig."""
    print(f"{prefix}🔍 BATCH COMPARISON:")
    
    results = {}
    
    # GT Tensors
    gt_keys = ['gt_boxes_b_normalized', 'gt_labels_b', 'gt_attributes_b', 'gt_valid_mask_b']
    for key in gt_keys:
        if key in batch1 and key in batch2:
            results[key] = compare_tensors(batch1[key], batch2[key], key)
        else:
            print(f"    ❌ {key}: Missing in one batch")
            results[key] = False
    
    # Sensor Data
    if 'sensor_data' in batch1 and 'sensor_data' in batch2:
        sensor1 = batch1['sensor_data']
        sensor2 = batch2['sensor_data']
        
        if set(sensor1.keys()) != set(sensor2.keys()):
            print(f"    ❌ Sensor modalities mismatch: {set(sensor1.keys())} vs {set(sensor2.keys())}")
            results['sensor_data'] = False
        else:
            sensor_match = True
            for sensor_name in sensor1.keys():
                print(f"    📡 {sensor_name}:")
                for tensor_key in ['features', 'metadata', 'centers', 'mask', 'boxes']:
                    if tensor_key in sensor1[sensor_name] and tensor_key in sensor2[sensor_name]:
                        match = compare_tensors(
                            sensor1[sensor_name][tensor_key], 
                            sensor2[sensor_name][tensor_key], 
                            f"{sensor_name}_{tensor_key}"
                        )
                        if not match:
                            sensor_match = False
                    else:
                        print(f"      ❌ {tensor_key}: Missing")
                        sensor_match = False
            results['sensor_data'] = sensor_match
    else:
        print(f"    ❌ sensor_data: Missing in one batch")
        results['sensor_data'] = False
    
    return results


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🎯 KRITISCHER TEST: Collate Function Konsistenz")
    print("="*70)
    
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    
    # ================================================================
    # 1. LIVE MODE (wie ursprünglich im Dataset)
    # ================================================================
    print(f"\n📦 SCHRITT 1: LIVE MODE Dataset")
    print("-" * 40)
    
    live_dataset = ObjectFusionGTDatasetStaged(
        dataroot=cfg_dict["dataset"]["dataroot"],
        version=cfg_dict["dataset"]["version"],
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None  # ← LIVE MODE (kein Cache)
    )
    
    print(f"Live Dataset Size: {len(live_dataset)}")
    
    # Lade 5 Samples
    test_samples = []
    for i in range(5):
        frame = live_dataset[i]
        test_samples.append(frame)
        print(f"Sample {i}: {frame['sample_token']}")
    
    # ================================================================
    # 2. COLLATE FUNCTION DIREKT (wie Training)
    # ================================================================
    print(f"\n🔧 SCHRITT 2: DIREKTE COLLATE FUNCTION")
    print("-" * 40)
    
    # Teste verschiedene Batch-Größen
    batch_sizes = [1, 2, 5]
    collated_batches = {}
    
    for batch_size in batch_sizes:
        print(f"\nBatch Size {batch_size}:")
        samples_subset = test_samples[:batch_size]
        
        try:
            batch = object_fusion_gt_collate_fn_autoregressive(samples_subset)
            
            gt_shape = batch['gt_boxes_b_normalized'].shape
            valid_count = batch['gt_valid_mask_b'].sum().item()
            
            print(f"  GT Shape: {gt_shape}")
            print(f"  Valid GT Objects: {valid_count}")
            
            collated_batches[batch_size] = batch
            
        except Exception as e:
            print(f"  ❌ Collating failed: {e}")
    
    # ================================================================
    # 3. DATALOADER (wie echtes Training)
    # ================================================================
    print(f"\n🚀 SCHRITT 3: TRAINING DATALOADER")
    print("-" * 40)
    
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        dataloaders = build_autoregressive_dataloaders(
            cfg_dict, 
            logger=logger,
            splits_to_build=[val_split]
        )
        
        if val_split in dataloaders:
            dataloader = dataloaders[val_split]
            dataloader_batch = next(iter(dataloader))
            
            print(f"DataLoader Batch Size: {dataloader.batch_size}")
            print(f"DataLoader GT Shape: {dataloader_batch['gt_boxes_b_normalized'].shape}")
            print(f"DataLoader Valid GT: {dataloader_batch['gt_valid_mask_b'].sum().item()}")
            
        else:
            print(f"❌ DataLoader konnte nicht erstellt werden")
            return
    
    except Exception as e:
        print(f"❌ DataLoader Test failed: {e}")
        return
    
    # ================================================================
    # 4. VERGLEICH: Direkte Collate vs DataLoader
    # ================================================================
    print(f"\n⚖️ SCHRITT 4: KONSISTENZ-VERGLEICH")
    print("-" * 40)
    
    # Vergleiche Batch Size 1 (sollte identisch sein)
    if 1 in collated_batches:
        print(f"\nDirekte Collate (Batch=1) vs DataLoader (Batch={dataloader.batch_size}):")
        comparison_results = compare_batches(
            collated_batches[1], 
            dataloader_batch,
            "  "
        )
        
        all_match = all(comparison_results.values())
        print(f"\n🎯 FINALE BEWERTUNG:")
        print(f"  Alle Tensoren identisch: {'✅' if all_match else '❌'}")
        
        if not all_match:
            print(f"  Fehlende Übereinstimmungen:")
            for key, match in comparison_results.items():
                if not match:
                    print(f"    • {key}")
    
    # ================================================================
    # 5. MULTI-SAMPLE KONSISTENZ
    # ================================================================
    print(f"\n🔄 SCHRITT 5: MULTI-SAMPLE KONSISTENZ")
    print("-" * 40)
    
    # Teste ob verschiedene Samples aus dem DataLoader mit direkter Collate übereinstimmen
    print(f"Teste 3 weitere DataLoader Batches...")
    
    for i in range(3):
        try:
            dl_batch = next(iter(dataloader))
            
            # Finde entsprechende Samples im Live Dataset
            sample_tokens = dl_batch['sample_tokens']
            print(f"\nBatch {i+1}: {sample_tokens}")
            
            # Lade dieselben Samples direkt
            corresponding_frames = []
            for token in sample_tokens:
                for j, frame in enumerate(test_samples):
                    if frame['sample_token'] == token:
                        corresponding_frames.append(frame)
                        break
            
            if len(corresponding_frames) == len(sample_tokens):
                direct_batch = object_fusion_gt_collate_fn_autoregressive(corresponding_frames)
                
                # Kurzer Vergleich
                gt_match = torch.allclose(
                    dl_batch['gt_boxes_b_normalized'], 
                    direct_batch['gt_boxes_b_normalized'], 
                    atol=1e-6
                )
                
                print(f"  GT Boxes Match: {'✅' if gt_match else '❌'}")
            else:
                print(f"  ⚠️ Nicht alle Samples gefunden")
                
        except Exception as e:
            print(f"  ❌ Batch {i+1} Test failed: {e}")
    
    print(f"\n🎉 KONSISTENZ-TEST ABGESCHLOSSEN!")


if __name__ == "__main__":
    main()
