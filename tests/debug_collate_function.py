"""
Debug script: Überprüfe Collate Function
Findet heraus warum GT-Labels nach Collating -1 werden.
"""

from __future__ import annotations

import torch
import hydra
from omegaconf import DictConfig
from typing import Dict, Any

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 DEBUG: Collate Function")
    print("="*50)
    
    # Dataset laden
    droot = cfg_dict["dataset"]["dataroot"]
    version = cfg_dict["dataset"]["version"]
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    
    val_dataset = ObjectFusionGTDatasetStaged(
        dataroot=droot,
        version=version,
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None,
        devkit_ram=None,
    )
    
    # Teste ersten Sample
    sample_idx = 0
    frame = val_dataset[sample_idx]
    sample_token = frame['sample_token']
    
    print(f"\n🔍 Sample {sample_idx}: {sample_token}")
    
    # VOR Collating
    print("\n1. VOR COLLATING:")
    gt_normalized = frame.get("ground_truth", {}).get("normalized", [])
    gt_labels_raw = frame.get("ground_truth", {}).get("labels", [])
    
    print(f"   GT normalized Anzahl: {len(gt_normalized)}")
    print(f"   GT labels raw shape: {gt_labels_raw.shape if hasattr(gt_labels_raw, 'shape') else type(gt_labels_raw)}")
    print(f"   GT labels raw Inhalt: {gt_labels_raw[:10]}...")  # Erste 10
    print(f"   GT labels unique: {set(gt_labels_raw) if len(gt_labels_raw) > 0 else 'LEER'}")
    
    if len(gt_normalized) > 0:
        print(f"   Erstes GT normalized: {gt_normalized[0]}")
        
    # NACH Collating 
    print("\n2. NACH COLLATING:")
    batch = object_fusion_gt_collate_fn_autoregressive([frame])
    
    gt_boxes_norm_batch = batch['gt_boxes_b_normalized']
    gt_labels_batch = batch['gt_labels_b']
    gt_mask_batch = batch['gt_valid_mask_b']
    
    print(f"   gt_boxes_b_normalized shape: {gt_boxes_norm_batch.shape}")
    print(f"   gt_labels_b shape: {gt_labels_batch.shape}")
    print(f"   gt_valid_mask_b shape: {gt_mask_batch.shape}")
    
    print(f"   gt_labels_b Inhalt: {gt_labels_batch[0][:10]}...")  # Erste 10 vom ersten Batch
    print(f"   gt_labels_b unique: {set(gt_labels_batch[0].tolist())}")
    print(f"   gt_valid_mask_b sum: {gt_mask_batch[0].sum().item()}")  # Anzahl valider Objekte
    print(f"   gt_valid_mask_b (erste 10): {gt_mask_batch[0][:10]}")
    
    # Detailanalyse: Schaue welche Indizes valide sind
    print("\n3. DETAILANALYSE:")
    valid_mask = gt_labels_batch[0] >= 0
    invalid_mask = gt_labels_batch[0] == -1
    
    print(f"   Valide GT (labels >= 0): {valid_mask.sum().item()}")
    print(f"   Invalide GT (labels == -1): {invalid_mask.sum().item()}")
    
    if valid_mask.any():
        valid_indices = torch.where(valid_mask)[0]
        print(f"   Valide Indizes: {valid_indices[:10]}...")  # Erste 10
        print(f"   Valide Labels: {gt_labels_batch[0][valid_indices][:10]}...")
    else:
        print("   ❌ KEINE VALIDEN LABELS GEFUNDEN!")
    
    # Cross-check mit raw data
    print("\n4. CROSS-CHECK:")
    print(f"   Raw GT Labels Anzahl: {len(gt_labels_raw)}")
    print(f"   Batch GT Labels Shape: {gt_labels_batch.shape}")
    print(f"   Sind alle raw labels >= 0? {all(label >= 0 for label in gt_labels_raw)}")
    
    # Schaue was in der Collate-Function passiert
    print("\n5. COLLATE FUNCTION TRACE:")
    print("   Checking collate_functions.py logic...")
    
    # Simuliere Collate-Logik manuell
    gt_boxes_list = [frame.get("ground_truth", {}).get("normalized", [])]
    gt_classes_list = [frame.get("ground_truth", {}).get("labels", [])]
    
    print(f"   gt_boxes_list Anzahl: {len(gt_boxes_list[0])}")
    print(f"   gt_classes_list: {gt_classes_list[0][:10] if len(gt_classes_list[0]) > 0 else 'LEER'}...")
    
    # Max length
    max_len_gt = max(len(gts) for gts in gt_boxes_list) if any(len(gt) > 0 for gt in gt_boxes_list) else 0
    print(f"   max_len_gt: {max_len_gt}")
    
    # Tensor erstellen wie in Collate-Function
    B = 1
    gt_labels_manual = torch.full((B, max_len_gt), -1, dtype=torch.long)
    
    for i in range(B):
        gt_classes = gt_classes_list[i]
        if len(gt_classes) > 0:
            n = len(gt_classes)
            for j in range(min(n, max_len_gt)):
                gt_labels_manual[i, j] = int(gt_classes[j])
    
    print(f"   Manual labels: {gt_labels_manual[0][:10]}...")
    print(f"   Manual labels unique: {set(gt_labels_manual[0].tolist())}")
    
    print("\n✅ Debug abgeschlossen!")


if __name__ == "__main__":
    main()
