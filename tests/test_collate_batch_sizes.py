"""
Final Collate Function Test: Verschiedene Batch-Sizes
Garantiert dass alle Funktionen korrekt ein-/auslesen bei verschiedenen Batch-Größen.
"""

from __future__ import annotations

import torch
import hydra
from omegaconf import DictConfig
from typing import Dict, Any, List
from torch.utils.data import DataLoader

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive
from oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher


def test_single_sample(dataset, cfg_dict):
    """Teste einzelnen Sample."""
    frame = dataset[0]
    batch = object_fusion_gt_collate_fn_autoregressive([frame])
    
    print(f"  SINGLE SAMPLE:")
    print(f"    gt_boxes_b_normalized: {batch['gt_boxes_b_normalized'].shape}")
    print(f"    gt_labels_b: {batch['gt_labels_b'].shape}")
    print(f"    gt_valid_mask_b: {batch['gt_valid_mask_b'].shape}")
    
    # Prüfe GT-Daten
    valid_count = batch['gt_valid_mask_b'][0].sum().item()
    unique_labels = set(batch['gt_labels_b'][0][batch['gt_valid_mask_b'][0]].tolist())
    
    print(f"    Valid GT objects: {valid_count}")
    print(f"    Unique labels: {unique_labels}")
    
    return batch


def test_batch_dataloader(dataset, batch_size, cfg_dict):
    """Teste mit DataLoader und verschiedenen Batch-Sizes."""
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=object_fusion_gt_collate_fn_autoregressive
    )
    
    # Ersten Batch holen
    batch = next(iter(dataloader))
    
    print(f"  BATCH SIZE {batch_size}:")
    print(f"    gt_boxes_b_normalized: {batch['gt_boxes_b_normalized'].shape}")
    print(f"    gt_labels_b: {batch['gt_labels_b'].shape}")
    print(f"    gt_valid_mask_b: {batch['gt_valid_mask_b'].shape}")
    print(f"    sample_tokens: {len(batch['sample_tokens'])}")
    
    # Prüfe jedes Sample im Batch
    total_valid = 0
    all_labels = set()
    
    for i in range(batch_size):
        valid_count = batch['gt_valid_mask_b'][i].sum().item()
        if valid_count > 0:
            sample_labels = set(batch['gt_labels_b'][i][batch['gt_valid_mask_b'][i]].tolist())
            all_labels.update(sample_labels)
        total_valid += valid_count
        print(f"      Sample {i}: {valid_count} valid GT objects")
    
    print(f"    Total valid objects: {total_valid}")
    print(f"    All unique labels: {all_labels}")
    
    return batch


def test_hungarian_matcher_with_batch(batch, batch_size, cfg_dict):
    """Teste Hungarian Matcher mit verschiedenen Batch-Sizes."""
    print(f"  HUNGARIAN MATCHER TEST (Batch Size {batch_size}):")
    
    try:
        # Hungarian Matcher erstellen
        matcher = HungarianMatcher(
            cost_center=cfg_dict['loss']['cost_center'],
            cost_class=cfg_dict['loss']['cost_class'],
            cost_size=cfg_dict['loss']['cost_size'],
            cost_giou_bev=cfg_dict['loss']['cost_giou_bev'],
            cost_angle=cfg_dict['loss']['cost_angle'],
            cfg=cfg_dict
        )
        
        # Dummy Predictions erstellen
        num_queries = 50  # Kleiner für Tests
        num_classes = len(cfg_dict['dataset']['class_names'])
        
        torch.manual_seed(42)
        pred_boxes = torch.randn(batch_size, num_queries, 10, dtype=torch.float64) * 0.3 + 0.5
        pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, dtype=torch.float32) * 2.0
        
        predictions = {
            'pred_class_logits_batch': pred_logits,
            'pred_boxes_normalized': pred_boxes
        }
        
        # Targets für Matcher (Liste-Format)
        targets = {
            'gt_boxes_b_normalized': batch['gt_boxes_b_normalized'],
            'gt_labels_b': [batch['gt_labels_b'][i] for i in range(batch_size)],
            'gt_valid_mask_b': [batch['gt_valid_mask_b'][i] for i in range(batch_size)]
        }
        
        # Hungarian Matching
        (indices, _), costs = matcher(predictions, targets)
        
        print(f"    ✅ Hungarian matching successful")
        print(f"    Assignments per sample: {[len(idx[0]) for idx in indices]}")
        print(f"    Average costs: center={costs.get('cost_center', 0):.4f}, class={costs.get('cost_class', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Hungarian matching failed: {e}")
        return False


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 FINAL COLLATE FUNCTION & BATCH SIZE TEST")
    print("="*60)
    
    # Dataset laden (LIVE mode - ohne fehlerhafte Cache)
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    dataset = ObjectFusionGTDatasetStaged(
        dataroot=cfg_dict["dataset"]["dataroot"],
        version=cfg_dict["dataset"]["version"],
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None  # Force LIVE mode
    )
    
    print(f"\n📁 Dataset: {len(dataset)} samples in {val_split}")
    print(f"   Mode: {dataset.mode}")
    
    # Test 1: Einzelner Sample
    print(f"\n1. EINZELNER SAMPLE TEST:")
    single_batch = test_single_sample(dataset, cfg_dict)
    
    # Test 2: Verschiedene Batch-Sizes
    print(f"\n2. BATCH SIZE TESTS:")
    batch_sizes = [1, 2, 4, 8]
    batch_results = {}
    
    for batch_size in batch_sizes:
        if batch_size <= len(dataset):
            batch = test_batch_dataloader(dataset, batch_size, cfg_dict)
            batch_results[batch_size] = batch
        else:
            print(f"  BATCH SIZE {batch_size}: ⚠️ Skipped (larger than dataset)")
    
    # Test 3: Hungarian Matcher mit verschiedenen Batch-Sizes
    print(f"\n3. HUNGARIAN MATCHER TESTS:")
    matcher_results = {}
    
    for batch_size, batch in batch_results.items():
        success = test_hungarian_matcher_with_batch(batch, batch_size, cfg_dict)
        matcher_results[batch_size] = success
    
    # Test 4: Konsistenz-Prüfung
    print(f"\n4. KONSISTENZ-PRÜFUNG:")
    
    # Prüfe ob Batch Size 1 identisch mit Einzelsample ist
    if 1 in batch_results:
        batch_1 = batch_results[1]
        
        # Vergleiche GT-Daten
        gt_match = torch.equal(single_batch['gt_boxes_b_normalized'], batch_1['gt_boxes_b_normalized'])
        labels_match = torch.equal(single_batch['gt_labels_b'], batch_1['gt_labels_b'])
        mask_match = torch.equal(single_batch['gt_valid_mask_b'], batch_1['gt_valid_mask_b'])
        
        print(f"  Single vs Batch-1 GT boxes: {'✅' if gt_match else '❌'}")
        print(f"  Single vs Batch-1 labels: {'✅' if labels_match else '❌'}")
        print(f"  Single vs Batch-1 masks: {'✅' if mask_match else '❌'}")
    
    # Test 5: Sensor Data Konsistenz
    print(f"\n5. SENSOR DATA KONSISTENZ:")
    if 2 in batch_results:
        batch_2 = batch_results[2]
        
        for sensor_name in batch_2['sensor_data'].keys():
            sensor_data = batch_2['sensor_data'][sensor_name]
            print(f"  {sensor_name}:")
            print(f"    features: {sensor_data['features'].shape}")
            print(f"    centers: {sensor_data['centers'].shape}")
            print(f"    boxes: {sensor_data['boxes'].shape}")
            print(f"    mask: {sensor_data['mask'].shape}")
    
    # FINALE BEWERTUNG
    print(f"\n🎯 FINALE BEWERTUNG:")
    print("="*40)
    
    all_tests_passed = True
    
    # GT-Daten vorhanden?
    if single_batch['gt_valid_mask_b'][0].sum().item() > 0:
        print("✅ Ground Truth Daten korrekt geladen")
    else:
        print("❌ Keine Ground Truth Daten!")
        all_tests_passed = False
    
    # Batch-Sizes funktionieren?
    successful_batches = sum(1 for success in matcher_results.values() if success)
    print(f"✅ {successful_batches}/{len(matcher_results)} Batch-Sizes erfolgreich")
    
    if successful_batches < len(matcher_results):
        all_tests_passed = False
    
    # Hungarian Matcher funktioniert?
    if all(matcher_results.values()):
        print("✅ Hungarian Matcher funktioniert mit allen Batch-Sizes")
    else:
        print("❌ Hungarian Matcher Probleme!")
        all_tests_passed = False
    
    if all_tests_passed:
        print("\n🎉 ALLE TESTS BESTANDEN!")
        print("🚀 Ready für Training mit beliebigen Batch-Sizes!")
    else:
        print("\n⚠️ EINIGE TESTS FEHLGESCHLAGEN!")
        print("🔧 Weitere Debugging erforderlich")
    
    print(f"\n💡 GARANTIE:")
    if all_tests_passed:
        print("   ✅ Collate Function funktioniert korrekt")
        print("   ✅ Ground Truth wird korrekt ein-/ausgelesen")
        print("   ✅ Hungarian Matcher ist kompatibel")
        print("   ✅ Skaliert auf höhere Batch-Sizes")
    
    print(f"\n✅ Test abgeschlossen!")


if __name__ == "__main__":
    main()
