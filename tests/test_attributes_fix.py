"""
Test: Attribute Fix Verification
Prüft ob Attribute jetzt korrekt extrahiert werden.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from typing import Dict, Any

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 TEST: Attribute Fix Verification")
    print("="*50)
    
    # Dataset laden
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    dataset = ObjectFusionGTDatasetStaged(
        dataroot=cfg_dict["dataset"]["dataroot"],
        version=cfg_dict["dataset"]["version"],
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None
    )
    
    # Ersten Sample laden
    frame = dataset[0]
    
    print(f"\n1. VOR COLLATING:")
    gt_normalized = frame.get("ground_truth", {}).get("normalized", [])
    print(f"   GT objects: {len(gt_normalized)}")
    
    if len(gt_normalized) > 0:
        # Extrahiere class_idx und attribute_idx direkt
        classes_direct = [gt_obj['class_idx'] for gt_obj in gt_normalized]
        attributes_direct = [gt_obj['attribute_idx'] for gt_obj in gt_normalized]
        
        print(f"   Direct classes: {classes_direct[:10]}...")
        print(f"   Direct attributes: {attributes_direct[:10]}...")
        print(f"   Unique classes: {set(classes_direct)}")
        print(f"   Unique attributes: {set(attributes_direct)}")
    
    print(f"\n2. NACH COLLATING:")
    try:
        batch = object_fusion_gt_collate_fn_autoregressive([frame])
        
        gt_boxes = batch['gt_boxes_b_normalized']
        gt_labels = batch['gt_labels_b']
        gt_attributes = batch['gt_attributes_b']
        gt_mask = batch['gt_valid_mask_b']
        
        print(f"   gt_boxes_b_normalized: {gt_boxes.shape}")
        print(f"   gt_labels_b: {gt_labels.shape}")
        print(f"   gt_attributes_b: {gt_attributes.shape}")
        print(f"   gt_valid_mask_b: {gt_mask.shape}")
        
        # Prüfe Inhalte
        valid_indices = gt_mask[0]
        if valid_indices.any():
            valid_labels = gt_labels[0][valid_indices]
            valid_attributes = gt_attributes[0][valid_indices]
            
            print(f"   Valid labels: {valid_labels[:10].tolist()}...")
            print(f"   Valid attributes: {valid_attributes[:10].tolist()}...")
            print(f"   Unique labels: {set(valid_labels.tolist())}")
            print(f"   Unique attributes: {set(valid_attributes.tolist())}")
            
            # Vergleiche mit direkter Extraktion
            if len(gt_normalized) > 0:
                classes_match = valid_labels.tolist() == classes_direct
                attrs_match = valid_attributes.tolist() == attributes_direct
                
                print(f"\n3. VERGLEICH:")
                print(f"   Classes match: {'✅' if classes_match else '❌'}")
                print(f"   Attributes match: {'✅' if attrs_match else '❌'}")
                
                if not classes_match:
                    print(f"   Classes diff: {valid_labels.tolist()[:5]} vs {classes_direct[:5]}")
                if not attrs_match:
                    print(f"   Attributes diff: {valid_attributes.tolist()[:5]} vs {attributes_direct[:5]}")
        
    except Exception as e:
        print(f"   ❌ Collating failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test mit Batch Size 2
    print(f"\n4. BATCH SIZE 2 TEST:")
    try:
        frame2 = dataset[1]
        batch2 = object_fusion_gt_collate_fn_autoregressive([frame, frame2])
        
        print(f"   Batch shapes:")
        print(f"     gt_labels_b: {batch2['gt_labels_b'].shape}")
        print(f"     gt_attributes_b: {batch2['gt_attributes_b'].shape}")
        
        # Prüfe beide Samples
        for i in range(2):
            valid_mask = batch2['gt_valid_mask_b'][i]
            valid_count = valid_mask.sum().item()
            print(f"   Sample {i}: {valid_count} valid objects")
            
            if valid_count > 0:
                sample_labels = set(batch2['gt_labels_b'][i][valid_mask].tolist())
                sample_attrs = set(batch2['gt_attributes_b'][i][valid_mask].tolist())
                print(f"     Classes: {sample_labels}")
                print(f"     Attributes: {sample_attrs}")
        
    except Exception as e:
        print(f"   ❌ Batch-2 test failed: {e}")
    
    print(f"\n✅ Attribute Fix Test completed!")


if __name__ == "__main__":
    main()
