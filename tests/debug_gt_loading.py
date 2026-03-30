"""
Debug script: Überprüfe Ground Truth Loading
Findet heraus warum alle GT-Labels -1 sind.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from typing import Dict, Any

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 DEBUG: Ground Truth Loading")
    print("="*50)
    
    # Dataset laden
    droot = cfg_dict["dataset"]["dataroot"]
    version = cfg_dict["dataset"]["version"]
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    
    print(f"📁 Dataset: {version} / {val_split}")
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
    
    # Raw GT direkt von der Methode laden
    print("\n1. DIRECT _get_full_box_data_for_token_world:")
    try:
        raw_gt = val_dataset._get_full_box_data_for_token_world(sample_token)
        print(f"   Raw GT Objekte: {len(raw_gt)}")
        if len(raw_gt) > 0:
            print(f"   Erstes GT Objekt: {raw_gt[0]}")
        else:
            print("   ❌ LEER! Keine Ground Truth Objekte geladen!")
    except Exception as e:
        print(f"   ❌ FEHLER beim GT-Loading: {e}")
    
    # Processed GT aus Frame
    print("\n2. PROCESSED GT aus Frame:")
    gt_normalized = frame.get("ground_truth", {}).get("normalized", [])
    gt_labels = frame.get("ground_truth", {}).get("labels", [])
    
    print(f"   GT normalized: {len(gt_normalized)}")
    print(f"   GT labels: {gt_labels}")
    
    if len(gt_normalized) > 0:
        print(f"   Erstes normalized GT: {gt_normalized[0]}")
    else:
        print("   ❌ LEER! Keine normalisierten GT Objekte!")
    
    # TruckScenes DevKit direkt testen
    print("\n3. DIREKT TRUCKSCENES DEVKIT:")
    try:
        ts = val_dataset.ts
        sample_record = ts.get('sample', sample_token)
        print(f"   Sample timestamp: {sample_record['timestamp']}")
        print(f"   Sample annotation tokens: {len(sample_record.get('anns', []))}")
        
        # Annotations direkt laden
        ann_tokens = sample_record.get('anns', [])
        print(f"   Annotation tokens: {ann_tokens[:5]}...")  # Erste 5
        
        valid_anns = 0
        for ann_token in ann_tokens[:10]:  # Teste nur erste 10
            ann_record = ts.get('sample_annotation', ann_token)
            category_token = ann_record['category_token']
            category_record = ts.get('category', category_token)
            category_name = category_record['name']
            
            print(f"   Ann {ann_token[:8]}: {category_name}")
            if category_name in cfg_dict['dataset']['class_names']:
                valid_anns += 1
        
        print(f"   Valide Annotations (in class_names): {valid_anns}")
        
    except Exception as e:
        print(f"   ❌ FEHLER bei DevKit: {e}")
    
    print("\n4. DATASET CONFIG:")
    print(f"   Class names: {cfg_dict['dataset']['class_names']}")
    print(f"   Cutoff distance: {cfg_dict['dataset']['cutoff_dist']}")
    
    # Teste mehrere Samples
    print(f"\n5. QUICK CHECK MEHRERE SAMPLES:")
    for i in range(min(10, len(val_dataset))):
        frame_i = val_dataset[i]
        gt_norm_i = frame_i.get("ground_truth", {}).get("normalized", [])
        print(f"   Sample {i}: {len(gt_norm_i)} GT Objekte")
    
    print("\n✅ Debug abgeschlossen!")


if __name__ == "__main__":
    main()
