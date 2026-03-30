"""
Debug script: Attribute Structure Analysis
Analysiert wie Attribute korrekt aus Ground Truth extrahiert werden.
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
    
    print("🔍 DEBUG: Attribute Structure Analysis")
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
    sample_token = frame['sample_token']
    
    print(f"\n🔍 Sample: {sample_token}")
    
    # Ground Truth Struktur analysieren
    gt_data = frame.get("ground_truth", {})
    
    print(f"\n1. GROUND TRUTH KEYS:")
    print(f"   Keys: {list(gt_data.keys())}")
    
    # Normalized GT analysieren
    gt_normalized = gt_data.get("normalized", [])
    print(f"\n2. NORMALIZED GT STRUCTURE:")
    print(f"   Count: {len(gt_normalized)}")
    
    if len(gt_normalized) > 0:
        first_gt = gt_normalized[0]
        print(f"   First GT keys: {list(first_gt.keys())}")
        print(f"   First GT: {first_gt}")
        
        # Prüfe alle GT-Objekte auf Attribute
        print(f"\n3. ATTRIBUTE ANALYSIS:")
        attribute_indices = []
        class_indices = []
        
        for i, gt_obj in enumerate(gt_normalized[:10]):  # Erste 10
            class_idx = gt_obj.get('class_idx', -1)
            attr_idx = gt_obj.get('attribute_idx', -1)
            
            class_indices.append(class_idx)
            attribute_indices.append(attr_idx)
            
            if i < 3:  # Zeige Details für erste 3
                print(f"   GT {i}: class_idx={class_idx}, attribute_idx={attr_idx}")
        
        print(f"   All class indices: {set(class_indices)}")
        print(f"   All attribute indices: {set(attribute_indices)}")
    
    # Labels aus GT
    gt_labels = gt_data.get("labels", [])
    print(f"\n4. GT LABELS:")
    print(f"   GT labels type: {type(gt_labels)}")
    print(f"   GT labels shape/length: {gt_labels.shape if hasattr(gt_labels, 'shape') else len(gt_labels)}")
    print(f"   GT labels sample: {gt_labels[:10] if hasattr(gt_labels, '__getitem__') else gt_labels}")
    
    # Attribute Vocabulary
    from oft.transformer.datasets.truckscenes.dataset import ATTRIBUTE_VOCAB
    print(f"\n5. ATTRIBUTE VOCABULARY:")
    print(f"   Vocab size: {len(ATTRIBUTE_VOCAB)}")
    print(f"   Vocab: {ATTRIBUTE_VOCAB[:5]}...")  # Erste 5
    
    # Dataset Attribute-Mapping
    print(f"\n6. DATASET ATTRIBUTE MAPPING:")
    print(f"   attribute_to_idx: {list(dataset.attribute_to_idx.items())[:5]}...")  # Erste 5
    print(f"   num_attributes: {dataset.num_attributes}")
    
    print(f"\n💡 KORREKTE EXTRACTION:")
    print(f"   ✅ class_idx: Verfügbar in normalized GT")
    print(f"   ✅ attribute_idx: Verfügbar in normalized GT")  
    print(f"   ❌ Problem: Collate Function verwendet falsche Keys!")
    
    print(f"\n🔧 FIX NEEDED:")
    print(f"   Collate Function sollte verwenden:")
    print(f"   • gt_classes_list: [gt['class_idx'] for gt in normalized]")
    print(f"   • gt_attributes_list: [gt['attribute_idx'] for gt in normalized]")


if __name__ == "__main__":
    main()
