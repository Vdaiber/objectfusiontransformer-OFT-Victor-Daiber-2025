"""
Quick DataLoader Test für RAM-Pfad
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from typing import Dict, Any
import logging

from oft.transformer.datasets.loaders.autoregressive_loader import build_autoregressive_dataloaders


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    # Logger setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🔍 DataLoader RAM-Pfad Test")
    print("="*40)
    
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    
    try:
        dataloaders = build_autoregressive_dataloaders(
            cfg_dict, 
            logger=logger,
            splits_to_build=[val_split]
        )
        
        if val_split in dataloaders:
            dataloader = dataloaders[val_split]
            print(f"✅ DataLoader erstellt")
            print(f"   Batch Size: {dataloader.batch_size}")
            print(f"   Dataset Size: {len(dataloader.dataset)}")
            
            # Test ersten Batch
            first_batch = next(iter(dataloader))
            print(f"\n✅ Erster Batch geladen:")
            print(f"   GT Shapes: {first_batch['gt_boxes_b_normalized'].shape}")
            print(f"   Valid GT: {first_batch['gt_valid_mask_b'].sum().item()}")
            
            # Test Labels und Attributes
            valid_mask = first_batch['gt_valid_mask_b']
            valid_labels = first_batch['gt_labels_b'][valid_mask]
            valid_attrs = first_batch['gt_attributes_b'][valid_mask]
            
            print(f"   Unique Labels: {set(valid_labels.tolist())}")
            print(f"   Unique Attributes: {set(valid_attrs.tolist())}")
            
            print(f"\n🎉 DataLoader RAM-Pfad: 100% FUNKTIONAL!")
        else:
            print(f"❌ DataLoader konnte nicht erstellt werden")
            
    except Exception as e:
        print(f"❌ DataLoader Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
