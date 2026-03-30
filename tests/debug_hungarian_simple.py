"""
Simplified Hungarian Matcher Test um den exakten Fehler zu finden.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from typing import Dict, Any
import torch

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive
from oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🔍 SIMPLIFIED HUNGARIAN MATCHER TEST")
    print("="*50)
    
    # Dataset laden
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    val_dataset = ObjectFusionGTDatasetStaged(
        dataroot=cfg_dict["dataset"]["dataroot"],
        version=cfg_dict["dataset"]["version"],
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None
    )
    
    # Ersten Sample laden
    frame = val_dataset[0]
    batch = object_fusion_gt_collate_fn_autoregressive([frame])
    sample_token = batch['sample_tokens'][0]
    
    print(f"\n🔍 Sample: {sample_token}")
    
    # Ground Truth extrahieren  
    gt_boxes_norm = batch['gt_boxes_b_normalized'][0]
    gt_labels = batch['gt_labels_b'][0]
    gt_mask = batch['gt_valid_mask_b'][0]
    
    valid_mask = gt_labels >= 0
    gt_boxes_valid = gt_boxes_norm[valid_mask]
    gt_labels_valid = gt_labels[valid_mask]
    
    print(f"   GT Objects: {len(gt_boxes_valid)}")
    print(f"   GT Labels: {gt_labels_valid[:10].tolist()}...")
    
    # Hungarian Matcher erstellen
    try:
        matcher = HungarianMatcher(
            cost_center=cfg_dict['loss']['cost_center'],
            cost_class=cfg_dict['loss']['cost_class'],
            cost_size=cfg_dict['loss']['cost_size'],
            cost_giou_bev=cfg_dict['loss']['cost_giou_bev'],
            cost_angle=cfg_dict['loss']['cost_angle'],
            cfg=cfg_dict
        )
        print(f"   ✅ Matcher created successfully")
    except Exception as e:
        print(f"   ❌ Matcher creation failed: {e}")
        return
    
    # Test Predictions
    num_queries = 100
    num_classes = len(cfg_dict['dataset']['class_names'])
    
    torch.manual_seed(42)
    pred_boxes_norm = torch.randn(num_queries, 10, dtype=torch.float64) * 0.3 + 0.5
    pred_logits = torch.randn(num_queries, num_classes + 1, dtype=torch.float32) * 2.0
    
    # Predictions Dict
    predictions_dict = {
        'pred_class_logits_batch': pred_logits.unsqueeze(0),
        'pred_boxes_normalized': pred_boxes_norm.unsqueeze(0)
    }
    
    # Targets Dict
    targets_dict = {
        'gt_boxes_b_normalized': gt_boxes_valid.unsqueeze(0),
        'gt_labels_b': [gt_labels_valid],
        'gt_valid_mask_b': [torch.ones(len(gt_labels_valid), dtype=torch.bool)]
    }
    
    print(f"   Predictions shape: {predictions_dict['pred_boxes_normalized'].shape}")
    print(f"   Targets shape: {targets_dict['gt_boxes_b_normalized'].shape}")
    
    # Hungarian Matching durchführen
    try:
        indices = matcher(predictions_dict, targets_dict)
        print(f"   ✅ Hungarian matching successful")
        print(f"   Assignments: {len(indices[0][0])} matches")
        
        # Test Config-Zugriff
        print(f"\n🔍 CONFIG ACCESS TEST:")
        print(f"   loss keys: {list(cfg_dict['loss'].keys())}")
        print(f"   cost_center: {cfg_dict['loss']['cost_center']}")
        print(f"   cost_class: {cfg_dict['loss']['cost_class']}")
        print(f"   huber delta: {cfg_dict['loss']['huber']['delta']}")
        
    except Exception as e:
        print(f"   ❌ Hungarian matching failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Test completed!")


if __name__ == "__main__":
    main()
