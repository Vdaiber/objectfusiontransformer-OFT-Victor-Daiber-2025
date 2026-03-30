"""
BIT-GENAUE VERGLEICH: Pre-Processing vs Training (Batch Size 1)
Testet ob bei Batch Size 1 die Daten 100% identisch sind.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from typing import Dict, Any
import torch
import numpy as np
import logging

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive
from oft.transformer.datasets.loaders.autoregressive_loader import build_autoregressive_dataloaders


def bit_exact_compare(arr1, arr2, name: str, tolerance: float = 0.0) -> bool:
    """Bit-genauer Vergleich zweier Arrays."""
    
    # Type Check
    if type(arr1) != type(arr2):
        print(f"    ❌ {name}: Type mismatch - {type(arr1)} vs {type(arr2)}")
        return False
    
    # Handle different types
    if isinstance(arr1, torch.Tensor) and isinstance(arr2, torch.Tensor):
        # Shape check
        if arr1.shape != arr2.shape:
            print(f"    ❌ {name}: Shape mismatch - {arr1.shape} vs {arr2.shape}")
            return False
        
        # Dtype check
        if arr1.dtype != arr2.dtype:
            print(f"    ❌ {name}: Dtype mismatch - {arr1.dtype} vs {arr2.dtype}")
            return False
        
        # Exact comparison
        if torch.equal(arr1, arr2):
            print(f"    ✅ {name}: BIT-EXACT match!")
            return True
        else:
            # Show differences
            if tolerance > 0 and torch.allclose(arr1, arr2, atol=tolerance):
                max_diff = torch.max(torch.abs(arr1 - arr2)).item()
                print(f"    ⚠️ {name}: Close match (max diff: {max_diff:.2e})")
                return True
            else:
                max_diff = torch.max(torch.abs(arr1 - arr2)).item()
                print(f"    ❌ {name}: Different (max diff: {max_diff:.2e})")
                
                # Show first few different values
                diff_mask = ~torch.equal(arr1, arr2)
                if diff_mask.any():
                    flat1 = arr1.flatten()
                    flat2 = arr2.flatten()
                    diff_indices = diff_mask.flatten().nonzero().flatten()[:5]  # First 5 differences
                    print(f"      First differences:")
                    for idx in diff_indices:
                        print(f"        [{idx}]: {flat1[idx].item()} vs {flat2[idx].item()}")
                return False
    
    elif isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        # Convert to torch for comparison
        t1 = torch.from_numpy(arr1)
        t2 = torch.from_numpy(arr2)
        return bit_exact_compare(t1, t2, name, tolerance)
    
    elif isinstance(arr1, (list, tuple)) and isinstance(arr2, (list, tuple)):
        if len(arr1) != len(arr2):
            print(f"    ❌ {name}: Length mismatch - {len(arr1)} vs {len(arr2)}")
            return False
        
        for i, (item1, item2) in enumerate(zip(arr1, arr2)):
            if not bit_exact_compare(item1, item2, f"{name}[{i}]", tolerance):
                return False
        
        print(f"    ✅ {name}: BIT-EXACT list match!")
        return True
    
    else:
        # Scalar comparison
        if arr1 == arr2:
            print(f"    ✅ {name}: Exact scalar match!")
            return True
        else:
            print(f"    ❌ {name}: Scalar mismatch - {arr1} vs {arr2}")
            return False


def compare_samples_detailed(sample1: Dict[str, Any], sample2: Dict[str, Any]) -> Dict[str, bool]:
    """Detaillierter Bit-genauer Vergleich zweier Samples."""
    print("🔍 BIT-EXACT SAMPLE COMPARISON:")
    
    results = {}
    
    # Sample Token
    token1 = sample1.get('sample_token', 'MISSING')
    token2 = sample2.get('sample_token', 'MISSING')
    results['sample_token'] = bit_exact_compare(token1, token2, "sample_token")
    
    # Ground Truth
    print("\n  📊 GROUND TRUTH:")
    gt1 = sample1.get('ground_truth', {})
    gt2 = sample2.get('ground_truth', {})
    
    # GT normalized
    gt_norm1 = gt1.get('normalized', [])
    gt_norm2 = gt2.get('normalized', [])
    
    if len(gt_norm1) != len(gt_norm2):
        print(f"    ❌ GT normalized count: {len(gt_norm1)} vs {len(gt_norm2)}")
        results['gt_normalized'] = False
    else:
        gt_match = True
        print(f"    GT Object Count: {len(gt_norm1)}")
        
        # Compare first few GT objects in detail
        for i in range(min(3, len(gt_norm1))):
            obj1 = gt_norm1[i]
            obj2 = gt_norm2[i]
            
            print(f"    GT Object {i}:")
            
            # Box 10D
            box1 = obj1.get('box_10d_normalized')
            box2 = obj2.get('box_10d_normalized')
            if not bit_exact_compare(box1, box2, f"  box_10d[{i}]"):
                gt_match = False
            
            # Class idx
            class1 = obj1.get('class_idx')
            class2 = obj2.get('class_idx')
            if not bit_exact_compare(class1, class2, f"  class_idx[{i}]"):
                gt_match = False
            
            # Attribute idx
            attr1 = obj1.get('attribute_idx')
            attr2 = obj2.get('attribute_idx')
            if not bit_exact_compare(attr1, attr2, f"  attribute_idx[{i}]"):
                gt_match = False
        
        results['gt_normalized'] = gt_match
    
    # GT labels
    labels1 = gt1.get('labels', [])
    labels2 = gt2.get('labels', [])
    results['gt_labels'] = bit_exact_compare(labels1, labels2, "gt_labels")
    
    # Sensor Data
    print("\n  📡 SENSOR DATA:")
    sensor1 = sample1.get('sensor_data', {})
    sensor2 = sample2.get('sensor_data', {})
    
    if set(sensor1.keys()) != set(sensor2.keys()):
        print(f"    ❌ Sensor keys mismatch: {set(sensor1.keys())} vs {set(sensor2.keys())}")
        results['sensor_data'] = False
    else:
        sensor_match = True
        for sensor_name in sensor1.keys():
            print(f"    {sensor_name}:")
            
            s1 = sensor1[sensor_name]
            s2 = sensor2[sensor_name]
            
            for key in ['features', 'metadata', 'centers', 'boxes']:
                if key in s1 and key in s2:
                    if not bit_exact_compare(s1[key], s2[key], f"    {key}"):
                        sensor_match = False
                else:
                    print(f"      ❌ {key}: Missing in one sensor")
                    sensor_match = False
        
        results['sensor_data'] = sensor_match
    
    # Other fields
    print("\n  🗃️ METADATA:")
    other_fields = ['ego_translation_world', 'ego_rotation_world_quat', 'scene_meta', 'ego_motion', 'temporal_info']
    for field in other_fields:
        if field in sample1 and field in sample2:
            results[field] = bit_exact_compare(sample1[field], sample2[field], field)
        else:
            print(f"    ⚠️ {field}: Missing in one sample")
            results[field] = False
    
    return results


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🎯 BIT-GENAUE VERGLEICH: Pre-Processing vs Training (Batch Size 1)")
    print("="*80)
    
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    
    # ================================================================
    # 1. RAM MODE SAMPLE (direkt aus Cache)
    # ================================================================
    print(f"\n📦 SCHRITT 1: RAM MODE Sample (direkt)")
    print("-" * 50)
    
    try:
        from oft.transformer.utils.virtual_sensor_cache import load_complete_dataset_to_ram
        from oft.transformer.utils.config_hash_utils import generate_unified_config_hash
        
        config_hash = generate_unified_config_hash(cfg_dict)
        cache_dir = cfg_dict.get('training', {}).get('preprocessed_cache_dir', '/data/daiber_fent/virtual_sensor_cache')
        
        ram_cache = load_complete_dataset_to_ram(
            cache_dir=cache_dir,
            config_hash=config_hash,
            split_name=val_split,
            logger=None
        )
        
        ram_dataset = ObjectFusionGTDatasetStaged(
            dataroot=cfg_dict["dataset"]["dataroot"],
            version=cfg_dict["dataset"]["version"],
            split_name=val_split,
            pipeline_config=cfg_dict,
            verbose=False,
            ram_cache=ram_cache
        )
        
        ram_sample = ram_dataset[0]
        print(f"RAM Sample Token: {ram_sample['sample_token']}")
        print(f"RAM GT Objects: {len(ram_sample['ground_truth']['normalized'])}")
        
        for sensor_name, sensor_data in ram_sample['sensor_data'].items():
            count = len(sensor_data['features'])
            print(f"RAM {sensor_name}: {count} objects")
    
    except Exception as e:
        print(f"❌ RAM Mode failed: {e}")
        return
    
    # ================================================================
    # 2. TRAINING DATALOADER (Batch Size 1)
    # ================================================================
    print(f"\n🚀 SCHRITT 2: Training DataLoader (Batch Size 1)")
    print("-" * 50)
    
    try:
        # Setze Batch Size auf 1 und Sequential Sampler für deterministische Reihenfolge
        cfg_dict['training']['batch_size'] = 1
        cfg_dict['training']['sampler'] = 'sequential'  # ← WICHTIG: Sequential für gleiche Reihenfolge
        
        # Setup Logger
        logging.basicConfig(level=logging.WARNING)  # Reduce noise
        logger = logging.getLogger(__name__)
        
        dataloaders = build_autoregressive_dataloaders(
            cfg_dict, 
            logger=logger,
            splits_to_build=[val_split]
        )
        
        dataloader = dataloaders[val_split]
        
        # Get first batch (should be single sample)
        batch = next(iter(dataloader))
        
        print(f"DataLoader Batch Size: {dataloader.batch_size}")
        print(f"DataLoader Sample Tokens: {batch['sample_tokens']}")
        print(f"DataLoader GT Shape: {batch['gt_boxes_b_normalized'].shape}")
        
        # Extract single sample from batch
        training_sample = {
            'sample_token': batch['sample_tokens'][0],
            'ground_truth': {
                'normalized': [],
                'labels': batch['gt_labels_b'][0][batch['gt_valid_mask_b'][0]].cpu().numpy(),
            },
            'sensor_data': {}
        }
        
        # Reconstruct GT normalized from batch
        valid_mask = batch['gt_valid_mask_b'][0]
        valid_count = valid_mask.sum().item()
        
        gt_boxes = batch['gt_boxes_b_normalized'][0][valid_mask]
        gt_labels = batch['gt_labels_b'][0][valid_mask]
        gt_attributes = batch['gt_attributes_b'][0][valid_mask]
        
        for i in range(valid_count):
            gt_obj = {
                'box_10d_normalized': gt_boxes[i].cpu().numpy(),
                'class_idx': gt_labels[i].cpu().item(),
                'attribute_idx': gt_attributes[i].cpu().item()
            }
            training_sample['ground_truth']['normalized'].append(gt_obj)
        
        # Reconstruct sensor data
        for sensor_name, sensor_tensors in batch['sensor_data'].items():
            sensor_mask = ~sensor_tensors['mask'][0]  # Invert mask (True = padding)
            sensor_count = sensor_mask.sum().item()
            
            training_sample['sensor_data'][sensor_name] = {
                'features': sensor_tensors['features'][0][sensor_mask].cpu().numpy(),
                'metadata': sensor_tensors['metadata'][0][sensor_mask].cpu().numpy(),
                'centers': sensor_tensors['centers'][0][sensor_mask].cpu().numpy(),
                'boxes': sensor_tensors['boxes'][0][sensor_mask].cpu().numpy(),
            }
            
            print(f"Training {sensor_name}: {sensor_count} objects")
        
        # Copy other fields if available
        if 'ego_translation_world' in batch:
            training_sample['ego_translation_world'] = batch['ego_translation_world'][0].cpu().numpy()
        if 'ego_rotation_world_quat' in batch:
            training_sample['ego_rotation_world_quat'] = batch['ego_rotation_world_quat'][0].cpu().numpy()
        if 'scene_meta' in batch:
            training_sample['scene_meta'] = batch['scene_meta'][0]
        if 'ego_motion' in batch:
            training_sample['ego_motion'] = batch['ego_motion']
        if 'temporal_info' in batch:
            training_sample['temporal_info'] = batch['temporal_info'][0]
        
    except Exception as e:
        print(f"❌ Training DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ================================================================
    # 3. BIT-EXACT COMPARISON
    # ================================================================
    print(f"\n⚖️ SCHRITT 3: BIT-EXACT COMPARISON")
    print("-" * 50)
    
    if ram_sample['sample_token'] != training_sample['sample_token']:
        print(f"❌ DIFFERENT SAMPLES!")
        print(f"  RAM: {ram_sample['sample_token']}")
        print(f"  Training: {training_sample['sample_token']}")
        print(f"  Cannot compare different samples!")
        return
    
    # Detailed comparison
    comparison_results = compare_samples_detailed(ram_sample, training_sample)
    
    # ================================================================
    # 4. SUMMARY
    # ================================================================
    print(f"\n🎯 FINALE BEWERTUNG:")
    print("-" * 30)
    
    all_match = all(comparison_results.values())
    
    print(f"All components BIT-EXACT: {'✅' if all_match else '❌'}")
    
    if not all_match:
        print(f"\nMismatches:")
        for component, match in comparison_results.items():
            if not match:
                print(f"  ❌ {component}")
    
    if all_match:
        print(f"\n🎉 PERFEKT! Pre-Processing und Training sind BIT-EXACT identisch!")
        print(f"   Das bedeutet dein System funktioniert 100% korrekt!")
    else:
        print(f"\n🚨 PROBLEM: Es gibt Unterschiede zwischen Pre-Processing und Training!")
        print(f"   Das erklärt möglicherweise die Null-Losses!")


if __name__ == "__main__":
    main()
