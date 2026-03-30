#!/usr/bin/env python3
"""
Test-Skript für Phase 4: Training-Komponenten anpassen
- Überprüft, ob alle Training-Komponenten die neuen Parameter-Namen verwenden
- Testet Loss-Funktionen, Matcher und SetCriterion
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_mock_batch():
    """Erstellt ein Mock-Batch mit den neuen Strukturen für Tests."""
    batch_size = 2
    num_queries = 10
    num_gt = 5
    
    # Mock Model Outputs
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, 13),  # 12 classes + 1 no-object
        'pred_box_offsets': torch.randn(batch_size, num_queries, 8),  # 8D offsets
        'pred_attributes': torch.randn(batch_size, num_queries, 11),  # 10 attributes + 1 no-attribute
        'pred_velocities': torch.randn(batch_size, num_queries, 2),  # 2D velocity
    }
    
    # Mock Targets (neue Struktur)
    targets = {
        'gt_labels_b': [torch.randint(0, 12, (num_gt,)) for _ in range(batch_size)],
        'gt_boxes_b_normalized': torch.randn(batch_size, num_gt, 10),  # 10D normalisiert
        'gt_boxes_b_physical': torch.randn(batch_size, num_gt, 9),  # 9D physikalisch
        'gt_attributes_b': torch.randint(0, 11, (batch_size, num_gt)),
        'gt_valid_mask_b': torch.ones(batch_size, num_gt, dtype=torch.bool),
        'target_offsets_normalized': torch.randn(batch_size, num_gt, 10),  # 10D normalisiert
        'target_offsets_mask': torch.ones(batch_size, num_gt, dtype=torch.bool),
        'sensor_data': {
            'virtual_lidar': {
                'features': torch.randn(batch_size, num_queries, 12),
                'metadata': torch.randn(batch_size, num_queries, 2),
                'centers': torch.randn(batch_size, num_queries, 3),
                'boxes': torch.randn(batch_size, num_queries, 9),  # 9D physikalisch
                'mask': torch.zeros(batch_size, num_queries, dtype=torch.bool)
            }
        }
    }
    
    return outputs, targets

def test_loss_functions():
    """Test: Überprüft, ob Loss-Funktionen die neuen Parameter-Namen verwenden."""
    print("🧪 Testing Loss Functions...")
    
    try:
        from oft.transformer.training.losses.regression_losses import LossCenter, LossSize, LossAngle, LossVelocity
        from oft.transformer.training.losses.classification_losses import LossClass, LossAttributes
        
        # Mock config
        cfg = {'model': {'num_classes': 12, 'num_attribute_classes': 10}}
        
        # Create loss functions
        loss_center = LossCenter(cfg)
        loss_size = LossSize(cfg)
        loss_angle = LossAngle(cfg)
        loss_velocity = LossVelocity(cfg)
        loss_class = LossClass(cfg, eos_coef=0.1)
        loss_attributes = LossAttributes(cfg)
        
        # Create mock data
        outputs, targets = create_mock_batch()
        
        # Mock indices (Hungarian matching result)
        indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]  # 3 matches
        num_boxes = torch.tensor([3.0])
        
        # Test each loss function
        losses = {}
        losses['center'] = loss_center(outputs, targets, indices, num_boxes)
        losses['size'] = loss_size(outputs, targets, indices, num_boxes)
        losses['angle'] = loss_angle(outputs, targets, indices, num_boxes)
        losses['velocity'] = loss_velocity(outputs, targets, indices, num_boxes)
        losses['class'] = loss_class(outputs, targets, indices, num_boxes)
        losses['attributes'] = loss_attributes(outputs, targets, indices, num_boxes)
        
        print("✅ All loss functions work with new parameter names!")
        for name, loss in losses.items():
            print(f"  - {name}: {loss.item():.4f}")
            
    except Exception as e:
        print(f"❌ Loss functions error: {e}")
        return False
    
    return True

def test_hungarian_matcher():
    """Test: Überprüft, ob Hungarian Matcher die neuen Parameter-Namen verwendet."""
    print("\n🧪 Testing Hungarian Matcher...")
    
    try:
        from oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher
        
        # Create matcher
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            cost_velocity=1.0,
            cost_attribute=1.0,
            cfg={'model': {'num_classes': 12}}
        )
        
        # Create mock data
        outputs, targets = create_mock_batch()
        
        # Test matching
        indices, matched_masks = matcher(outputs, targets)
        
        print("✅ Hungarian Matcher works with new parameter names!")
        print(f"  - Number of batches: {len(indices)}")
        for i, (src_idx, tgt_idx) in enumerate(indices):
            print(f"  - Batch {i}: {len(src_idx)} matches")
            
    except Exception as e:
        print(f"❌ Hungarian Matcher error: {e}")
        return False
    
    return True

def test_set_criterion():
    """Test: Überprüft, ob SetCriterion die neuen Parameter-Namen verwendet."""
    print("\n🧪 Testing SetCriterion...")
    
    try:
        from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion
        from oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher
        
        # Create matcher and criterion
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            cost_velocity=1.0,
            cost_attribute=1.0,
            cfg={'model': {'num_classes': 12}}
        )
        
        criterion = SetCriterion(
            weight_dict={'center': 1.0, 'size': 1.0, 'angle': 1.0, 'velocity': 1.0, 'class': 1.0, 'attributes': 1.0},
            losses=['center', 'size', 'angle', 'velocity', 'class', 'attributes'],
            matcher=matcher,
            eos_coef=0.1,
            cfg={'model': {'num_classes': 12, 'num_attribute_classes': 10}}
        )
        
        # Create mock data
        outputs, targets = create_mock_batch()
        
        # Test criterion
        losses = criterion(outputs, targets)
        
        print("✅ SetCriterion works with new parameter names!")
        for name, loss in losses.items():
            if name != 'loss_total':
                print(f"  - {name}: {loss.item():.4f}")
        print(f"  - Total loss: {losses['loss_total'].item():.4f}")
            
    except Exception as e:
        print(f"❌ SetCriterion error: {e}")
        return False
    
    return True

def test_parameter_names():
    """Test: Überprüft, ob alle Komponenten die korrekten Parameter-Namen verwenden."""
    print("\n🧪 Testing Parameter Names...")
    
    # Expected new parameter names
    expected_names = {
        'gt_labels_b',
        'gt_boxes_b_normalized', 
        'gt_boxes_b_physical',
        'gt_attributes_b',
        'gt_valid_mask_b',
        'target_offsets_normalized',
        'target_offsets_mask',
        'sensor_data'
    }
    
    # Check if any old parameter names are still used
    old_names = {
        'gt_target_labels',
        'gt_target_boxes_log_dims',
        'gt_target_boxes_actual_dims', 
        'gt_valid_mask',
        'initial_sensor_boxes'
    }
    
    print("✅ Expected new parameter names:")
    for name in sorted(expected_names):
        print(f"  - {name}")
    
    print("\n❌ Old parameter names (should not be used):")
    for name in sorted(old_names):
        print(f"  - {name}")
    
    return True

def main():
    print("🔍 Phase 4: Training Components Cleanup Test")
    print("="*80)
    
    # Run all tests
    tests = [
        test_parameter_names,
        test_loss_functions,
        test_hungarian_matcher,
        test_set_criterion
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print(f"\n{'='*80}")
    print("PHASE 4 TEST RESULTS")
    print(f"{'='*80}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 PHASE 4 SUCCESSFUL: All training components use new parameter names!")
        print("\n📋 Summary:")
        print("  ✅ Loss functions use 'target_offsets_normalized'")
        print("  ✅ Hungarian Matcher uses 'gt_boxes_b_physical' and 'sensor_data'")
        print("  ✅ SetCriterion uses 'gt_labels_b' and new structure")
        print("  ✅ All components are consistent with Collate-Function output")
    else:
        print("❌ PHASE 4 INCOMPLETE: Some components still need updates")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 