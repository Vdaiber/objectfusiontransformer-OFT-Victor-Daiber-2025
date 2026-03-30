#!/usr/bin/env python3
"""
Unit Tests für Batch Size Kompatibilität im Training Pipeline

Testet kritische Komponenten auf Skalierbarkeit mit verschiedenen Batch Sizes:
1. Hungarian Matcher
2. Loss Functions  
3. Box Reconstruction
4. Memory Usage

FOKUS: Identifiziere Probleme bei Batch Size Erhöhung von 1 → 4, 8, 16
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
import sys
import os
sys.path.append('/app/src')

from oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher
from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion
from oft.transformer.training.losses.classification_losses import LossClass
from oft.transformer.training.losses.regression_losses import LossCenter, LossSize, LossAngle, LossVelocity
from oft.transformer.training.losses.giou_losses import LossGIoUBEV
from oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive


class TestBatchSizeCompatibility:
    """Test Suite für Batch Size Skalierung"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def cfg(self):
        """Minimal Config für Tests"""
        return {
            'dataset': {'num_classes': 13},
            'model': {
                'd_model': 256,
                'num_queries': 300,
                'sensor_fusion': {
                    'virtual_lidar': {'enabled': True},
                    'virtual_camera': {'enabled': True}, 
                    'virtual_radar': {'enabled': True}
                },
                'encoders': {
                    'intra_modal': {'nhead': 8, 'num_layers': 3},
                    'fusion': {'nhead': 8, 'num_layers': 3}
                },
                            'decoder': {
                'center_head': {'layers': [256, 256]},
                'size_head': {'layers': [256, 256]},
                'yaw_head': {'layers': [256, 256]},
                'velocity_head': {'layers': [256, 256]},
                'class_head': {'layers': [256, 256]},
                'attribute_head': {'layers': [256, 256]},
                'num_classes': 12  # 12 real classes + 1 no_object = 13 total
            },
            'metadata_encoder': {
                'object_metadata': {'enabled': True},
                'scene_metadata': {'enabled': True}
            }
            },
            'training': {
                'matcher': {
                    'cost_center': 100.0,
                    'cost_class': 10.0,
                    'cost_giou_bev': 20.0,
                    'cost_size': 0.0,
                    'cost_angle': 0.0
                }
                    },
        'loss': {
            'huber': {'delta': 0.01},
            'class_eos_coefficient': 0.8,
            'classification_loss': 'WeightedCrossEntropy',
            'weights': {
                'loss_center': 1.0,
                'loss_size': 1.0,
                'loss_angle': 1.0,
                'loss_velocity': 1.0,
                'loss_class': 1.0,
                'loss_attributes': 1.0,
                'loss_giou_bev': 1.0
            }
        }
        }
    
    def create_sample_data(self, batch_size: int, num_queries: int = 300, num_classes: int = 13, device: torch.device = None):
        """Erstellt Sample-Daten für verschiedene Batch Sizes"""
        if device is None:
            device = torch.device('cpu')
            
        # Predictions (Model Output Format)
        predictions = {
            'pred_class_logits_batch': torch.randn(batch_size, num_queries, num_classes, device=device),
            'pred_boxes_normalized': torch.randn(batch_size, num_queries, 10, device=device),
            'pred_attributes_logits_batch': torch.randn(batch_size, num_queries, 11, device=device),
            'pred_velocity_offsets_normalized': torch.randn(batch_size, num_queries, 2, device=device)
        }
        
        # Ground Truth (Variable per Batch)
        targets = {
            'gt_labels_b': [],
            'gt_boxes_b_normalized': torch.zeros(batch_size, 50, 10, device=device),  # Max 50 GT per batch
            'gt_valid_mask_b': [],
            'gt_attributes_b': [],
            'gt_velocity_b_normalized': torch.zeros(batch_size, 50, 2, device=device)
        }
        
        # Create realistic GT distribution per batch element
        for b in range(batch_size):
            # Simulate variable number of GT objects (5-15 per sample)
            num_gt = np.random.randint(5, 16)
            
            # Ground truth classes
            gt_classes = torch.randint(0, num_classes-1, (num_gt,), device=device)  # Exclude no_object
            targets['gt_labels_b'].append(gt_classes)
            
            # Ground truth boxes (normalized)
            targets['gt_boxes_b_normalized'][b, :num_gt] = torch.randn(num_gt, 10, device=device)
            
            # Valid mask (True for actual GT, False for padding)
            valid_mask = torch.zeros(50, dtype=torch.bool, device=device)
            valid_mask[:num_gt] = True
            targets['gt_valid_mask_b'].append(valid_mask)
            
            # Ground truth attributes
            gt_attributes = torch.randint(0, 11, (num_gt,), device=device)
            targets['gt_attributes_b'].append(gt_attributes)
            
            # Ground truth velocity
            targets['gt_velocity_b_normalized'][b, :num_gt] = torch.randn(num_gt, 2, device=device)
        
        return predictions, targets
    
    def test_hungarian_matcher_batch_scaling(self, cfg, device):
        """Test Hungarian Matcher mit verschiedenen Batch Sizes"""
        print(f"\n=== TEST: Hungarian Matcher Batch Scaling ===")
        
        matcher = HungarianMatcher(
            cost_center=cfg['training']['matcher']['cost_center'],
            cost_class=cfg['training']['matcher']['cost_class'],
            cost_giou_bev=cfg['training']['matcher']['cost_giou_bev'],
            cost_size=cfg['training']['matcher']['cost_size'],
            cost_angle=cfg['training']['matcher']['cost_angle'],
            cfg=cfg
        ).to(device)
        
        batch_sizes = [1, 2, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n--- Testing Batch Size: {batch_size} ---")
            
            # Create test data
            predictions, targets = self.create_sample_data(batch_size, device=device)
            
            # Test matcher forward pass
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if start_time:
                start_time.record()
            
            try:
                (indices, matched_masks), avg_costs = matcher(predictions, targets)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)
                else:
                    elapsed_time = 0.0
                
                # Verify output structure
                assert len(indices) == batch_size, f"Expected {batch_size} indices, got {len(indices)}"
                assert len(matched_masks) == batch_size, f"Expected {batch_size} matched_masks, got {len(matched_masks)}"
                
                # Verify indices format per batch element
                total_matches = 0
                for b, (src_idx, tgt_idx) in enumerate(indices):
                    assert len(src_idx) == len(tgt_idx), f"Batch {b}: Mismatched indices lengths"
                    assert src_idx.max() < 300 if len(src_idx) > 0 else True, f"Batch {b}: Invalid src index"
                    assert tgt_idx.max() < len(targets['gt_labels_b'][b]) if len(tgt_idx) > 0 else True, f"Batch {b}: Invalid tgt index"
                    total_matches += len(src_idx)
                
                results[batch_size] = {
                    'success': True,
                    'elapsed_time': elapsed_time,
                    'total_matches': total_matches,
                    'memory_allocated': torch.cuda.memory_allocated() if device.type == 'cuda' else 0
                }
                
                print(f"✅ SUCCESS: {total_matches} total matches, {elapsed_time:.2f}ms")
                
            except Exception as e:
                results[batch_size] = {
                    'success': False,
                    'error': str(e),
                    'memory_allocated': torch.cuda.memory_allocated() if device.type == 'cuda' else 0
                }
                print(f"❌ FAILED: {e}")
        
        # Analyze scaling behavior
        print(f"\n=== SCALING ANALYSIS ===")
        for batch_size, result in results.items():
            if result['success']:
                print(f"Batch {batch_size}: {result['elapsed_time']:.2f}ms, {result['total_matches']} matches, {result['memory_allocated']/1024/1024:.1f}MB")
            else:
                print(f"Batch {batch_size}: FAILED - {result['error']}")
        
        return results
    
    def test_loss_functions_batch_scaling(self, cfg, device):
        """Test Loss Functions mit verschiedenen Batch Sizes"""
        print(f"\n=== TEST: Loss Functions Batch Scaling ===")
        
        # Initialize loss functions
        losses = {
            'center': LossCenter(cfg),
            'size': LossSize(cfg),
            'angle': LossAngle(cfg),
            'velocity': LossVelocity(cfg),
            'class': LossClass(cfg),
            'giou': LossGIoUBEV(cfg)
        }
        
        for name, loss_fn in losses.items():
            loss_fn.to(device)
        
        batch_sizes = [1, 2, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n--- Testing Batch Size: {batch_size} ---")
            
            predictions, targets = self.create_sample_data(batch_size, device=device)
            
            # Create dummy indices (Hungarian matcher results)
            indices = []
            for b in range(batch_size):
                num_gt = len(targets['gt_labels_b'][b])
                # Simulate matching: first num_gt queries match to all GT
                src_indices = torch.arange(num_gt, device=device)
                tgt_indices = torch.arange(num_gt, device=device)
                indices.append((src_indices, tgt_indices))
            
            num_boxes = torch.tensor(sum(len(targets['gt_labels_b'][b]) for b in range(batch_size)), 
                                   dtype=torch.float, device=device)
            
            batch_results = {}
            
            # Test each loss function
            for loss_name, loss_fn in losses.items():
                try:
                    if loss_name == 'class':
                        # Special handling for classification loss
                        # Create unified targets (matched + no_object)
                        target_classes = torch.full(
                            (batch_size, 300), 12, dtype=torch.long, device=device  # no_object = 12
                        )
                        for b, (src_idx, tgt_idx) in enumerate(indices):
                            if len(src_idx) > 0:
                                target_classes[b, src_idx] = targets['gt_labels_b'][b][tgt_idx]
                        
                        targets_modified = targets.copy()
                        targets_modified['gt_labels_b'] = target_classes
                        loss_value = loss_fn(predictions, targets_modified, indices, num_boxes)
                    else:
                        loss_value = loss_fn(predictions, targets, indices, num_boxes)
                    
                    batch_results[loss_name] = {
                        'success': True,
                        'loss_value': loss_value.item(),
                        'loss_shape': loss_value.shape
                    }
                    print(f"  ✅ {loss_name}: {loss_value.item():.4f}")
                    
                except Exception as e:
                    batch_results[loss_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  ❌ {loss_name}: FAILED - {e}")
            
            results[batch_size] = batch_results
        
        return results
    
    def test_criterion_batch_scaling(self, cfg, device):
        """Test SetCriterion (komplette Loss Pipeline) mit verschiedenen Batch Sizes"""
        print(f"\n=== TEST: SetCriterion Batch Scaling ===")
        
        # Create criterion
        criterion = SetCriterion(
            weight_dict=cfg['loss']['weights'],
            losses=['center', 'size', 'angle', 'velocity', 'class', 'attributes', 'giou_bev'],
            cfg=cfg
        ).to(device)
        
        batch_sizes = [1, 2, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n--- Testing Batch Size: {batch_size} ---")
            
            predictions, targets = self.create_sample_data(batch_size, device=device)
            
            try:
                # Forward pass through criterion
                loss_dict = criterion(predictions, targets)
                
                # Verify output structure
                expected_losses = ['loss_center', 'loss_size', 'loss_angle', 'loss_velocity', 
                                 'loss_class', 'loss_attributes', 'loss_giou_bev']
                
                for loss_name in expected_losses:
                    assert loss_name in loss_dict, f"Missing loss: {loss_name}"
                    assert isinstance(loss_dict[loss_name], torch.Tensor), f"Invalid loss type: {loss_name}"
                    assert loss_dict[loss_name].dim() == 0, f"Loss should be scalar: {loss_name}"
                
                total_loss = sum(loss_dict.values())
                
                results[batch_size] = {
                    'success': True,
                    'total_loss': total_loss.item(),
                    'individual_losses': {k: v.item() for k, v in loss_dict.items()}
                }
                
                print(f"✅ SUCCESS: Total Loss = {total_loss.item():.4f}")
                for loss_name, loss_value in loss_dict.items():
                    print(f"    {loss_name}: {loss_value.item():.4f}")
                
            except Exception as e:
                results[batch_size] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ FAILED: {e}")
        
        return results
    
    def test_memory_scaling(self, cfg, device):
        """Test Memory Usage mit verschiedenen Batch Sizes"""
        if device.type != 'cuda':
            print("Skipping memory test on CPU")
            return {}
            
        print(f"\n=== TEST: Memory Scaling ===")
        
        batch_sizes = [1, 2, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated()
            
            try:
                predictions, targets = self.create_sample_data(batch_size, device=device)
                
                # Create matcher
                matcher = HungarianMatcher(
                    cost_center=100.0, cost_class=10.0, cost_giou_bev=20.0,
                    cost_size=0.0, cost_angle=0.0, cfg=cfg
                ).to(device)
                
                # Forward pass
                (indices, matched_masks), avg_costs = matcher(predictions, targets)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - initial_memory
                
                results[batch_size] = {
                    'success': True,
                    'memory_used_mb': memory_used / 1024 / 1024,
                    'memory_per_sample_mb': memory_used / batch_size / 1024 / 1024
                }
                
                print(f"Batch {batch_size}: {memory_used/1024/1024:.1f}MB total, {memory_used/batch_size/1024/1024:.1f}MB per sample")
                
            except Exception as e:
                results[batch_size] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"Batch {batch_size}: FAILED - {e}")
        
        return results


def run_full_test_suite():
    """Führt alle Tests aus und erstellt einen Bericht"""
    print("="*80)
    print("BATCH SIZE COMPATIBILITY TEST SUITE")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test configuration
    cfg = {
        'dataset': {'num_classes': 13},
        'model': {
            'd_model': 256,
            'num_queries': 300,
            'sensor_fusion': {
                'virtual_lidar': {'enabled': True},
                'virtual_camera': {'enabled': True}, 
                'virtual_radar': {'enabled': True}
            },
            'encoders': {
                'intra_modal': {'nhead': 8, 'num_layers': 3},
                'fusion': {'nhead': 8, 'num_layers': 3}
            },
            'decoder': {
                'center_head': {'layers': [256, 256]},
                'size_head': {'layers': [256, 256]},
                'yaw_head': {'layers': [256, 256]},
                'velocity_head': {'layers': [256, 256]},
                'class_head': {'layers': [256, 256]},
                'attribute_head': {'layers': [256, 256]},
                'num_classes': 12  # 12 real classes + 1 no_object = 13 total
            },
            'metadata_encoder': {
                'object_metadata': {'enabled': True},
                'scene_metadata': {'enabled': True}
            }
        },
        'training': {
            'matcher': {
                'cost_center': 100.0,
                'cost_class': 10.0,
                'cost_giou_bev': 20.0,
                'cost_size': 0.0,
                'cost_angle': 0.0
            }
        },
        'loss': {
            'huber': {'delta': 0.01},
            'class_eos_coefficient': 0.8,
            'classification_loss': 'WeightedCrossEntropy',
            'weights': {
                'loss_center': 1.0,
                'loss_size': 1.0,
                'loss_angle': 1.0,
                'loss_velocity': 1.0,
                'loss_class': 1.0,
                'loss_attributes': 1.0,
                'loss_giou_bev': 1.0
            }
        }
    }
    
    test_suite = TestBatchSizeCompatibility()
    
    # Run all tests
    matcher_results = test_suite.test_hungarian_matcher_batch_scaling(cfg, device)
    loss_results = test_suite.test_loss_functions_batch_scaling(cfg, device)
    criterion_results = test_suite.test_criterion_batch_scaling(cfg, device)
    memory_results = test_suite.test_memory_scaling(cfg, device)
    
    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print("\n🔍 HUNGARIAN MATCHER:")
    for batch_size, result in matcher_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  Batch {batch_size}: {status}")
        if not result['success']:
            print(f"    Error: {result['error']}")
    
    print("\n🔍 LOSS FUNCTIONS:")
    for batch_size, batch_results in loss_results.items():
        print(f"  Batch {batch_size}:")
        for loss_name, result in batch_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"    {loss_name}: {status}")
            if not result['success']:
                print(f"      Error: {result['error']}")
    
    print("\n🔍 CRITERION PIPELINE:")
    for batch_size, result in criterion_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  Batch {batch_size}: {status}")
        if not result['success']:
            print(f"    Error: {result['error']}")
    
    if memory_results:
        print("\n🔍 MEMORY SCALING:")
        for batch_size, result in memory_results.items():
            if result['success']:
                print(f"  Batch {batch_size}: {result['memory_used_mb']:.1f}MB total, {result['memory_per_sample_mb']:.1f}MB per sample")
            else:
                print(f"  Batch {batch_size}: FAILED - {result['error']}")
    
    # Identify critical issues
    print("\n" + "="*80)
    print("CRITICAL ISSUES IDENTIFIED")
    print("="*80)
    
    issues_found = False
    
    # Check for matcher failures
    failed_matchers = [bs for bs, result in matcher_results.items() if not result['success']]
    if failed_matchers:
        print(f"🚨 HUNGARIAN MATCHER failures at batch sizes: {failed_matchers}")
        issues_found = True
    
    # Check for loss function failures
    for batch_size, batch_results in loss_results.items():
        failed_losses = [loss for loss, result in batch_results.items() if not result['success']]
        if failed_losses:
            print(f"🚨 LOSS FUNCTION failures at batch size {batch_size}: {failed_losses}")
            issues_found = True
    
    # Check for criterion failures
    failed_criteria = [bs for bs, result in criterion_results.items() if not result['success']]
    if failed_criteria:
        print(f"🚨 CRITERION failures at batch sizes: {failed_criteria}")
        issues_found = True
    
    if not issues_found:
        print("✅ No critical issues found! System appears to scale properly.")
    
    return {
        'matcher': matcher_results,
        'losses': loss_results,
        'criterion': criterion_results,
        'memory': memory_results
    }


if __name__ == '__main__':
    results = run_full_test_suite()
