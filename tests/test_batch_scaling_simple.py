#!/usr/bin/env python3
"""
VEREINFACHTER Batch Size Compatibility Test

Fokus auf die KRITISCHEN Komponenten ohne komplizierte Config-Dependencies
"""

import torch
import numpy as np
import sys
import os
sys.path.append('/app/src')

from oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher
from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion


def load_real_config():
    """Lädt die echte Pipeline-Config"""
    import hydra
    from hydra import initialize, compose
    from omegaconf import DictConfig
    
    try:
        with initialize(config_path="/app/config", version_base=None):
            cfg = compose(config_name="pipeline_staged.yaml")
            return cfg
    except Exception as e:
        print(f"Failed to load real config: {e}")
        return None


def create_sample_data(batch_size: int, device: torch.device):
    """Erstellt realistische Sample-Daten für Tests"""
    num_queries = 300
    num_classes = 13
    
    # Model Predictions
    predictions = {
        'pred_class_logits_batch': torch.randn(batch_size, num_queries, num_classes, device=device),
        'pred_boxes_normalized': torch.randn(batch_size, num_queries, 10, device=device),
        'pred_attributes_logits_batch': torch.randn(batch_size, num_queries, 11, device=device),
        'pred_velocity_offsets_normalized': torch.randn(batch_size, num_queries, 2, device=device)
    }
    
    # Ground Truth (Variable pro Batch)
    targets = {
        'gt_labels_b': [],
        'gt_boxes_b_normalized': torch.zeros(batch_size, 50, 10, device=device),
        'gt_valid_mask_b': [],
        'gt_attributes_b': [],
        'gt_velocity_b_normalized': torch.zeros(batch_size, 50, 2, device=device)
    }
    
    # Erstelle realistische GT pro Batch-Element
    for b in range(batch_size):
        # 5-15 GT Objekte pro Sample (realistisch)
        num_gt = np.random.randint(5, 16)
        
        # GT Classes (ohne no_object)
        gt_classes = torch.randint(0, 12, (num_gt,), device=device)
        targets['gt_labels_b'].append(gt_classes)
        
        # GT Boxes
        targets['gt_boxes_b_normalized'][b, :num_gt] = torch.randn(num_gt, 10, device=device)
        
        # Valid Mask
        valid_mask = torch.zeros(50, dtype=torch.bool, device=device)
        valid_mask[:num_gt] = True
        targets['gt_valid_mask_b'].append(valid_mask)
        
        # GT Attributes
        gt_attributes = torch.randint(0, 11, (num_gt,), device=device)
        targets['gt_attributes_b'].append(gt_attributes)
        
        # GT Velocity
        targets['gt_velocity_b_normalized'][b, :num_gt] = torch.randn(num_gt, 2, device=device)
    
    return predictions, targets


def test_hungarian_matcher_scaling():
    """Test Hungarian Matcher mit verschiedenen Batch Sizes"""
    print("="*80)
    print("HUNGARIAN MATCHER BATCH SCALING TEST")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Lade echte Config
    cfg = load_real_config()
    if cfg is None:
        print("❌ FAILED: Kann echte Config nicht laden")
        return
    
    # Erstelle Matcher
    matcher = HungarianMatcher(
        cost_center=100.0,
        cost_class=10.0,
        cost_giou_bev=20.0,
        cost_size=0.0,
        cost_angle=0.0,
        cfg=cfg
    ).to(device)
    
    batch_sizes = [1, 2, 4, 8, 16]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing Batch Size: {batch_size} ---")
        
        try:
            # Erstelle Test-Daten
            predictions, targets = create_sample_data(batch_size, device)
            
            # Memory vor Test
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            # Matcher Forward Pass
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if start_time:
                start_time.record()
            
            (indices, matched_masks), avg_costs = matcher(predictions, targets)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
            else:
                elapsed_time = 0.0
            
            # Memory nach Test
            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - initial_memory
            else:
                memory_used = 0
            
            # Verify Results
            assert len(indices) == batch_size, f"Expected {batch_size} indices, got {len(indices)}"
            assert len(matched_masks) == batch_size, f"Expected {batch_size} matched_masks, got {len(matched_masks)}"
            
            # Verify per-batch struktur
            total_matches = 0
            for b, (src_idx, tgt_idx) in enumerate(indices):
                assert len(src_idx) == len(tgt_idx), f"Batch {b}: Index length mismatch"
                assert src_idx.max() < 300 if len(src_idx) > 0 else True, f"Batch {b}: Invalid src index"
                assert tgt_idx.max() < len(targets['gt_labels_b'][b]) if len(tgt_idx) > 0 else True, f"Batch {b}: Invalid tgt index"
                total_matches += len(src_idx)
            
            results[batch_size] = {
                'success': True,
                'elapsed_time': elapsed_time,
                'total_matches': total_matches,
                'matches_per_sample': total_matches / batch_size,
                'memory_used_mb': memory_used / 1024 / 1024 if memory_used > 0 else 0,
                'avg_costs': avg_costs
            }
            
            print(f"✅ SUCCESS: {total_matches} matches ({total_matches/batch_size:.1f} avg), {elapsed_time:.2f}ms, {memory_used/1024/1024:.1f}MB")
            
        except Exception as e:
            results[batch_size] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ FAILED: {e}")
    
    # Analyse der Skalierung
    print(f"\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80)
    
    print(f"{'Batch':<8} {'Time(ms)':<10} {'Matches':<8} {'Avg/Sample':<10} {'Memory(MB)':<12} {'Status':<10}")
    print("-" * 70)
    
    for batch_size, result in results.items():
        if result['success']:
            print(f"{batch_size:<8} {result['elapsed_time']:<10.2f} {result['total_matches']:<8} {result['matches_per_sample']:<10.1f} {result['memory_used_mb']:<12.1f} SUCCESS")
        else:
            print(f"{batch_size:<8} {'N/A':<10} {'N/A':<8} {'N/A':<10} {'N/A':<12} FAILED")
    
    # Performance Analysis
    successful_results = {k: v for k, v in results.items() if v['success']}
    if len(successful_results) >= 2:
        print(f"\n=== PERFORMANCE SCALING ===")
        
        times = [(bs, r['elapsed_time']) for bs, r in successful_results.items()]
        times.sort()
        
        for i in range(1, len(times)):
            prev_bs, prev_time = times[i-1]
            curr_bs, curr_time = times[i]
            
            time_scaling_factor = curr_time / prev_time
            batch_scaling_factor = curr_bs / prev_bs
            efficiency = batch_scaling_factor / time_scaling_factor
            
            print(f"Batch {prev_bs} → {curr_bs}: {time_scaling_factor:.2f}x time, {efficiency:.2f}x efficiency")
    
    return results


def test_criterion_scaling():
    """Test SetCriterion mit verschiedenen Batch Sizes"""
    print("\n" + "="*80)
    print("SET CRITERION BATCH SCALING TEST")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Lade echte Config
    cfg = load_real_config()
    if cfg is None:
        print("❌ FAILED: Kann echte Config nicht laden")
        return
    
    # Erstelle Criterion
    try:
        criterion = SetCriterion(
            weight_dict={
                'loss_center': 30.0,
                'loss_size': 1.0, 
                'loss_angle': 2.0,
                'loss_velocity': 3.0,
                'loss_class': 1.0,
                'loss_attributes': 0.5
            },
            losses=['center', 'size', 'angle', 'velocity', 'class', 'attributes'],
            cfg=cfg
        ).to(device)
    except Exception as e:
        print(f"❌ FAILED to create criterion: {e}")
        return
    
    batch_sizes = [1, 2, 4, 8]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing Batch Size: {batch_size} ---")
        
        try:
            # Erstelle Test-Daten
            predictions, targets = create_sample_data(batch_size, device)
            
            # Forward Pass durch Criterion
            loss_dict = criterion(predictions, targets)
            
            # Verify Loss Structure
            expected_losses = ['loss_center', 'loss_size', 'loss_angle', 'loss_velocity', 'loss_class', 'loss_attributes']
            
            for loss_name in expected_losses:
                if loss_name not in loss_dict:
                    raise ValueError(f"Missing loss: {loss_name}")
                if not isinstance(loss_dict[loss_name], torch.Tensor):
                    raise ValueError(f"Invalid loss type for {loss_name}")
                if loss_dict[loss_name].dim() != 0:
                    raise ValueError(f"Loss {loss_name} should be scalar, got shape {loss_dict[loss_name].shape}")
            
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
            import traceback
            traceback.print_exc()
    
    return results


def analyze_loop_dependencies():
    """Analysiert explizite Batch-Loops in der Codebase"""
    print("\n" + "="*80)
    print("BATCH LOOP DEPENDENCY ANALYSIS") 
    print("="*80)
    
    print("\n🔍 Analysiere kritische Dateien für explizite Batch-Loops...")
    
    # Files to analyze
    files_to_check = [
        '/app/src/oft/transformer/training/matchers/hungarian_matcher.py',
        '/app/src/oft/transformer/training/criteria/autoregressive_criterion.py',
        '/app/src/oft/transformer/training/losses/classification_losses.py',
        '/app/src/oft/transformer/training/losses/regression_losses.py',
        '/app/src/oft/transformer/models/architectures/autoregressive_architecture.py'
    ]
    
    batch_loop_patterns = [
        'for.*range.*batch_size',
        'for.*enumerate.*batch',
        'for.*in.*indices',
        'for.*batch_idx',
        'for.*b.*in.*range'
    ]
    
    findings = {}
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        print(f"\n📁 {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            file_findings = []
            for i, line in enumerate(lines, 1):
                for pattern in batch_loop_patterns:
                    import re
                    if re.search(pattern, line, re.IGNORECASE):
                        file_findings.append({
                            'line': i,
                            'code': line.strip(),
                            'pattern': pattern
                        })
            
            if file_findings:
                findings[file_path] = file_findings
                for finding in file_findings:
                    print(f"  🚨 Line {finding['line']}: {finding['code']}")
            else:
                print(f"  ✅ No explicit batch loops found")
                
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    
    print(f"\n" + "="*80)
    print("SUMMARY: BATCH LOOP DEPENDENCIES")
    print("="*80)
    
    if findings:
        print("\n⚠️  EXPLICIT BATCH LOOPS FOUND:")
        for file_path, file_findings in findings.items():
            print(f"\n📁 {os.path.basename(file_path)} ({len(file_findings)} loops):")
            for finding in file_findings:
                print(f"  Line {finding['line']}: {finding['code']}")
    else:
        print("\n✅ No explicit batch loops found in critical files!")
    
    return findings


def main():
    """Main Test Function"""
    print("BATCH SIZE COMPATIBILITY ANALYSIS")
    print("="*80)
    
    # Test 1: Hungarian Matcher
    matcher_results = test_hungarian_matcher_scaling()
    
    # Test 2: SetCriterion
    criterion_results = test_criterion_scaling()
    
    # Test 3: Code Analysis
    loop_findings = analyze_loop_dependencies()
    
    # Final Summary
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    
    print("\n🔍 HUNGARIAN MATCHER SCALING:")
    matcher_success = all(r['success'] for r in matcher_results.values())
    print(f"  Status: {'✅ ALL PASSED' if matcher_success else '❌ SOME FAILED'}")
    if not matcher_success:
        failed = [bs for bs, r in matcher_results.items() if not r['success']]
        print(f"  Failed batch sizes: {failed}")
    
    print("\n🔍 CRITERION SCALING:")
    criterion_success = all(r['success'] for r in criterion_results.values())
    print(f"  Status: {'✅ ALL PASSED' if criterion_success else '❌ SOME FAILED'}")
    if not criterion_success:
        failed = [bs for bs, r in criterion_results.items() if not r['success']]
        print(f"  Failed batch sizes: {failed}")
    
    print("\n🔍 BATCH LOOP ANALYSIS:")
    if loop_findings:
        print(f"  Status: ⚠️  {len(loop_findings)} FILES WITH EXPLICIT LOOPS")
        for file_path in loop_findings.keys():
            print(f"    • {os.path.basename(file_path)}")
    else:
        print(f"  Status: ✅ NO EXPLICIT BATCH LOOPS FOUND")
    
    # Critical Issues
    critical_issues = []
    if not matcher_success:
        critical_issues.append("Hungarian Matcher fails at higher batch sizes")
    if not criterion_success:
        critical_issues.append("Criterion fails at higher batch sizes") 
    if loop_findings:
        critical_issues.append("Explicit batch loops may limit scalability")
    
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES IDENTIFIED:")
        for issue in critical_issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ NO CRITICAL BATCH SIZE ISSUES IDENTIFIED!")
    
    return {
        'matcher': matcher_results,
        'criterion': criterion_results,
        'loops': loop_findings,
        'critical_issues': critical_issues
    }


if __name__ == '__main__':
    results = main()
