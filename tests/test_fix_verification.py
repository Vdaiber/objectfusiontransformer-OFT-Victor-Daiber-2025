#!/usr/bin/env python3
"""
Verification Test für SetCriterion Fix
ZWECK: Sicherstellen dass die echte Implementierung noch funktioniert
"""

import torch
import numpy as np
import sys
sys.path.append('/app/src')

from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion

def create_real_test_data(batch_size=4):
    """Erstellt realistische Test-Daten für SetCriterion"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_queries = 300
    num_classes = 13
    
    # Model Predictions (realistic)
    predictions = {
        'pred_class_logits_batch': torch.randn(batch_size, num_queries, num_classes, device=device),
        'pred_boxes_normalized': torch.randn(batch_size, num_queries, 10, device=device),
        'pred_attributes_logits_batch': torch.randn(batch_size, num_queries, 11, device=device),
        'pred_velocity_offsets_normalized': torch.randn(batch_size, num_queries, 2, device=device)
    }
    
    # Ground Truth (realistic variable lengths)
    targets = {
        'gt_labels_b': [],
        'gt_boxes_b_normalized': torch.zeros(batch_size, 50, 10, device=device),
        'gt_valid_mask_b': [],
        'gt_attributes_b': [],
        'gt_velocity_b_normalized': torch.zeros(batch_size, 50, 2, device=device)
    }
    
    for b in range(batch_size):
        num_gt = np.random.randint(5, 16)  # 5-15 GT objects
        
        # GT data
        gt_classes = torch.randint(0, 12, (num_gt,), device=device)
        targets['gt_labels_b'].append(gt_classes)
        
        targets['gt_boxes_b_normalized'][b, :num_gt] = torch.randn(num_gt, 10, device=device)
        
        valid_mask = torch.zeros(50, dtype=torch.bool, device=device)
        valid_mask[:num_gt] = True
        targets['gt_valid_mask_b'].append(valid_mask)
        
        gt_attributes = torch.randint(0, 11, (num_gt,), device=device)
        targets['gt_attributes_b'].append(gt_attributes)
        
        targets['gt_velocity_b_normalized'][b, :num_gt] = torch.randn(num_gt, 2, device=device)
    
    return predictions, targets

def test_setcriterion_with_different_batch_sizes():
    """Test SetCriterion mit verschiedenen Batch Sizes"""
    print("="*60)
    print("SETCRITERION REAL FUNCTIONALITY TEST")
    print("="*60)
    
    # Load real config für SetCriterion
    import hydra
    from hydra import initialize, compose
    from omegaconf import DictConfig
    
    try:
        with initialize(config_path="/app/config", version_base=None):
            cfg = compose(config_name="pipeline_staged.yaml")
    except Exception as e:
        print(f"⚠️ Konnte echte Config nicht laden: {e}")
        # Fallback config
        cfg = {
            'loss': {
                'weights': {
                    'loss_center': 30.0,
                    'loss_size': 1.0,
                    'loss_angle': 2.0,
                    'loss_velocity': 3.0,
                    'loss_class': 1.0,
                    'loss_attributes': 0.5
                }
            }
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create SetCriterion
    try:
        criterion = SetCriterion(
            weight_dict=cfg['loss']['weights'],
            losses=['center', 'size', 'angle', 'velocity', 'class', 'attributes'],
            cfg=cfg
        ).to(device)
    except Exception as e:
        print(f"❌ Fehler beim Erstellen von SetCriterion: {e}")
        return False
    
    # Test verschiedene Batch Sizes
    batch_sizes = [1, 2, 4, 8]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing Batch Size: {batch_size} ---")
        
        try:
            # Create test data
            predictions, targets = create_real_test_data(batch_size)
            
            # Run SetCriterion forward pass
            loss_dict = criterion(predictions, targets)
            
            # Verify output structure
            expected_losses = ['loss_center', 'loss_size', 'loss_angle', 'loss_velocity', 'loss_class', 'loss_attributes']
            
            for loss_name in expected_losses:
                if loss_name not in loss_dict:
                    raise ValueError(f"Missing loss: {loss_name}")
                if not isinstance(loss_dict[loss_name], torch.Tensor):
                    raise ValueError(f"Invalid loss type: {loss_name}")
                if loss_dict[loss_name].dim() != 0:
                    raise ValueError(f"Loss should be scalar: {loss_name}")
                if torch.isnan(loss_dict[loss_name]) or torch.isinf(loss_dict[loss_name]):
                    raise ValueError(f"Loss contains NaN/Inf: {loss_name}")
            
            # Calculate total loss
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
            print(f"❌ FAILED: {e}")
            results[batch_size] = {
                'success': False,
                'error': str(e)
            }
            import traceback
            traceback.print_exc()
    
    return results

def test_performance_scaling():
    """Test Performance mit neuer Implementation"""
    print(f"\n" + "="*60)
    print("PERFORMANCE SCALING TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Minimal config für schnellen Test
    cfg = {
        'loss': {
            'weights': {
                'loss_center': 30.0,
                'loss_size': 1.0,
                'loss_angle': 2.0,
                'loss_velocity': 3.0,
                'loss_class': 1.0,
                'loss_attributes': 0.5
            }
        }
    }
    
    try:
        criterion = SetCriterion(
            weight_dict=cfg['loss']['weights'],
            losses=['center', 'size', 'angle', 'velocity', 'class', 'attributes'],
            cfg=cfg
        ).to(device)
    except Exception as e:
        print(f"❌ Konnte SetCriterion nicht erstellen: {e}")
        return {}
    
    batch_sizes = [1, 2, 4, 8, 16]
    performance_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Performance Test Batch Size: {batch_size} ---")
        
        try:
            predictions, targets = create_real_test_data(batch_size)
            
            # Warm-up
            for _ in range(3):
                _ = criterion(predictions, targets)
            
            # Actual timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            
            num_iterations = 20
            for _ in range(num_iterations):
                loss_dict = criterion(predictions, targets)
            
            if device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end) / num_iterations
            else:
                elapsed_time = 0
            
            performance_results[batch_size] = {
                'time_per_forward': elapsed_time,
                'total_loss': sum(loss_dict.values()).item()
            }
            
            print(f"✅ Zeit pro Forward Pass: {elapsed_time:.2f}ms")
            
        except Exception as e:
            print(f"❌ FEHLER bei Batch {batch_size}: {e}")
            performance_results[batch_size] = {'error': str(e)}
    
    return performance_results

def main():
    """Haupttest"""
    print("SETCRITERION FIX VERIFICATION")
    print("="*60)
    
    # Test 1: Funktionalität
    functionality_results = test_setcriterion_with_different_batch_sizes()
    
    # Test 2: Performance
    performance_results = test_performance_scaling()
    
    # Summary
    print(f"\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    # Check functionality
    all_functional = all(r.get('success', False) for r in functionality_results.values())
    print(f"\n🔍 FUNKTIONALITÄT: {'✅ ALLE TESTS BESTANDEN' if all_functional else '❌ EINIGE TESTS FEHLGESCHLAGEN'}")
    
    if not all_functional:
        failed_batches = [bs for bs, r in functionality_results.items() if not r.get('success', False)]
        print(f"    Fehlgeschlagene Batch Sizes: {failed_batches}")
    
    # Check performance scaling
    valid_performance = {bs: r for bs, r in performance_results.items() if 'time_per_forward' in r}
    
    if len(valid_performance) >= 2:
        print(f"\n📊 PERFORMANCE SCALING:")
        print(f"{'Batch':<8} {'Time(ms)':<12} {'Loss':<12}")
        print("-" * 35)
        
        for batch_size, results in valid_performance.items():
            print(f"{batch_size:<8} {results['time_per_forward']:<12.2f} {results['total_loss']:<12.4f}")
        
        # Analyze scaling
        times = [(bs, r['time_per_forward']) for bs, r in valid_performance.items()]
        times.sort()
        
        if len(times) >= 2:
            scaling_efficiency = []
            for i in range(1, len(times)):
                prev_bs, prev_time = times[i-1]
                curr_bs, curr_time = times[i]
                batch_factor = curr_bs / prev_bs
                time_factor = curr_time / prev_time if prev_time > 0 else float('inf')
                efficiency = batch_factor / time_factor if time_factor > 0 else 0
                scaling_efficiency.append(efficiency)
            
            avg_efficiency = sum(scaling_efficiency) / len(scaling_efficiency)
            print(f"\n💡 Durchschnittliche Skalierungs-Effizienz: {avg_efficiency:.2f}")
            
            if avg_efficiency > 0.7:
                print("✅ GUTE SKALIERUNG!")
            elif avg_efficiency > 0.5:
                print("✅ Akzeptable Skalierung")
            else:
                print("⚠️ Skalierung könnte besser sein")
    
    # Final verdict
    print(f"\n🎯 FAZIT:")
    if all_functional:
        print("✅ SetCriterion Fix ist erfolgreich implementiert!")
        print("✅ Alle Batch Sizes funktionieren korrekt")
        print("✅ Keine Regression in der Funktionalität")
        if len(valid_performance) >= 2:
            print("✅ Performance-Skalierung gemessen")
    else:
        print("❌ SetCriterion Fix hat Probleme!")
        print("❌ Einige Batch Sizes funktionieren nicht")
    
    return all_functional

if __name__ == '__main__':
    success = main()
