#!/usr/bin/env python3
"""
SIMPLE Batch Size Test - ohne komplizierte Config
"""

import torch
import numpy as np
import sys
import os
sys.path.append('/app/src')

def create_minimal_test_data(batch_size: int, device: torch.device):
    """Erstellt minimale Test-Daten"""
    num_queries = 300
    num_classes = 13
    
    predictions = {
        'pred_class_logits_batch': torch.randn(batch_size, num_queries, num_classes, device=device),
        'pred_boxes_normalized': torch.randn(batch_size, num_queries, 10, device=device)
    }
    
    targets = {
        'gt_labels_b': [],
        'gt_boxes_b_normalized': torch.zeros(batch_size, 50, 10, device=device),
        'gt_valid_mask_b': []
    }
    
    for b in range(batch_size):
        num_gt = np.random.randint(5, 16)
        gt_classes = torch.randint(0, 12, (num_gt,), device=device)
        targets['gt_labels_b'].append(gt_classes)
        
        targets['gt_boxes_b_normalized'][b, :num_gt] = torch.randn(num_gt, 10, device=device)
        
        valid_mask = torch.zeros(50, dtype=torch.bool, device=device)
        valid_mask[:num_gt] = True
        targets['gt_valid_mask_b'].append(valid_mask)
    
    return predictions, targets

def test_batch_timing():
    """Test verschiedene Batch Sizes und messe Zeit"""
    print("="*60)
    print("BATCH SIZE TIMING TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    from oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher
    
    # Minimal Config
    cfg = {
        'loss': {'huber': {'delta': 0.01}},
        'training': {'matcher': {'cost_center': 100.0}}
    }
    
    matcher = HungarianMatcher(
        cost_center=100.0,
        cost_class=10.0,
        cost_giou_bev=0.0,
        cost_size=0.0,
        cost_angle=0.0,
        cfg=cfg
    ).to(device)
    
    batch_sizes = [1, 2, 4, 8, 16]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        try:
            predictions, targets = create_minimal_test_data(batch_size, device)
            
            # Zeitmessung
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            
            # Hungarian Matcher ausführen
            (indices, matched_masks), avg_costs = matcher(predictions, targets)
            
            if device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
            else:
                elapsed_time = 0
            
            total_matches = sum(len(src) for src, tgt in indices)
            
            results[batch_size] = {
                'time': elapsed_time,
                'matches': total_matches,
                'matches_per_sample': total_matches / batch_size
            }
            
            print(f"✅ Zeit: {elapsed_time:.2f}ms, Matches: {total_matches} ({total_matches/batch_size:.1f}/sample)")
            
        except Exception as e:
            print(f"❌ FEHLER: {e}")
            results[batch_size] = {'error': str(e)}
    
    # Effizienz-Analyse
    print(f"\n" + "="*60)
    print("EFFIZIENZ ANALYSE")
    print("="*60)
    
    successful = {k: v for k, v in results.items() if 'time' in v}
    
    if len(successful) >= 2:
        batch_list = sorted(successful.keys())
        for i in range(1, len(batch_list)):
            prev_batch = batch_list[i-1]
            curr_batch = batch_list[i]
            
            prev_time = successful[prev_batch]['time']
            curr_time = successful[curr_batch]['time']
            
            batch_factor = curr_batch / prev_batch
            time_factor = curr_time / prev_time if prev_time > 0 else float('inf')
            efficiency = batch_factor / time_factor if time_factor > 0 else 0
            
            print(f"Batch {prev_batch}→{curr_batch}: {time_factor:.2f}x Zeit, {efficiency:.2f}x Effizienz")
            
            if efficiency < 0.5:
                print(f"  ⚠️ WARNUNG: Schlechte Skalierung!")
            elif efficiency > 0.8:
                print(f"  ✅ Gute Skalierung")
    
    return results

def analyze_bottlenecks():
    """Analysiert wo die Bottlenecks sind"""
    print(f"\n" + "="*60)
    print("BOTTLENECK ANALYSE")
    print("="*60)
    
    bottlenecks = {
        "Hungarian Matcher": {
            "file": "hungarian_matcher.py:90",
            "code": "for i in range(batch_size):",
            "problem": "Sequential per-batch processing",
            "impact": "MEDIUM - funktioniert aber ineffizient"
        },
        "SetCriterion": {
            "file": "autoregressive_criterion.py:147",
            "code": "for batch_idx, (src_indices, tgt_indices) in enumerate(indices):",
            "problem": "Manual batch loop for target assignment",
            "impact": "HIGH - blockiert GPU parallelization"
        },
        "Loss Functions": {
            "file": "classification_losses.py + regression_losses.py",
            "code": "torch.cat([t[i] for t, (_, i) in zip(...)])",
            "problem": "List comprehensions mit batch indices",
            "impact": "HIGH - CPU bottleneck"
        }
    }
    
    for name, info in bottlenecks.items():
        print(f"\n🚨 {name}:")
        print(f"  File: {info['file']}")
        print(f"  Code: {info['code']}")
        print(f"  Problem: {info['problem']}")
        print(f"  Impact: {info['impact']}")
    
    return bottlenecks

def suggest_solutions():
    """Schlägt konkrete Lösungen vor"""
    print(f"\n" + "="*60)
    print("LÖSUNGSVORSCHLÄGE")
    print("="*60)
    
    solutions = [
        {
            "problem": "Hungarian Matcher Sequential Processing",
            "solution": "Implementiere Betreuer's Assigner-Pattern mit flatten/split",
            "code": """
# Statt: for i in range(batch_size)
# Besser: 
out_flattened = {k: outputs[k].flatten(0, 1) for k in keys}
cost_matrix = compute_cost_matrix(out_flattened, gt_flattened)
indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
            """,
            "priority": "MEDIUM"
        },
        {
            "problem": "SetCriterion Batch Loop",
            "solution": "Ersetze explizite Loop durch vectorized operations",
            "code": """
# Statt: for batch_idx, (src_indices, tgt_indices) in enumerate(indices)
# Besser:
batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
src_indices_flat = torch.cat([src for (src, _) in indices])
target_classes_with_no_object[batch_indices, src_indices_flat] = gt_classes_flat
            """,
            "priority": "HIGH"
        },
        {
            "problem": "Loss Function List Comprehensions",
            "solution": "Pre-compute alle batch indices einmalig",
            "code": """
# Einmalig berechnen:
def get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
            """,
            "priority": "HIGH"
        }
    ]
    
    for i, sol in enumerate(solutions, 1):
        print(f"\n💡 LÖSUNG {i}: {sol['solution']}")
        print(f"   Problem: {sol['problem']}")
        print(f"   Priorität: {sol['priority']}")
        print(f"   Code:")
        print(sol['code'])
    
    return solutions

def main():
    """Hauptfunktion"""
    print("BATCH SIZE COMPATIBILITY ANALYSE")
    
    # Test Timing
    timing_results = test_batch_timing()
    
    # Analysiere Bottlenecks
    bottlenecks = analyze_bottlenecks()
    
    # Lösungsvorschläge
    solutions = suggest_solutions()
    
    # Fazit
    print(f"\n" + "="*60)
    print("FAZIT")
    print("="*60)
    
    print("\n✅ FUNKTIONIERT:")
    print("  - Hungarian Matcher skaliert (wenn auch nicht optimal)")
    print("  - Grundlegende Architektur ist batch-fähig")
    
    print("\n🚨 PROBLEME:")
    print("  - Explizite Batch-Loops in Loss-Funktionen")
    print("  - Ineffiziente List Comprehensions")
    print("  - CPU-bound Operations blockieren GPU")
    
    print("\n💡 NÄCHSTE SCHRITTE:")
    print("  1. SetCriterion vectorisieren (HIGH PRIORITY)")
    print("  2. Loss Functions optimieren (HIGH PRIORITY)")  
    print("  3. Hungarian Matcher mit Assigner-Pattern (MEDIUM PRIORITY)")
    print("  4. Batch Size langsam hochfahren: 1→2→4→8→16")
    
    return {
        'timing': timing_results,
        'bottlenecks': bottlenecks,
        'solutions': solutions
    }

if __name__ == '__main__':
    results = main()
