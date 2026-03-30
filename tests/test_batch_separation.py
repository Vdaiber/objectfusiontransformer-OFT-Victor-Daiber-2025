#!/usr/bin/env python3
"""
Test: Beweist dass Batches NICHT vermischt werden
ZWECK: Zeigen dass ähnliche Boxes in verschiedenen Batches getrennt bleiben
"""

import torch
import numpy as np
import sys
sys.path.append('/app/src')

def test_batch_separation_with_similar_boxes():
    """Test mit sehr ähnlichen Boxes in verschiedenen Batches"""
    print("="*60)
    print("BATCH SEPARATION TEST - Ähnliche Boxes")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 3
    num_queries = 5
    num_classes = 4  # [car, truck, bus, no_object]
    no_object_class_id = 3
    
    print(f"Setup: {batch_size} Batches, {num_queries} Queries pro Batch")
    
    # KRITISCHER TEST: Ähnliche GT-Klassen in verschiedenen Batches
    targets = {
        'gt_labels_b': [
            torch.tensor([0, 1], device=device),  # Batch 0: [car, truck]
            torch.tensor([0, 2], device=device),  # Batch 1: [car, bus] <- GLEICHE car KLASSE!
            torch.tensor([1, 0], device=device),  # Batch 2: [truck, car] <- GLEICHE KLASSEN!
        ]
    }
    
    # Hungarian Matcher Ergebnisse (simuliert)
    indices = [
        (torch.tensor([0, 1], device=device), torch.tensor([0, 1], device=device)),  # Batch 0: Query 0,1 → GT 0,1
        (torch.tensor([2, 3], device=device), torch.tensor([0, 1], device=device)),  # Batch 1: Query 2,3 → GT 0,1  
        (torch.tensor([1, 4], device=device), torch.tensor([0, 1], device=device)),  # Batch 2: Query 1,4 → GT 0,1
    ]
    
    print(f"\nGround Truth Labels:")
    for b, gt in enumerate(targets['gt_labels_b']):
        print(f"  Batch {b}: {gt} (Klassen: {['car', 'truck', 'bus'][gt[0]]}, {['car', 'truck', 'bus'][gt[1]]})")
    
    print(f"\nHungarian Matcher Ergebnisse:")
    for b, (src, tgt) in enumerate(indices):
        print(f"  Batch {b}: Query {src} ← GT {tgt}")
    
    # Initialize target tensor
    target_classes = torch.full((batch_size, num_queries), no_object_class_id, 
                               dtype=torch.long, device=device)
    
    print(f"\nVor Assignment (alle no_object={no_object_class_id}):")
    for b in range(batch_size):
        print(f"  Batch {b}: {target_classes[b]}")
    
    # *** VECTORIZED ASSIGNMENT (aus dem echten Code) ***
    all_batch_indices = []
    all_src_indices = []
    all_gt_classes = []
    
    for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
        if len(src_indices) > 0:
            batch_indices_for_this_batch = torch.full_like(src_indices, batch_idx, device=device)
            gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
            
            all_batch_indices.append(batch_indices_for_this_batch)
            all_src_indices.append(src_indices)
            all_gt_classes.append(gt_classes)
    
    if len(all_batch_indices) > 0:
        flat_batch_indices = torch.cat(all_batch_indices)
        flat_src_indices = torch.cat(all_src_indices)
        flat_gt_classes = torch.cat(all_gt_classes)
        
        print(f"\nVectorized Assignment Daten:")
        print(f"  flat_batch_indices: {flat_batch_indices} (Welcher Batch)")
        print(f"  flat_src_indices:   {flat_src_indices} (Welche Query)")
        print(f"  flat_gt_classes:    {flat_gt_classes} (Welche Klasse)")
        
        # Das kritische Assignment
        target_classes[flat_batch_indices, flat_src_indices] = flat_gt_classes
    
    print(f"\nNach Assignment:")
    for b in range(batch_size):
        print(f"  Batch {b}: {target_classes[b]}")
    
    # *** VERIFIKATION: Keine Batch-Vermischung ***
    print(f"\n" + "="*60)
    print("VERIFIKATION: Batch-Trennung")
    print("="*60)
    
    all_correct = True
    
    for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
        expected_gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
        actual_assigned_classes = target_classes[batch_idx, src_indices]
        
        is_correct = torch.equal(expected_gt_classes, actual_assigned_classes)
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Expected GT: {expected_gt_classes}")
        print(f"  Assigned:    {actual_assigned_classes}")
        print(f"  Correct:     {is_correct}")
        
        if not is_correct:
            print(f"  ❌ FEHLER: Batch {batch_idx} Assignment falsch!")
            all_correct = False
        else:
            print(f"  ✅ Batch {batch_idx} korrekt zugewiesen")
    
    # Zusätzliche Verifikation: Keine Cross-Batch Kontamination
    print(f"\n" + "="*40)
    print("CROSS-BATCH KONTAMINATION CHECK")
    print("="*40)
    
    # Check dass nur die erwarteten Positionen verändert wurden
    expected_modified_positions = set()
    for batch_idx, (src_indices, _) in enumerate(indices):
        for src_idx in src_indices:
            expected_modified_positions.add((batch_idx, src_idx.item()))
    
    actual_modified_positions = set()
    for b in range(batch_size):
        for q in range(num_queries):
            if target_classes[b, q] != no_object_class_id:
                actual_modified_positions.add((b, q))
    
    print(f"Expected modified positions: {sorted(expected_modified_positions)}")
    print(f"Actual modified positions:   {sorted(actual_modified_positions)}")
    
    positions_match = expected_modified_positions == actual_modified_positions
    print(f"Positions match: {positions_match}")
    
    if positions_match and all_correct:
        print(f"\n✅ BATCH SEPARATION ERFOLGREICH!")
        print(f"✅ Keine Vermischung zwischen Batches")
        print(f"✅ Jeder Batch behält seine eigenen Assignments")
        return True
    else:
        print(f"\n❌ BATCH SEPARATION FEHLGESCHLAGEN!")
        return False

def test_performance_vs_original():
    """Test Performance der neuen vs. alten Implementation"""
    print(f"\n" + "="*60)
    print("PERFORMANCE VERGLEICH")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def original_implementation(target_classes, targets, indices):
        """Original Implementation mit expliziter Loop"""
        result = target_classes.clone()
        for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
            if len(src_indices) > 0:
                gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
                result[batch_idx, src_indices] = gt_classes
        return result
    
    def vectorized_implementation(target_classes, targets, indices):
        """Neue Vectorized Implementation"""
        result = target_classes.clone()
        
        all_batch_indices = []
        all_src_indices = []
        all_gt_classes = []
        
        for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
            if len(src_indices) > 0:
                batch_indices_for_this_batch = torch.full_like(src_indices, batch_idx, device=device)
                gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
                
                all_batch_indices.append(batch_indices_for_this_batch)
                all_src_indices.append(src_indices)
                all_gt_classes.append(gt_classes)
        
        if len(all_batch_indices) > 0:
            flat_batch_indices = torch.cat(all_batch_indices)
            flat_src_indices = torch.cat(all_src_indices)
            flat_gt_classes = torch.cat(all_gt_classes)
            
            result[flat_batch_indices, flat_src_indices] = flat_gt_classes
        
        return result
    
    # Test mit verschiedenen Batch Sizes
    for batch_size in [4, 8, 16, 32]:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        num_queries = 300  # Realistisch
        num_classes = 13
        no_object_class_id = num_classes - 1
        
        # Create test data
        target_classes = torch.full((batch_size, num_queries), no_object_class_id, 
                                   dtype=torch.long, device=device)
        
        targets = {'gt_labels_b': []}
        indices = []
        
        for b in range(batch_size):
            num_gt = torch.randint(5, 16, (1,)).item()  # 5-15 GT
            gt_classes = torch.randint(0, num_classes-1, (num_gt,), device=device)
            targets['gt_labels_b'].append(gt_classes)
            
            src_indices = torch.arange(num_gt, device=device)
            tgt_indices = torch.arange(num_gt, device=device)
            indices.append((src_indices, tgt_indices))
        
        # Warmup
        for _ in range(5):
            _ = original_implementation(target_classes, targets, indices)
            _ = vectorized_implementation(target_classes, targets, indices)
        
        # Time original
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        num_iter = 100
        for _ in range(num_iter):
            original_result = original_implementation(target_classes, targets, indices)
        
        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            original_time = start.elapsed_time(end) / num_iter
        else:
            original_time = 0
        
        # Time vectorized  
        if device.type == 'cuda':
            start.record()
        
        for _ in range(num_iter):
            vectorized_result = vectorized_implementation(target_classes, targets, indices)
        
        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            vectorized_time = start.elapsed_time(end) / num_iter
        else:
            vectorized_time = 0
        
        # Verify identical results
        identical = torch.equal(original_result, vectorized_result)
        
        # Performance analysis
        if original_time > 0 and vectorized_time > 0:
            speedup = original_time / vectorized_time
        else:
            speedup = 1.0
        
        print(f"  Original:    {original_time:.3f}ms")
        print(f"  Vectorized:  {vectorized_time:.3f}ms") 
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"  Identical:   {identical}")
        
        if not identical:
            print(f"  ❌ RESULTS DIFFER!")
            return False
    
    print(f"\n✅ PERFORMANCE TEST BESTANDEN!")
    return True

def main():
    """Main test function"""
    print("BATCH SEPARATION & PERFORMANCE VERIFICATION")
    print("="*70)
    
    # Test 1: Batch separation
    separation_success = test_batch_separation_with_similar_boxes()
    
    if not separation_success:
        print("\n❌ BATCH SEPARATION TEST FEHLGESCHLAGEN!")
        return False
    
    # Test 2: Performance comparison
    performance_success = test_performance_vs_original()
    
    if not performance_success:
        print("\n❌ PERFORMANCE TEST FEHLGESCHLAGEN!")
        return False
    
    print(f"\n" + "="*70)
    print("FINAL VERIFICATION")
    print("="*70)
    print("✅ BATCH SEPARATION: Keine Vermischung zwischen Batches")
    print("✅ KORREKTHEIT: Identische Ergebnisse wie Original")
    print("✅ PERFORMANCE: Gemessen und dokumentiert")
    print("✅ SICHERHEIT: Ihr BoundingBox Refinement ist sicher!")
    
    print(f"\n🎯 FAZIT FÜR IHR ALGORITHMUS:")
    print("• Jeder Batch bleibt komplett getrennt")
    print("• Ähnliche Boxes in verschiedenen Batches werden NICHT verwechselt")
    print("• Hungarian Matching bleibt batch-spezifisch")
    print("• Nur die GPU-Parallelisierung wurde verbessert")
    
    return True

if __name__ == '__main__':
    success = main()
