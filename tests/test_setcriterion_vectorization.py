#!/usr/bin/env python3
"""
Test für SetCriterion Vectorization
ZWECK: Sicherstellen dass die Vectorisierung funktioniert UND identische Ergebnisse liefert
"""

import torch
import numpy as np
import sys
import os
sys.path.append('/app/src')

def create_test_data(batch_size=4, num_queries=10, num_classes=5):
    """Erstellt Test-Daten für SetCriterion"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Predictions (Model Output)
    predictions = {
        'pred_class_logits_batch': torch.randn(batch_size, num_queries, num_classes, device=device)
    }
    
    # Targets mit variable GT pro Batch
    targets = {
        'gt_labels_b': []
    }
    
    # Simuliere Hungarian Matcher Indices
    indices = []
    
    for b in range(batch_size):
        # Variable Anzahl GT pro Batch (1-3)
        num_gt = np.random.randint(1, 4)
        
        # GT Classes (ohne no_object)
        gt_classes = torch.randint(0, num_classes-1, (num_gt,), device=device)
        targets['gt_labels_b'].append(gt_classes)
        
        # Hungarian Indices: erste num_gt queries matched
        src_indices = torch.arange(num_gt, device=device)
        tgt_indices = torch.arange(num_gt, device=device)
        indices.append((src_indices, tgt_indices))
    
    return predictions, targets, indices

def original_implementation(predictions, targets, indices):
    """ORIGINALE SetCriterion Implementation (mit Batch-Loop)"""
    batch_size = predictions['pred_class_logits_batch'].shape[0]
    num_queries = predictions['pred_class_logits_batch'].shape[1]
    num_classes = predictions['pred_class_logits_batch'].shape[2]
    no_object_class_id = num_classes - 1  # Letzter Klasse ist no_object
    
    # Initialize all predictions as no_object
    target_classes_with_no_object = torch.full(
        (batch_size, num_queries), 
        no_object_class_id, 
        dtype=torch.long, 
        device=predictions['pred_class_logits_batch'].device
    )
    
    # *** ORIGINAL: EXPLIZITE BATCH-LOOP ***
    for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
        if len(src_indices) > 0:
            gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
            target_classes_with_no_object[batch_idx, src_indices] = gt_classes
    
    return target_classes_with_no_object

def vectorized_implementation(predictions, targets, indices):
    """NEUE Vectorized Implementation (ohne Batch-Loop)"""
    batch_size = predictions['pred_class_logits_batch'].shape[0]
    num_queries = predictions['pred_class_logits_batch'].shape[1]
    num_classes = predictions['pred_class_logits_batch'].shape[2]
    no_object_class_id = num_classes - 1
    
    # Initialize all predictions as no_object (GLEICH)
    target_classes_with_no_object = torch.full(
        (batch_size, num_queries), 
        no_object_class_id, 
        dtype=torch.long, 
        device=predictions['pred_class_logits_batch'].device
    )
    
    # *** NEUE VECTORIZED APPROACH ***
    # Schritt 1: Sammle alle batch_indices, src_indices, tgt_indices
    all_batch_indices = []
    all_src_indices = []
    all_tgt_indices = []
    
    for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
        if len(src_indices) > 0:
            # Erstelle batch_indices für jedes src/tgt pair
            batch_indices_for_this_batch = torch.full_like(src_indices, batch_idx)
            
            all_batch_indices.append(batch_indices_for_this_batch)
            all_src_indices.append(src_indices)
            all_tgt_indices.append(tgt_indices)
    
    if len(all_batch_indices) > 0:
        # Schritt 2: Concatenate alle indices
        flat_batch_indices = torch.cat(all_batch_indices)  # [total_matches]
        flat_src_indices = torch.cat(all_src_indices)      # [total_matches]
        flat_tgt_indices = torch.cat(all_tgt_indices)      # [total_matches]
        
        # Schritt 3: Sammle alle GT classes
        all_gt_classes = []
        for batch_idx, tgt_indices in enumerate([indices[i][1] for i in range(len(indices))]):
            if len(tgt_indices) > 0:
                gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
                all_gt_classes.append(gt_classes)
        
        if len(all_gt_classes) > 0:
            flat_gt_classes = torch.cat(all_gt_classes)  # [total_matches]
            
            # Schritt 4: VECTORIZED assignment (KEIN LOOP!)
            target_classes_with_no_object[flat_batch_indices, flat_src_indices] = flat_gt_classes
    
    return target_classes_with_no_object

def test_vectorization_correctness():
    """Test dass beide Implementationen identische Ergebnisse liefern"""
    print("="*60)
    print("TEST: Vectorization Correctness")
    print("="*60)
    
    # Test mit verschiedenen Batch Sizes
    for batch_size in [1, 2, 4, 8]:
        print(f"\n--- Testing Batch Size: {batch_size} ---")
        
        # Erstelle Test-Daten
        predictions, targets, indices = create_test_data(batch_size)
        
        # Beide Implementationen ausführen
        original_result = original_implementation(predictions, targets, indices)
        vectorized_result = vectorized_implementation(predictions, targets, indices)
        
        # Vergleiche Ergebnisse
        are_equal = torch.equal(original_result, vectorized_result)
        
        if are_equal:
            print(f"✅ PASS: Ergebnisse sind identisch")
        else:
            print(f"❌ FAIL: Ergebnisse unterscheiden sich!")
            print(f"Original shape: {original_result.shape}")
            print(f"Vectorized shape: {vectorized_result.shape}")
            print(f"Difference locations: {(original_result != vectorized_result).nonzero()}")
            return False
    
    print(f"\n✅ ALLE TESTS BESTANDEN: Vectorized implementation ist korrekt!")
    return True

def test_performance_comparison():
    """Test Performance-Unterschied zwischen Original und Vectorized"""
    print(f"\n" + "="*60)
    print("TEST: Performance Comparison")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_sizes = [1, 2, 4, 8, 16]
    num_queries = 300  # Realistisch
    num_classes = 13   # Realistisch
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        # Erstelle größere Test-Daten
        predictions, targets, indices = create_test_data(batch_size, num_queries, num_classes)
        
        # Zeitmessung Original
        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        for _ in range(10):  # 10 Wiederholungen für bessere Messung
            original_result = original_implementation(predictions, targets, indices)
        
        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            original_time = start.elapsed_time(end) / 10  # Durchschnitt
        else:
            original_time = 0
        
        # Zeitmessung Vectorized
        if device.type == 'cuda':
            start.record()
        
        for _ in range(10):
            vectorized_result = vectorized_implementation(predictions, targets, indices)
        
        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            vectorized_time = start.elapsed_time(end) / 10
        else:
            vectorized_time = 0
        
        # Performance Vergleich
        if original_time > 0:
            speedup = original_time / vectorized_time if vectorized_time > 0 else float('inf')
        else:
            speedup = 1.0
        
        results[batch_size] = {
            'original_time': original_time,
            'vectorized_time': vectorized_time,
            'speedup': speedup
        }
        
        print(f"Original: {original_time:.2f}ms")
        print(f"Vectorized: {vectorized_time:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
    
    return results

def main():
    """Haupttest-Funktion"""
    print("SETCRITERION VECTORIZATION TESTS")
    print("="*60)
    
    # Test 1: Korrektheit
    correctness_passed = test_vectorization_correctness()
    
    if not correctness_passed:
        print("\n❌ KORREKTHEIT-TEST FEHLGESCHLAGEN!")
        print("Vectorization kann nicht implementiert werden.")
        return False
    
    # Test 2: Performance
    performance_results = test_performance_comparison()
    
    # Zusammenfassung
    print(f"\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    
    print(f"\n✅ KORREKTHEIT: Vectorized implementation liefert identische Ergebnisse")
    
    print(f"\n📊 PERFORMANCE:")
    print(f"{'Batch':<8} {'Original(ms)':<12} {'Vector(ms)':<12} {'Speedup':<10}")
    print("-" * 50)
    for batch_size, results in performance_results.items():
        print(f"{batch_size:<8} {results['original_time']:<12.2f} {results['vectorized_time']:<12.2f} {results['speedup']:<10.2f}x")
    
    # Analyse
    avg_speedup = sum(r['speedup'] for r in performance_results.values()) / len(performance_results)
    print(f"\n💡 Durchschnittlicher Speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1.2:
        print("✅ SIGNIFIKANTE PERFORMANCE-VERBESSERUNG!")
    elif avg_speedup > 1.0:
        print("✅ Leichte Performance-Verbesserung")
    else:
        print("⚠️ Keine Performance-Verbesserung")
    
    print(f"\n🎯 FAZIT: Vectorization ist sicher zu implementieren!")
    return True

if __name__ == '__main__':
    success = main()
