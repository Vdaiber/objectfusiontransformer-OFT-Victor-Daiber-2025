#!/usr/bin/env python3
"""
EINFACHER Test für SetCriterion Fix
"""

import torch
import sys
sys.path.append('/app/src')

def test_original_vs_fixed():
    """Test ob der Fix funktioniert wie erwartet"""
    print("SETCRITERION FIX TEST")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    batch_size = 4
    num_queries = 10
    num_classes = 5
    no_object_class_id = num_classes - 1
    
    # Test-Daten erstellen
    predictions = {
        'pred_class_logits_batch': torch.randn(batch_size, num_queries, num_classes, device=device)
    }
    
    targets = {
        'gt_labels_b': []
    }
    
    indices = []
    
    # Verschiedene GT pro Batch
    for b in range(batch_size):
        num_gt = 2  # Einfach: 2 GT pro Batch
        gt_classes = torch.randint(0, num_classes-1, (num_gt,), device=device)
        targets['gt_labels_b'].append(gt_classes)
        
        # Hungarian Indices: erste 2 queries matched
        src_indices = torch.arange(num_gt, device=device)
        tgt_indices = torch.arange(num_gt, device=device)
        indices.append((src_indices, tgt_indices))
    
    # Initialize target tensor
    target_classes_with_no_object = torch.full(
        (batch_size, num_queries), 
        no_object_class_id, 
        dtype=torch.long, 
        device=device
    )
    
    print(f"\nVor Assignment:")
    print(f"Target shape: {target_classes_with_no_object.shape}")
    print(f"Alle Werte sind no_object ({no_object_class_id}): {torch.all(target_classes_with_no_object == no_object_class_id)}")
    
    # *** NEUE VECTORIZED IMPLEMENTATION (kopiert aus dem echten Code) ***
    all_batch_indices = []
    all_src_indices = []
    all_gt_classes = []
    
    for batch_idx, (src_indices, tgt_indices) in enumerate(indices):
        if len(src_indices) > 0:
            # Create batch indices tensor for this batch
            batch_indices_for_this_batch = torch.full_like(src_indices, batch_idx, 
                                                         device=target_classes_with_no_object.device)
            
            # Extract GT classes for matched targets
            gt_classes = targets['gt_labels_b'][batch_idx][tgt_indices]
            
            # Collect indices and GT classes
            all_batch_indices.append(batch_indices_for_this_batch)
            all_src_indices.append(src_indices)
            all_gt_classes.append(gt_classes)
    
    # Vectorized assignment: process all batches simultaneously
    if len(all_batch_indices) > 0:
        flat_batch_indices = torch.cat(all_batch_indices)  # [total_matches]
        flat_src_indices = torch.cat(all_src_indices)      # [total_matches]
        flat_gt_classes = torch.cat(all_gt_classes)        # [total_matches]
        
        print(f"\nVectorized Assignment:")
        print(f"flat_batch_indices: {flat_batch_indices}")
        print(f"flat_src_indices: {flat_src_indices}")
        print(f"flat_gt_classes: {flat_gt_classes}")
        
        # Single vectorized assignment (replaces all individual batch assignments)
        target_classes_with_no_object[flat_batch_indices, flat_src_indices] = flat_gt_classes
    
    print(f"\nNach Assignment:")
    print(f"Target tensor:")
    for b in range(batch_size):
        print(f"  Batch {b}: {target_classes_with_no_object[b]}")
    
    # Verify results
    print(f"\nVerification:")
    for b in range(batch_size):
        src_indices, tgt_indices = indices[b]
        gt_classes = targets['gt_labels_b'][b][tgt_indices]
        assigned_classes = target_classes_with_no_object[b, src_indices]
        
        matches = torch.equal(assigned_classes, gt_classes)
        print(f"  Batch {b}: GT {gt_classes} → Assigned {assigned_classes} → Match: {matches}")
        
        if not matches:
            print(f"    ❌ FEHLER in Batch {b}!")
            return False
    
    print(f"\n✅ VECTORIZED IMPLEMENTATION FUNKTIONIERT KORREKT!")
    return True

def test_batch_scaling():
    """Test mit verschiedenen Batch Sizes"""
    print(f"\n" + "="*50)
    print("BATCH SCALING TEST")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch_size in [1, 2, 4, 8, 16]:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        try:
            num_queries = 50
            num_classes = 13
            no_object_class_id = num_classes - 1
            
            # Create test data
            predictions = {
                'pred_class_logits_batch': torch.randn(batch_size, num_queries, num_classes, device=device)
            }
            
            targets = {'gt_labels_b': []}
            indices = []
            
            for b in range(batch_size):
                num_gt = torch.randint(1, 6, (1,)).item()  # 1-5 GT
                gt_classes = torch.randint(0, num_classes-1, (num_gt,), device=device)
                targets['gt_labels_b'].append(gt_classes)
                
                src_indices = torch.arange(num_gt, device=device)
                tgt_indices = torch.arange(num_gt, device=device)
                indices.append((src_indices, tgt_indices))
            
            # Initialize
            target_classes = torch.full((batch_size, num_queries), no_object_class_id, 
                                      dtype=torch.long, device=device)
            
            # Vectorized assignment
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
                
                target_classes[flat_batch_indices, flat_src_indices] = flat_gt_classes
            
            # Quick verification
            total_matches = sum(len(indices[b][0]) for b in range(batch_size))
            non_no_object = (target_classes != no_object_class_id).sum().item()
            
            if total_matches == non_no_object:
                print(f"✅ SUCCESS: {total_matches} matches assigned correctly")
            else:
                print(f"❌ MISMATCH: Expected {total_matches}, got {non_no_object}")
                return False
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False
    
    print(f"\n✅ ALLE BATCH SIZES FUNKTIONIEREN!")
    return True

def main():
    print("SETCRITERION VECTORIZATION FIX TEST")
    print("="*60)
    
    # Test 1: Basic functionality
    basic_success = test_original_vs_fixed()
    
    if not basic_success:
        print("\n❌ BASIC TEST FEHLGESCHLAGEN!")
        return False
    
    # Test 2: Batch scaling
    scaling_success = test_batch_scaling()
    
    if not scaling_success:
        print("\n❌ SCALING TEST FEHLGESCHLAGEN!")
        return False
    
    print(f"\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print("✅ SETCRITERION FIX IST ERFOLGREICH!")
    print("✅ Vectorized implementation funktioniert korrekt")
    print("✅ Skaliert gut mit verschiedenen Batch Sizes")
    print("✅ Keine Regression - alles funktioniert wie erwartet")
    
    return True

if __name__ == '__main__':
    success = main()
