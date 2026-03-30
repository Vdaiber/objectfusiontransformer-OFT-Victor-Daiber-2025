#!/usr/bin/env python3
"""
COMPREHENSIVE UNIT TEST FOR HUNGARIAN MATCHER
===============================================

Dieser Test analysiert den HungarianMatcher zu 100% und deckt alle möglichen Probleme auf:
1. Cost-Matrix Berechnung
2. Gewichtung der verschiedenen Costs  
3. Bipartite Matching Algorithmus
4. Edge Cases (keine Predictions, keine Targets, etc.)
5. Gradient Flow durch den Matcher

Ziel: Herausfinden warum der Matcher nicht korrekt funktioniert!
"""

import sys
import os
sys.path.insert(0, './src')

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any

# Import the matcher
from src.oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher

def create_test_predictions() -> Dict[str, torch.Tensor]:
    """Erstellt realistische Test-Predictions."""
    batch_size = 2
    num_queries = 50  # Typische Anzahl Queries
    
    return {
        # Classification logits: (B, N, num_classes) - KORRIGIERTER KEY!
        'pred_class_logits_batch': torch.randn(batch_size, num_queries, 12, requires_grad=True),
        
        # Box predictions: (B, N, 8) - [x,y,z,w,l,h,sin,cos] in normalized ego coordinates
        'pred_box_offsets_ego_norm_8d_batch': torch.randn(batch_size, num_queries, 8, requires_grad=True) * 0.1,
        
        # Velocity predictions: (B, N, 2) - [vx, vy] 
        'pred_velocity_offsets_ego_norm_2d_batch': torch.randn(batch_size, num_queries, 2, requires_grad=True) * 0.05,
        
        # Reconstructed boxes (normalized): (B, N, 10) - [x,y,z,w,l,h,sin,cos,vx,vy] - Das erwartet der Matcher!
        'pred_boxes_normalized': torch.randn(batch_size, num_queries, 10, requires_grad=True) * 0.1,
        
        # Anchor boxes für die Predictions: (B, N, 9) - [x,y,z,w,l,h,yaw,vx,vy] physical
        'anchor_boxes_ego_phys_9d_batch': torch.tensor([
            # Batch 0: 3 gute Anchor Boxen
            [[10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0, 0.0],  # Car-like box
             [20.0, -5.0, 0.0, 12.0, 3.0, 3.0, 1.57, 0.0, 0.0], # Truck-like box
             [-5.0, 15.0, 0.0, 2.0, 1.0, 1.8, 0.0, 0.5, -0.5]] + # Pedestrian-like box
             [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * (num_queries - 3),  # Padding
            
            # Batch 1: 2 gute Anchor Boxen
            [[15.0, 10.0, 0.0, 6.0, 3.0, 2.0, 0.5, -1.0, 1.0], # Vehicle
             [-10.0, -8.0, 0.0, 2.5, 1.5, 1.7, 0.0, 0.0, 0.0]] + # Pedestrian  
             [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * (num_queries - 2)   # Padding
        ], dtype=torch.float32)
    }

def create_test_targets() -> Dict[str, torch.Tensor]:
    """Erstellt realistische Ground Truth Targets."""
    return {
        # Ground truth boxes: (B, max_gt, 10) - [x,y,z,w,l,h,sin,cos,vx,vy] normalized ego
        'gt_boxes_b_normalized': torch.tensor([
            # Batch 0: 2 GT objects
            [[0.1, 0.05, 0.0, 0.08, 0.04, 0.03, 0.0, 1.0, 0.02, 0.0],    # Car
             [0.2, -0.05, 0.0, 0.24, 0.06, 0.06, 1.0, 0.0, 0.0, 0.0],     # Truck
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],         # Padding
            
            # Batch 1: 3 GT objects  
            [[0.15, 0.1, 0.0, 0.12, 0.06, 0.04, 0.48, 0.88, -0.02, 0.02], # Vehicle
             [-0.1, -0.08, 0.0, 0.05, 0.03, 0.034, 0.0, 1.0, 0.0, 0.0],   # Pedestrian
             [0.25, 0.15, 0.0, 0.04, 0.02, 0.025, 0.0, 1.0, 0.01, -0.01]] # Bicycle
        ], dtype=torch.float32),
        
        # Ground truth labels: (B, max_gt)
        'gt_labels_b': torch.tensor([
            [2, 7, -1],  # car, truck, padding  
            [5, 1, 4]    # vehicle, pedestrian, bicycle
        ], dtype=torch.long),
        
        # Valid mask: (B, max_gt) - True für valide Targets
        'gt_valid_mask_b': torch.tensor([
            [True, True, False],   # 2 valide Targets in Batch 0
            [True, True, True]     # 3 valide Targets in Batch 1
        ], dtype=torch.bool)
    }

def test_matcher_basic_functionality():
    """Test 1: Grundlegende Matcher-Funktionalität."""
    print("\n" + "="*80)
    print("TEST 1: GRUNDLEGENDE MATCHER-FUNKTIONALITÄT")
    print("="*80)
    
    # Standard Matcher mit ausgewogenen Kosten und minimaler Config
    test_cfg = {
        'loss': {
            'huber': {
                'delta': 0.1
            }
        }
    }
    
    matcher = HungarianMatcher(
        cost_center=1.0,
        cost_class=1.0, 
        cost_size=1.0,
        cost_angle=1.0,
        cost_giou_bev=0.0,  # Deaktiviert für diesen Test
        cfg=test_cfg
    )
    
    predictions = create_test_predictions()
    targets = create_test_targets()
    
    print(f"Input shapes:")
    print(f"  Predictions: {predictions['pred_class_logits_batch'].shape}")
    print(f"  GT Boxes: {targets['gt_boxes_b_normalized'].shape}")
    print(f"  GT Labels: {targets['gt_labels_b'].shape}")
    print(f"  Valid Mask: {targets['gt_valid_mask_b'].shape}")
    
    # Führe Matching durch
    try:
        (indices, matched_masks), cost_dict = matcher(predictions, targets)
        
        print(f"\n✅ Matching erfolgreich!")
        print(f"Anzahl Batches: {len(indices)}")
        
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            print(f"\nBatch {batch_idx}:")
            print(f"  Matched Predictions: {pred_idx.tolist()}")
            print(f"  Matched Targets: {tgt_idx.tolist()}")
            print(f"  Anzahl Matches: {len(pred_idx)}")
            
            # Validiere die Indices
            num_valid_targets = targets['gt_valid_mask_b'][batch_idx].sum().item()
            print(f"  Erwartete Matches (valid targets): {num_valid_targets}")
            
            if len(pred_idx) != num_valid_targets:
                print(f"  ⚠️  WARNING: Match-Anzahl stimmt nicht überein!")
            else:
                print(f"  ✅ Match-Anzahl korrekt!")
        
        # Analysiere Cost Dictionary
        print(f"\nCost Dictionary:")
        for cost_name, cost_tensor in cost_dict.items():
            if cost_tensor is not None:
                print(f"  {cost_name}: shape={cost_tensor.shape}, mean={cost_tensor.mean().item():.4f}")
            else:
                print(f"  {cost_name}: None")
                
        return True
        
    except Exception as e:
        print(f"❌ MATCHER FEHLER: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cost_weights_impact():
    """Test 2: Auswirkung verschiedener Cost-Gewichte."""
    print("\n" + "="*80) 
    print("TEST 2: COST-GEWICHTE AUSWIRKUNG")
    print("="*80)
    
    test_cfg = {
        'loss': {
            'huber': {
                'delta': 0.1
            }
        }
    }
    
    predictions = create_test_predictions()
    targets = create_test_targets()
    
    # Test verschiedene Gewichtungs-Szenarien
    weight_scenarios = [
        ("Balanced", {"cost_center": 1.0, "cost_class": 1.0, "cost_size": 1.0, "cost_angle": 1.0, "cfg": test_cfg}),
        ("Center Dominant", {"cost_center": 100.0, "cost_class": 1.0, "cost_size": 1.0, "cost_angle": 1.0, "cfg": test_cfg}),
        ("Class Dominant", {"cost_center": 1.0, "cost_class": 100.0, "cost_size": 1.0, "cost_angle": 1.0, "cfg": test_cfg}),
        ("Size Dominant", {"cost_center": 1.0, "cost_class": 1.0, "cost_size": 100.0, "cost_angle": 1.0, "cfg": test_cfg}),
    ]
    
    results = {}
    
    for scenario_name, weights in weight_scenarios:
        print(f"\n--- Szenario: {scenario_name} ---")
        print(f"Gewichte: {weights}")
        
        matcher = HungarianMatcher(**weights)
        (indices, matched_masks), cost_dict = matcher(predictions, targets)
        
        # Berechne gewichtete Gesamtkosten pro Batch
        total_costs = []
        for batch_idx in range(len(indices)):
            batch_cost = 0.0
            for cost_name, cost_tensor in cost_dict.items():
                if cost_tensor is not None and cost_name.startswith('cost_'):
                    weight_key = cost_name  # cost_center, cost_class, etc.
                    weight = weights.get(weight_key, 1.0)
                    # Handle 0-dim tensors (averaged over batch)
                    batch_cost += (cost_tensor * weight).item()
            total_costs.append(batch_cost)
        
        print(f"Gesamtkosten pro Batch: {[f'{cost:.2f}' for cost in total_costs]}")
        print(f"Durchschnittliche Kosten: {np.mean(total_costs):.2f}")
        
        # Analysiere welche Kosten dominieren
        print(f"Cost-Anteile:")
        for cost_name, cost_tensor in cost_dict.items():
            if cost_tensor is not None and cost_name.startswith('cost_'):
                weight = weights.get(cost_name, 1.0)
                weighted_cost = (cost_tensor * weight).sum().item()
                percentage = (weighted_cost / np.sum(total_costs)) * 100
                print(f"  {cost_name}: {percentage:.1f}%")
        
        results[scenario_name] = {
            'indices': indices,
            'total_costs': total_costs,
            'cost_dict': cost_dict
        }
    
    # Vergleiche Matching-Ergebnisse
    print(f"\n--- MATCHING-VERGLEICH ---")
    for scenario_name, result in results.items():
        print(f"\n{scenario_name}:")
        for batch_idx, (pred_idx, tgt_idx) in enumerate(result['indices']):
            print(f"  Batch {batch_idx}: Pred {pred_idx.tolist()} -> GT {tgt_idx.tolist()}")
    
    return results

def test_edge_cases():
    """Test 3: Edge Cases und Fehlerbehandlung."""
    print("\n" + "="*80)
    print("TEST 3: EDGE CASES UND FEHLERBEHANDLUNG") 
    print("="*80)
    
    test_cfg = {
        'loss': {
            'huber': {
                'delta': 0.1
            }
        }
    }
    
    matcher = HungarianMatcher(cost_center=1.0, cost_class=1.0, cost_size=1.0, cost_angle=1.0, cfg=test_cfg)
    
    # Edge Case 1: Keine Targets
    print("\n--- Edge Case 1: Keine validen Targets ---")
    predictions = create_test_predictions()
    empty_targets = {
        'gt_boxes_b_normalized': torch.zeros(2, 3, 10),
        'gt_labels_b': torch.full((2, 3), -1, dtype=torch.long),  # Alle padding
        'gt_valid_mask_b': torch.zeros(2, 3, dtype=torch.bool)    # Alle invalid
    }
    
    try:
        (indices, matched_masks), cost_dict = matcher(predictions, empty_targets)
        print(f"✅ Erfolgreich behandelt: {len(indices)} Batches")
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            print(f"  Batch {batch_idx}: {len(pred_idx)} Matches (sollte 0 sein)")
    except Exception as e:
        print(f"❌ Fehler bei leeren Targets: {e}")
    
    # Edge Case 2: Sehr wenige Predictions (Confidence Threshold simulieren)
    print("\n--- Edge Case 2: Sehr wenige Predictions ---")
    few_pred = create_test_predictions()
    # Simuliere dass nur erste 3 Predictions über Confidence-Threshold sind
    few_pred['pred_class_logits_batch'][:, 3:, :] = -float('inf')  # Sehr niedrige Confidence
    
    try:
        targets = create_test_targets()
        (indices, matched_masks), cost_dict = matcher(few_pred, targets)
        print(f"✅ Erfolgreich behandelt: {len(indices)} Batches") 
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            print(f"  Batch {batch_idx}: Pred {pred_idx.tolist()} -> GT {tgt_idx.tolist()}")
    except Exception as e:
        print(f"❌ Fehler bei wenigen Predictions: {e}")
    
    # Edge Case 3: Extreme Kostenwerte
    print("\n--- Edge Case 3: Extreme Kostenwerte ---")
    extreme_matcher = HungarianMatcher(cost_center=1000.0, cost_class=0.001, cost_size=0.001, cost_angle=0.001, cfg=test_cfg)
    
    try:
        predictions = create_test_predictions()
        targets = create_test_targets()
        (indices, matched_masks), cost_dict = extreme_matcher(predictions, targets)
        print(f"✅ Extreme Gewichte erfolgreich behandelt")
        
        # Prüfe ob Center-Cost wirklich dominiert
        for cost_name, cost_tensor in cost_dict.items():
            if cost_tensor is not None:
                print(f"  {cost_name}: mean={cost_tensor.mean().item():.6f}")
                
    except Exception as e:
        print(f"❌ Fehler bei extremen Gewichten: {e}")

def test_gradient_flow():
    """Test 4: Gradient Flow durch den Matcher."""
    print("\n" + "="*80)
    print("TEST 4: GRADIENT FLOW DURCH MATCHER")
    print("="*80)
    
    test_cfg = {
        'loss': {
            'huber': {
                'delta': 0.1
            }
        }
    }
    
    matcher = HungarianMatcher(cost_center=5.0, cost_class=2.0, cost_size=1.0, cost_angle=1.0, cfg=test_cfg)
    
    predictions = create_test_predictions()
    targets = create_test_targets()
    
    # Berechne Matching
    (indices, matched_masks), cost_dict = matcher(predictions, targets)
    
    # Simuliere Loss Berechnung basierend auf Matching
    total_loss = 0.0
    src_idx_all = []
    tgt_idx_all = []
    
    for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
        if len(pred_idx) > 0:
            # Erstelle globale Indices
            batch_offset_pred = batch_idx * predictions['pred_logits'].shape[1]
            batch_offset_tgt = batch_idx * targets['gt_boxes_b_normalized'].shape[1]
            
            src_idx_all.extend(batch_offset_pred + pred_idx)
            tgt_idx_all.extend(batch_offset_tgt + tgt_idx)
    
    if len(src_idx_all) > 0:
        # Flatten für Loss-Berechnung  
        pred_logits_flat = predictions['pred_class_logits_batch'].view(-1, predictions['pred_class_logits_batch'].shape[-1])
        gt_labels_flat = targets['gt_labels_b'].view(-1)
        
        # Nur matched predictions und targets
        matched_pred_logits = pred_logits_flat[src_idx_all]
        matched_gt_labels = gt_labels_flat[tgt_idx_all]
        
        # Simuliere CrossEntropy Loss
        loss_fn = nn.CrossEntropyLoss()
        classification_loss = loss_fn(matched_pred_logits, matched_gt_labels)
        
        print(f"Simulierte Classification Loss: {classification_loss.item():.4f}")
        print(f"Anzahl gematchte Paare: {len(src_idx_all)}")
        
        # Teste Gradient Propagation
        classification_loss.backward()
        
        # Prüfe Gradienten
        pred_logits_grad = predictions['pred_class_logits_batch'].grad
        box_offsets_grad = predictions['pred_box_offsets_ego_norm_8d_batch'].grad
        
        if pred_logits_grad is not None:
            print(f"✅ Gradienten für pred_logits: mean={pred_logits_grad.mean().item():.6f}, max={pred_logits_grad.max().item():.6f}")
        else:
            print(f"❌ Keine Gradienten für pred_logits!")
            
        if box_offsets_grad is not None:
            print(f"✅ Gradienten für box_offsets: mean={box_offsets_grad.mean().item():.6f}, max={box_offsets_grad.max().item():.6f}")
        else:
            print(f"❌ Keine Gradienten für box_offsets!")
        
        # Analysiere Gradient-Verteilung
        matched_indices = set(src_idx_all)
        num_total_pred = pred_logits_flat.shape[0]
        num_matched = len(matched_indices)
        
        print(f"\nGradient-Verteilung:")
        print(f"  Gesamte Predictions: {num_total_pred}")
        print(f"  Gematchte Predictions: {num_matched}")
        print(f"  Ungematchte Predictions: {num_total_pred - num_matched}")
        print(f"  Match-Rate: {num_matched/num_total_pred*100:.1f}%")
        
    else:
        print(f"❌ Keine Matches gefunden - kann Gradients nicht testen!")

def main():
    """Führt alle Matcher-Tests durch."""
    print("🔍 COMPREHENSIVE HUNGARIAN MATCHER UNIT TEST")
    print("=" * 80)
    print("Ziel: 100%ige Analyse des Matchers und aller möglichen Probleme")
    print("=" * 80)
    
    torch.manual_seed(42)  # Reproducibility
    
    # Test 1: Grundfunktionalität
    success_basic = test_matcher_basic_functionality()
    
    if success_basic:
        # Test 2: Cost Weight Impact
        test_cost_weights_impact()
        
        # Test 3: Edge Cases
        test_edge_cases()
        
        # Test 4: Gradient Flow
        test_gradient_flow()
        
        print("\n" + "="*80)
        print("✅ ALLE MATCHER-TESTS ABGESCHLOSSEN!")
        print("Bitte analysieren Sie die Ausgaben auf Anomalien oder unerwartetes Verhalten.")
        print("=" * 80)
    else:
        print("\n" + "="*80)
        print("❌ BASIC MATCHER TEST FEHLGESCHLAGEN!")
        print("Der Matcher hat grundlegende Probleme - weitere Tests übersprungen.")
        print("=" * 80)

if __name__ == "__main__":
    main()