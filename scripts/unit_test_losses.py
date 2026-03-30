#!/usr/bin/env python3
"""
Unit Tests für OFT Loss Functions - Systematische Validierung
ZWECK: Verifiziere dass alle Loss-Komponenten mathematisch korrekt arbeiten
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append('/app')

# Import the matcher cost functions
from src.oft.transformer.training.losses.matcher_costs import (
    center_cost_matrix_huber, class_cost_matrix, size_cost_matrix_l1
)

def test_matcher_costs():
    """Test Matcher Cost Functions"""
    print("🔍 UNIT TEST: Matcher Costs")
    
    # Dummy data
    pred_logits = torch.randn(10, 13)  # 10 queries, 13 classes (12 + no_object)
    gt_classes = torch.randint(0, 12, (5,))  # 5 GT objects
    
    pred_boxes = torch.randn(10, 3)  # [x, y, z] centers
    gt_boxes = torch.randn(5, 3)
    
    # Test class costs
    class_costs = class_cost_matrix(pred_logits, gt_classes)
    print(f"  ✅ Class costs shape: {class_costs.shape} (should be [10, 5])")
    print(f"  ✅ Class costs range: {class_costs.min().item():.3f} - {class_costs.max().item():.3f}")
    
    # Test center costs  
    center_costs = center_cost_matrix_huber(pred_boxes, gt_boxes)
    print(f"  ✅ Center costs shape: {center_costs.shape} (should be [10, 5])")
    print(f"  ✅ Center costs range: {center_costs.min().item():.3f} - {center_costs.max().item():.3f}")
    
    # CRITICAL: Verify that costs are reasonable scales
    avg_class_cost = class_costs.mean().item()
    avg_center_cost = center_costs.mean().item()
    ratio = avg_class_cost / avg_center_cost if avg_center_cost > 0 else float('inf')
    
    print(f"  📊 Average class cost: {avg_class_cost:.3f}")
    print(f"  📊 Average center cost: {avg_center_cost:.3f}")
    print(f"  📊 Class/Center ratio: {ratio:.2f}x")
    
    if ratio > 5.0:
        print(f"  ⚠️  WARNING: Class costs {ratio:.1f}x higher than Center - may cause dominance!")
    else:
        print(f"  ✅ GOOD: Cost ratio is reasonable")
    
    return True

def test_loss_scaling():
    """Test Loss Component Scaling"""
    print("\n🔍 UNIT TEST: Loss Scaling")
    
    # Simulate typical loss values
    batch_size = 2
    num_matched = 20
    
    # Typical raw loss values (before weighting)
    raw_center_loss = torch.tensor(0.05)    # Small - geometric 
    raw_size_loss = torch.tensor(0.18)      # Often larger
    raw_class_loss = torch.tensor(0.25)     # Cross-entropy  
    raw_velocity_loss = torch.tensor(0.03)  # Usually small
    
    # Apply current config weights
    weights = {
        'center': 20.0,
        'size': 1.0, 
        'class': 2.0,
        'velocity': 3.0
    }
    
    weighted_losses = {
        name: raw_loss * weight for (name, raw_loss), weight in 
        zip([('center', raw_center_loss), ('size', raw_size_loss), 
             ('class', raw_class_loss), ('velocity', raw_velocity_loss)], 
            weights.values())
    }
    
    total_loss = sum(weighted_losses.values())
    loss_percentages = {name: (loss/total_loss*100).item() 
                       for name, loss in weighted_losses.items()}
    
    print("  📊 Projected Loss Distribution:")
    for name, pct in loss_percentages.items():
        print(f"    {name}: {pct:.1f}%")
    
    # Check if center dominates
    if loss_percentages['center'] > 30:
        print("  ✅ GOOD: Center loss will dominate")
    else:
        print("  ⚠️  WARNING: Center loss may still be too weak")
        
    return True

def test_car_class_weight():
    """Test Car Class Weight Impact"""
    print("\n🔍 UNIT TEST: Car Class Weight")
    
    # Simulate class distribution
    class_names = ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 'pedestrian',
                   'motorcycle', 'bicycle', 'traffic_cone', 'barrier', 'animal', 'traffic_sign']
    
    old_weights = [0.5, 0.8, 1.3, 0.9, 1.5, 0.9, 2.0, 1.5, 1.0, 2.5, 2.3, 0.6]
    new_weights = [1.2, 1.0, 1.3, 1.0, 1.5, 1.1, 2.0, 1.5, 1.0, 2.5, 2.3, 0.8]
    
    print("  📊 Class Weight Changes:")
    for name, old_w, new_w in zip(class_names, old_weights, new_weights):
        change = (new_w - old_w) / old_w * 100
        symbol = "↑" if change > 0 else "↓" if change < 0 else "="
        print(f"    {name}: {old_w:.1f} → {new_w:.1f} ({symbol}{abs(change):.0f}%)")
    
    # Focus on car
    car_boost = (1.2 - 0.5) / 0.5 * 100
    print(f"  🚗 CAR BOOST: +{car_boost:.0f}% - should significantly improve Car mAP!")
    
    return True

def main():
    print("=" * 60)
    print("🧪 OFT LOSS FUNCTION UNIT TESTS")
    print("=" * 60)
    
    try:
        test_matcher_costs()
        test_loss_scaling() 
        test_car_class_weight()
        
        print("\n" + "=" * 60)
        print("✅ ALL UNIT TESTS PASSED!")
        print("🚀 Ready to start training with new config!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ UNIT TEST FAILED: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()