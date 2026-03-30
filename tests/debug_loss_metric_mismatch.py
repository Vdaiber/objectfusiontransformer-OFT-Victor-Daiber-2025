#!/usr/bin/env python3
"""
Debug the mismatch between perfect regression losses and poor metrics.
"""

import torch
import json
import numpy as np
from pathlib import Path

def analyze_predictions_vs_gt():
    """Compare predictions with ground truth to understand the mismatch"""
    
    # Load the latest predictions
    pred_path = "/app/output/testruns/confidence_fix/eval_results/epoch_0/predictions.json"
    
    if not Path(pred_path).exists():
        print(f"❌ Predictions file not found: {pred_path}")
        return
    
    with open(pred_path, 'r') as f:
        predictions_data = json.load(f)
    
    results = predictions_data['results']
    
    print("🔍 LOSS vs METRIC MISMATCH ANALYSIS")
    print("=" * 50)
    
    sample_tokens = list(results.keys())
    print(f"Total samples: {len(sample_tokens)}")
    
    total_predictions = 0
    high_conf_predictions = 0
    class_distribution = {}
    translation_magnitudes = []
    
    for sample_token, predictions in results.items():
        total_predictions += len(predictions)
        
        for pred in predictions:
            # Count high confidence predictions
            score = pred['detection_score']
            if score > 0.5:  # High confidence
                high_conf_predictions += 1
            
            # Class distribution
            class_name = pred['detection_name']
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
            
            # Translation magnitudes (world coordinates)
            translation = pred['translation']
            magnitude = np.linalg.norm(translation)
            translation_magnitudes.append(magnitude)
    
    print(f"\n📊 PREDICTION STATISTICS:")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Average per sample: {total_predictions / len(sample_tokens):.2f}")
    print(f"  High confidence (>0.5): {high_conf_predictions} ({100*high_conf_predictions/total_predictions:.1f}%)")
    
    print(f"\n📊 CLASS DISTRIBUTION:")
    for class_name, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} ({100*count/total_predictions:.1f}%)")
    
    print(f"\n📊 TRANSLATION STATISTICS:")
    translation_magnitudes = np.array(translation_magnitudes)
    print(f"  Mean magnitude: {translation_magnitudes.mean():.1f} meters")
    print(f"  Std magnitude: {translation_magnitudes.std():.1f} meters")
    print(f"  Min magnitude: {translation_magnitudes.min():.1f} meters")
    print(f"  Max magnitude: {translation_magnitudes.max():.1f} meters")
    
    # Sample a few predictions for detailed analysis
    print(f"\n🔍 SAMPLE PREDICTIONS (first 3):")
    sample_count = 0
    for sample_token, predictions in results.items():
        if sample_count >= 3:
            break
        
        print(f"\n  Sample {sample_token[:8]}... ({len(predictions)} predictions):")
        for i, pred in enumerate(predictions[:3]):  # First 3 predictions per sample
            print(f"    Pred {i}: {pred['detection_name']} (score: {pred['detection_score']:.3f})")
            print(f"      Translation: {pred['translation']}")
            print(f"      Size: {pred['size']}")
            print(f"      Velocity: {pred['velocity']}")
        
        sample_count += 1
    
    # Check if we have ground truth info
    print(f"\n🔍 ESTIMATED DETECTION RATE:")
    # Assuming ~14 GT objects per sample (as mentioned)
    estimated_gt_per_sample = 14
    estimated_total_gt = len(sample_tokens) * estimated_gt_per_sample
    detection_rate = total_predictions / estimated_total_gt
    print(f"  Estimated GT objects: {estimated_total_gt}")
    print(f"  Detection rate: {detection_rate:.2f} (1.0 = perfect, >1.0 = false positives)")
    
    if detection_rate > 2.0:
        print(f"  ⚠️  HIGH FALSE POSITIVE RATE: Too many predictions!")
    elif detection_rate < 0.5:
        print(f"  ⚠️  LOW RECALL: Missing many objects!")
    else:
        print(f"  ✅ Detection rate seems reasonable")


def analyze_loss_calculation():
    """Analyze what the perfect losses actually mean"""
    
    print(f"\n🔍 LOSS INTERPRETATION ANALYSIS")
    print("=" * 40)
    
    print(f"📊 REGRESSION LOSS VALUES (from latest run):")
    print(f"  loss_center: 0.0001  -> Avg error ~0.1mm in normalized coords")
    print(f"  loss_size: 0.0158    -> Avg error ~1.6% in size")
    print(f"  loss_angle: 0.0051   -> Avg error ~0.1 radians = ~6 degrees")
    print(f"  loss_velocity: 0.0001 -> Avg error ~0.01 m/s")
    
    print(f"\n💡 WHAT PERFECT LOSSES COULD MEAN:")
    print(f"  1. ✅ Model predicts boxes very accurately FOR DETECTED OBJECTS")
    print(f"  2. ❌ But model detects WRONG OBJECTS (false positives)")
    print(f"  3. ❌ Or model MISSES many real objects (false negatives)")
    print(f"  4. ❌ Classification/matching is wrong despite good box regression")
    
    print(f"\n🎯 KEY INSIGHT:")
    print(f"  Regression losses only measure QUALITY of detected boxes")
    print(f"  They DON'T measure:")
    print(f"    - Whether we detect the RIGHT objects")
    print(f"    - Whether we detect ALL objects")
    print(f"    - Whether classifications are correct")


def main():
    """Run the analysis"""
    try:
        analyze_predictions_vs_gt()
        analyze_loss_calculation()
        
        print(f"\n🎯 CONCLUSION:")
        print(f"The perfect regression losses indicate the model is very good at")
        print(f"predicting box coordinates, BUT:")
        print(f"  1. It might be detecting the WRONG objects")
        print(f"  2. It might be MISSING many real objects") 
        print(f"  3. It might have classification problems")
        print(f"  4. There might be coordinate transformation bugs")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"  1. Check if high-confidence predictions are on correct objects")
        print(f"  2. Check recall rate (are we missing GT objects?)")
        print(f"  3. Verify coordinate transformations are consistent")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
