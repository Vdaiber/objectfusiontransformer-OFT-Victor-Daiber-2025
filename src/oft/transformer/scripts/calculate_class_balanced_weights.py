#!/usr/bin/env python3
"""
Simple Class-Balanced Weights Calculator

Analyzes ground truth class distribution from TruckScenes dataset and calculates
optimal Class-Balanced Loss parameters following Cui et al. CVPR 2019.

NOW WITH CONFIG INTEGRATION: Automatically reads dataset settings from 
your pipeline_staged.yaml configuration file.

Saves results to JSON file for later use, eliminating need for runtime calculation.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
import argparse

# Import TruckScenes DevKit
from truckscenes import TruckScenes
from truckscenes.utils.splits import create_splits_scenes
from truckscenes.eval.detection.utils import category_to_detection_name

# NEW: Config integration
import sys
import os
sys.path.append('/app')
try:
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️  Hydra not available - using command line arguments instead")

def load_config_automatically() -> Dict[str, Any]:
    """Load dataset configuration from pipeline_staged.yaml automatically."""
    if not CONFIG_AVAILABLE:
        return {}
        
    try:
        # Initialize Hydra with project config directory
        config_dir = "/app/config"
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Load the main pipeline configuration
            cfg = compose(config_name="pipeline_staged")
            
            # Convert to regular dict for easier handling
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            
            print(f"✅ Loaded configuration from pipeline_staged.yaml")
            print(f"   Dataset version: {config_dict.get('dataset', {}).get('version', 'not found')}")
            print(f"   Data root: {config_dict.get('dataset', {}).get('dataroot', 'not found')}")
            
            return config_dict
            
    except Exception as e:
        print(f"⚠️  Could not load config automatically: {e}")
        print(f"   Falling back to command line arguments")
        return {}

def calculate_effective_numbers(class_counts: List[int], beta: float) -> List[float]:
    """Calculate effective number of samples for each class using Cui et al. formula.
    
    Effective Number: E_n = (1 - β^n) / (1 - β)
    where n is the number of samples in the class and β is the re-weighting hyperparameter.
    """
    effective_nums = []
    for n in class_counts:
        if n == 0:
            effective_nums.append(0.0)
        else:
            effective_num = (1.0 - np.power(beta, n)) / (1.0 - beta)
            effective_nums.append(effective_num)
    return effective_nums

def calculate_class_weights(class_counts: List[int], beta: float, normalize: bool = True) -> List[float]:
    """Calculate Class-Balanced weights from effective numbers.
    
    Class-Balanced Weight: w_j = (1-β) / (1-β^n_j)
    """
    effective_nums = calculate_effective_numbers(class_counts, beta)
    
    # Calculate weights as inverse of effective numbers
    weights = []
    for effective_num in effective_nums:
        if effective_num == 0:
            weights.append(0.0)
        else:
            weight = 1.0 / effective_num
            weights.append(weight)
    
    # Normalize weights to have reasonable scale
    if normalize and weights and max(weights) > 0:
        weight_sum = sum(w for w in weights if w > 0)
        num_valid_classes = sum(1 for w in weights if w > 0)
        if weight_sum > 0 and num_valid_classes > 0:
            # Normalize so average weight of valid classes is 1.0
            scale_factor = num_valid_classes / weight_sum
            weights = [w * scale_factor if w > 0 else 0.0 for w in weights]
    
    return weights

def recommend_beta(total_samples: int, imbalance_ratio: float, num_classes: int) -> float:
    """Recommend optimal beta parameter based on dataset characteristics.
    
    Following Cui et al. CVPR 2019 recommendations:
    - Small datasets (< 10K): β = 0.99 - 0.999
    - Medium datasets (10K - 100K): β = 0.999 - 0.9999  
    - Large datasets (> 100K): β = 0.9999 - 0.99999
    
    Also consider class imbalance ratio:
    - High imbalance (ratio > 100): Use higher β
    - Low imbalance (ratio < 10): Use lower β
    """
    
    # Base recommendation by dataset size
    if total_samples < 1000:
        base_beta = 0.9
    elif total_samples < 10000:
        base_beta = 0.99
    elif total_samples < 100000:
        base_beta = 0.999
    else:
        base_beta = 0.9999
    
    # Adjust for imbalance ratio
    if imbalance_ratio > 1000:
        beta = min(0.9999, base_beta + 0.009)  # Higher β for extreme imbalance
    elif imbalance_ratio > 100:
        beta = min(0.9999, base_beta + 0.005)  # Higher β for high imbalance
    elif imbalance_ratio < 10:
        beta = max(0.9, base_beta - 0.05)     # Lower β for low imbalance
    else:
        beta = base_beta
    
    return beta

def analyze_truckscenes_gt_distribution(dataroot: str, version: str, split_name: str) -> Dict[str, Any]:
    """Analyze ground truth class distribution from TruckScenes dataset."""
    
    try:
        # Initialize TruckScenes
        truckscenes = TruckScenes(version=version, dataroot=dataroot, verbose=False)
        
        # Get split scenes (these are scene NAMES, not tokens!)
        split_scene_names = create_splits_scenes()[split_name]
        print(f"   Analyzing {len(split_scene_names)} scenes from {split_name}")
        
        # Convert scene names to actual scene tokens
        split_scene_tokens = []
        for scene_name in split_scene_names:
            # Find scene record by name
            scene_records = [s for s in truckscenes.scene if s['name'] == scene_name]
            if scene_records:
                split_scene_tokens.append(scene_records[0]['token'])
            else:
                print(f"   ⚠️  Scene '{scene_name}' not found in dataset")
        
        print(f"   Found {len(split_scene_tokens)} valid scene tokens")
        
        # Count classes across all annotations
        class_counter = Counter()
        total_annotations = 0
        
        for scene_token in split_scene_tokens:
            scene = truckscenes.get('scene', scene_token)
            
            # Get all samples in this scene
            sample_token = scene['first_sample_token']
            while sample_token != '':
                sample = truckscenes.get('sample', sample_token)
                
                # Get all annotations for this sample
                for ann_token in sample['anns']:
                    ann = truckscenes.get('sample_annotation', ann_token)
                    
                    # Convert category to simplified detection name
                    category_name = ann['category_name']
                    
                    # Use official DevKit mapping function (automatically stays synchronized)
                    simplified_class = category_to_detection_name(category_name)
                    
                    # Skip categories that don't have a valid mapping in the DevKit
                    if simplified_class is None:
                        continue
                    class_counter[simplified_class] += 1
                    total_annotations += 1
                
                # Move to next sample
                sample_token = sample['next']
                
                # Break on empty token
                if sample_token == '':
                    break
        
        print(f"   Total annotations analyzed: {total_annotations:,}")
        
    except Exception as e:
        print(f"❌ Error analyzing {version} {split_name}: {e}")
        return {}
    
    
    all_class_names = ['car', 'truck', 'bus', 'trailer', 'other_vehicle', 'pedestrian', 
                      'motorcycle', 'bicycle', 'traffic_cone', 'barrier', 'animal', 'traffic_sign']
    
    class_counts = [class_counter.get(cls, 0) for cls in all_class_names]
    

    # The no_object class is dynamically generated by the Hungarian Matcher
    # during training and does not exist in the TruckScenes dataset annotations.
    # Adding artificial counts here completely destroys the class balance calculations!
    
    # Calculate statistics (using only the 12 real TruckScenes classes)
    max_samples = max(class_counts) if class_counts else 0
    min_samples = min([c for c in class_counts if c > 0]) if any(class_counts) else 1
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else 1.0
    
    # Recommend beta based on real class distribution only
    recommended_beta = recommend_beta(total_annotations, imbalance_ratio, len(all_class_names))
    
    # Calculate weights for real classes only (12 TruckScenes classes)
    class_weights_real = calculate_class_weights(class_counts, recommended_beta)
    
    return {
        'dataset_info': {
            'version': version,
            'split': split_name,
            'total_annotations': total_annotations,
            'num_scenes': len(split_scene_tokens),
            'analysis_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown'
        },
        'class_distribution': {
            'class_names': all_class_names,        
            'class_counts': class_counts,          
            'class_percentages': [c/total_annotations*100 if total_annotations > 0 else 0 for c in class_counts],
            'note': 'no_object class is dynamically generated by Hungarian Matcher, not in dataset'
        },
        'statistics': {
            'max_samples_per_class': max_samples,
            'min_samples_per_class': min_samples,
            'imbalance_ratio': imbalance_ratio,
            'total_classes': len(all_class_names),  # 12 real classes
            'classes_with_data': sum(1 for c in class_counts if c > 0)
        },
        'class_balanced_loss': {
            'recommended_beta': recommended_beta,
            'class_weights': class_weights_real,  
            'formula': 'weight = (1-beta) / (1-beta^n) where n=class_count',
            'paper': 'Cui et al. CVPR 2019: Class-Balanced Loss Based on Effective Number of Samples',
            'note': 'These weights are for the 12 TruckScenes classes only. no_object handling is separate.'
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate Class-Balanced Loss weights from TruckScenes GT data')
    parser.add_argument('--dataroot', type=str, default=None, 
                       help='Path to TruckScenes dataset root (overrides config)')
    parser.add_argument('--output_dir', type=str, default='/app/config',
                       help='Directory to save weight files')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Dataset versions to analyze (overrides config): v1.0-mini, v1.0')
    parser.add_argument('--force_manual', action='store_true',
                       help='Force manual mode (ignore config file)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🎯 Class-Balanced Loss Weight Calculator (with Config Integration)")
    print("=" * 70)
    
    # Load configuration automatically unless forced manual mode
    config = {}
    if not args.force_manual:
        config = load_config_automatically()
    
    # Determine dataroot (priority: command line > config > default)
    if args.dataroot:
        dataroot = args.dataroot
        print(f"📁 Using dataroot from command line: {dataroot}")
    elif config and config.get('dataset', {}).get('dataroot'):
        dataroot = config['dataset']['dataroot']
        print(f"📁 Using dataroot from config: {dataroot}")
    else:
        dataroot = '/data'
        print(f"📁 Using default dataroot: {dataroot}")
    
    # Determine dataset versions (priority: command line > config > default)
    if args.datasets:
        datasets_to_analyze = [(v, ['train', 'val'] if v == 'v1.0' else ['mini_train', 'mini_val']) for v in args.datasets]
        print(f"📊 Using datasets from command line: {args.datasets}")
    elif config and config.get('dataset', {}).get('version'):
        config_version = config['dataset']['version']
        if config_version == 'v1.0':
            datasets_to_analyze = [('v1.0', ['train', 'val'])]
        else:
            datasets_to_analyze = [('v1.0-mini', ['mini_train', 'mini_val'])]
        print(f"📊 Using dataset from config: {config_version}")
    else:
        # Default: analyze both datasets
        datasets_to_analyze = [
            ('v1.0-mini', ['mini_train', 'mini_val']),
            ('v1.0', ['train', 'val'])  # Full dataset support for when user switches to v1.0
        ]
        print(f"📊 Using default datasets: v1.0-mini, v1.0")
    
    for version, splits in datasets_to_analyze:
        print(f"\n🔍 Analyzing {version}...")
        
        # For each dataset version, analyze the training split for Class-Balanced weights
        train_split = splits[0]  # Use first split as training split
        
        try:
            analysis = analyze_truckscenes_gt_distribution(dataroot, version, train_split)
            
            if analysis:
                # Save to JSON file
                output_file = output_dir / f"class_balanced_weights_{version.replace('.', '_').replace('-', '_')}.json"
                
                with open(output_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                
                print(f"✅ Saved analysis to: {output_file}")
                
                # Print summary
                stats = analysis['statistics']
                cb_loss = analysis['class_balanced_loss']
                
                print(f"   📊 Summary:")
                print(f"      - Total annotations: {analysis['dataset_info']['total_annotations']:,}")
                print(f"      - Classes with data: {stats['classes_with_data']}/{stats['total_classes']}")
                print(f"      - Imbalance ratio: {stats['imbalance_ratio']:.2f}")
                print(f"      - Recommended β: {cb_loss['recommended_beta']:.4f}")
                print(f"      - Weight range: {min(cb_loss['class_weights']):.4f} - {max(cb_loss['class_weights']):.4f}")
                
            else:
                print(f"❌ Failed to analyze {version}")
                
        except Exception as e:
            print(f"❌ Error processing {version}: {e}")
            continue
    
    print(f"\n🎉 Class-Balanced weight calculation complete!")
    print(f"   💾 Files saved to: {output_dir}")
    print(f"   🚀 Ready for training with automatic Class-Balanced Loss!")

if __name__ == "__main__":
    main() 
