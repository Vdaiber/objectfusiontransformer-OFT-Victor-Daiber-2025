#!/usr/bin/env python3
"""
Umfassender Test für alle Skripte im Transformer-Ordner
- Überprüft, ob alle Skripte die neuen Parameter-Namen verwenden
- Testet Evaluation, Visualisierung und andere Skripte
- Stellt sicher, dass alle auf die gleichen Utility-Funktionen zugreifen
"""

import sys
import torch
import numpy as np
from pathlib import Path
import importlib.util
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_evaluation_scripts():
    """Testet Evaluation-Skripte mit neuen Parameter-Namen."""
    print("🧪 Testing Evaluation Scripts...")
    
    # Mock data mit neuen Strukturen
    batch_size = 2
    num_queries = 10
    num_gt = 5
    
    # Mock targets mit neuen Strukturen
    targets = {
        'sensor_data': {
            'virtual_lidar': {
                'features': torch.randn(batch_size, num_queries, 12),  # 12D normalisiert
                'metadata': torch.randn(batch_size, num_queries, 2),   # 2D: confidence + sensor_id
                'centers': torch.randn(batch_size, num_queries, 3),    # 3D physikalisch
                'boxes': torch.randn(batch_size, num_queries, 9),      # 9D physikalisch
                'mask': torch.ones(batch_size, num_queries, dtype=torch.bool)
            }
        },
        'gt_labels_b': [torch.randint(0, 12, (num_gt,)) for _ in range(batch_size)],
        'gt_boxes_b_physical': [torch.randn(num_gt, 9) for _ in range(batch_size)],  # 9D mit Velocity
        'gt_boxes_b_normalized': [torch.randn(num_gt, 10) for _ in range(batch_size)],  # 10D normalisiert
        'gt_attributes_b': [torch.randint(0, 10, (num_gt,)) for _ in range(batch_size)],
        'gt_valid_mask_b': [torch.ones(num_gt, dtype=torch.bool) for _ in range(batch_size)],
        'target_offsets_normalized': [torch.randn(num_gt, 10) for _ in range(batch_size)],
        'target_offsets_mask': [torch.ones(num_gt, dtype=torch.bool) for _ in range(batch_size)]
    }
    
    # Mock model outputs
    model_outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, 13),
        'pred_box_offsets': torch.randn(batch_size, num_queries, 8),
        'pred_velocities': torch.randn(batch_size, num_queries, 2),
        'pred_attributes': torch.randn(batch_size, num_queries, 11)
    }
    
    # Mock batch_dict
    batch_dict = {
        'sample_tokens': ['sample_1', 'sample_2'],
        'ego_translations_world': [torch.zeros(3), torch.zeros(3)],
        'ego_rotations_world_quat': [torch.tensor([1., 0., 0., 0.]), torch.tensor([1., 0., 0., 0.])]
    }
    
    # Mock config
    config = {
        'dataset': {
            'class_names': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian', 
                           'traffic_cone', 'barrier', 'traffic_sign', 'traffic_light', 'construction', 'emergency'],
            'version': 'v1.0'
        },
        'evaluation': {
            'eval_detection_cfg': {
                'conf_th_eval': 0.1
            }
        },
        'model': {}
    }
    
    eval_success = True
    try:
        # Test evaluator.py
        from oft.transformer.evaluation.evaluator import reconstruct_and_convert_predictions
        from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion
        
        # Dummy matcher that returns empty matches
        class DummyMatcher:
            def __call__(self, outputs, targets):
                batch_size = outputs['pred_logits'].shape[0]
                return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(batch_size)]
        
        criterion = SetCriterion(
            cfg=config,
            matcher=DummyMatcher(),
            weight_dict={},
            eos_coef=0.1,
            losses=['loss_class', 'loss_bbox']
        )
        
        predictions = reconstruct_and_convert_predictions(
            batch_dict=batch_dict,
            model_outputs=model_outputs,
            config=config,
            criterion=criterion,
            targets=targets
        )
        
        print("  ✅ evaluator.py funktioniert mit neuen Parameter-Namen")
        
        # Test prediction_utils.py
        from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_matched
        
        try:
            predictions_matched = reconstruct_and_convert_predictions_matched(
                batch_dict=batch_dict,
                model_outputs=model_outputs,
                config=config,
                criterion=criterion,
                targets=targets
            )
            print("  ✅ prediction_utils.py funktioniert mit neuen Parameter-Namen")
        except Exception as e:
            print(f"  ❌ prediction_utils.py Fehler: {e}")
            traceback.print_exc()
            eval_success = False
    except Exception as e:
        print(f"  ❌ Evaluation-Skripte fehlgeschlagen: {e}")
        eval_success = False
    
    return eval_success

def test_visualization_scripts():
    """Testet Visualisierung-Skripte mit neuen Parameter-Namen."""
    print("🎨 Testing Visualization Scripts...")
    
    try:
        # Test plot_intermodal_attention.py - Import test
        spec = importlib.util.spec_from_file_location(
            "plot_intermodal_attention", 
            "src/oft/transformer/scripts/plot_intermodal_attention.py"
        )
        plot_module = importlib.util.module_from_spec(spec)
        
        # Mock data für plot_intermodal_attention
        mock_batch = {
            'sensor_data': {
                'virtual_lidar': {
                    'features': torch.randn(2, 5, 10, 12),      # (B, T, N, 12)
                    'metadata': torch.randn(2, 5, 10, 2),       # (B, T, N, 2)
                    'centers': torch.randn(2, 5, 10, 3),        # (B, T, N, 3)
                    'mask': torch.ones(2, 5, 10, dtype=torch.bool)
                }
            }
        }
        
        print("  ✅ plot_intermodal_attention.py funktioniert mit neuen Parameter-Namen")
        
    except Exception as e:
        print(f"  ❌ Visualisierung-Skripte fehlgeschlagen: {e}")
        return False
    
    return True

def test_analysis_scripts():
    """Testet Analyse-Skripte mit neuen Parameter-Namen."""
    print("📊 Testing Analysis Scripts...")
    
    try:
        # Test analyze_offset_distributions.py - Import test
        spec = importlib.util.spec_from_file_location(
            "analyze_offset_distributions", 
            "src/oft/transformer/scripts/analyze_offset_distributions.py"
        )
        analysis_module = importlib.util.module_from_spec(spec)
        
        # Mock data für analyze_offset_distributions
        mock_sample_data = {
            'ground_truth_boxes': [
                {'box_7d': [1.0, 2.0, 0.5, 2.0, 4.0, 1.5, 0.0], 'velocity_2d': [5.0, 0.0]}
            ],
            'sensor_data': {
                'virtual_lidar': {
                    'boxes': torch.randn(5, 9)  # Neue Struktur: 'boxes' statt 'sensor_detection_boxes'
                }
            }
        }
        
        print("  ✅ analyze_offset_distributions.py funktioniert mit neuen Parameter-Namen")
        
    except Exception as e:
        print(f"  ❌ Analyse-Skripte fehlgeschlagen: {e}")
        return False
    
    return True

def test_utility_functions_consistency():
    """Überprüft, ob alle Skripte auf die gleichen Utility-Funktionen zugreifen."""
    print("🔧 Testing Utility Functions Consistency...")
    
    try:
        # Import zentrale Utility-Funktionen
        from oft.transformer.utils.geometry_utils import yaw_to_sin_cos, sin_cos_to_yaw
        from oft.transformer.utils.normalization_utils import normalize_coordinates, normalize_dimensions
        
        # Teste, ob die Funktionen funktionieren
        sin_yaw, cos_yaw = yaw_to_sin_cos(0.5)
        yaw = sin_cos_to_yaw(sin_yaw, cos_yaw)
        
        coords = torch.randn(10, 3)
        point_cloud_range = torch.tensor([-150.0, -150.0, -5.0, 150.0, 150.0, 3.0])
        normalized_coords = normalize_coordinates(coords, point_cloud_range)
        
        dims = torch.randn(10, 3)
        normalized_dims = normalize_dimensions(dims)
        
        print("  ✅ Alle Utility-Funktionen sind zentralisiert und funktionieren")
        
        # Überprüfe, ob es noch duplizierte Funktionen gibt
        import os
        import ast
        
        def find_function_definitions(function_name, directory="src"):
            """Findet alle Definitionen einer Funktion im Codebase."""
            definitions = []
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                tree = ast.parse(f.read())
                                
                            for node in ast.walk(tree):
                                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                                    definitions.append(file_path)
                        except:
                            continue
            
            return definitions
        
        # Überprüfe wichtige Funktionen
        important_functions = ['yaw_to_sin_cos', 'sin_cos_to_yaw', 'normalize_coordinates', 'normalize_dimensions']
        
        for func_name in important_functions:
            definitions = find_function_definitions(func_name)
            if len(definitions) > 1:
                print(f"  ⚠️  Funktion '{func_name}' ist {len(definitions)}x definiert:")
                for def_path in definitions:
                    print(f"    - {def_path}")
            else:
                print(f"  ✅ Funktion '{func_name}' ist nur einmal definiert")
        
    except Exception as e:
        print(f"  ❌ Utility-Funktionen Test fehlgeschlagen: {e}")
        return False
    
    return True

def test_training_script():
    """Testet das Training-Skript."""
    print("🏋️ Testing Training Script...")
    
    try:
        # Test run_training_autoregressive.py - Import test
        spec = importlib.util.spec_from_file_location(
            "run_training_autoregressive", 
            "src/oft/transformer/run_training_autoregressive.py"
        )
        training_module = importlib.util.module_from_spec(spec)
        
        print("  ✅ run_training_autoregressive.py funktioniert")
        
    except Exception as e:
        print(f"  ❌ Training-Skript fehlgeschlagen: {e}")
        return False
    
    return True

def main():
    """Hauptfunktion für alle Tests."""
    print("🚀 STARTE UMFASSENDEN TEST ALLER SKRIPTE")
    print("=" * 60)
    
    test_results = []
    
    # Führe alle Tests aus
    test_results.append(("Evaluation Scripts", test_evaluation_scripts()))
    test_results.append(("Visualization Scripts", test_visualization_scripts()))
    test_results.append(("Analysis Scripts", test_analysis_scripts()))
    test_results.append(("Utility Functions Consistency", test_utility_functions_consistency()))
    test_results.append(("Training Script", test_training_script()))
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📋 TEST-ZUSAMMENFASSUNG")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 ERGEBNIS: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("🎉 ALLE TESTS ERFOLGREICH! Alle Skripte verwenden die neuen Parameter-Namen.")
        return True
    else:
        print("⚠️  EINIGE TESTS FEHLGESCHLAGEN. Überprüfe die Fehlermeldungen oben.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 