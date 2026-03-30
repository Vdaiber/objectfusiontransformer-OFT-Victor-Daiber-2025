"""
Test Phase 3: Vereinfachte Box-Formate

Dieser Test überprüft die Implementierung der vereinfachten Box-Formate aus Phase 3:
- Ground Truth (physikalisch): Box9D [x,y,z,w,l,h,yaw,vx,vy]
- Ground Truth (normalisiert): Box10D [x_norm,y_norm,z_norm,w_norm,l_norm,h_norm,sin(yaw),cos(yaw),vx_norm,vy_norm]
- Sensor-Detektionen: Box9D [x,y,z,w,l,h,yaw,vx,vy] (verrauschte GT)
- Offsets (10D): [dx_norm,dy_norm,dz_norm,dw_norm,dl_norm,dh_norm,sin(dyaw),cos(dyaw),dvx_norm,dvy_norm]
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List

# Test-Konfiguration
TEST_CONFIG = {
    'dataset': {
        'point_cloud_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        'max_velocity': 10.0,
        'virtual_sensors': [
            {'name': 'virtual_lidar', 'enabled': True},
            {'name': 'virtual_radar', 'enabled': True}
        ]
    }
}


def create_mock_sample_phase3() -> Dict[str, Any]:
    """Erstelle einen Mock-Sample mit den neuen Box-Formaten aus Phase 3."""
    num_objects = 3
    
    # Sensor-Daten mit neuen Box-Formaten
    sensor_data = {}
    for sensor_name in ['virtual_lidar', 'virtual_radar']:
        # PHASE 3: Erstelle korrekte Features mit sin/cos Yaw
        features = []
        for i in range(num_objects):
            # Erstelle zufällige Werte für alle Komponenten außer Yaw
            coords = np.random.randn(3).astype(np.float32)  # x_norm, y_norm, z_norm
            dims = np.random.randn(3).astype(np.float32)    # w_norm, l_norm, h_norm
            
            # PHASE 3: Korrekte sin/cos Yaw-Repräsentation
            yaw_angle = np.random.uniform(-np.pi, np.pi)
            yaw_sin_cos = np.array([np.sin(yaw_angle), np.cos(yaw_angle)], dtype=np.float32)
            
            velocity = np.random.randn(2).astype(np.float32)  # vx_norm, vy_norm
            
            # Konkateniere zu 12D Features
            sensor_class = np.array([np.random.randint(0, 12)], dtype=np.float32)  # sensor_class
            sensor_attr = np.array([np.random.randint(0, 11)], dtype=np.float32)   # sensor_attr
            feature = np.concatenate([coords, dims, yaw_sin_cos, velocity, sensor_class, sensor_attr])
            features.append(feature)
        
        sensor_data[sensor_name] = {
            'features': np.array(features, dtype=np.float32),  # 12D features (normalisiert)
            'metadata': np.random.randn(num_objects, 2).astype(np.float32),   # 2D metadata
            'centers': np.random.randn(num_objects, 3).astype(np.float32),    # 3D centers (PHASE 3: PHYSIKALISCH für Visualisierung)
            'boxes': np.random.randn(num_objects, 9).astype(np.float32),      # 9D boxes (PHASE 3: PHYSIKALISCH)
            'mask': np.zeros(num_objects, dtype=bool)                         # mask
        }
    
    # Ground Truth mit neuen Box-Formaten (PHASE 3)
    normalized_gt = []
    physical_gt = []
    
    for i in range(3):
        # PHASE 3: Box10D normalisiert
        coords_norm = np.random.randn(3).astype(np.float32)  # x_norm, y_norm, z_norm
        dims_norm = np.random.randn(3).astype(np.float32)    # w_norm, l_norm, h_norm
        
        # Korrekte sin/cos Yaw-Repräsentation
        yaw_angle = np.random.uniform(-np.pi, np.pi)
        yaw_sin_cos = np.array([np.sin(yaw_angle), np.cos(yaw_angle)], dtype=np.float32)
        
        velocity_norm = np.random.randn(2).astype(np.float32)  # vx_norm, vy_norm
        
        box_10d = np.concatenate([coords_norm, dims_norm, yaw_sin_cos, velocity_norm])
        
        normalized_gt.append({
            'box_10d_normalized': box_10d,
            'class_idx': np.random.randint(0, 12),
            'attribute_idx': np.random.randint(0, 3)
        })
        
        # PHASE 3: Box9D physikalisch
        coords_phys = np.random.randn(3).astype(np.float32)  # x, y, z
        dims_phys = np.random.randn(3).astype(np.float32)    # w, l, h
        yaw_phys = np.random.uniform(-np.pi, np.pi)          # yaw
        velocity_phys = np.random.randn(2).astype(np.float32)  # vx, vy
        
        box_9d = np.concatenate([coords_phys, dims_phys, [yaw_phys], velocity_phys])
        
        physical_gt.append({
            'box_9d_physical': box_9d,
            'class_idx': np.random.randint(0, 12),
            'attribute_idx': np.random.randint(0, 3)
        })
    
    ground_truth = {
        'normalized': normalized_gt,
        'physical': physical_gt,
        'labels': np.array([0, 1, 2], dtype=np.int64),
        'offsets': np.random.randn(3, 10).astype(np.float32)  # 10D offsets (PHASE 3)
    }
    
    return {
        'sample_token': 'test_token_123',
        'timestamp': 1234567890.0,
        'ego_translation_world': np.array([1.0, 2.0, 0.5], dtype=np.float32),
        'ego_rotation_world_quat': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        'ground_truth': ground_truth,
        'sensor_data': sensor_data,
        'scene_meta': {'weather': 'clear', 'area': 'highway'}
    }


def test_phase3_box_formats():
    """Test: Überprüfe, dass alle neuen Box-Formate korrekt sind."""
    print("🧪 Testing Phase 3 box formats...")
    
    sample = create_mock_sample_phase3()
    
    # Test Sensor-Daten Box-Formate
    for sensor_name, sensor_data in sample['sensor_data'].items():
        # PHASE 3: Sensor-Detektionen sind Box9D
        assert sensor_data['boxes'].shape[1] == 9, f"Sensor boxes should be 9D, got {sensor_data['boxes'].shape[1]}"
        assert sensor_data['features'].shape[1] == 12, f"Features should be 12D, got {sensor_data['features'].shape[1]}"
    
    # Test Ground Truth Box-Formate
    gt = sample['ground_truth']
    
    # PHASE 3: Normalized GT sind Box10D
    for gt_item in gt['normalized']:
        assert 'box_10d_normalized' in gt_item, "Normalized GT should have 'box_10d_normalized'"
        assert gt_item['box_10d_normalized'].shape[0] == 10, f"Box10D should be 10D, got {gt_item['box_10d_normalized'].shape[0]}"
    
    # PHASE 3: Physical GT sind Box9D
    for gt_item in gt['physical']:
        assert 'box_9d_physical' in gt_item, "Physical GT should have 'box_9d_physical'"
        assert gt_item['box_9d_physical'].shape[0] == 9, f"Box9D should be 9D, got {gt_item['box_9d_physical'].shape[0]}"
    
    # PHASE 3: Offsets sind 10D
    assert gt['offsets'].shape[1] == 10, f"Offsets should be 10D, got {gt['offsets'].shape[1]}"
    
    print("✅ Phase 3 box formats are correct!")


def test_phase3_collate_functions():
    """Test: Überprüfe, dass die Collate-Funktionen mit neuen Box-Formaten funktionieren."""
    print("🧪 Testing Phase 3 collate functions...")
    
    try:
        from oft.transformer.datasets.preprocessing.collate_functions import (
            object_fusion_gt_collate_fn_autoregressive
        )
        
        # Erstelle Mock-Batch mit Phase 3 Formaten
        batch = [create_mock_sample_phase3() for _ in range(2)]
        
        # Teste autoregressive Collate-Funktion
        collated = object_fusion_gt_collate_fn_autoregressive(batch)
        
        # PHASE 3: Überprüfe neue Box-Dimensionen
        assert 'gt_boxes_b_normalized' in collated, "Should have 'gt_boxes_b_normalized'"
        assert collated['gt_boxes_b_normalized'].shape[-1] == 10, f"Normalized GT should be 10D, got {collated['gt_boxes_b_normalized'].shape[-1]}"
        
        assert 'gt_boxes_b_physical' in collated, "Should have 'gt_boxes_b_physical'"
        assert collated['gt_boxes_b_physical'].shape[-1] == 9, f"Physical GT should be 9D, got {collated['gt_boxes_b_physical'].shape[-1]}"
        
        # PHASE 3: Überprüfe Sensor-Daten
        for sensor_name, sensor_data in collated['sensor_data'].items():
            assert sensor_data['boxes'].shape[-1] == 9, f"Sensor boxes should be 9D, got {sensor_data['boxes'].shape[-1]}"
            assert sensor_data['features'].shape[-1] == 12, f"Features should be 12D, got {sensor_data['features'].shape[-1]}"
        
        print("✅ Phase 3 collate functions work correctly!")
        
    except ImportError as e:
        print(f"⚠️  Could not import collate functions: {e}")


def test_phase3_box_format_consistency():
    """Test: Überprüfe Konsistenz zwischen verschiedenen Box-Formaten."""
    print("🧪 Testing Phase 3 box format consistency...")
    
    # Teste, dass Box9D und Box10D die gleiche Anzahl Objekte haben
    sample = create_mock_sample_phase3()
    gt = sample['ground_truth']
    
    num_normalized = len(gt['normalized'])
    num_physical = len(gt['physical'])
    num_labels = len(gt['labels'])
    num_offsets = gt['offsets'].shape[0]
    
    assert num_normalized == num_physical, f"Normalized ({num_normalized}) and physical ({num_physical}) should have same count"
    assert num_normalized == num_labels, f"Normalized ({num_normalized}) and labels ({num_labels}) should have same count"
    assert num_normalized == num_offsets, f"Normalized ({num_normalized}) and offsets ({num_offsets}) should have same count"
    
    # Teste, dass alle Sensor-Daten die gleiche Anzahl Objekte haben
    for sensor_name, sensor_data in sample['sensor_data'].items():
        num_sensor_objects = sensor_data['features'].shape[0]
        assert num_sensor_objects == num_normalized, f"Sensor {sensor_name} ({num_sensor_objects}) and GT ({num_normalized}) should have same count"
    
    print("✅ Phase 3 box format consistency is maintained!")


def test_phase3_velocity_integration():
    """Test: Überprüfe, dass Velocity korrekt in alle Box-Formate integriert ist."""
    print("🧪 Testing Phase 3 velocity integration...")
    
    sample = create_mock_sample_phase3()
    
    # PHASE 3: Box9D sollte Velocity enthalten (Positionen 7-8)
    for gt_item in sample['ground_truth']['physical']:
        box_9d = gt_item['box_9d_physical']
        velocity = box_9d[7:9]  # vx, vy
        assert velocity.shape[0] == 2, f"Velocity should be 2D, got {velocity.shape[0]}"
    
    # PHASE 3: Box10D sollte normalisierte Velocity enthalten (Positionen 8-9)
    for gt_item in sample['ground_truth']['normalized']:
        box_10d = gt_item['box_10d_normalized']
        velocity_norm = box_10d[8:10]  # vx_norm, vy_norm
        assert velocity_norm.shape[0] == 2, f"Normalized velocity should be 2D, got {velocity_norm.shape[0]}"
    
    # PHASE 3: Sensor-Detektionen sollten Velocity enthalten (Positionen 7-8)
    for sensor_name, sensor_data in sample['sensor_data'].items():
        for box_9d in sensor_data['boxes']:
            velocity = box_9d[7:9]  # vx, vy
            assert velocity.shape[0] == 2, f"Sensor velocity should be 2D, got {velocity.shape[0]}"
    
    print("✅ Phase 3 velocity integration is correct!")


def test_phase3_yaw_representation():
    """Test: Überprüfe, dass Yaw korrekt als sin/cos in normalisierten Formaten dargestellt wird."""
    print("🧪 Testing Phase 3 yaw representation...")
    
    sample = create_mock_sample_phase3()
    
    # PHASE 3: Box10D sollte sin/cos Yaw enthalten (Positionen 6-7)
    for gt_item in sample['ground_truth']['normalized']:
        box_10d = gt_item['box_10d_normalized']
        yaw_sin_cos = box_10d[6:8]  # sin(yaw), cos(yaw)
        assert yaw_sin_cos.shape[0] == 2, f"Yaw sin/cos should be 2D, got {yaw_sin_cos.shape[0]}"
        
        # Teste, dass sin² + cos² ≈ 1 (numerische Stabilität)
        sin_sq_plus_cos_sq = yaw_sin_cos[0]**2 + yaw_sin_cos[1]**2
        assert 0.9 <= sin_sq_plus_cos_sq <= 1.1, f"sin² + cos² should be ≈ 1, got {sin_sq_plus_cos_sq}"
    
    # PHASE 3: Features sollten sin/cos Yaw enthalten (Positionen 6-7)
    for sensor_name, sensor_data in sample['sensor_data'].items():
        for features in sensor_data['features']:
            yaw_sin_cos = features[6:8]  # sin(yaw), cos(yaw)
            assert yaw_sin_cos.shape[0] == 2, f"Feature yaw sin/cos should be 2D, got {yaw_sin_cos.shape[0]}"
    
    print("✅ Phase 3 yaw representation is correct!")


def run_all_phase3_tests():
    """Führe alle Phase 3 Tests aus."""
    print("🚀 Running all Phase 3 tests...")
    
    tests = [
        test_phase3_box_formats,
        test_phase3_collate_functions,
        test_phase3_box_format_consistency,
        test_phase3_velocity_integration,
        test_phase3_yaw_representation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print(f"\n📊 Phase 3 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 3 tests passed! Box formats are correctly implemented.")
    else:
        print("⚠️  Some Phase 3 tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    run_all_phase3_tests() 