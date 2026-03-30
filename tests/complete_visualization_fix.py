
# Complete fix for epoch_visualization_workflow.py

import os
import numpy as np
import cv2
from pyquaternion import Quaternion
from truckscenes.utils.data_classes import Box as DevkitBox
from truckscenes.utils.geometry_utils import BoxVisibility, box_in_image
from oft.transformer.datasets.truckscenes.transforms import transform_box_vehicle_to_world
from truckscenes.utils.geometry_utils import transform_matrix
from oft.transformer.utils.epoch_visualization import visualize_epoch_sample
from oft.transformer.utils.normalization_utils import denormalize_coordinates, denormalize_dimensions, get_global_normalizer
from oft.transformer.utils.geometry_utils import sin_cos_to_yaw
import torch
from typing import List


# Class mapping from prediction names to DevKit GT names
PREDICTION_TO_DEVKIT_CLASS_MAPPING = {
    "car": "vehicle.car",
    "pedestrian": "human.pedestrian.adult", 
    "truck": "vehicle.truck",
    "trailer": "vehicle.trailer",
    "bicycle": "vehicle.bicycle",
    "bus": "vehicle.bus",
    "barrier": "movable_object.barrier",
    "traffic_cone": "movable_object.trafficcone",
    "motorcycle": "vehicle.motorcycle",
    "other_vehicle": "vehicle.other",
    "animal": "animal",
    "traffic_sign": "static_object.trafficsign"
}


def _filter_boxes_with_devkit_logic(boxes: List[DevkitBox], K: np.ndarray, img_shape) -> List[DevkitBox]:
    """Filter boxes using exact DevKit visibility logic to match GT box count."""
    h, w = img_shape[:2]
    imsize = (w, h)  # DevKit expects (width, height)
    
    visible_boxes = []
    for box in boxes:
        # Use DevKit's exact box_in_image function
        if box_in_image(box, K, imsize, vis_level=BoxVisibility.ANY):
            visible_boxes.append(box)
    
    return visible_boxes


def visualize_val_sample_from_dataset(cfg, val_dataset, epoch, run_dir, trucksc=None, predicted_boxes_normalized=None):
    """
    FIXED: Visualisiert ein Val-Sample nach DevKit-Standard mit korrekten Transformationen.
    """
    sample_idx = cfg['visualization']['sample_idx']
    camera_channel = cfg['visualization']['camera_channel']

    # 1. Lade das Val-Sample mit Sicherheitsprüfung
    try:
        if sample_idx >= len(val_dataset):
            print(f"[VIS ERROR] sample_idx {sample_idx} >= dataset_length {len(val_dataset)}")
            sample_idx = min(sample_idx, len(val_dataset) - 1)
            print(f"[VIS FIX] Using corrected sample_idx: {sample_idx}")
        
        sample = val_dataset[sample_idx]
        sample_token = sample["sample_token"]
        print(f"[VIS DEBUG] Loading sample_idx={sample_idx}, sample_token={sample_token}")
        
    except Exception as e:
        print(f"[VIS ERROR] Failed to load val_dataset[{sample_idx}]: {e}")
        sample = val_dataset[0]
        sample_token = sample["sample_token"]
        print(f"[VIS FALLBACK] Using first sample: {sample_token}")

    # 2. DevKit setup
    ts = val_dataset.ts if hasattr(val_dataset, 'ts') and val_dataset.ts is not None else trucksc
    
    if ts is None:
        from truckscenes import TruckScenes
        ts = TruckScenes(version=val_dataset.version, dataroot=val_dataset.dataroot, verbose=False)
        print(f"[VIS DEBUG] Loaded DevKit for visualization (RAM mode fallback)")
    
    try:
        sample_record = ts.get('sample', sample_token)
        print(f"[VIS DEBUG] Successfully loaded sample_record from DevKit")
    except Exception as e:
        print(f"[VIS ERROR] Failed to get sample_record from DevKit: {e}")
        raise e
    
    # Verify timestamps
    dataset_timestamp = sample.get('timestamp')
    devkit_timestamp = sample_record['timestamp']
    
    if dataset_timestamp is not None and dataset_timestamp == devkit_timestamp:
        print(f'[VIS TIMESTAMP OK] Timestamps are consistent: {dataset_timestamp}')
    else:
        print(f'[VIS TIMESTAMP ERROR] Timestamp mismatch!')
        print(f'  Dataset: {dataset_timestamp}')
        print(f'  DevKit:  {devkit_timestamp}')
        print(f'  Difference: {abs(dataset_timestamp - devkit_timestamp) / 1000:.1f} ms')
        
    cam_token = sample_record['data'][camera_channel]
    sd_record = ts.get('sample_data', cam_token)
    cs_record = ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    image_path = os.path.join(ts.dataroot, sd_record['filename'])
    image = cv2.imread(image_path)
    K = np.array(cs_record['camera_intrinsic'], dtype=np.float64).reshape(3, 3)

    # 3. GT-Boxen: DevKit-Standard
    print(f'=== GT-Boxen laden (DevKit-Standard) ===')
    print(f'Sample Token: {sample_token}')
    print(f'Kamera Channel: {camera_channel}')
    print(f'Sample Data Token: {cam_token}')
    
    _, gt_boxes_cam, _ = ts.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    print(f'DevKit get_sample_data() gab {len(gt_boxes_cam)} sichtbare Boxen zurück')
    gt_boxes = gt_boxes_cam
    print(f'Verwende {len(gt_boxes)} GT-Boxen in Kamera-Koordinaten')

    # 4. FIXED: Predicted-Boxen laden und verarbeiten
    fused_boxes = []
    
    import json
    devkit_pred_path = os.path.join(run_dir, 'eval_results', f'epoch_{epoch}', 'predictions.json')
    
    if os.path.exists(devkit_pred_path):
        print(f'=== Lade Predicted-Boxen aus DevKit-Evaluation: {devkit_pred_path} ===')
        with open(devkit_pred_path, 'r') as f:
            devkit_predictions = json.load(f)
        
        sample_predictions = devkit_predictions['results'].get(sample_token, [])
        
        if sample_predictions:
            # FIXED: Konsistente Ego-Pose
            pose_record = ts.getclosest('ego_pose', sample_record['timestamp'])
            
            for i, pred in enumerate(sample_predictions):
                center_world = np.array(pred['translation'])
                size_world = np.array(pred['size'])
                rotation_world_quat = Quaternion(pred['rotation'])
                
                # FIXED: Map prediction class to DevKit class
                pred_class = pred['detection_name']
                devkit_class = PREDICTION_TO_DEVKIT_CLASS_MAPPING.get(pred_class, pred_class)
                
                # Erstelle DevBox in Weltkoordinaten
                world_box = DevkitBox(
                    center=center_world,
                    size=size_world,
                    orientation=rotation_world_quat,
                    name=devkit_class,  # FIXED: Use mapped class name
                    score=pred['detection_score'],
                    token=f'pred_{i}'
                )
                
                # FIXED: Korrekte Koordinatentransformation
                # World → Ego
                ego_box = world_box.copy()
                ego_box.translate(-np.array(pose_record['translation']))
                ego_box.rotate(Quaternion(pose_record['rotation']).inverse)
                
                # Ego → Camera (FIXED: NO .inverse!)
                camera_box = ego_box.copy()
                camera_box.translate(-np.array(cs_record['translation']))
                camera_box.rotate(Quaternion(cs_record['rotation']))  # FIXED: Removed .inverse
                
                fused_boxes.append(camera_box)
                
            print(f'Erstellte {len(fused_boxes)} Predicted-Boxen in Kamera-Koordinaten (DevKit-Standard)')

            # FIXED: DevKit-kompatible Sichtbarkeitsfilterung
            fused_boxes_before = len(fused_boxes)
            fused_boxes = _filter_boxes_with_devkit_logic(fused_boxes, K, image.shape)
            print(f'FOV-Filter: {fused_boxes_before} → {len(fused_boxes)} sichtbare Predicted-Boxen im Bildbereich')
    else:
        print(f'WARNUNG: Keine DevKit-Evaluation gefunden: {devkit_pred_path}')
        print('Keine Predicted-Boxen verfügbar für Visualisierung')

    # 5. Visualisierung
    visualize_epoch_sample(
        image=image,
        fused_boxes=fused_boxes,
        gt_boxes=gt_boxes,
        camera_k_matrix=K,
        run_dir=run_dir,
        epoch=epoch,
        trucksc=ts
    )
