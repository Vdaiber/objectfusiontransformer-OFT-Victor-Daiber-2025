# epoch_visualization_workflow.py
import os
import numpy as np
import cv2
from pyquaternion import Quaternion
from truckscenes.utils.data_classes import Box as DevkitBox
from truckscenes.utils.geometry_utils import BoxVisibility
from oft.transformer.datasets.truckscenes.transforms import transform_box_vehicle_to_world
from truckscenes.utils.geometry_utils import transform_matrix
from oft.transformer.utils.epoch_visualization import visualize_epoch_sample
from oft.transformer.utils.normalization_utils import denormalize_coordinates, denormalize_dimensions, get_global_normalizer
from oft.transformer.utils.geometry_utils import sin_cos_to_yaw
import torch
from typing import List


def _filter_boxes_in_camera_fov(boxes: List[DevkitBox], K: np.ndarray, img_shape) -> List[DevkitBox]:
    """Filtert Boxen, die im Bildbereich der Kamera sichtbar sind.

    Eine Box gilt als sichtbar, wenn (a) alle Ecken positive Tiefenwerte (z>0) besitzen
    UND (b) mindestens eine Ecke nach Projektion innerhalb der Bildgrenzen liegt.
    """
    h, w = img_shape[:2]
    visible_boxes = []
    for box in boxes:
        # Eckpunkte der Box in Kamera­koordinaten (3×8)
        corners_cam = box.corners()
        # Tiefenprüfung – alle Punkte vor der Kamera
        if np.any(corners_cam[2, :] <= 0):
            continue
        # Projektion ins Bild
        homog_pts = K @ corners_cam  # 3×8
        u = homog_pts[0, :] / homog_pts[2, :]
        v = homog_pts[1, :] / homog_pts[2, :]
        # Sichtbarkeitsbedingung: Mindestens ein Punkt im Bildrahmen
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.any(in_img):
            visible_boxes.append(box)
    return visible_boxes


def visualize_val_sample_from_dataset(cfg, val_dataset, epoch, run_dir, trucksc=None, predicted_boxes_normalized=None):
    """
    Visualisiert ein Val-Sample nach DevKit-Standard:
    - GT-Boxen: DevKit's get_sample_data() für sichtbare Boxen in Welt-Koordinaten
    - Predicted-Boxen: Denormalisieren, Ego→Welt, Mapping zu DevKit-Klassenname
    - render_box_cv2 erwartet Welt-Koordinaten und macht Transformation intern
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
        # Fallback: Use first sample in dataset
        sample = val_dataset[0]
        sample_token = sample["sample_token"]
        print(f"[VIS FALLBACK] Using first sample: {sample_token}")

    # 2. Lade das Kamerabild und die Kameramatrix
    ts = val_dataset.ts if hasattr(val_dataset, 'ts') and val_dataset.ts is not None else trucksc
    
    # Fallback: Load DevKit if needed for visualization
    if ts is None:
        from truckscenes import TruckScenes
        ts = TruckScenes(version=val_dataset.version, dataroot=val_dataset.dataroot, verbose=False)
        print(f"[VIS DEBUG] Loaded DevKit for visualization (RAM mode fallback)")
    
    try:
        sample_record = ts.get('sample', sample_token)
        print(f"[VIS DEBUG] Successfully loaded sample_record from DevKit")
    except Exception as e:
        print(f"[VIS ERROR] Failed to get sample_record from DevKit: {e}")
        print(f"[VIS DEBUG] sample_token: {sample_token}")
        print(f"[VIS DEBUG] DevKit version: {ts.version}")
        raise e
    
    # Verify timestamps are consistent
    dataset_timestamp = sample.get('timestamp')  # Handle potential None case
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

    # 3. GT-Boxen: DevKit-Standard - nur sichtbare Boxen für diese Kamera
    print(f'=== GT-Boxen laden (DevKit-Standard) ===')
    print(f'Sample Token: {sample_token}')
    print(f'Kamera Channel: {camera_channel}')
    print(f'Sample Data Token: {cam_token}')
    
    # DevKit's get_sample_data() gibt Boxen in Kamera-Koordinaten zurück
    # render_box_cv2 erwartet Kamera-Koordinaten, nicht Welt-Koordinaten!
    _, gt_boxes_cam, _ = ts.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    print(f'DevKit get_sample_data() gab {len(gt_boxes_cam)} sichtbare Boxen zurück')
    
    # Verwende die Boxen direkt in Kamera-Koordinaten (render_box_cv2 erwartet das!)
    gt_boxes = gt_boxes_cam
    
    print(f'Verwende {len(gt_boxes)} GT-Boxen in Kamera-Koordinaten')

    # 4. Predicted-Boxen laden und verarbeiten (NUR aus DevKit-Evaluation)
    fused_boxes = []
    
    # Lade Predicted-Boxen aus DevKit-Evaluation (Welt-Koordinaten)
    import json
    
    devkit_pred_path = os.path.join(run_dir, 'eval_results', f'epoch_{epoch}', 'predictions.json')
    if os.path.exists(devkit_pred_path):
        print(f'=== Lade Predicted-Boxen aus DevKit-Evaluation: {devkit_pred_path} ===')
        with open(devkit_pred_path, 'r') as f:
            devkit_predictions = json.load(f)
        
        # Finde Predictions für dieses Sample
        sample_predictions = devkit_predictions['results'].get(sample_token, [])

        
        if sample_predictions:
            # FIXED: Verwende KONSISTENTE Timestamp-basierte Ego-Pose Auswahl
            # Problem: sd_record['ego_pose_token'] kann unterschiedliche Timestamp haben
            # Lösung: Verwende die gleiche Timestamp-basierte Methode wie im Dataset
            
            # Dataset verwendet: ts.getclosest('ego_pose', sample_record['timestamp'])
            # Wir müssen das GLEICHE verwenden für konsistente Transformation
            pose_record = ts.getclosest('ego_pose', sample_record['timestamp'])
            
            
            for i, pred in enumerate(sample_predictions):
                center_world = np.array(pred['translation'])
                size_world = np.array(pred['size'])
                rotation_world_quat = Quaternion(pred['rotation'])
                
                # Erstelle DevBox in Weltkoordinaten (wie DevKit es macht)
                world_box = DevkitBox(
                    center=center_world,
                    size=size_world,
                    orientation=rotation_world_quat,
                    name=pred['detection_name'],
                    score=pred['detection_score'],
                    token=f'pred_{i}'
                )
                
                # SYNCHRONISIERTE DEVKIT-TRANSFORMATION: World → Ego → Camera
                # Schritt 1: World → Ego (mit KONSISTENTER Ego-Pose)
                ego_box = world_box.copy()
                ego_box.translate(-np.array(pose_record['translation']))
                ego_box.rotate(Quaternion(pose_record['rotation']).inverse)
                
                # Schritt 2: Ego → Camera (DevKit Standard)
                camera_box = ego_box.copy()
                camera_box.translate(-np.array(cs_record['translation']))
                camera_box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                fused_boxes.append(camera_box)
                
            print(f'Erstellte {len(fused_boxes)} Predicted-Boxen in Kamera-Koordinaten (DevKit-Standard)')

            # --- EINHEITLICHER FOV-FILTER ---
            fused_boxes_before = len(fused_boxes)
            fused_boxes = _filter_boxes_in_camera_fov(fused_boxes, K, image.shape)
            print(f'FOV-Filter: {fused_boxes_before} → {len(fused_boxes)} sichtbare Predicted-Boxen im Bildbereich')
    else:
        print(f'WARNUNG: Keine DevKit-Evaluation gefunden: {devkit_pred_path}')
        print('Keine Predicted-Boxen verfügbar für Visualisierung')

    # 5. Visualisierung aufrufen (Boxen sind jetzt in Kamera-Koordinaten)
    visualize_epoch_sample(
        image=image,
        fused_boxes=fused_boxes,
        gt_boxes=gt_boxes,
        camera_k_matrix=K,
        run_dir=run_dir,
        epoch=epoch,
        trucksc=ts
    ) 