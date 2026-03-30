# scripts/calculate_norm_stats.py
# Berechnet die Min/Max-Koordinatenstatistiken UND die Standardabweichung der Box-Offsets
# für den (geclippten) Trainings-Datensatz. Angepasst an die "Staged Fusion"-Architektur.

import hydra
from omegaconf import DictConfig
import numpy as np
import os
import sys
import json
from tqdm import tqdm
from pyquaternion import Quaternion as PyQuaternion
import math

# Füge das 'src'-Verzeichnis des Projekts zum Python-Pfad hinzu
project_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_src_root not in sys.path:
    sys.path.insert(0, project_src_root)

# Korrekter Import für die "Staged"-Architektur
from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged

def calculate_and_save_stats(cfg: DictConfig):
    """
    Instanziiert das Staged-Dataset, iteriert durch alle Samples und berechnet:
    1. Min/Max-Normalisierungsstatistiken für die Fahrzeug-Koordinaten (x, y, z).
    2. Die Standardabweichung der Box-Offsets [dx, dy, dz, d(log w), d(log l), d(log h), d(yaw)].
    """
    print("--- Initializing Staged Dataset for statistics calculation ---")
    train_split = cfg.training.train_split_name
    print(f"Using split: '{train_split}'")

    dataset = ObjectFusionGTDatasetStaged(
        dataroot=cfg.dataset.dataroot,
        version=cfg.dataset.version,
        split_name=train_split,
        pipeline_config=cfg,
        verbose=False # Weniger Output für die Berechnung
    )
    print(f"Dataset initialized with {len(dataset)} samples.")
    print(f"Clipping distance is set to: {dataset.cutoff_dist}m")
    print("-" * 50)

    all_initial_x, all_initial_y, all_initial_z = [], [], []
    all_offsets = []

    print("--- Iterating over dataset to collect coordinates and offsets ---")
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        # Das Dataset gibt eine Sequenz zurück, wir benötigen nur den letzten (aktuellen) Frame
        sample_data = dataset[i]
        print("DEBUG sample_data.keys():", sample_data.keys())
        if 'sensor_data' not in sample_data:
            print("FEHLER: sample_data hat keinen 'sensor_data'-Key! Sample wird übersprungen.")
            continue
        # 1. Sammle Koordinaten von allen initialen (verrauschten) Boxen für Min/Max
        for sensor_name, data in sample_data['sensor_data'].items():
            sensor_detection_boxes = data['sensor_detection_boxes']
            if sensor_detection_boxes.shape[0] > 0:
                all_initial_x.extend(sensor_detection_boxes[:, 0])
                all_initial_y.extend(sensor_detection_boxes[:, 1])
                all_initial_z.extend(sensor_detection_boxes[:, 2])

        # 2. Rekonstruiere die Paare aus (initial_box, gt_box) um die Offsets zu berechnen
        gt_targets_for_sample = sample_data['ground_truth_boxes']
        
        # Erstelle eine Map von GTs, um sie den Sensor-Detections zuzuordnen.
        # Wir gehen davon aus, dass die Reihenfolge innerhalb eines Sensors erhalten bleibt.
        for sensor_cfg in dataset.virtual_sensors_cfg:
            sensor_name = sensor_cfg['name']
            sensor_detection_boxes_for_sensor = sample_data['sensor_data'][sensor_name]['sensor_detection_boxes']

            # Annahme: Dropout ist für die Stats-Berechnung nicht aktiv, daher 1-zu-1-Mapping.
            # Wenn GTs und Initial-Boxen nicht übereinstimmen, überspringen wir, um Fehler zu vermeiden.
            if len(sensor_detection_boxes_for_sensor) != len(gt_targets_for_sample):
                continue
            
            for sensor_detection_box, gt_target in zip(sensor_detection_boxes_for_sensor, gt_targets_for_sample):
                gt_box = gt_target['box_7d']
                
                # Berechne den Offset im realen Raum
                offset_center = gt_box[:3] - sensor_detection_box[:3]
                # Log-Differenz für Dimensionen
                offset_dims = np.log(np.maximum(gt_box[3:6], 1e-5)) - np.log(np.maximum(sensor_detection_box[3:6], 1e-5))
                # Winkeldifferenz
                offset_yaw = gt_box[6] - sensor_detection_box[6]
                offset_yaw = (offset_yaw + math.pi) % (2 * math.pi) - math.pi

                offset_7d = np.concatenate([offset_center, offset_dims, [offset_yaw]])
                all_offsets.append(offset_7d)

    if not all_initial_x:
        print("Warning: No initial boxes found. Cannot calculate coordinate stats.")
        coord_stats = {}
    else:
        coord_stats = {
            "x_min": float(np.min(all_initial_x)), "x_max": float(np.max(all_initial_x)),
            "y_min": float(np.min(all_initial_y)), "y_max": float(np.max(all_initial_y)),
            "z_min": float(np.min(all_initial_z)), "z_max": float(np.max(all_initial_z)),
        }

    if not all_offsets:
        print("Warning: No offsets calculated. Cannot calculate offset stats.")
        offset_stats = {"box_std": [1.0] * 7} # Fallback
    else:
        all_offsets_np = np.array(all_offsets)
        box_std_devs = np.std(all_offsets_np, axis=0).tolist()
        offset_stats = {"box_std": box_std_devs}

    # Kombiniere beide Statistik-Typen
    final_stats = {**coord_stats, **offset_stats}
    
    print("\n" + "="*25)
    print("--- Calculated Final Statistics ---")
    print(json.dumps(final_stats, indent=4))
    print("="*25 + "\n")

    # Speichere die Statistiken im Hydra-Output-Verzeichnis
    output_filename = "norm_stats.json" 
    output_path = os.path.join(os.getcwd(), output_filename) 

    with open(output_path, 'w') as f:
        json.dump(final_stats, f, indent=4)
        
    print(f"✅ Successfully saved final normalization stats to: {output_path}")


@hydra.main(config_path="../../../../config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    print("--- Configuration for Stats Calculation Loaded ---")
    # Stelle sicher, dass die Konfiguration für die Berechnung korrekt ist (z.B. kein Dropout)
    for sensor in cfg.dataset.virtual_sensors:
        if 'dropout_rate' in sensor:
            print(f"Temporarily setting dropout_rate for '{sensor.name}' to 0.0 for stats calculation.")
            sensor.dropout_rate = 0.0
            
    calculate_and_save_stats(cfg)

if __name__ == '__main__':
    main()