# Transformer Fusion Dataset - Technische Funktions- und Klassenübersicht

## 📊 Strukturübersicht

```
/app/src/oft/transformer/datasets/
├── __init__.py                                  # Haupt-Modul-Exporte
├── truckscenes/                                 # TruckScenes Dataset-spezifische Module
│   ├── __init__.py                              # Submodul-Exporte  
│   ├── dataset.py                               # Haupt-Dataset-Klasse
│   └── transforms.py                            # Koordinatensystem-Transformationen
├── preprocessing/                               # Datenvorverarbeitung
│   ├── __init__.py                              # Submodul-Exporte
│   ├── data_augmentation.py                     # Sensorsimulation und Rausch-Modelle
│   └── collate_functions.py                     # Batch-Kollation für Transformer
└── loaders/                                     # DataLoader-Konstruktoren
    ├── __init__.py                              # Submodul-Exporte
    └── autoregressive_loader.py                 # Autoregressive DataLoader
```

**Importbeziehungen:**
- `dataset.py` → `transforms.py`, `data_augmentation.py`, `normalization_utils.py`, `geometry_utils.py`
- `autoregressive_loader.py` → `dataset.py`, `collate_functions.py`
- `data_augmentation.py` → YAML config files (scene conditioning)

---

## 🏗️ Hauptkomponenten

### Class: `ObjectFusionGTDatasetStaged`
- **Type:** class
- **File:** `/app/src/oft/transformer/datasets/truckscenes/dataset.py:L45–716`
- **Inputs:** `dataroot: str, version: str, split_name: str, pipeline_config: Dict[str, Any], verbose: bool = True`
- **Outputs:** PyTorch Dataset object
- **Config Dependencies:** 
  - `dataset.virtual_sensors` → Sensorkonfiguration
  - `dataset.simulation` → False Positive/Negative Simulation
  - `dataset.scene_conditioning` → Umgebungsabhängige Rauschmodellierung
  - `dataset.normalization_stats_path` → Normalisierungsstatistiken
- **Calls:** 
  - `get_virtual_sensor_cache()` @ virtual_sensor_cache.py
  - `load_scene_conditioning_config()` @ data_augmentation.py:L20–42
  - `parse_scene_description()` @ geometry_utils.py
- **Description:** Hauptdataset für gestuftes Sensorfusion-Training. Simuliert virtuelle Sensoren (LiDAR, Kamera, Radar) mit realistischen Fehlern und verwaltet temporale Sequenzen mit Cache-System.
- **Pipeline Usage:** Zentraler Datenlieferant für das autoregressive Transformer-Training mit Multi-Modal-Sensordaten.

---

## 🔄 Koordinatentransformationen (transforms.py)

### Function: `transform_box_vehicle_to_world`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/truckscenes/transforms.py:L16–58`
- **Inputs:** `box_ego_phys_7d, ego_translation_world, ego_rotation_world`
- **Outputs:** `np.ndarray` (7D Box in world coordinates)
- **Config Dependencies:** None
- **Calls:** PyQuaternion operations
- **Description:** Transformiert 7D Bounding Box von Ego-Fahrzeug zu Weltkoordinaten mit Quaternion-Rotationen.
- **Pipeline Usage:** Konvertierung für Evaluationsmetriken und Visualisierung.

### Function: `transform_world_to_ego_frame`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/truckscenes/transforms.py:L61–92`
- **Inputs:** `box_world_phys_7d: np.ndarray, ego_translation: np.ndarray, ego_rotation: PyQuaternion`
- **Outputs:** `np.ndarray` (7D Box in ego coordinates)
- **Config Dependencies:** None
- **Calls:** PyQuaternion inverse operations
- **Description:** Transformiert Bounding Boxes von Welt- zu Ego-Koordinaten mit nur Yaw-Komponente zur Vermeidung von Roll/Pitch-Kontamination.
- **Pipeline Usage:** Haupttransformation für alle Sensor- und GT-Daten in Modell-Pipeline.

### Function: `transform_velocity_world_to_ego_frame`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/truckscenes/transforms.py:L94–116`
- **Inputs:** `object_velocity_world_phys_2d: np.ndarray, ego_rotation: PyQuaternion, ego_velocity_world_phys_2d: np.ndarray`
- **Outputs:** `np.ndarray` (2D relative velocity in ego frame)
- **Config Dependencies:** None
- **Calls:** PyQuaternion rotation operations
- **Description:** Berechnet relative Geschwindigkeit zwischen Objekt und Ego-Fahrzeug im Ego-Koordinatensystem.
- **Pipeline Usage:** Verarbeitung von Geschwindigkeitsdaten für Motion Prediction.

### Function: `transform_relative_velocity_ego_to_world_frame`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/truckscenes/transforms.py:L118–140`
- **Inputs:** `relative_velocity_ego_2d: np.ndarray, ego_rotation: PyQuaternion, ego_velocity_world_phys_2d: np.ndarray`
- **Outputs:** `np.ndarray` (2D absolute velocity in world frame)
- **Config Dependencies:** None
- **Calls:** PyQuaternion rotation operations
- **Description:** Konvertiert relative Geschwindigkeit zurück zu absoluter Weltgeschwindigkeit.
- **Pipeline Usage:** Rekonstruktion für Evaluationsausgaben.

---

## 🎭 Datenaugmentierung (data_augmentation.py)

### Function: `load_scene_conditioning_config`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/data_augmentation.py:L20–42`
- **Inputs:** `config_path: str`
- **Outputs:** `Dict[str, Any]`
- **Config Dependencies:** `scene_conditioning_config.yaml`
- **Calls:** `yaml.safe_load()`
- **Description:** Lädt YAML-Konfiguration für umgebungsabhängige Rauschmodellierung.
- **Pipeline Usage:** Initialisierung der szenenabhängigen Sensorsimulation.

### Function: `calculate_scene_dependent_factor`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/data_augmentation.py:L44–116`
- **Inputs:** `sensor_name: str, scene_meta: Dict[str, str], scene_conditioning_cfg: Dict[str, Any]`
- **Outputs:** `float` (multiplicative noise factor)
- **Config Dependencies:** `scene_conditioning_cfg.scene_conditional_noise`
- **Calls:** None
- **Description:** Berechnet kombinierten Umgebungsfaktor basierend auf Wetter, Beleuchtung, Gebiet, etc.
- **Pipeline Usage:** Szenenabhängige Verstärkung der Sensor-Rauschparameter.

### Function: `enhance_noise_parameters`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/data_augmentation.py:L118–166`
- **Inputs:** `sensor_cfg: Dict[str, Any], scene_meta: Dict[str, str], scene_conditioning_cfg: Dict[str, Any]`
- **Outputs:** `Dict[str, Any]` (enhanced sensor config)
- **Config Dependencies:** `scene_conditioning_cfg.component_sensitivity`
- **Calls:** `calculate_scene_dependent_factor()`
- **Description:** Verstärkt Sensor-Rauschparameter basierend auf Szenenbedingungen und komponentenspezifischer Sensitivität.
- **Pipeline Usage:** Dynamische Anpassung der Sensorfehler-Simulation.

### Function: `add_noise_to_box`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/data_augmentation.py:L244–312`
- **Inputs:** `sensor_box_ego_phys_7d: np.ndarray, sensor_velocity_ego_phys_2d: np.ndarray, sensor_noise_cfg: Dict[str, Any], scene_meta: Dict[str, str] = None, scene_conditioning_cfg: Dict[str, Any] = None, sample_token: str = None, global_seed: int = 42`
- **Outputs:** `Tuple[np.ndarray, np.ndarray]` (9D noisy box, 2D noisy velocity)
- **Config Dependencies:** 
  - `sensor_cfg.pos_noise_std`
  - `sensor_cfg.dim_noise_std`
  - `sensor_cfg.yaw_noise_std`
  - `sensor_cfg.velocity_noise_std`
- **Calls:** `enhance_noise_parameters()`
- **Description:** Fügt physikbasiertes Gaußsches Rauschen zu Bounding Box und Geschwindigkeitsmessungen hinzu mit deterministischer Sample-spezifischer Saat.
- **Pipeline Usage:** Hauptfunktion für realistische Sensorfehler-Simulation.

### Function: `simulate_false_negatives`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/data_augmentation.py:L314–336`
- **Inputs:** `gt_detections_raw_world: List[Dict[str, Any]], fn_rate: float, sample_token: str = None, global_seed: int = 42`
- **Outputs:** `List[Dict[str, Any]]` (filtered GT detections)
- **Config Dependencies:** `simulation.fn_rate`
- **Calls:** None
- **Description:** Simuliert Fehldetektionen durch zufälliges Entfernen von Ground Truth Objekten.
- **Pipeline Usage:** Training-Robustheit gegen verpasste Detektionen.

### Function: `simulate_false_positives`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/data_augmentation.py:L338–401`
- **Inputs:** `is_training: bool, simulation_cfg: Dict[str, Any], sample_token: str = None, global_seed: int = 42`
- **Outputs:** `List[Dict[str, Any]]` (synthetic false positive detections)
- **Config Dependencies:** 
  - `simulation.num_fps`
  - `simulation.fp_pos_range`
  - `simulation.fp_dim_range`
  - `simulation.fp_velocity_range`
- **Calls:** None
- **Description:** Generiert synthetische Fehlerkennungen mit `class_idx=-1` zur Robustheitssteigerung.
- **Pipeline Usage:** Simulation von Sensor-Artefakten und Störungen.

### Function: `add_class_noise`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/data_augmentation.py:L402–507`
- **Inputs:** `gt_class_idx: int, sensor_cfg: Dict[str, Any], class_names: List[str], scene_meta: Dict[str, str] = None, scene_conditioning_cfg: Dict[str, Any] = None`
- **Outputs:** `int` (sensor class index with errors)
- **Config Dependencies:** `sensor_cfg.class_accuracy`
- **Calls:** `enhance_classification_accuracy()`
- **Description:** Simuliert realistische Klassifikationsfehler mit sensorspezifischen Verwirrungsmatrizen (Radar: Größen-/Geschwindigkeitsverwirrung, LiDAR: Form-Verwirrung, Kamera: visuelle Ähnlichkeit).
- **Pipeline Usage:** Training für robuste Klassifikation unter Sensorfehlern.

### Function: `add_attribute_noise`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/data_augmentation.py:L509–601`
- **Inputs:** `gt_attribute_idx: int, sensor_cfg: Dict[str, Any], attribute_vocab: List[str], scene_meta: Dict[str, str] = None, scene_conditioning_cfg: Dict[str, Any] = None`
- **Outputs:** `int` (sensor attribute index with errors)
- **Config Dependencies:** `sensor_cfg.attribute_accuracy`
- **Calls:** `enhance_classification_accuracy()`
- **Description:** Simuliert Attributfehler mit sensorspezifischen Charakteristika (Radar: Geschwindigkeits-gut, Zustand-schlecht; Kamera: visuelle Zustände gut; LiDAR: Geometrie gut, Motion schlecht).
- **Pipeline Usage:** Training für robuste Attributvorhersage.

---

## 🧩 Batch-Kollation (collate_functions.py)

### Function: `object_fusion_gt_collate_fn_autoregressive`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/preprocessing/collate_functions.py:L15–208`
- **Inputs:** `batch_of_frames: List[Dict[str, Any]]`
- **Outputs:** `Dict[str, Any]` (collated batch tensors)
- **Config Dependencies:** None
- **Calls:** `torch.from_numpy()`, `torch.stack()`
- **Description:** Kollationiert variable Anzahl von Sensordetektionen zu paddierten Tensoren mit einheitlichen Dimensionen für Batch-Verarbeitung.
- **Pipeline Usage:** Umwandlung einzelner Frames zu Batch-Tensoren für Transformer-Training.

---

## 🚀 DataLoader (autoregressive_loader.py)

### Function: `build_autoregressive_dataloaders`
- **Type:** function
- **File:** `/app/src/oft/transformer/datasets/loaders/autoregressive_loader.py:L19–129`
- **Inputs:** `cfg: Dict[str, Any], logger: logging.Logger, splits_to_build: List[str] = ['train', 'val']`
- **Outputs:** `Dict[str, DataLoader]`
- **Config Dependencies:** 
  - `dataset.dataroot`
  - `dataset.version`
  - `training.batch_size`
  - `training.num_workers`
  - `training.train_split_name`
  - `training.val_split_name`
  - `training.sampler` → "sequential", "random", "auto"
- **Calls:** 
  - `ObjectFusionGTDatasetStaged()`
  - `object_fusion_gt_collate_fn_autoregressive()`
- **Description:** Konstruiert DataLoaders mit konfigurierbarer Sampling-Strategie (Sequential für temporale Konsistenz, Random für Klassendiversität).
- **Pipeline Usage:** Zentrale DataLoader-Erstellung für Training und Validation.

---

## 🎯 Dataset-Methoden (dataset.py)

### Method: `__init__`
- **Type:** method
- **File:** `/app/src/oft/transformer/datasets/truckscenes/dataset.py:L54–196`
- **Inputs:** Constructor parameters
- **Outputs:** None
- **Config Dependencies:** Vollständige pipeline_config
- **Calls:** 
  - `create_splits_scenes()` @ TruckScenes devkit
  - `get_virtual_sensor_cache()`
  - `load_scene_conditioning_config()`
- **Description:** Initialisiert Dataset mit TruckScenes-Anbindung, virtueller Sensor-Cache und Szenen-Konditionierung.
- **Pipeline Usage:** Setup des kompletten Dataset-Systems.

### Method: `__getitem__`
- **Type:** method
- **File:** `/app/src/oft/transformer/datasets/truckscenes/dataset.py:L205–231`
- **Inputs:** `idx: int`
- **Outputs:** `Dict[str, Any]` (complete sample data)
- **Config Dependencies:** None
- **Calls:** `_get_single_sample_data()`
- **Description:** Liefert verarbeitete Einzelframe-Daten für den gegeben Index mit Ego-Koordinatentransformation.
- **Pipeline Usage:** DataLoader-Interface für Batch-Generierung.

### Method: `_get_single_sample_data`
- **Type:** method
- **File:** `/app/src/oft/transformer/datasets/truckscenes/dataset.py:L232–584`
- **Inputs:** `sample_token: str, ref_ego_translation: np.ndarray, ref_ego_rotation: PyQuaternion`
- **Outputs:** `Dict[str, Any]` (processed sample with all augmentations)
- **Config Dependencies:** Alle Sensor- und Simulation-Configs
- **Calls:** 
  - `transform_world_to_ego_frame()` @ transforms.py
  - `simulate_false_negatives()` @ data_augmentation.py
  - `simulate_false_positives()` @ data_augmentation.py
  - `add_noise_to_box()` @ data_augmentation.py
  - Cache-Funktionen
- **Description:** Verarbeitet einen Complete Sample mit Cache-Prüfung, GT-Transformation, Sensorsimulation und Normalisierung. Kern der Datenverarbeitung.
- **Pipeline Usage:** Zentrale Verarbeitungslogik mit allen Augmentierungen.

### Method: `_get_full_box_data_for_token_world`
- **Type:** method
- **File:** `/app/src/oft/transformer/datasets/truckscenes/dataset.py:L649–716`
- **Inputs:** `sample_token: str`
- **Outputs:** `List[Dict[str, Any]]` (GT detections in world frame)
- **Config Dependencies:** `dataset.class_names`
- **Calls:** 
  - `category_to_detection_name()` @ TruckScenes devkit
  - `_calculate_velocity_from_temporal_tracking()`
- **Description:** Extrahiert Ground Truth Annotations aus TruckScenes mit Geschwindigkeitsberechnung und Klassfilterung.
- **Pipeline Usage:** GT-Datengewinnung für Training und Evaluation.

### Method: `_calculate_velocity_from_temporal_tracking`
- **Type:** method
- **File:** `/app/src/oft/transformer/datasets/truckscenes/dataset.py:L586–648`
- **Inputs:** `ann_record: Dict[str, Any]`
- **Outputs:** `np.ndarray` (2D velocity vector in world frame)
- **Config Dependencies:** None
- **Calls:** TruckScenes devkit annotation API
- **Description:** Berechnet Objektgeschwindigkeit aus temporalen Annotationen mit Vor-/Rückwärts-Differenzierung.
- **Pipeline Usage:** Geschwindigkeitsdaten für Motion Prediction.

---

## 📊 Globale Konstanten

### Constant: `ATTRIBUTE_VOCAB`
- **Type:** constant
- **File:** `/app/src/oft/transformer/datasets/truckscenes/dataset.py:L37–43`
- **Value:** List of 11 TruckScenes attribute names
- **Description:** Vollständiges Attributvokabular für Objektzustände (moving, parked, etc.).
- **Pipeline Usage:** Attribut-Encoding und -Validation.

---

## 🔗 Externe Abhängigkeiten

**Hauptabhängigkeiten:**
- `TruckScenes` devkit → Dataset-API und Evaluation
- `oft.transformer.utils.normalization_utils` → Zentrale Normalisierung
- `oft.transformer.utils.geometry_utils` → Geometrische Hilfsfunktionen
- `oft.transformer.utils.virtual_sensor_cache` → Performance-Cache
- `PyQuaternion` → Quaternion-Operationen
- `torch`, `numpy` → Tensor-Operationen

**Config-Abhängigkeiten:**
- `config/pipeline_staged.yaml` → Hauptkonfiguration
- `config/normalization_stats.yaml` → Normalisierungsparameter  
- `config/scene_conditioning_config.yaml` → Umgebungsabhängige Modellierung

---

## 💼 Pipeline-Integration

**Datenfluss:**
1. **`ObjectFusionGTDatasetStaged`** → Lädt und verarbeitet TruckScenes-Daten
2. **`transforms.py`** → Koordinatensystem-Konvertierungen (World ↔ Ego)
3. **`data_augmentation.py`** → Realistische Sensorfehler-Simulation
4. **`collate_functions.py`** → Batch-Kollation mit dynamischem Padding
5. **`autoregressive_loader.py`** → DataLoader-Konstruktion mit Sampling-Strategien

**Verwendung:** Zentrale Datenversorgungs-Pipeline für das autoregressive Transformer-Training mit Multi-Modal-Sensorfusion (LiDAR, Kamera, Radar) und realistischer Fehlermodellierung.