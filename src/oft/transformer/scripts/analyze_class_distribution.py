# scripts/analyze_class_distribution.py
"""
Dieses Skript analysiert die Verteilung der Objektklassen in einem
gegebenen Datensatz-Split der "Staged Fusion"-Architektur.

Es lädt die Konfiguration, initialisiert das Dataset und zählt die
Anzahl der Ground-Truth-Instanzen für jede definierte Klasse.

Ausführung aus dem /app Verzeichnis:
python3 src/oft/transformer/scripts/analyze_class_distribution.py
"""

import hydra
from omegaconf import DictConfig
import os
import sys
from collections import Counter
from tqdm import tqdm

# Füge das 'src'-Verzeichnis des Projekts zum Python-Pfad hinzu
# Passt den Pfad an, da das Skript jetzt in einem Unterverzeichnis liegt
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from oft.transformer.datasets.truckscenes.dataset_staged import ObjectFusionGTDatasetStaged

def analyze_distribution(cfg: DictConfig):
    """
    Instanziiert das Dataset und zählt die Klasseninstanzen.
    """
    print("--- Initializing Staged Dataset for Class Distribution Analysis ---")
    dataset_cfg = cfg.dataset
    train_split = cfg.training.train_split_name
    class_names = list(dataset_cfg.class_names)
    print(f"Analyzing split: '{train_split}'")

    dataset = ObjectFusionGTDatasetStaged(
        dataroot=dataset_cfg.dataroot,
        version=dataset_cfg.version,
        split_name=train_split,
        pipeline_config=cfg,
        verbose=False 
    )
    print(f"Dataset initialized with {len(dataset)} samples.")
    print("-" * 50)

    class_counter = Counter()

    print("--- Iterating over dataset to count ground truth instances ---")
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        sequence_data = dataset[i]
        # Wir zählen nur die GT-Objekte im letzten (aktuellen) Frame der Sequenz
        sample_data = sequence_data[-1] 
        
        gt_targets = sample_data.get('gt_detections_actual_dims', [])
        for target in gt_targets:
            class_idx = target.get('class_idx')
            if class_idx is not None and 0 <= class_idx < len(class_names):
                class_name = class_names[class_idx]
                class_counter[class_name] += 1
    
    print("\n" + "="*35)
    print("--- Final Class Distribution ---")
    print(f"{'Object Class':<20} | {'Instance Count'}")
    print("-" * 35)
    
    total_instances = sum(class_counter.values())
    
    sorted_classes = sorted(class_counter.items(), key=lambda item: item[1], reverse=True)
    
    for class_name, count in sorted_classes:
        percentage = (count / total_instances) * 100 if total_instances > 0 else 0
        print(f"{class_name:<20} | {count:<5} ({percentage:.2f}%)")
        
    print("-" * 35)
    print(f"{'Total Instances':<20} | {total_instances}")
    print("="*35 + "\n")
    print("✅ Analysis complete.")


@hydra.main(config_path="../../../../config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    analyze_distribution(cfg)

if __name__ == '__main__':
    # Diese Zeile ist notwendig, damit das Skript aus dem /app-Verzeichnis mit dem
    # relativen Pfad zur Konfiguration korrekt ausgeführt werden kann.
    # Es ändert temporär das Arbeitsverzeichnis zum Root des Projekts.
    os.chdir(project_root)
    main()
    