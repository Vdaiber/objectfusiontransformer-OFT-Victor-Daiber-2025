# src/oft/transformer/select_samples.py
#!/usr/bin/env python3
"""
Hilfsskript zur intelligenten Auswahl von Samples basierend auf verschiedenen Kriterien.
"""
import json
import os
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# KORREKTER IMPORT aus der neuen modularen Struktur
from .oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDataset
from oft.transformer.utils.geometry_utils import parse_scene_description

@dataclass
class SceneInfo:
    name: str
    description: str
    token: str
    first_sample_token: str
    last_sample_token: str
    nbr_samples: int
    weather: str
    area: str
    daytime: str
    season: str
    lighting: str
    structure: str
    construction: str



def load_scenes(dataroot: str) -> List[SceneInfo]:
    """Load and parse scene information."""
    scene_file = Path(dataroot) / "v1.0-mini" / "scene.json"
    with open(scene_file, 'r') as f:
        scenes_data = json.load(f)
    
    scenes = []
    for scene in scenes_data:
        desc_dict = parse_scene_description(scene['description'])
        scenes.append(SceneInfo(
            name=scene['name'],
            description=scene['description'],
            token=scene['token'],
            first_sample_token=scene['first_sample_token'],
            last_sample_token=scene['last_sample_token'],
            nbr_samples=scene['nbr_samples'],
            weather=desc_dict.get('weather', ''),
            area=desc_dict.get('area', ''),
            daytime=desc_dict.get('daytime', ''),
            season=desc_dict.get('season', ''),
            lighting=desc_dict.get('lighting', ''),
            structure=desc_dict.get('structure', ''),
            construction=desc_dict.get('construction', '')
        ))
    return scenes

def find_samples_by_criteria(scenes: List[SceneInfo], 
                           weather: str = None,
                           area: str = None,
                           daytime: str = None,
                           season: str = None,
                           lighting: str = None,
                           structure: str = None,
                           construction: str = None) -> List[SceneInfo]:
    """Find scenes matching the given criteria."""
    matching_scenes = []
    for scene in scenes:
        matches = True
        if weather and scene.weather != weather:
            matches = False
        if area and scene.area != area:
            matches = False
        if daytime and scene.daytime != daytime:
            matches = False
        if season and scene.season != season:
            matches = False
        if lighting and scene.lighting != lighting:
            matches = False
        if structure and scene.structure != structure:
            matches = False
        if construction and scene.construction != construction:
            matches = False
        if matches:
            matching_scenes.append(scene)
    return matching_scenes

def get_category_options(scenes: List[SceneInfo]) -> Dict[str, List[str]]:
    """Get all available options for each category."""
    return {
        "Wetter": sorted(set(s.weather for s in scenes)),
        "Bereich": sorted(set(s.area for s in scenes)),
        "Tageszeit": sorted(set(s.daytime for s in scenes)),
        "Jahreszeit": sorted(set(s.season for s in scenes)),
        "Beleuchtung": sorted(set(s.lighting for s in scenes)),
        "Struktur": sorted(set(s.structure for s in scenes)),
        "Baustelle": sorted(set(s.construction for s in scenes))
    }

def get_category_mapping() -> Dict[str, str]:
    """Map German category names to English attribute names."""
    return {
        "wetter": "weather",
        "bereich": "area",
        "tageszeit": "daytime",
        "jahreszeit": "season",
        "beleuchtung": "lighting",
        "struktur": "structure",
        "baustelle": "construction"
    }

def get_similar_scenes(scenes: List[SceneInfo], criteria: Dict[str, str]) -> List[SceneInfo]:
    """Find scenes that match some of the criteria."""
    similar_scenes = []
    category_mapping = get_category_mapping()
    
    for scene in scenes:
        match_count = 0
        total_criteria = 0
        for category, value in criteria.items():
            if value:  # Only count non-None criteria
                total_criteria += 1
                english_category = category_mapping.get(category.lower())
                if english_category and getattr(scene, english_category) == value:
                    match_count += 1
        if match_count > 0:  # Include scenes that match at least one criterion
            similar_scenes.append((scene, match_count/total_criteria))
    
    # Sort by match percentage
    similar_scenes.sort(key=lambda x: x[1], reverse=True)
    return [scene for scene, _ in similar_scenes]

def get_scene_split_indices(vis_dataset, scene_token):
    """Returns the start and end indices in the split for a given scene_token."""
    indices = [i for i, s in enumerate(vis_dataset.sample_tokens) if vis_dataset.ts.get('sample', s)['scene_token'] == scene_token]
    if not indices:
        return None, None
    return indices[0], indices[-1]

def print_scene_details(scene: SceneInfo, vis_dataset=None):
    """Print detailed information about a scene."""
    print(f"\nSzene: {scene.name}")
    print(f"Beschreibung: {scene.description}")
    print(f"Anzahl Samples: {scene.nbr_samples}")
    if vis_dataset is not None:
        start_idx, end_idx = get_scene_split_indices(vis_dataset, scene.name)
        if start_idx is not None:
            print(f"Split-Indizes: {start_idx} bis {end_idx}")
            print("\nKopieren Sie diese Zeilen in Ihre Pipeline-Datei:")
            print(f"visualization.sample_idx: {start_idx}  # Erster Frame dieser Szene")
            print(f"visualization.sample_idx: {end_idx}  # Letzter Frame dieser Szene")
        else:
            print("Szene ist im aktuellen Split nicht enthalten!")
    else:
        print("(Split-Indizes können nicht angezeigt werden, da kein Dataset geladen ist.)")

def select_scene_and_sample(scenes: List[SceneInfo], vis_dataset=None):
    """Let user select a scene and then choose a sample."""
    print("\nVerfügbare Szenen:")
    print("-----------------")
    for i, scene in enumerate(scenes, 1):
        print(f"\n{i}. Szene:")
        print(f"   Name: {scene.name}")
        print(f"   Beschreibung: {scene.description}")
        print(f"   Verfügbare Indizes: 0 bis {scene.nbr_samples-1}")
    
    while True:
        try:
            scene_choice = int(input("\nWählen Sie eine Szene (Nummer eingeben): "))
            if 1 <= scene_choice <= len(scenes):
                selected_scene = scenes[scene_choice-1]
                break
            else:
                print(f"Bitte eine Zahl zwischen 1 und {len(scenes)} eingeben.")
        except ValueError:
            print("Bitte eine gültige Zahl eingeben.")
    
    print(f"\nSie haben Szene {scene_choice} ausgewählt:")
    print_scene_details(selected_scene, vis_dataset)
    print(f"\nOder wählen Sie einen Index im Split-Bereich für diese Szene.")
    print("\nMöchten Sie eine andere Szene auswählen? (j/n)")
    if input().lower() == 'j':
        select_scene_and_sample(scenes, vis_dataset)

def interactive_menu():
    """Interactive menu for sample selection."""
    print("\n=== Sample-Auswahl Menü ===")
    print("1. Nach Kriterien suchen")
    print("2. Verfügbare Kategorien anzeigen")
    print("3. Beenden")
    
    choice = input("\nBitte wählen Sie eine Option (1-3): ")
    
    if choice == "1":
        search_by_criteria()
    elif choice == "2":
        show_categories()
    elif choice == "3":
        print("Auf Wiedersehen!")
        return
    else:
        print("Ungültige Auswahl. Bitte versuchen Sie es erneut.")
        interactive_menu()

def show_categories():
    """Show all available categories and their options."""
    scenes = load_scenes("datasets")
    categories = get_category_options(scenes)
    
    print("\nVerfügbare Kategorien und Optionen:")
    print("-----------------------------------")
    for category, options in categories.items():
        print(f"\n{category}:")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
    
    input("\nDrücken Sie Enter, um zum Hauptmenü zurückzukehren...")
    interactive_menu()

def search_by_criteria():
    """Interactive search by criteria."""
    scenes = load_scenes("datasets")
    categories = get_category_options(scenes)
    selected_criteria = {}
    print("\nWählen Sie Ihre Suchkriterien:")
    print("-----------------------------")
    for category, options in categories.items():
        print(f"\n{category}:")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        print("  0. Keine Auswahl")
        choice = input(f"\nBitte wählen Sie eine Option für {category} (0-{len(options)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            selected_criteria[category.lower()] = options[int(choice)-1]
    # Suche nach Szenen mit den ausgewählten Kriterien
    matching_scenes = find_samples_by_criteria(
        scenes,
        weather=selected_criteria.get('wetter'),
        area=selected_criteria.get('bereich'),
        daytime=selected_criteria.get('tageszeit'),
        season=selected_criteria.get('jahreszeit'),
        lighting=selected_criteria.get('beleuchtung'),
        structure=selected_criteria.get('struktur'),
        construction=selected_criteria.get('baustelle')
    )
    # Lade das Dataset für die Split-Index-Bestimmung
    from oft.data.transformer_gt_dataset import ObjectFusionGTDataset
    from omegaconf import OmegaConf
    import yaml
    # Lade die Pipeline-Konfiguration (z.B. pipeline_c_modules.yaml)
    config_path = "config/pipeline_c_modules.yaml"
    with open(config_path, 'r') as f:
        pipeline_cfg = yaml.safe_load(f)
    dataset_cfg = pipeline_cfg.get('dataset', {})
    split_name = pipeline_cfg.get('training', {}).get('val_split_name', 'mini_val')
    vis_dataset = ObjectFusionGTDataset(
        dataroot=dataset_cfg["dataroot"],
        version=dataset_cfg["version"],
        split_name=split_name,
        pipeline_config=pipeline_cfg,
        verbose=False
    )
    if not matching_scenes:
        print("\nKeine exakten Übereinstimmungen gefunden.")
        print("Suche nach ähnlichen Szenen...")
        similar_scenes = get_similar_scenes(scenes, selected_criteria)
        if similar_scenes:
            print("\nÄhnliche Szenen gefunden:")
            print("------------------------")
            select_scene_and_sample(similar_scenes[:5], vis_dataset)
        else:
            print("\nKeine ähnlichen Szenen gefunden. Versuchen Sie es mit weniger Kriterien.")
    else:
        print(f"\nGefundene Szenen ({len(matching_scenes)}):")
        print("-----------------------------")
        select_scene_and_sample(matching_scenes, vis_dataset)
    input("\nDrücken Sie Enter, um zum Hauptmenü zurückzukehren...")
    interactive_menu()

def main():
    parser = argparse.ArgumentParser(description="Sample-Auswahl und Visualisierung")
    parser.add_argument("--weather", help="Wetterbedingung (clear, overcast, rain, snow)")
    parser.add_argument("--area", help="Bereich (city, highway, terminal)")
    parser.add_argument("--daytime", help="Tageszeit (morning, noon)")
    parser.add_argument("--lighting", help="Beleuchtung (dark, illuminated, other_lighting, twilight)")
    parser.add_argument("--list-categories", action="store_true", help="Liste alle verfügbaren Kategorien")
    parser.add_argument("--interactive", action="store_true", help="Starte den interaktiven Modus")
    args = parser.parse_args()

    if args.interactive:
        interactive_menu()
        return

    # Konfiguriere den Datensatz-Pfad
    dataroot = "datasets"
    
    # Lade Szenen
    scenes = load_scenes(dataroot)
    
    if args.list_categories:
        print("\nVerfügbare Szenen-Kategorien:")
        print("-----------------------------")
        categories = get_category_options(scenes)
        
        for category, values in categories.items():
            print(f"\n{category}:")
            for value in values:
                print(f"  - {value}")
        return

    # Suche nach Szenen mit den angegebenen Kriterien
    matching_scenes = find_samples_by_criteria(
        scenes,
        weather=args.weather,
        area=args.area,
        daytime=args.daytime,
        lighting=args.lighting
    )

    if not matching_scenes:
        print("\nKeine Szenen gefunden, die den angegebenen Kriterien entsprechen.")
        print("Verwenden Sie --list-categories, um verfügbare Optionen zu sehen.")
        return

    print(f"\nGefundene Szenen ({len(matching_scenes)}):")
    print("-----------------------------")
    
    for scene in matching_scenes:
        print_scene_details(scene)

if __name__ == "__main__":
    main() 