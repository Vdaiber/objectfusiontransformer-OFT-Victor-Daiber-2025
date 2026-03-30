"""
Debug script: Hungarian Matcher Analysis
Analysiert die Hungarian Matcher-Zuordnungen um zu verstehen warum Regression Losses "hard stuck" sind.

Zeigt:
- Cost Matrix Komponenten (center, class, giou, size, angle)
- Finale Zuordnungen (pred_idx -> gt_idx)  
- Qualität der Zuordnungen (Distanzen, IoU, Class-Matches)
- Problematische Zuordnungen identifizieren

Usage:
python debug_hungarian_matcher.py
"""

from __future__ import annotations

import os
import torch
import numpy as np
from typing import Any, Dict, List, Tuple
from omegaconf import DictConfig
import hydra

from oft.transformer.datasets.truckscenes.dataset import ObjectFusionGTDatasetStaged
from oft.transformer.datasets.preprocessing.collate_functions import object_fusion_gt_collate_fn_autoregressive
from oft.transformer.training.matchers.hungarian_matcher import HungarianMatcher
from oft.transformer.training.losses.matcher_costs import (
    center_cost_matrix_huber, 
    class_cost_matrix, 
    giou_cost_matrix,
    size_cost_matrix_l1,
    angle_cost_matrix_huber
)
from oft.transformer.utils.normalization_utils import get_global_normalizer, denormalize_coordinates, denormalize_dimensions
from oft.transformer.utils.iou import generalized_box_iou_bev


def analyze_cost_matrix_components(
    pred_boxes: torch.Tensor, 
    gt_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    cfg: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """
    Analysiere alle Cost-Matrix-Komponenten einzeln.
    
    Args:
        pred_boxes: [num_queries, 10] (EGO, normalized)
        gt_boxes: [num_gt, 10] (EGO, normalized) 
        pred_logits: [num_queries, num_classes+1]
        gt_labels: [num_gt]
        cfg: Configuration
        
    Returns:
        Dict mit Cost-Matrizen für jede Komponente
    """
    delta = cfg['loss']['huber']['delta']
    
    # Cost Matrix Komponenten berechnen
    center_cost = center_cost_matrix_huber(pred_boxes, gt_boxes, delta=delta)
    class_cost = class_cost_matrix(pred_logits, gt_labels)
    size_cost = size_cost_matrix_l1(pred_boxes[:, 3:6], gt_boxes[:, 3:6])
    angle_cost = angle_cost_matrix_huber(pred_boxes, gt_boxes, delta=delta)
    giou_cost = giou_cost_matrix(pred_boxes, gt_boxes)
    
    return {
        'center_cost': center_cost,
        'class_cost': class_cost, 
        'size_cost': size_cost,
        'angle_cost': angle_cost,
        'giou_cost': giou_cost
    }


def analyze_assignment_quality(
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor, 
    pred_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    cost_components: Dict[str, torch.Tensor],
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analysiere die Qualität der Hungarian-Zuordnungen.
    
    Args:
        indices: Hungarian Matcher Ergebnisse [(src_idx, tgt_idx), ...]
        pred_boxes: [num_queries, 10] (EGO, normalized)
        gt_boxes: [num_gt, 10] (EGO, normalized)
        pred_logits: [num_queries, num_classes+1] 
        gt_labels: [num_gt]
        cost_components: Cost-Matrix-Komponenten
        cfg: Configuration
        
    Returns:
        Dict mit Analyse-Ergebnissen
    """
    analysis = {
        'num_assignments': 0,
        'center_distances': [],
        'giou_values': [], 
        'class_matches': [],
        'assignment_costs': [],
        'problematic_assignments': []
    }
    
    if len(indices) == 0 or len(indices[0]) == 0:
        print("⚠️  Keine Zuordnungen gefunden!")
        return analysis
        
    # Für Batch-Size 1 (erstes Element)
    src_idx, tgt_idx = indices[0]
    analysis['num_assignments'] = len(src_idx)
    
    if len(src_idx) == 0:
        print("⚠️  Keine Zuordnungen in diesem Batch!")
        return analysis
    
    # Denormalisierung für GIoU-Berechnung
    normalizer = get_global_normalizer()
    point_cloud_range = torch.tensor(
        normalizer.stats['metadata']['point_cloud_range'], 
        device=pred_boxes.device, 
        dtype=pred_boxes.dtype
    )
    
    for i, (pred_i, gt_i) in enumerate(zip(src_idx, tgt_idx)):
        pred_i, gt_i = pred_i.item(), gt_i.item()
        
        # Center-Distanz (normalisiert)
        center_dist = torch.norm(pred_boxes[pred_i, :3] - gt_boxes[gt_i, :3]).item()
        analysis['center_distances'].append(center_dist)
        
        # GIoU berechnen (denormalisiert)
        pred_box_denorm = denormalize_single_box(pred_boxes[pred_i], point_cloud_range)
        gt_box_denorm = denormalize_single_box(gt_boxes[gt_i], point_cloud_range)
        
        giou = generalized_box_iou_bev(
            pred_box_denorm.unsqueeze(0), 
            gt_box_denorm.unsqueeze(0), 
            exact=True
        )[0, 0].item()
        analysis['giou_values'].append(giou)
        
        # Class-Match prüfen
        pred_class = torch.argmax(pred_logits[pred_i, :-1]).item()  # Ohne no_object
        gt_class = gt_labels[gt_i].item()
        class_match = (pred_class == gt_class)
        analysis['class_matches'].append(class_match)
        
        # Zuordnungskosten
        total_cost = (
            cfg['loss']['cost_center'] * cost_components['center_cost'][pred_i, gt_i] +
            cfg['loss']['cost_class'] * cost_components['class_cost'][pred_i, gt_i] +
            cfg['loss']['cost_size'] * cost_components['size_cost'][pred_i, gt_i] +
            cfg['loss']['cost_angle'] * cost_components['angle_cost'][pred_i, gt_i] +
            cfg['loss']['cost_giou_bev'] * cost_components['giou_cost'][pred_i, gt_i]
        ).item()
        analysis['assignment_costs'].append(total_cost)
        
        # Problematische Zuordnungen identifizieren
        if giou < 0.1 or center_dist > 0.5 or not class_match:
            analysis['problematic_assignments'].append({
                'assignment_idx': i,
                'pred_idx': pred_i,
                'gt_idx': gt_i,
                'center_distance': center_dist,
                'giou': giou,
                'class_match': class_match,
                'pred_class': pred_class,
                'gt_class': gt_class,
                'total_cost': total_cost
            })
    
    return analysis


def denormalize_single_box(box_norm: torch.Tensor, point_cloud_range: torch.Tensor) -> torch.Tensor:
    """
    Denormalisiere eine einzelne Box für GIoU-Berechnung.
    
    Args:
        box_norm: [10] (EGO, normalized)
        point_cloud_range: [6] (x_min, y_min, z_min, x_max, y_max, z_max)
        
    Returns:
        box_denorm: [7] (x, y, z, w, l, h, yaw) in Metern
    """
    # Koordinaten denormalisieren
    coords_denorm = denormalize_coordinates(box_norm[:3].unsqueeze(0), point_cloud_range)[0]
    
    # Dimensionen denormalisieren (log-space -> linear)
    dims_denorm = denormalize_dimensions(box_norm[3:6].unsqueeze(0))[0]
    
    # Yaw aus sin/cos
    yaw = torch.atan2(box_norm[6], box_norm[7])
    
    return torch.cat([coords_denorm, dims_denorm, yaw.unsqueeze(0)])


def print_analysis_summary(analysis: Dict[str, Any], cfg: Dict[str, Any]):
    """Drucke eine übersichtliche Zusammenfassung der Analyse."""
    
    print("\n" + "="*80)
    print("🔍 HUNGARIAN MATCHER ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Anzahl Zuordnungen: {analysis['num_assignments']}")
    
    if analysis['num_assignments'] > 0:
        print(f"   • Durchschnittliche Center-Distanz: {np.mean(analysis['center_distances']):.4f}")
        print(f"   • Durchschnittlicher GIoU: {np.mean(analysis['giou_values']):.4f}")
        print(f"   • Class-Match Rate: {np.mean(analysis['class_matches'])*100:.1f}%")
        print(f"   • Durchschnittliche Zuordnungskosten: {np.mean(analysis['assignment_costs']):.4f}")
    
    print(f"\n⚙️ MATCHER CONFIGURATION:")
    print(f"   • cost_center: {cfg['loss']['cost_center']}")
    print(f"   • cost_class: {cfg['loss']['cost_class']}")
    print(f"   • cost_size: {cfg['loss']['cost_size']}")
    print(f"   • cost_angle: {cfg['loss']['cost_angle']}")
    print(f"   • cost_giou_bev: {cfg['loss']['cost_giou_bev']}")
    print(f"   • huber_delta: {cfg['loss']['huber']['delta']}")
    
    print(f"\n🚨 PROBLEMATIC ASSIGNMENTS: {len(analysis['problematic_assignments'])}")
    for prob in analysis['problematic_assignments'][:5]:  # Zeige nur erste 5
        print(f"   • Assignment {prob['assignment_idx']}: "
              f"Pred[{prob['pred_idx']}] -> GT[{prob['gt_idx']}]")
        print(f"     - Center Distance: {prob['center_distance']:.4f}")
        print(f"     - GIoU: {prob['giou']:.4f}")
        print(f"     - Class Match: {prob['class_match']} "
              f"(Pred: {prob['pred_class']}, GT: {prob['gt_class']})")
        print(f"     - Total Cost: {prob['total_cost']:.4f}")
    
    if len(analysis['problematic_assignments']) > 5:
        print(f"   ... und {len(analysis['problematic_assignments']) - 5} weitere")


def analyze_sample(
    dataset: ObjectFusionGTDatasetStaged, 
    matcher: HungarianMatcher,
    sample_idx: int,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analysiere einen einzelnen Sample mit dem Hungarian Matcher.
    
    Args:
        dataset: Validation Dataset
        matcher: Hungarian Matcher Instanz
        sample_idx: Index des zu analysierenden Samples
        cfg: Configuration
        
    Returns:
        Analyse-Ergebnisse für diesen Sample
    """
    # Sample laden und batchen
    frame = dataset[sample_idx]
    batch = object_fusion_gt_collate_fn_autoregressive([frame])
    
    sample_token = batch['sample_tokens'][0]
    print(f"\n🔍 ANALYSIERE SAMPLE: {sample_token} (Index: {sample_idx})")
    
    # Ground Truth extrahieren
    gt_boxes_norm = batch['gt_boxes_b_normalized'][0]  # [num_gt, 10]
    gt_labels = batch['gt_labels_b'][0]  # [num_gt]
    
    # Valide GT finden (label != -1)
    valid_mask = gt_labels >= 0
    if not valid_mask.any():
        print("⚠️  Keine validen Ground Truth Objekte in diesem Sample!")
        return {}
    
    gt_boxes_valid = gt_boxes_norm[valid_mask]
    gt_labels_valid = gt_labels[valid_mask]
    
    print(f"   • Ground Truth Objekte: {len(gt_boxes_valid)}")
    print(f"   • GT Classes: {gt_labels_valid.tolist()}")
    
    # Dummy Predictions erstellen (100 Queries)
    num_queries = 100
    num_classes = len(cfg['dataset']['class_names'])
    
    # Für realistische Analyse: Random Predictions in sinnvollem Bereich
    torch.manual_seed(42)  # Reproduzierbarkeit
    pred_boxes_norm = torch.randn(num_queries, 10, dtype=torch.float64) * 0.3 + 0.5  # Um GT-Bereich
    pred_logits = torch.randn(num_queries, num_classes + 1, dtype=torch.float32) * 2.0
    
    print(f"   • Prediction Queries: {num_queries}")
    
    # Cost Matrix Komponenten analysieren
    cost_components = analyze_cost_matrix_components(
        pred_boxes_norm, gt_boxes_valid, pred_logits, gt_labels_valid, cfg
    )
    
    print(f"   • Center Cost Range: [{cost_components['center_cost'].min():.4f}, {cost_components['center_cost'].max():.4f}]")
    print(f"   • Class Cost Range: [{cost_components['class_cost'].min():.4f}, {cost_components['class_cost'].max():.4f}]")
    print(f"   • GIoU Cost Range: [{cost_components['giou_cost'].min():.4f}, {cost_components['giou_cost'].max():.4f}]")
    
    # Hungarian Matching durchführen
    predictions_dict = {
        'pred_class_logits_batch': pred_logits.unsqueeze(0),  # [1, num_queries, num_classes+1]
        'pred_boxes_normalized': pred_boxes_norm.unsqueeze(0)  # [1, num_queries, 10]
    }
    
    targets_dict = {
        'gt_boxes_b_normalized': gt_boxes_valid.unsqueeze(0),  # [1, num_gt, 10]
        'gt_labels_b': [gt_labels_valid],  # List format for matcher
        'gt_valid_mask_b': [torch.ones(len(gt_labels_valid), dtype=torch.bool)]  # List format for matcher
    }
    
    (indices, _), _ = matcher(predictions_dict, targets_dict)
    
    # Zuordnungsqualität analysieren
    analysis = analyze_assignment_quality(
        indices, pred_boxes_norm, gt_boxes_valid, 
        pred_logits, gt_labels_valid, cost_components, cfg
    )
    
    return analysis


@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hauptfunktion für Hungarian Matcher Analyse."""
    
    from omegaconf import OmegaConf
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    
    print("🚀 HUNGARIAN MATCHER DIAGNOSTIC TOOL")
    print("="*50)
    
    # Dataset laden
    droot = cfg_dict["dataset"]["dataroot"]
    version = cfg_dict["dataset"]["version"]
    val_split = cfg_dict["training"].get("val_split_name", "mini_val")
    
    print(f"📁 Loading dataset: {version} / {val_split}")
    val_dataset = ObjectFusionGTDatasetStaged(
        dataroot=droot,
        version=version,
        split_name=val_split,
        pipeline_config=cfg_dict,
        verbose=False,
        ram_cache=None,
        devkit_ram=None,
    )
    
    # Hungarian Matcher erstellen
    matcher = HungarianMatcher(
        cost_center=cfg_dict['loss']['cost_center'],
        cost_class=cfg_dict['loss']['cost_class'],
        cost_size=cfg_dict['loss']['cost_size'],
        cost_giou_bev=cfg_dict['loss']['cost_giou_bev'],
        cost_angle=cfg_dict['loss']['cost_angle'],
        cfg=cfg_dict
    )
    print(f"🎯 Matcher erstellt mit Konfiguration:")
    print(f"   • cost_center: {cfg_dict['loss']['cost_center']}")
    print(f"   • cost_class: {cfg_dict['loss']['cost_class']}")
    print(f"   • cost_giou_bev: {cfg_dict['loss']['cost_giou_bev']}")
    
    # Mehrere Samples analysieren - probiere mehr Samples um valide GT zu finden
    max_samples = min(20, len(val_dataset))
    all_analyses = []
    
    print(f"📊 Dataset hat {len(val_dataset)} Samples insgesamt")
    
    for i in range(max_samples):
        try:
            analysis = analyze_sample(val_dataset, matcher, i, cfg_dict)
            if analysis:  # Nur wenn valide Daten
                all_analyses.append(analysis)
        except Exception as e:
            print(f"❌ Fehler bei Sample {i}: {e}")
            continue
    
    # Gesamtstatistik
    if all_analyses:
        print("\n" + "="*80)
        print("📈 OVERALL STATISTICS")
        print("="*80)
        
        total_assignments = sum(a['num_assignments'] for a in all_analyses)
        total_problematic = sum(len(a['problematic_assignments']) for a in all_analyses)
        
        print(f"Samples analysiert: {len(all_analyses)}")
        print(f"Gesamte Zuordnungen: {total_assignments}")
        print(f"Problematische Zuordnungen: {total_problematic} ({total_problematic/max(total_assignments,1)*100:.1f}%)")
        
        if total_assignments > 0:
            all_center_dists = [d for a in all_analyses for d in a['center_distances']]
            all_gious = [g for a in all_analyses for g in a['giou_values']]
            all_class_matches = [c for a in all_analyses for c in a['class_matches']]
            
            print(f"Durchschnittliche Center-Distanz: {np.mean(all_center_dists):.4f}")
            print(f"Durchschnittlicher GIoU: {np.mean(all_gious):.4f}")
            print(f"Globale Class-Match Rate: {np.mean(all_class_matches)*100:.1f}%")
            
            # Detailanalyse des ersten Samples
            if all_analyses:
                print_analysis_summary(all_analyses[0], cfg_dict)
    
    print("\n✅ Analyse abgeschlossen!")
    print("\n💡 EMPFEHLUNGEN:")
    if all_analyses and total_problematic / max(total_assignments, 1) > 0.3:
        print("   • Hohe Rate problematischer Zuordnungen!")
        print("   • Prüfe Matcher-Gewichte: cost_center vs cost_class vs cost_giou_bev")
        print("   • Eventuell GIoU-Gewicht erhöhen für bessere geometrische Zuordnungen")
    else:
        print("   • Zuordnungen scheinen größtenteils korrekt zu sein")
        print("   • Problem könnte in Loss-Berechnung oder anderen Komponenten liegen")


if __name__ == "__main__":
    main()
