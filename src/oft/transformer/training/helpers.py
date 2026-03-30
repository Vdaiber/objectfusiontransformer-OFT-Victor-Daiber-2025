# src/oft/transformer/training/helpers.py
"""
Training helper functions for the Object Fusion Transformer pipeline.

This module contains helper functions for the training and evaluation process,
particularly logic for single epoch processing and model evaluation.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, List, Tuple
import torch.nn.functional as F
from pyquaternion import Quaternion as PyQuaternion

from ..utils.logging import MetricLogger
from ..evaluation.evaluator import reconstruct_and_convert_predictions
from ..training.criteria import SetCriterion
from ..models.fusion_transformer import ObjectFusionTransformerModel

@torch.no_grad()
def evaluate_model_internally(model: ObjectFusionTransformerModel,
                              criterion: SetCriterion,
                              data_loader: DataLoader,
                              device: torch.device,
                              cfg_dict: Dict[str, Any],
                              logger: logging.Logger,
                              epoch: int):
    """Performs internal validation for a single epoch."""
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    header = f'Epoch: [{epoch}] (Validation)'
    all_predictions_for_devkit = []
    
    for batch_dict in metric_logger.log_every(data_loader, 10, header):
        # Move data to target device
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.to(device)

        predictions = model(
            batch_dict['encoder_input_features'], 
            batch_dict['encoder_input_xyz_centers'], 
            batch_dict['encoder_input_mask']
        )
        
        losses_dict_unweighted = criterion(
            decoder_outputs=predictions,
            gt_labels_b=batch_dict['gt_labels_b'],
            gt_boxes_b_log_dims=batch_dict['gt_boxes_b_log_dims'],
            gt_boxes_b_actual_dims=batch_dict['gt_boxes_b_actual_dims'],
            gt_valid_mask_b=batch_dict['gt_valid_mask_b']
        )

        total_loss = sum(criterion.weight_dict.get(k, 1.0) * v for k, v in losses_dict_unweighted.items())
        metric_logger.update(loss=total_loss.item())
        for k, v in losses_dict_unweighted.items():
            metric_logger.update(**{f'{k}': v.item()})

        # Reconstruction for DevKit evaluation
        sample_predictions = reconstruct_and_convert_predictions(
            batch_dict_for_eval=batch_dict,
            predictions=predictions,
            config=cfg_dict,
            criterion=criterion
        )
        all_predictions_for_devkit.extend(sample_predictions)

    logger.info(f"Averaged validation stats (internal loss) epoch {epoch}: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, all_predictions_for_devkit
