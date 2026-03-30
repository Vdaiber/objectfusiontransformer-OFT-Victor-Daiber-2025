# File: src/oft/transformer/training/trainers/autoregressive_trainer.py
"""
Autoregressive trainer for sequential prediction training in sensor fusion.

This module implements training logic for transformer models that generate predictions
sequentially, where each prediction depends on previous outputs. The trainer manages
temporal state transitions, handles loss computation with Hungarian matching, and
maintains autoregressive memory across temporal sequences.
"""

import torch
import math
from typing import Dict, Any, List, Optional, Tuple

from oft.transformer.training.trainers.base_trainer import BaseTrainer
from oft.transformer.training.utils.metric_logger import MetricLogger, SmoothedValue
from oft.transformer.datasets.truckscenes.dataset import ATTRIBUTE_VOCAB
from oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive
from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive
from oft.transformer.training.utils.diagnostic_logger import DiagnosticLogger


class AutoregressiveTrainer(BaseTrainer):
    """Trainer for autoregressive transformer models with temporal state management.
    
    Manages training and evaluation loops for models that generate predictions
    sequentially. Handles temporal memory states, scene boundary detection,
    and proper loss computation with Hungarian matching.
    
    Attributes:
        memory: Transformer memory state for temporal consistency
        memory_anchor_boxes: Anchor boxes maintained across temporal sequences
        ego_pose_previous: Previous ego pose for motion compensation
        last_timestamp_us: Previous timestamp for delta time calculation
    """

    def __init__(self, model: ObjectFusionTransformerAutoregressive, criterion, optimizer, lr_scheduler, dataloaders, device, cfg, output_dir, logger):
        """Initialize trainer with model and training components.
        
        This trainer specifically handles autoregressive transformer models that
        maintain temporal state across sequential predictions. It manages memory
        states, scene transitions, and coordinate transformations between ego-vehicle
        and world reference frames.
        
        Args:
            model: Autoregressive transformer model with temporal memory
            criterion: Loss function implementing Hungarian bipartite matching
            optimizer: Parameter optimization algorithm (e.g., AdamW)
            lr_scheduler: Learning rate scheduling strategy
            dataloaders: Training and validation data loaders with temporal sequences
            device: Computational device (CUDA/CPU)
            cfg: Configuration dictionary containing hyperparameters
            output_dir: Output directory for checkpoints and logs
            logger: Logging interface for training diagnostics
        """
        super().__init__(model, criterion, optimizer, lr_scheduler, dataloaders, device, cfg, output_dir, logger)
        
        # Initialize temporal state variables
        self.memory = None
        self.memory_anchor_boxes = None
        self.ego_pose_previous = None
        self.last_timestamp_us = None
        
        # Initialize Diagnostic Logger
        self.diag_logger = DiagnosticLogger(model=self.model, criterion=self.criterion, output_dir=str(self.output_dir))
        
    def _prepare_batch_for_device(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensors to device and harmonize dtypes for stable model execution.
        
        Args:
            batch_dict: Batch data from dataloader
            
        Returns:
            Batch with all tensors moved to device
        """
        # Use non_blocking transfers if DataLoader uses pin_memory for faster host→device copies
        is_pinned = self.cfg['training'].get('pin_memory', False)

        # Traverse nested batch structures and move tensors to device
        # Preserve float64 for world-frame ego pose/motion; cast others to float32
        preserve_fp64_keys = {"ego_translation_world", "ego_rotation_world_quat", "ego_motion"}
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                # Cast all float tensors to float32 before moving to device to avoid dtype mismatches
                if key not in preserve_fp64_keys and value.is_floating_point() and value.dtype != torch.float32:
                    value = value.float()
                batch_dict[key] = value.to(self.device, non_blocking=is_pinned)
            elif isinstance(value, dict):
                # Process nested dictionary (sensor_data)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        if key != "ego_motion" and sub_value.is_floating_point() and sub_value.dtype != torch.float32:
                            sub_value = sub_value.float()
                        value[sub_key] = sub_value.to(self.device, non_blocking=is_pinned)
                    elif isinstance(sub_value, dict):
                        # Process sensor-specific nested data
                        for sensor_key, sensor_value in sub_value.items():
                            if isinstance(sensor_value, torch.Tensor):
                                if sensor_value.is_floating_point() and sensor_value.dtype != torch.float32:
                                    sensor_value = sensor_value.float()
                                sub_value[sensor_key] = sensor_value.to(self.device, non_blocking=is_pinned)
        return batch_dict

    def _prepare_targets_for_criterion(self, batch_dict: Dict[str, Any], predictions: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """Prepare ground truth targets and get matcher diagnostics.
        
        Args:
            batch_dict: Batch containing ground truth data
            predictions: Model predictions for matching
            
        Returns:
            - Prepared targets with Hungarian matching indices
            - Dictionary with average matcher costs for diagnostics
        """
        # Extract prediction components for matching
        initial_boxes_b = predictions['anchor_boxes']
        padding_mask_b = predictions['fused_padding_mask']

        # Get batch size for tensor splitting
        batch_size = batch_dict['gt_labels_b'].shape[0]
        
        # Convert batch tensors to lists for Hungarian matcher
        gt_labels_list = [batch_dict['gt_labels_b'][i] for i in range(batch_size)]
        gt_attributes_list = [batch_dict['gt_attributes_b'][i] for i in range(batch_size)]
        gt_valid_mask_list = [batch_dict['gt_valid_mask_b'][i] for i in range(batch_size)]

        # Create standardized targets dictionary
        targets_for_criterion = {
            'gt_labels_b': batch_dict['gt_labels_b'],
            'gt_boxes_b_normalized': batch_dict['gt_boxes_b_normalized'],
            'gt_boxes_b_physical': batch_dict['gt_boxes_b_physical'],
            'gt_attributes_b': batch_dict.get('gt_attributes_b', None),
            'gt_valid_mask_b': batch_dict['gt_valid_mask_b'],
        }

        # Perform Hungarian matching and get diagnostics
        (indices, _), matcher_costs_diag = self.criterion.matcher(
            predictions={
                'pred_class_logits_batch': predictions['pred_class_logits_batch'],
                'pred_boxes_normalized': predictions['pred_boxes_normalized'],
                'pred_attributes_logits_batch': predictions.get('pred_attributes_logits_batch', torch.empty_like(predictions['pred_class_logits_batch']))
            },
            targets={
                'gt_labels_b': gt_labels_list,  # List format for matcher
                'gt_boxes_b_normalized': batch_dict['gt_boxes_b_normalized'],
                'gt_valid_mask_b': gt_valid_mask_list  # List format for matcher
            }
        )
        
        targets_for_criterion['indices'] = indices
        return targets_for_criterion, matcher_costs_diag

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute training loop for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average training metrics
        """
        # Set training mode
        self.model.train()
        self.criterion.train()
        
        # Initialize metric tracking
        metric_logger = MetricLogger(delimiter="  ", logger=self.logger)
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Train Epoch: [{epoch}]'
        
        # Process training batches
        for batch_idx, batch_dict in enumerate(metric_logger.log_every(self.dataloaders['train'], self.cfg['training']['print_freq'], header)):
            # --- LR Warmup for first epoch ---
            if epoch == 0:
                target_lr = self.cfg['training']['learning_rate']
                warmup_steps = len(self.dataloaders['train'])
                initial_lr = 1e-6
                if warmup_steps > 0:
                    current_lr = initial_lr + (target_lr - initial_lr) * (batch_idx / warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr

            # --- Batch Preparation ---
            # Move all tensors to correct device (CPU/GPU) for model processing
            batch_dict = self._prepare_batch_for_device(batch_dict)
            
            # Extract batch components
            sensor_data_b = batch_dict['sensor_data']
            # Ego pose handling for coordinate transformations:
            # - Single-frame mode (autoregressive.enabled=false): Set to None to allow
            #   per-sample ego pose extraction from batch_dict['ego_translation_world'][i]
            # - Temporal mode (autoregressive.enabled=true): Would use batch[0] for consistency
            ego_pose_current = None  # Enables per-sample coordinate transformation
            is_scene_start = bool(any(batch_dict.get('is_scene_start', [False])))

            # --- Autoregressive State Management ---
            # Reset memory states at scene boundaries to prevent temporal contamination
            if is_scene_start:
                self.logger.info("New scene started, resetting memory.")
                self.memory = None
                self.memory_anchor_boxes = None
                self.ego_pose_previous = None
                self.last_timestamp_us = None
            
            # Calculate delta time for temporal prediction
            # Only compute dt if autoregressive temporal pathway is enabled
            model_auto = self.cfg['model']['autoregressive']
            if model_auto.get('enabled', False) and model_auto.get('memory_enabled', False):
                current_timestamp_us = batch_dict['temporal_info'][0].get('timestamp')
                dt = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                if self.last_timestamp_us and current_timestamp_us:
                    delta_time_s = (current_timestamp_us - self.last_timestamp_us) / 1_000_000.0
                    if 0 < delta_time_s < 1.0:
                        dt = torch.tensor(delta_time_s, device=self.device, dtype=torch.float64)
                if current_timestamp_us is not None:
                    self.last_timestamp_us = current_timestamp_us
            else:
                dt = torch.tensor(0.0, device=self.device, dtype=torch.float64)

            # --- Model Forward Pass ---
            # Execute autoregressive transformer model with temporal state
            predictions, new_memory, new_memory_anchor_boxes = self.model(
                sensor_data=sensor_data_b,  # Multi-modal sensor data (EGO, normalized)
                memory=self.memory,  # Previous transformer memory (EGO, normalized)
                memory_anchor_boxes=self.memory_anchor_boxes,  # Previous anchor boxes (EGO, unnormalized)
                dt=dt,  # Delta time for temporal prediction (seconds, unnormalized)
                ego_pose_current=ego_pose_current,  # None in single-frame mode
                ego_pose_previous=self.ego_pose_previous,  # None in single-frame mode
                scene_meta=batch_dict['scene_meta'],  # Scene metadata information
                logger=self.logger  # Logger for model diagnostics
            )

            # --- Target Preparation & Matcher Diagnostics ---
            targets_for_criterion, matcher_costs_diag = self._prepare_targets_for_criterion(
                batch_dict=batch_dict,
                predictions=predictions
            )

            # --- Loss Calculation with Hungarian Matching ---
            # Compute all loss components (classification, regression, attributes)
            # The criterion applies Hungarian bipartite matching between predictions and GT
            loss_dict = self.criterion(predictions, targets_for_criterion)
            weight_dict = self.criterion.weight_dict
            
            # Use pre-weighted total loss from criterion
            # (loss_total = sum of individually weighted loss components)
            losses = loss_dict['loss_total']
            
            # OPTIMIZATION: Keep losses on GPU, avoid .item() calls during training
            # Only transfer to CPU when absolutely necessary (logging, stability checks)
            
            # Check for numerical stability WITHOUT .item() - use tensor operations
            if not torch.isfinite(losses).all():
                loss_cpu_for_error = losses.item()  # Only now transfer for error message
                self.logger.error(f"Loss is {loss_cpu_for_error}, stopping training")
                raise SystemExit("Stopping training due to non-finite loss.")

            # Backpropagation
            self.optimizer.zero_grad()
            losses.backward()
            
            # Apply gradient clipping if specified to prevent exploding gradients
            clip_max_norm = self.cfg['training'].get('clip_max_norm', 0)
            module_clip_norms = self.cfg['training'].get('module_clip_norms', {})
            
            if clip_max_norm > 0 or module_clip_norms:
                # MODULSPEZIFISCHES GRADIENT CLIPPING
                total_grad_norm_before = 0.0
                any_module_clipped = False
                
                if module_clip_norms:
                    # Define module groups for targeted clipping
                    module_groups = {
                        'class_head': [p for n, p in self.model.named_parameters() if 'decoder.class_head' in n and p.grad is not None],
                        'attribute_head': [p for n, p in self.model.named_parameters() if 'decoder.attribute_head' in n and p.grad is not None],
                        'center_head': [p for n, p in self.model.named_parameters() if 'decoder.center_offset_head' in n and p.grad is not None],
                        'size_head': [p for n, p in self.model.named_parameters() if 'decoder.size_offset_head' in n and p.grad is not None],
                        'yaw_head': [p for n, p in self.model.named_parameters() if 'decoder.yaw_offset_head' in n and p.grad is not None],
                        'velocity_head': [p for n, p in self.model.named_parameters() if 'decoder.velocity_head_mlp' in n and p.grad is not None],
                        'fusion_encoder': [p for n, p in self.model.named_parameters() if 'inter_modal_fusion' in n and p.grad is not None],
                        'intra_modal_encoders': [p for n, p in self.model.named_parameters() if 'intra_modal_encoders' in n and p.grad is not None]
                    }
                    
                    # Calculate total gradient norm before any clipping
                    all_params = [p for p in self.model.parameters() if p.grad is not None]
                    if not all_params:
                        self.logger.warning("No gradients found to clip.")
                        total_grad_norm_before = 0.0
                    else:
                        total_grad_norm_before_tensor = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
                    
                    # Apply module-specific clipping
                    module_grad_norms = {}
                    for module_name, clip_norm in module_clip_norms.items():
                        if module_name in module_groups and module_groups[module_name]:
                            module_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(module_groups[module_name], clip_norm)
                            module_grad_norms[module_name] = module_grad_norm_tensor
                            # Mark as clipped only if pre-clip norm exceeded threshold
                            if module_grad_norm_tensor.item() > float(clip_norm):
                                any_module_clipped = True
                        
                # Calculate final total norm for monitoring after all clipping
                if all_params:
                    total_grad_norm_after_tensor = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
                else:
                    total_grad_norm_after_tensor = torch.tensor(0.0, device=self.device)
                
                # OPTIMIZATION: Batch all gradient norm .item() calls
                grad_norm_tensors = {'total_before': total_grad_norm_before_tensor if 'total_grad_norm_before_tensor' in locals() else torch.tensor(0.0, device=self.device),
                                   'total_after': total_grad_norm_after_tensor}
                grad_norm_tensors.update(module_grad_norms)
                grad_norm_values = {k: v.item() for k, v in grad_norm_tensors.items()}
                
                total_grad_norm_before = grad_norm_values['total_before']  
                total_grad_norm_after = grad_norm_values['total_after']
                
                # Clipping warnings (using already computed CPU values)
                for module_name, clip_norm in module_clip_norms.items():
                    if module_name in grad_norm_values:
                        module_grad_norm = grad_norm_values[module_name]
                        if module_grad_norm > clip_norm:
                            self.logger.info(f"Clipping '{module_name}': norm={module_grad_norm:.4f} -> clip_val={clip_norm}")
                        if module_grad_norm > clip_norm * 1.5:
                            self.logger.warning(f"Severe gradient explosion in '{module_name}': norm={module_grad_norm:.4f}")

                # Print total grad norm line only if at least one module was actually clipped
                if module_clip_norms and any_module_clipped and total_grad_norm_before > 0:
                    self.logger.info(f"Total grad norm: Before={total_grad_norm_before:.2f} -> After={total_grad_norm_after:.2f} (module clipping applied)")
            
            # Global clipping (if enabled)
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_max_norm)
            
            self.optimizer.step()  # Update model parameters using computed gradients
            
            # --- DIAGNOSTIC LOGGING ---
            if batch_idx % self.cfg['training']['print_freq'] == 0:
                # Pass GPU tensors directly to diagnostic logger (it handles .item() internally)
                self.diag_logger.log_batch_stats(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss_dict=loss_dict,  # Keep as GPU tensors
                    weight_dict=weight_dict,
                    total_loss=losses.item(),  # Only this one needs .item()
                    matcher_costs=matcher_costs_diag
                )

            # --- Logging and State Update ---
            # OPTIMIZATION: Only transfer losses to CPU for logging (minimal transfers)
            # Transfer losses only every N batches to minimize GPU-CPU synchronization
            log_this_batch = (batch_idx % self.cfg['training']['print_freq'] == 0) or (batch_idx == len(self.dataloaders['train']) - 1)
            
            if log_this_batch:
                # Single batched GPU→CPU transfer for logging
                loss_tensors_for_log = {'total_loss': losses}
                # Log individual (unweighted) losses for transparency
                loss_tensors_for_log.update({k: v for k, v in loss_dict.items() if k.startswith('loss_') and k != 'loss_total'})
                loss_values_cpu = {k: v.item() for k, v in loss_tensors_for_log.items()}
                
                # Update metric logger with CPU values
                loss_cpu = loss_values_cpu['total_loss']
                individual_losses_cpu = {k: v for k, v in loss_values_cpu.items() if k != 'total_loss'}
                metric_logger.update(loss=loss_cpu, **individual_losses_cpu)
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            else:
                # No GPU→CPU transfer - just update LR
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            
            # Update ego pose for next iteration (temporal state management)
            self.ego_pose_previous = ego_pose_current  # (WORLD, unnormalized)

            # --- Memory State Update (Detached) ---
            # Update autoregressive memory states, detaching from computation graph
            # This prevents gradient accumulation across temporal sequences
            self.memory = new_memory.detach()  # (B, N, d_model) (EGO, normalized)
            if new_memory_anchor_boxes is not None:
                self.memory_anchor_boxes = new_memory_anchor_boxes.detach()  # (B, N, 9) (EGO, unnormalized)
            else:
                self.memory_anchor_boxes = None

        # Return average metrics for the epoch
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def _evaluate_model(self, epoch: int) -> tuple[Dict[str, Any], list]:
        """Execute evaluation loop for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of average metrics and reconstructed predictions
        """
        # Set model and criterion to evaluation mode (no gradients, no dropout)
        self.model.eval()  # Disable dropout, use batch normalization statistics
        self.criterion.eval()  # Use evaluation-specific loss computation
        
        # Reset temporal state for clean evaluation
        self.memory = None
        self.memory_anchor_boxes = None
        self.ego_pose_previous = None
        self.last_timestamp_us = None
        
        # Initialize metric tracking
        metric_logger = MetricLogger(delimiter="  ", logger=self.logger)
        header = f'Test Epoch: [{epoch}]'
        
        # Collect predictions for evaluation
        all_predictions = []
        
        # Iterate through validation batches with progress logging
        for batch_idx, batch_dict in enumerate(metric_logger.log_every(self.dataloaders['val'], self.cfg['training']['print_freq'], header)):
            # --- Batch Preparation ---
            # Move all tensors to correct device (CPU/GPU) for model processing
            batch_dict = self._prepare_batch_for_device(batch_dict)
            
            # Extract sensor data and ego pose information from batch
            sensor_data_b = batch_dict['sensor_data']  # Multi-modal sensor data 
            # In single-frame mode: Use None so evaluation uses per-sample ego poses
            ego_pose_current = None  # Let evaluation use batch_dict['ego_translation_world'][i] per sample
            is_scene_start = bool(any(batch_dict.get('is_scene_start', [False])))  # Scene boundary flag (boolean)

            # --- Autoregressive State Management ---
            # Reset memory states at scene boundaries to prevent temporal contamination
            if is_scene_start:
                self.logger.info("New scene started, resetting memory.")
                self.memory = None
                self.memory_anchor_boxes = None
                self.ego_pose_previous = None
                self.last_timestamp_us = None

            # --- DT Calculation ---
            # Compute delta time between consecutive frames for temporal prediction
            model_auto = self.cfg['model']['autoregressive']
            if model_auto.get('enabled', False) and model_auto.get('memory_enabled', False):
                current_timestamp_us = batch_dict['temporal_info'][0].get('timestamp')  # Current timestamp (microseconds)
                dt = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                if self.last_timestamp_us and current_timestamp_us:
                    delta_time_s = (current_timestamp_us - self.last_timestamp_us) / 1_000_000.0
                    if 0 < delta_time_s < 1.0:
                        dt = torch.tensor(delta_time_s, device=self.device, dtype=torch.float64)
                if current_timestamp_us is not None:
                    self.last_timestamp_us = current_timestamp_us
            else:
                dt = torch.tensor(0.0, device=self.device, dtype=torch.float64)

            # --- Model Forward Pass ---
            # Execute autoregressive transformer model with temporal state
            predictions, new_memory, new_memory_anchor_boxes = self.model(
                sensor_data=sensor_data_b,  # Multi-modal sensor data (EGO, normalized)
                memory=self.memory,  # Previous transformer memory (EGO, normalized)
                memory_anchor_boxes=self.memory_anchor_boxes,  # Previous anchor boxes (EGO, unnormalized)
                dt=dt,  # Delta time for temporal prediction (seconds, unnormalized)
                ego_pose_current=ego_pose_current,  # None in single-frame mode
                ego_pose_previous=self.ego_pose_previous,  # None in single-frame mode
                scene_meta=batch_dict['scene_meta'],  # Scene metadata information
                logger=self.logger  # Logger for model debugging
            )

            # --- Target Preparation ---
            # Prepare ground truth targets for loss computation with Hungarian matching
            targets_for_criterion, matcher_costs_diag = self._prepare_targets_for_criterion(
                batch_dict=batch_dict,  # Ground truth data from DataLoader
                predictions=predictions  # Model predictions for matching
            )

            # --- Loss Calculation ---
            # Compute all loss components using Hungarian-matched predictions and targets
            loss_dict = self.criterion(predictions, targets_for_criterion)  # Dictionary with per-loss (unweighted) and loss_total
            weight_dict = self.criterion.weight_dict  # Loss weights (already applied in loss_total)
            
            # Use unified total loss computed by criterion (avoid double-weighting)
            losses = loss_dict['loss_total']
            
            # OPTIMIZATION: Keep losses on GPU during evaluation, minimal CPU transfers
            # Only transfer for logging every N batches
            log_this_batch = (batch_idx % self.cfg['training']['print_freq'] == 0) or (batch_idx == len(self.dataloaders['val']) - 1)
            
            if log_this_batch:
                # Single GPU→CPU transfer for logging
                loss_tensors_for_log = {'total_loss': losses}
                loss_tensors_for_log.update({k: v for k, v in loss_dict.items() if k.startswith('loss_') and k != 'loss_total'})
                loss_values_cpu = {k: v.item() for k, v in loss_tensors_for_log.items()}
                
                # Update metric logger with CPU values
                loss_cpu = loss_values_cpu['total_loss']
                individual_losses_cpu = {k: v for k, v in loss_values_cpu.items() if k != 'total_loss'}
                metric_logger.update(loss=loss_cpu, **individual_losses_cpu)

            # --- Prediction Post-processing ---
            # validate_one_epoch model outputs for numerical stability
            for key, tensor in predictions.items():
                if torch.is_tensor(tensor):
                    has_nan = torch.isnan(tensor).any().item()
                    has_inf = torch.isinf(tensor).any().item()
                    if has_nan or has_inf:
                        print(f"🚨 MODEL OUTPUT NAN/INF - {key}: nan={has_nan}, inf={has_inf}")
                        print(f"   Values: {tensor}")
            
            # Reconstruct and convert predictions to evaluation format
            reconstructed_preds = reconstruct_and_convert_predictions_autoregressive(
                batch_dict=batch_dict,  # Original batch data for reconstruction context
                predictions=predictions,  # Model predictions (EGO, normalized)
                cfg=self.cfg,  # Configuration for reconstruction parameters
                ego_pose_current=ego_pose_current  # Current ego pose for coordinate conversion (WORLD, unnormalized)
            )
            all_predictions.extend(reconstructed_preds)  # Add to evaluation list
            
            # --- State Update ---
            # Update ego pose for next iteration (temporal state management)
            self.ego_pose_previous = ego_pose_current  # (WORLD, unnormalized)

            # --- Memory State Update (Detached) ---
            # Update autoregressive memory states, detaching from computation graph
            # This prevents gradient accumulation across temporal sequences
            self.memory = new_memory.detach()  # (B, N, d_model) (EGO, normalized)
            if new_memory_anchor_boxes is not None:
                self.memory_anchor_boxes = new_memory_anchor_boxes.detach()  # (B, N, 9) (EGO, unnormalized)
            else:
                self.memory_anchor_boxes = None
        
        # Return average metrics and all predictions for evaluation
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, all_predictions
            
 