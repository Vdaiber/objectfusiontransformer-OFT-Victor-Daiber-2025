# ./app/src/oft/transformer/training/trainers/base_trainer.py
"""
Base Trainer for Transformer-Based Sensor Fusion Models.

This module provides the abstract base class for training transformer-based sensor fusion models. 
It defines the general structure for training, validation, checkpointing, and evaluation, and is 
intended to be subclassed for specific model architectures and training strategies.

The base trainer implements the template method pattern, where subclasses provide specific
implementations for forward passes while inheriting the overall training loop structure.

Pipeline Context:
- Training Phase: Orchestrates the complete training loop with epoch iteration, loss computation,
  optimization, evaluation, and checkpointing for transformer-based sensor fusion models
- Data Flow: Receives collated sensor data from DataLoader (EGO, normalized features), processes
  through model forward pass, computes losses, and manages model state persistence
- Coordinate Systems: Handles data in multiple coordinate frames - sensor features (EGO, normalized),
  ground truth boxes (EGO, normalized for training, EGO unnormalized for evaluation), and
  world coordinates for ego pose information (WORLD, unnormalized)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
import datetime
import time
import shutil
from typing import Dict, Any, List, Optional, Tuple

from ..utils.metric_logger import MetricLogger, SmoothedValue
from ..utils.checkpoint_utils import resume_from_checkpoint, save_checkpoint, save_best_nds_checkpoint
from ...utils.common import sanitize_for_json
from ...evaluation.devkit_evaluator import run_devkit_evaluation
from ..callbacks.early_stopping import ValidationLossEarlyStopping


class BaseTrainer:
    """Abstract base class for transformer-based sensor fusion model training.
    
    This class provides the foundational training infrastructure that can be extended
    by specialized trainers for different architectures (e.g., autoregressive).
    It implements the common training loop, evaluation, checkpointing, and logging functionality
    using the template method pattern.
    
    The trainer follows the template method pattern, where subclasses implement
    specific forward pass logic while inheriting the overall training structure.
    This design promotes code reuse and ensures consistent training behavior across
    different model architectures and training strategies.
    
    Data Flow Context:
    - Input: Collated sensor data from DataLoader containing multi-modal features (EGO, normalized)
      and ground truth annotations (EGO, normalized for training, EGO unnormalized for evaluation)
    - Processing: Model forward pass, loss computation, gradient backpropagation, parameter updates
    - Output: Training statistics, validation metrics, and model checkpoints for persistence
    
    Coordinate System Handling:
    - Sensor Features: (EGO, normalized) - 12D features [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin(yaw), cos(yaw), vx_norm, vy_norm, sensor_class, sensor_attr]
    - Ground Truth Training: (EGO, normalized) - 10D boxes for loss computation
    - Ground Truth Evaluation: (EGO, unnormalized) - 9D boxes [x, y, z, w, l, h, yaw, vx, vy] in meters/radians
    - Ego Pose: (WORLD, unnormalized) - translation and rotation quaternions for coordinate transformations
    
    Attributes:
        model (nn.Module): The neural network model to be trained - processes sensor features (EGO, normalized)
        criterion (nn.Module): Loss function for computing training objectives - operates on normalized predictions and targets
        optimizer (torch.optim.Optimizer): Optimization algorithm for updating model parameters
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for adaptive learning rate adjustment
        dataloaders (Dict[str, DataLoader]): Dictionary containing train and validation data loaders
            - Data from DataLoader: sensor_data (EGO, normalized), gt_boxes_b_normalized (EGO, normalized), gt_boxes_b_physical (EGO, unnormalized)
        device (torch.device): Computational device (CPU/GPU) for model execution
        cfg (Dict[str, Any]): Configuration dictionary containing training hyperparameters and pipeline settings
        output_dir (Path): Directory for saving outputs and checkpoints
        logger (logging.Logger): Logging interface for training progress and debugging
        start_epoch (int): Epoch number to start training from (for resuming) - unitless
    """
    
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler, dataloaders: Dict[str, DataLoader], 
                 device: torch.device, cfg: Dict[str, Any], output_dir: str, logger: logging.Logger):
        """Initialize the base trainer with all necessary components.
        
        This constructor sets up the training infrastructure including model, loss function,
        optimizer, data loaders, and output directory. It also creates the checkpoint
        directory structure for saving training states and handles checkpoint resumption.
        
        Data Flow Context:
        - Input Components: Model processes sensor features (EGO, normalized), criterion computes losses on normalized data
        - DataLoader Structure: Provides collated batches with sensor_data (EGO, normalized features), 
          gt_boxes_b_normalized (EGO, normalized for training), gt_boxes_b_physical (EGO, unnormalized for evaluation)
        - Configuration: Contains normalization parameters, training hyperparameters, and pipeline settings
        
        Args:
            model (nn.Module): Transformer model to train - processes sensor features (EGO, normalized) 
                and generates predictions in normalized coordinate space for loss computation
            criterion (nn.Module): Loss function for training objectives - operates on normalized predictions 
                and ground truth targets (EGO, normalized) to compute training losses
            optimizer (torch.optim.Optimizer): Optimization algorithm for updating model parameters - unitless
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for adaptive learning rates - unitless
            dataloaders (Dict[str, DataLoader]): Dictionary containing train and validation data loaders
                - Data structure: sensor_data (EGO, normalized features), gt_boxes_b_normalized (EGO, normalized), 
                  gt_boxes_b_physical (EGO, unnormalized), ego_translation_world (WORLD, unnormalized)
            device (torch.device): Training device (CPU/GPU) - unitless
            cfg (Dict[str, Any]): Configuration dictionary containing training hyperparameters, 
                normalization settings, and pipeline configuration - unitless
            output_dir (str): Directory path for saving outputs and checkpoints - unitless
            logger (logging.Logger): Logger instance for training progress and debugging - unitless
            
        Returns:
            None: Constructor initializes the trainer but does not return any value.
            
        Note:
            No normalization or scaling operations are performed within this function.
            All data normalization is handled upstream in the DataLoader and dataset preprocessing.
        """
        # Store all training components for use throughout training process
        # These components handle data in different coordinate systems and normalization states
        self.model = model  # Processes sensor features (EGO, normalized) and generates normalized predictions
        self.criterion = criterion  # Computes losses on normalized predictions and targets (EGO, normalized)
        self.optimizer = optimizer  # Updates model parameters based on computed gradients
        self.lr_scheduler = lr_scheduler  # Adjusts learning rate based on validation performance
        self.dataloaders = dataloaders  # Provides collated batches with sensor_data (EGO, normalized) and ground truth
        self.device = device  # Computational device for tensor operations
        self.cfg = cfg  # Configuration containing normalization parameters and training settings
        self.output_dir = Path(output_dir)  # Directory for saving checkpoints and logs
        self.logger = logger  # Logging interface for training progress
        self.start_epoch = 0  # Epoch counter for training resumption (unitless)
        
        # Create output directory structure for checkpoints and logs
        # This ensures persistent storage of model states and training statistics
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Create main output directory
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)  # Create checkpoint subdirectory
        
        # Initialize Early Stopping if enabled
        self.early_stopping = None
        if cfg.get('training', {}).get('early_stopping', {}).get('enabled', False):
            early_stopping_cfg = cfg['training']['early_stopping']
            self.early_stopping = ValidationLossEarlyStopping(
                patience=early_stopping_cfg.get('patience', 3),
                min_delta=early_stopping_cfg.get('min_delta', 0.005),
                restore_best_weights=early_stopping_cfg.get('restore_best_weights', True),
                logger=self.logger
            )
            self.logger.info(f"🛑 Early Stopping enabled: patience={self.early_stopping.patience}, "
                           f"min_delta={self.early_stopping.min_delta}")
        else:
            self.logger.info("🔄 Early Stopping disabled")
        
    def train(self) -> None:
        """Execute the complete training loop with checkpoint management.
        
        This method manages the training process including epoch iteration,
        loss computation, optimization, evaluation, and checkpointing. It implements
        the template method pattern where the overall training structure is defined
        here, while specific training and evaluation logic is delegated to subclasses.
        
        Data Flow Context:
        - Input: DataLoader provides collated batches with sensor_data (EGO, normalized features),
          gt_boxes_b_normalized (EGO, normalized for training), gt_boxes_b_physical (EGO, unnormalized for evaluation)
        - Processing: Model forward pass on normalized features, loss computation on normalized predictions,
          gradient backpropagation, parameter updates, validation on normalized data
        - Output: Training statistics, validation metrics, model checkpoints, and optional DevKit evaluation results
        
        Coordinate System Handling:
        - Training Data: sensor_data features (EGO, normalized), gt_boxes_b_normalized (EGO, normalized)
        - Evaluation Data: gt_boxes_b_physical (EGO, unnormalized) for metric computation
        - Ego Pose: ego_translation_world (WORLD, unnormalized) for coordinate transformations
        
        The training loop performs the following steps for each epoch:
        1. Training phase with forward/backward passes and parameter updates on normalized data
        2. Learning rate scheduling based on validation performance
        3. Validation phase with model evaluation and metric computation using normalized inputs
        4. Logging of training and validation statistics for monitoring
        5. Checkpoint saving for model state persistence and resumption
        6. Optional DevKit evaluation for comprehensive performance metrics using unnormalized predictions
        
        Args:
            None: Uses class attributes for training configuration and components.
            
        Returns:
            None: Executes training loop but does not return any value.
            
        Note:
            No normalization or scaling operations are performed within this function.
            All data normalization is handled upstream in the DataLoader and dataset preprocessing.
            The model operates entirely on normalized data (EGO, normalized) for training and validation.
        """
        # Log training start and attempt to resume from checkpoint
        # This enables training resumption from previous states for long-running experiments
        self.logger.info("Starting training loop...")
        self.start_epoch = resume_from_checkpoint(
            self.model, self.optimizer, self.lr_scheduler, 
            self.output_dir, self.device, self.logger
        )  # Loads model state, optimizer state, and epoch counter from checkpoint
        
        # Record training start time for total duration calculation
        # This provides performance metrics for training efficiency analysis
        start_time = time.time()  # Unix timestamp for duration calculation (seconds)
        
        # Iterate through all training epochs from start_epoch to configured maximum
        # Each epoch processes the entire training dataset with forward/backward passes
        for epoch in range(self.start_epoch, self.cfg['training']['epochs']):
            self.logger.info(f"--- Epoch {epoch}/{self.cfg['training']['epochs'] - 1} ---")
            
            # Execute training phase for current epoch using subclass implementation
            # This processes all training batches with normalized sensor data (EGO, normalized)
            train_stats = self._train_one_epoch(epoch)  # Returns training statistics (unitless)
            
            # Update learning rate according to scheduler configuration
            # This adapts the learning rate based on validation performance for optimal convergence
            self.lr_scheduler.step()  # Adjusts learning rate based on validation metrics
            
            # Execute validation phase and collect predictions for evaluation
            # This evaluates model performance on validation data with normalized inputs
            val_loss_stats, raw_predictions_list = self._evaluate_model(epoch)  # Returns validation stats and predictions
            
            # Check Early Stopping if enabled
            should_stop = False
            if self.early_stopping is not None:
                val_loss = val_loss_stats.get('loss', float('inf'))
                should_stop = self.early_stopping(epoch, val_loss, self.model)
                
                if should_stop:
                    self.logger.warning(f"Early Stopping triggered at epoch {epoch}")
                    # Restore best weights if configured
                    if self.early_stopping.restore_best_weights:
                        self.early_stopping.restore_best_model(self.model)
                    # Log early stopping summary
                    summary = self.early_stopping.get_validation_summary()
                    self.logger.info(f"Early Stopping Summary: {summary}")
                    break  
            
            # Combine training and validation statistics for comprehensive logging
            # This creates a unified log entry for both training and validation metrics
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 
                        **{f'val_{k}': v for k, v in val_loss_stats.items()}, 
                        'epoch': epoch}  # Combined statistics dictionary (unitless)
            
            # Save checkpoint for current epoch to enable training resumption
            # This persists model state, optimizer state, and training progress
            save_checkpoint(self.model, self.optimizer, self.lr_scheduler, epoch, self.output_dir)

            # Write log statistics to file for persistence and analysis
            # This creates a persistent record of training progress for later analysis
            if self.output_dir:
                 with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(sanitize_for_json(log_stats)) + "\n")  # JSON-formatted log entry
            
            # Execute DevKit evaluation if configured and predictions are available
            # This performs comprehensive evaluation using unnormalized predictions for metric computation
            if self.cfg['evaluation']['run_devkit_eval'] and raw_predictions_list and epoch >= 0:
                self.logger.info(f"Running DevKit evaluation for epoch {epoch}...")
                results_folder = self.output_dir / 'eval_results' / f'epoch_{epoch}'  # Evaluation output directory
                eval_result = run_devkit_evaluation(raw_predictions_list=raw_predictions_list,  # Unnormalized predictions (EGO, unnormalized)
                                   config=self.cfg,  # Configuration for evaluation settings
                                   output_dir=str(results_folder))  # Output directory for evaluation results
                
                # Extract NDS score from metrics_summary.json and save best NDS checkpoint if evaluation successful
                if eval_result:
                    try:
                        # Load NDS score from the metrics_summary.json file saved by DevKit
                        metrics_file = results_folder / 'metrics_summary.json'
                        if metrics_file.exists():
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                            nds_score = metrics_data.get('all', {}).get('nd_score', None)
                            
                            if nds_score is not None:
                                self.logger.info(f"📊 Current NDS Score: {nds_score:.6f}")
                                
                                is_best = save_best_nds_checkpoint(
                                    model=self.model,
                                    optimizer=self.optimizer, 
                                    lr_scheduler=self.lr_scheduler,
                                    epoch=epoch,
                                    nds_score=nds_score,
                                    output_dir=self.output_dir,
                                    logger=self.logger
                                )
                                
                                if is_best:
                                    self.logger.info(f"🎯 New best model saved with NDS: {nds_score:.6f}")
                            else:
                                self.logger.warning("NDS score not found in metrics_summary.json")
                                self.logger.warning("Best checkpoint saving skipped for this epoch.")
                        else:
                            self.logger.warning(f"Metrics summary file not found: {metrics_file}")
                            self.logger.warning("Best checkpoint saving skipped for this epoch.")
                            
                    except Exception as e:
                        self.logger.warning(f"Could not extract NDS score from evaluation: {e}")
                        self.logger.warning("Best checkpoint saving skipped for this epoch.")
                
            elif self.cfg['evaluation']['run_devkit_eval']:
                self.logger.info(f"Skipping DevKit evaluation for epoch {epoch} (unstable in early epochs).")

            # Execute visualization after each epoch if enabled (BEFORE cache cleanup)
            # This generates visual representations of model predictions for qualitative analysis
            if self.cfg['visualization']['enabled']:
                try:
                    from oft.transformer.utils.epoch_visualization_workflow import visualize_val_sample_from_dataset
                    val_dataset = self.dataloaders['val'].dataset if 'val' in self.dataloaders else None
                    if val_dataset is not None:
                        # Generate visualization using validation dataset and current epoch
                        visualize_val_sample_from_dataset(self.cfg, val_dataset, epoch, str(self.output_dir))
                        self.logger.info(f"Visualization for epoch {epoch} successfully generated.")
                    else:
                        self.logger.warning("Visualization skipped: No validation dataset found.")
                except Exception as e:
                    self.logger.error(f"Error during visualization in epoch {epoch}: {e}")

        # Clean up RAM cache after training completion (after all epochs and visualizations)
        # This keeps RAM cache persistent throughout training for optimal performance
        val_dataset = self.dataloaders['val'].dataset if 'val' in self.dataloaders else None
        if val_dataset and hasattr(val_dataset, 'cleanup_ram_cache'):
            val_dataset.cleanup_ram_cache()

        # Calculate and log total training time for performance analysis
        # This provides overall training duration for efficiency assessment
        total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))  # Formatted duration string
        self.logger.info(f'Training completed. Total time: {total_time_str}')
        
        # Optional model archiving for DoE System support
        if self.cfg.get('training', {}).get('archive_model', False):
            self._archive_model_to_collection()
        
        # Final early stopping summary if used
        if self.early_stopping is not None:
            final_summary = self.early_stopping.get_validation_summary()
            self.logger.info(f"🏁 Final Early Stopping Summary: {final_summary}")
    
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch. Must be implemented by subclasses.
        
        This abstract method defines the interface for epoch-level training.
        Subclasses must implement the specific training logic for their
        model architecture and data processing requirements. The method
        should handle all aspects of training for a single epoch including
        batch iteration, forward/backward passes, and parameter updates.
        
        Data Flow Context:
        - Input: DataLoader batches containing sensor_data (EGO, normalized features), 
          gt_boxes_b_normalized (EGO, normalized for training), and gt_boxes_b_physical (EGO, unnormalized for evaluation)
        - Processing: Model forward pass on normalized features, loss computation on normalized predictions,
          gradient backpropagation, parameter updates
        - Output: Training statistics for the epoch including loss values and other metrics
        
        Coordinate System Handling:
        - Input Features: sensor_data features (EGO, normalized) - 12D features for model input
        - Training Targets: gt_boxes_b_normalized (EGO, normalized) - 10D boxes for loss computation
        - Evaluation Targets: gt_boxes_b_physical (EGO, unnormalized) - 9D boxes for metric computation
        
        Args:
            epoch (int): Current epoch number for logging and scheduling purposes - unitless
            
        Returns:
            Dict[str, float]: Dictionary containing training statistics for the epoch,
                including loss values and other relevant metrics - all values unitless
                
        Raises:
            NotImplementedError: If subclass does not implement this method.
            
        Note:
            No normalization or scaling operations are performed within this function.
            All data normalization is handled upstream in the DataLoader and dataset preprocessing.
            The method operates entirely on normalized data (EGO, normalized) for training.
        """
        raise NotImplementedError("Subclasses must implement _train_one_epoch")

    @torch.no_grad()
    def _evaluate_model(self, epoch: int) -> Tuple[Dict[str, float], List[Any]]:
        """Evaluate the model. Must be implemented by subclasses.
        
        This abstract method defines the interface for model evaluation.
        Subclasses must implement the specific evaluation logic, including
        prediction generation, metric computation, and result collection.
        The method should set the model to evaluation mode and perform
        inference without gradient computation for efficiency.
        
        Data Flow Context:
        - Input: Validation DataLoader batches containing sensor_data (EGO, normalized features),
          gt_boxes_b_normalized (EGO, normalized for loss computation), gt_boxes_b_physical (EGO, unnormalized for metrics)
        - Processing: Model forward pass on normalized features, prediction generation, metric computation
        - Output: Validation statistics and raw predictions for external evaluation (e.g., DevKit)
        
        Coordinate System Handling:
        - Input Features: sensor_data features (EGO, normalized) - 12D features for model input
        - Loss Targets: gt_boxes_b_normalized (EGO, normalized) - 10D boxes for validation loss computation
        - Metric Targets: gt_boxes_b_physical (EGO, unnormalized) - 9D boxes for evaluation metrics
        - Predictions: Output in normalized space (EGO, normalized) for loss computation, 
          converted to unnormalized space (EGO, unnormalized) for metric computation
        
        Args:
            epoch (int): Current epoch number for logging purposes - unitless
            
        Returns:
            Tuple[Dict[str, float], List[Any]]: A tuple containing:
                - **validation_stats** (Dict[str, float]): Dictionary containing evaluation metrics - unitless
                - **predictions** (List[Any]): List of predictions for external evaluation (e.g., DevKit) 
                  - Format: Unnormalized predictions (EGO, unnormalized) for metric computation
                
        Raises:
            NotImplementedError: If subclass does not implement this method.
            
        Note:
            No normalization or scaling operations are performed within this function.
            All data normalization is handled upstream in the DataLoader and dataset preprocessing.
            The method operates on normalized inputs but may output unnormalized predictions for evaluation.
        """
        raise NotImplementedError("Subclasses must implement _evaluate_model")

    def _archive_model_to_collection(self) -> None:
        """Archive the best trained model to the central model collection.
        
        This method copies the configuration file and best NDS checkpoint to a central
        model collection directory structure for later use in DoE campaigns. Only
        executed if archive_model is True and model_identifier is provided in config.
        
        The archived model structure allows systematic model selection for testing
        across multiple DoE configurations while maintaining clear referenceability.
        
        Directory Structure Created:
        /data/daiber_fent/models/{model_identifier}/
        ├── config.yaml       # Complete training configuration for reproducibility
        └── best_nds.pth      # Best model weights based on NDS score
        
        Args:
            None: Uses class attributes for configuration and paths.
            
        Returns:
            None: Performs file operations without return value.
            
        Raises:
            FileNotFoundError: If best_nds.pth checkpoint file does not exist
            ValueError: If model_identifier is not provided in configuration
            OSError: If directory creation or file copying operations fail
        """
        # Extract model identifier from configuration
        model_identifier = self.cfg.get('training', {}).get('model_identifier', '').strip()
        
        if not model_identifier:
            self.logger.error("Model archiving enabled but no model_identifier provided in config!")
            self.logger.error("Please set training.model_identifier in your config (e.g., 'model_noisefree')")
            return
            
        # Define archive directory structure  
        models_root = Path("/data/daiber_fent/models")
        model_archive_dir = models_root / model_identifier
        
        try:
            # Create model archive directory
            model_archive_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Model archive directory created: {model_archive_dir}")
            
            # Archive training configuration for reproducibility
            config_source = Path("config/pipeline_staged.yaml")  # Current config used for training
            config_target = model_archive_dir / "config.yaml"
            
            if config_source.exists():
                shutil.copy2(config_source, config_target)
                self.logger.info(f"📋 Training config archived: {config_target}")
            else:
                self.logger.warning(f"Config file not found at {config_source}, skipping config archiving")
            
            # Archive best NDS checkpoint
            best_nds_source = self.output_dir / "checkpoints" / "best_nds.pth"
            best_nds_target = model_archive_dir / "best_nds.pth"
            
            if best_nds_source.exists():
                shutil.copy2(best_nds_source, best_nds_target)
                self.logger.info(f"🏆 Best NDS checkpoint archived: {best_nds_target}")
                
                # Log best NDS info if available
                best_info_path = self.output_dir / "checkpoints" / "best_nds_info.json"
                if best_info_path.exists():
                    try:
                        with open(best_info_path, 'r') as f:
                            best_info = json.load(f)
                            best_nds = best_info.get('best_nds', 'Unknown')
                            best_epoch = best_info.get('best_epoch', 'Unknown')
                            self.logger.info(f"📊 Archived model - Best NDS: {best_nds}, Best Epoch: {best_epoch}")
                    except Exception as e:
                        self.logger.warning(f"Could not read best NDS info: {e}")
            else:
                self.logger.error(f"Best NDS checkpoint not found at {best_nds_source}")
                self.logger.error("This usually means no DevKit evaluation was performed during training")
                return
            
            self.logger.info(f"✅ Model successfully archived to: {model_archive_dir}")
            self.logger.info(f"🔍 Use model_identifier '{model_identifier}' for DoE testing")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to archive model: {e}")
            self.logger.error("Model archiving failed, but training results are still saved in output directory") 