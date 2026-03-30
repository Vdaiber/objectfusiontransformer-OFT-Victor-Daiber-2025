# ./app/src/oft/transformer/run_training_autoregressive.py
"""
Autoregressive Training Pipeline for Transformer-Based Sensor Fusion Models.

This script implements the complete training pipeline for autoregressive transformer-based
sensor fusion models that generate predictions sequentially, where each prediction depends
on previous outputs. The pipeline handles data loading, model initialization, loss function
configuration with Hungarian matching, optimizer setup, and training execution with
integration to Hydra configuration management.


This file is essential for the training phase as it orchestrates the complete end-to-end
training pipeline, ensuring proper data flow from DataLoaders through model forward passes
to loss computation and optimization. It serves as the main entry point for autoregressive
training experiments and manages the complete training lifecycle including logging,
checkpointing, and experiment tracking.
"""

import torch
import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any

# Set timezone for consistent timestamps across all logging and experiment tracking
# This ensures reproducible experiment timestamps regardless of system timezone
os.environ['TZ'] = 'Europe/Berlin'
try:
    time.tzset()  # Only works on Unix systems
except AttributeError:
    pass  # Windows doesn't have tzset

import hydra
from omegaconf import DictConfig, OmegaConf

from oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive
from oft.transformer.datasets.loaders.autoregressive_loader import build_autoregressive_dataloaders
from oft.transformer.utils.reproducibility import set_seed
from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion
from oft.transformer.training.trainers.autoregressive_trainer import AutoregressiveTrainer
from oft.transformer.training.utils.baseline_integration import (
    run_baseline_evaluation_for_training,
    should_run_baseline_evaluation
)


def build_model_autoregressive(cfg: Dict[str, Any], logger: logging.Logger) -> ObjectFusionTransformerAutoregressive:
    """Initialize and configure the autoregressive transformer model for sensor fusion applications.
    
    Creates and configures the ObjectFusionTransformerAutoregressive model according to
    the provided configuration parameters. Performs comprehensive parameter counting and
    logging for model complexity assessment, experiment tracking, and computational
    resource planning. The function handles model architecture instantiation with
    proper parameter initialization and gradient computation setup.
    
    This function is called during the training initialization phase to create the
    autoregressive transformer model that will process multi-modal sensor data
    (LiDAR, camera, radar) in sequential frame-by-frame manner, maintaining temporal
    memory across frames for consistent object tracking and prediction.
    
    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing model architecture 
            parameters including transformer dimensions, attention mechanisms, 
            encoder/decoder configurations, and sensor fusion specifications.
            Expected structure:
            - **'model'** (Dict[str, Any]): Model architecture configuration
            - **'transformer'** (Dict[str, Any]): Transformer-specific parameters
            - **'sensor_fusion'** (Dict[str, Any]): Multi-modal fusion settings
        logger (logging.Logger): Logger instance for recording initialization progress,
            parameter counts, and potential configuration issues during model setup.
            Used for experiment tracking and architecture transparency.
        
    Returns:
        ObjectFusionTransformerAutoregressive: Fully initialized autoregressive 
            transformer model instance with all architectural components configured
            according to the provided configuration parameters. The model expects:
            - **Input data**: Multi-modal sensor features (EGO, normalized) from DataLoader
            - **Memory**: Previous frame predictions and anchor boxes (EGO, normalized)
            - **Output**: Object predictions with 10D boxes (EGO, normalized) for transformer
              and 9D boxes (WORLD, unnormalized) for evaluation
            
    Note:
        No normalization or scaling operations are performed within this function.
        The function focuses solely on model architecture initialization and parameter
        counting for computational complexity assessment. The model itself handles
        coordinate system transformations and normalization internally during forward passes.
    """
    logger.info("Initializing ObjectFusionTransformerAutoregressive...")
    
    # Instantiate model with provided configuration parameters
    # The model architecture is defined in autoregressive_architecture.py and handles
    # multi-modal sensor fusion with temporal memory for autoregressive prediction
    model = ObjectFusionTransformerAutoregressive(cfg)
    
    # Calculate total parameter count for model complexity assessment and experiment tracking
    # This includes all parameters regardless of training status for model size analysis
    total_params = sum(p.numel() for p in model.parameters())
    # Count only trainable parameters (excludes frozen layers if any) for gradient computation planning
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log parameter statistics for experiment tracking and computational resource planning
    # These values are used for model complexity analysis and training efficiency assessment
    logger.info(f"Model created. Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    return model


def build_criterion(cfg: Dict[str, Any], logger: logging.Logger) -> SetCriterion:
    """Initialize the loss computation pipeline with Hungarian matching algorithm for optimal assignment.
    
    Sets up the complete loss computation pipeline including the Hungarian matcher for
    optimal bipartite assignment between predictions and ground truth objects, and the
    SetCriterion for computing weighted multi-task losses. The Hungarian algorithm
    ensures optimal matching between predicted and ground truth objects by minimizing
    the total assignment cost based on classification, bounding box, velocity, and
    attribute similarities.
    
    This function is called during training initialization to create the loss computation
    pipeline that will evaluate model predictions against ground truth annotations.
    The criterion handles the complex task of matching predicted objects to ground truth
    objects and computing multiple loss components (classification, regression, attributes).
    
    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing loss function 
            parameters including cost weights for different loss components, 
            end-of-sequence coefficients, and loss computation specifications.
            Expected structure:
            - **'loss'** (Dict[str, Any]): Loss configuration containing:
                - **'losses_to_compute'** (List[str]): List of loss types to compute
                - **'loss_weight_dict'** (Dict[str, float]): Weight coefficients for each loss type
                - **'class_eos_coefficient'** (float): End-of-sequence coefficient for classification
        logger (logging.Logger): Logger instance for recording initialization progress
            and loss function configuration details.
        
    Returns:
        SetCriterion: Configured loss criterion instance with integrated Hungarian 
            matcher, containing:
            - **matcher** (HungarianMatcher): Optimal assignment algorithm for 
              prediction-ground truth matching using cost matrices based on:
              - Classification probabilities (unnormalized logits)
              - Bounding box similarities (EGO, normalized coordinates)
              - Velocity predictions (EGO, normalized)
              - Attribute classifications (unnormalized logits)
            - **weight_dict** (Dict[str, float]): Loss weight coefficients for different 
              loss components (classification, regression, attributes, etc.)
            - **eos_coef** (float): End-of-sequence coefficient for handling 
              variable-length sequences in classification loss computation
            - **losses** (List[str]): List of loss types to compute during training
              (e.g., 'loss_class', 'loss_center', 'loss_size', 'loss_angle', 'loss_velocity')
            
    Note:
        No normalization or scaling operations are performed within this function.
        The function configures loss computation parameters and matching algorithms
        without applying any data transformations. The criterion expects:
        - **Model outputs**: Predictions in normalized coordinate system (EGO, normalized)
        - **Ground truth**: Annotations in normalized coordinate system (EGO, normalized)
        - **Loss computation**: Internal normalization and coordinate transformations as needed
    """
    logger.info("Initializing loss function and matcher...")
    
    # Extract loss configuration with fallback to empty dict if not present
    # This ensures robust configuration handling even with incomplete config files
    loss_cfg = cfg.get('loss', {})
    
    # Extract loss computation specifications from configuration
    # These define which loss components will be computed during training
    criterion_losses = loss_cfg['losses_to_compute']  # List of loss types to compute (e.g., ['loss_class', 'loss_center'])
    criterion_weight_dict = loss_cfg['loss_weight_dict']  # Weight coefficients for each loss type (e.g., {'loss_class': 1.0, 'loss_center': 2.0})
        
    # Initialize SetCriterion with Hungarian matcher and loss configuration
    # The SetCriterion will create the HungarianMatcher internally with Loss Functions
    # This creates the complete loss computation pipeline for autoregressive training
    criterion = SetCriterion(
        weight_dict=criterion_weight_dict,  # Loss weight coefficients for multi-task learning
        losses=criterion_losses,            # List of loss types to compute during training
        matcher=None,                       # Will be created internally by SetCriterion with Hungarian algorithm
        # Use consistent key from YAML configuration for end-of-sequence coefficient
        eos_coef=float(loss_cfg['class_eos_coefficient']),  # End-of-sequence coefficient for classification loss (handles class imbalance)
        cfg=cfg                            # Full configuration for loss function setup
    )
    
    return criterion


@hydra.main(config_path="/app/config", config_name="pipeline_staged.yaml", version_base=None)
def main(cfg: DictConfig):
    """Orchestrate the complete autoregressive training pipeline for sensor fusion models.
    
    Executes the end-to-end training pipeline for autoregressive transformer-based
    sensor fusion models. The function handles configuration management through Hydra,
    data loading with staged processing, model initialization, optimization setup
    with AdamW optimizer and step learning rate scheduling, and training execution
    with comprehensive logging and experiment tracking. The pipeline ensures
    reproducibility through seed setting and provides automatic experiment
    management via Hydra's output directory structure.
    
    This function serves as the main entry point for autoregressive training experiments.
    It orchestrates the complete training lifecycle from data loading through model
    training to evaluation and checkpointing. The function manages the data flow
    from TruckScenes dataset through the autoregressive transformer model to loss
    computation and optimization.
    
    Args:
        cfg (DictConfig): Hydra configuration object containing all training 
            parameters including model architecture, data loading, optimization,
            and training specifications. The configuration is automatically
            resolved and converted to a Python dictionary for processing.
            Expected structure:
            - **'dataset'** (Dict[str, Any]): Dataset configuration (dataroot, version, splits)
            - **'training'** (Dict[str, Any]): Training hyperparameters (batch_size, learning_rate, etc.)
            - **'model'** (Dict[str, Any]): Model architecture configuration
            - **'loss'** (Dict[str, Any]): Loss function configuration
            - **'optimizer'** (Dict[str, Any]): Optimization settings
        
    Returns:
        None: The function executes the training pipeline without returning
            explicit values. Training artifacts, logs, and model checkpoints
            are saved to the Hydra-managed output directory structure.
            
    Note:
        No normalization or scaling operations are performed within this function.
        The function orchestrates the training pipeline components without
        applying data transformations directly. Data flow:
        1. DataLoader provides sensor data (EGO, normalized) and ground truth (EGO, normalized)
        2. Model processes data and produces predictions (EGO, normalized)
        3. Criterion computes losses using Hungarian matching and multi-task loss functions
        4. Optimizer updates model parameters based on computed gradients
    """
    # Initialize Hydra-managed output directory for experiment tracking
    # This creates a unique directory for each training run with timestamp and run ID
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
    
    # Configure logger with specific name for autoregressive training
    # This ensures clear identification of log sources during multi-process training
    logger = logging.getLogger("OFT_Autoregressive_Training")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate log messages from parent loggers
    
    # Add console handler if no handlers exist to ensure logging output
    # This guarantees that log messages are displayed even if no handlers were configured
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(f"Logs will be saved to: {output_dir}")

    # Convert Hydra configuration to Python dictionary and resolve all references
    # This ensures all configuration values are properly resolved and accessible
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Set random seed for reproducible training
    # This ensures consistent results across different training runs with same seed
    set_seed(cfg_dict['training']["seed"], logger)
    
    # Determine computational device with CUDA preference if available
    # This automatically selects the best available device for training
    device = torch.device(cfg_dict['training']['device'])
    
    # Initialize data loaders with autoregressive mode for sequential processing
    # This creates DataLoaders that provide sensor data (EGO, normalized) and ground truth (EGO, normalized)
    # The autoregressive mode ensures sequential frame processing for temporal consistency
    dataloaders = build_autoregressive_dataloaders(cfg_dict, logger)
    
    # Initialize model and criterion, then move to specified device
    # The model will process sensor data and produce predictions (EGO, normalized)
    # Use float32 for training; high-precision world transforms are handled selectively elsewhere
    model = build_model_autoregressive(cfg_dict, logger).to(device)
    # The criterion will compute losses using Hungarian matching and multi-task loss functions
    criterion = build_criterion(cfg_dict, logger).to(device)

    # Configure AdamW optimizer with component-specific learning rates
    # AdamW provides better weight decay behavior compared to standard Adam optimizer
    
    # Check if component-specific learning rates are configured
    class_head_lr_factor = cfg_dict['training'].get('class_head_lr_factor', 1.0)
    attribute_head_lr_factor = cfg_dict['training'].get('attribute_head_lr_factor', 1.0)
    base_lr = cfg_dict['training']['learning_rate']
    
    if class_head_lr_factor != 1.0 or attribute_head_lr_factor != 1.0:
        # Create parameter groups with different learning rates
        class_head_params = [p for n, p in model.named_parameters() if 'decoder.class_head' in n]
        attribute_head_params = [p for n, p in model.named_parameters() if 'decoder.attribute_head' in n]
        other_params = [p for n, p in model.named_parameters() if 'decoder.class_head' not in n and 'decoder.attribute_head' not in n]
        
        param_groups = []
        if class_head_lr_factor != 1.0:
            param_groups.append({'params': class_head_params, 'lr': base_lr * class_head_lr_factor})
        else:
            other_params.extend(class_head_params)
            
        if attribute_head_lr_factor != 1.0:
            param_groups.append({'params': attribute_head_params, 'lr': base_lr * attribute_head_lr_factor})
        else:
            other_params.extend(attribute_head_params)
            
        param_groups.append({'params': other_params, 'lr': base_lr})
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg_dict['training']['weight_decay'])
        
        logger.info(f"Component-specific learning rates:")
        if class_head_lr_factor != 1.0:
            logger.info(f"  - Class Head LR: {base_lr * class_head_lr_factor:.6f} (factor: {class_head_lr_factor})")
        if attribute_head_lr_factor != 1.0:
            logger.info(f"  - Attribute Head LR: {base_lr * attribute_head_lr_factor:.6f} (factor: {attribute_head_lr_factor})")
        logger.info(f"  - Other params LR: {base_lr:.6f}")
    else:
        # Standard optimizer with single learning rate
        optimizer = torch.optim.AdamW(
            model.parameters(),  # All trainable parameters from the autoregressive model
            lr=base_lr,  # Initial learning rate for gradient updates
            weight_decay=cfg_dict['training']['weight_decay']  # Weight decay coefficient for regularization
        )
    
    # Initialize step learning rate scheduler with configurable drop frequency
    # This reduces learning rate at specified epoch intervals to improve convergence
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg_dict['training']['lr_drop_epoch'],  # Epoch interval for learning rate reduction
        gamma=cfg_dict['training']['lr_scheduler_gamma']  # Learning rate reduction factor
    )
    
    # Initialize autoregressive trainer with all components
    # The trainer orchestrates the complete training loop including forward passes,
    # loss computation, optimization, and evaluation
    trainer = AutoregressiveTrainer(
        model=model,           # Autoregressive transformer model for sensor fusion
        criterion=criterion,   # Loss computation pipeline with Hungarian matching
        optimizer=optimizer,   # AdamW optimizer for parameter updates
        lr_scheduler=lr_scheduler,  # Learning rate scheduler for convergence optimization
        dataloaders=dataloaders,    # DataLoaders providing sensor data (EGO, normalized)
        device=device,              # Computational device (CUDA/CPU)
        cfg=cfg_dict,               # Complete configuration for training pipeline
        output_dir=output_dir,      # Directory for saving checkpoints and logs
        logger=logger               # Logger for training progress tracking
    )
    
    # *** BASELINE EVALUATION: Run before training for performance comparison ***
    if should_run_baseline_evaluation(cfg_dict):
        baseline_results = run_baseline_evaluation_for_training(cfg_dict, output_dir, logger)
        # Store baseline results in trainer for later comparison
        trainer.baseline_results = baseline_results
    else:
        logger.info("Skipping baseline evaluation (disabled or resumed training)")
        trainer.baseline_results = {}
    
    # Execute the complete training pipeline
    # This starts the training loop that processes sensor data sequentially,
    # computes losses using Hungarian matching, and updates model parameters
    trainer.train()


if __name__ == "__main__":
    main()