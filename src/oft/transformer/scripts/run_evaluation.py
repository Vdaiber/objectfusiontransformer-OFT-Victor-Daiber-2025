#!/usr/bin/env python3
"""
Standalone Model Evaluation Script.

This script loads a pre-trained model checkpoint and runs evaluation against a
specific dataset configuration. It is designed to replicate the validation loop
from the training pipeline, ensuring consistent evaluation metrics.

This script is the definitive method for evaluating a model's performance on a
dataset, as it bypasses any potential inconsistencies in archived configuration
files by using a specified, current configuration to build the model architecture
before loading the saved weights.

Usage:
    python src/oft/transformer/scripts/run_evaluation.py \\
        --model-identifier baseline_sensors_1 \\
        --config-path /path/to/your/evaluation_config.yaml
"""
import argparse
import logging
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive
from oft.transformer.datasets.loaders.autoregressive_loader import build_autoregressive_dataloaders
from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion
from oft.transformer.training.trainers.autoregressive_trainer import AutoregressiveTrainer
from oft.transformer.utils.reproducibility import set_seed

def main():
    """Main entry point for the standalone evaluation script."""
    parser = argparse.ArgumentParser(description="Run standalone model evaluation.")
    parser.add_argument('--model-identifier', type=str, required=True, help='Identifier of the model to evaluate (e.g., baseline_sensors_1)')
    parser.add_argument('--config-path', type=str, required=True, help='Path to the YAML configuration file to use for evaluation.')
    parser.add_argument('--models-root', type=str, default="/data/daiber_fent/models", help='Root directory of the model archive')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("StandaloneEvaluation")

    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {args.config_path}")
    cfg = OmegaConf.load(args.config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # --- 2. Setup Environment ---
    set_seed(cfg_dict['training']["seed"], logger)
    device = torch.device(cfg_dict['training']['device'])
    
    # --- 3. Build Model, Criterion, and DataLoaders ---
    logger.info("Building model, criterion, and dataloaders...")
    dataloaders = build_autoregressive_dataloaders(cfg_dict, logger)
    model = ObjectFusionTransformerAutoregressive(cfg_dict).to(device)
    criterion = SetCriterion(
        weight_dict=cfg_dict['loss']['loss_weight_dict'],
        losses=cfg_dict['loss']['losses_to_compute'],
        matcher=None,
        eos_coef=float(cfg_dict['loss']['class_eos_coefficient']),
        cfg=cfg_dict
    ).to(device)

    # --- 4. Load Model Weights ---
    checkpoint_path = Path(args.models_root) / args.model_identifier / "best_nds.pth"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)
    
    logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    logger.info(f"Successfully loaded model weights. Best NDS: {checkpoint.get('nds_score', 'N/A')}, Epoch: {checkpoint.get('epoch', 'N/A')}")

    # --- 5. Run Evaluation ---
    # We can reuse the trainer's validation method directly.
    # We don't need an optimizer or lr_scheduler for evaluation.
    trainer = AutoregressiveTrainer(
        model=model,
        criterion=criterion,
        optimizer=None,
        lr_scheduler=None,
        dataloaders=dataloaders,
        device=device,
        cfg=cfg_dict,
        output_dir=Path("./evaluation_output"), # Temporary output
        logger=logger
    )

    logger.info("Starting evaluation...")
    eval_stats = trainer._validate_one_epoch(0) # Epoch number is arbitrary for standalone eval
    
    logger.info("--- Evaluation Complete ---")
    final_nds = eval_stats.get('NDS', 0.0)
    logger.info(f"Final NDS Score: {final_nds:.6f}")
    logger.info("---------------------------")
    
    # Print the final score for orchestrator to capture
    print(f"FINAL_NDS_SCORE:{final_nds}")

if __name__ == "__main__":
    main()
