#!/usr/bin/env python3
"""
Single Evaluation Script - Hydra Application (like training script).

This script is structured exactly like run_training_autoregressive.py but only
runs evaluation. It uses the same Hydra configuration management.
"""
import logging
import sys
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive
from oft.transformer.training.trainers.autoregressive_trainer import AutoregressiveTrainer
from oft.transformer.datasets.loaders.autoregressive_loader import build_autoregressive_dataloaders
from oft.transformer.training.criteria.autoregressive_criterion import SetCriterion
from oft.transformer.utils.reproducibility import set_seed

@hydra.main(config_path="/app/config", config_name="pipeline_staged", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main evaluation function - structured like the training script."""
    
    # Setup exactly like training script
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
    logger = logging.getLogger("OFT_Single_Evaluation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    set_seed(cfg_dict['training']["seed"], logger)
    device = torch.device(cfg_dict['training']['device'])
    
    # Get model identifier from config
    model_identifier = cfg.training.model_identifier
    logger.info(f"Evaluating model: {model_identifier}")
    
    try:
        # Build components exactly like training
        dataloaders = build_autoregressive_dataloaders(cfg_dict, logger, splits_to_build=['val'])
        model = ObjectFusionTransformerAutoregressive(cfg_dict).to(device)
        criterion = SetCriterion(
            weight_dict=cfg_dict['loss']['loss_weight_dict'],
            losses=cfg_dict['loss']['losses_to_compute'],
            matcher=None,
            eos_coef=float(cfg_dict['loss']['class_eos_coefficient']),
            cfg=cfg_dict
        ).to(device)

        # Load model checkpoint
        model_dir = Path(cfg_dict['training']['model_archive_dir']) / model_identifier
        checkpoint_path = model_dir / "best_nds.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle different checkpoint formats and architectures
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out keys that don't exist in current model architecture
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state_dict:
                filtered_state_dict[key] = value
            else:
                logger.warning(f"Skipping checkpoint key '{key}' - not in current model architecture")
        
        # Load only compatible weights
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        model.eval()
        
        logger.info(f"✅ Model loaded. Best NDS from training: {checkpoint.get('nds_score', 'N/A')}")

        # Create trainer exactly like training script
        trainer = AutoregressiveTrainer(
            model=model,
            criterion=criterion,
            optimizer=None,
            lr_scheduler=None,
            dataloaders=dataloaders,
            device=device,
            cfg=cfg_dict,
            output_dir=output_dir,
            logger=logger
        )
        
        logger.info("🔬 Generating predictions for DevKit evaluation...")
        
        # Generate predictions without loss calculation (more efficient)
        all_predictions = []
        model.eval()
        
        with torch.no_grad():
            for batch_dict in dataloaders['val']:
                # Move batch to device
                batch_dict = trainer._prepare_batch_for_device(batch_dict)
                
                # Generate predictions (no loss calculation needed)
                predictions = model(
                    sensor_data=batch_dict['sensor_data'],
                    memory=None,  # Reset for each batch in evaluation
                    memory_anchor_boxes=None,
                    dt=None,
                    ego_pose_current=None,
                    ego_pose_previous=None,
                    scene_meta=batch_dict.get('scene_meta', None),
                    logger=logger
                )
                
                # Convert predictions to evaluation format
                from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive
                reconstructed_preds = reconstruct_and_convert_predictions_autoregressive(
                    batch_dict=batch_dict,
                    predictions=predictions[0],  # predictions is tuple (predictions, memory, memory_boxes)
                    cfg=cfg_dict
                )
                all_predictions.extend(reconstructed_preds)
        
        logger.info(f"✅ Prediction generation complete. Got {len(all_predictions)} predictions.")
        
        # CRITICAL: Cleanup DataLoader cache immediately (but keep predictions for DevKit)
        if 'val' in dataloaders and hasattr(dataloaders['val'].dataset, 'cleanup_ram_cache'):
            dataloaders['val'].dataset.cleanup_ram_cache()
            logger.info("🧹 DataLoader RAM cache cleaned up")
        
        # Clear GPU cache of intermediate tensors (but keep model loaded)
        torch.cuda.empty_cache()
        logger.info("🧹 GPU intermediate tensors cleared")
        
        # Now run DevKit evaluation (this is what you want!)
        from oft.transformer.evaluation.devkit_evaluator import run_devkit_evaluation
        
        logger.info("🔬 Running TruckScenes DevKit evaluation...")
        # Use temporary directory for DevKit output, then copy metrics to experiment folder
        import tempfile
        with tempfile.TemporaryDirectory() as temp_eval_dir:
            eval_result = run_devkit_evaluation(
                raw_predictions_list=all_predictions,
                config=cfg_dict,
                output_dir=temp_eval_dir
            )
            
            # Extract NDS score exactly like your training code does
            nds_score = 0.0
            if eval_result:
                try:
                    import json
                    import shutil
                    temp_metrics_file = Path(temp_eval_dir) / 'metrics_summary.json'
                    if temp_metrics_file.exists():
                        with open(temp_metrics_file, 'r') as f:
                            metrics_data = json.load(f)
                        nds_score = metrics_data.get('all', {}).get('nd_score', 0.0)
                        logger.info(f"📊 Extracted NDS from DevKit: {nds_score:.6f}")
                        
                        # Copy metrics_summary.json to experiment folder (passed by orchestrator)
                        if hasattr(cfg, 'experiment_output_dir') and cfg.experiment_output_dir:
                            final_metrics_path = Path(cfg.experiment_output_dir) / 'metrics_summary.json'
                        else:
                            final_metrics_path = output_dir / 'metrics_summary.json'
                        shutil.copy2(temp_metrics_file, final_metrics_path)
                        logger.info(f"💾 Metrics saved to: {final_metrics_path}")
                    else:
                        logger.warning("metrics_summary.json not found in DevKit output")
                except Exception as e:
                    logger.warning(f"Could not extract NDS score: {e}")
            
        logger.info(f"✅ DevKit evaluation complete. Final NDS: {nds_score:.6f}")
        
        # CRITICAL: Final cleanup after DevKit evaluation
        del all_predictions  # Now we can safely delete predictions
        torch.cuda.empty_cache()  # Final GPU cleanup
        logger.info("🧹 Final cleanup: predictions deleted, GPU cache cleared")
        
        # Print for orchestrator
        print(f"FINAL_NDS_SCORE:{nds_score}")

    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
