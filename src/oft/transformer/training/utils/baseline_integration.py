#!/usr/bin/env python3
"""
Integration utility for baseline evaluation in training pipeline.

This module provides functions to integrate baseline sensor evaluation
into the training process for performance comparison.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any


def run_baseline_evaluation_for_training(cfg: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Run baseline evaluation before training starts.
    
    Args:
        cfg: Training configuration dictionary
        output_dir: Training output directory
        logger: Logger instance
        
    Returns:
        Dictionary with baseline results per sensor
    """
    logger.info("\n" + "="*70)
    logger.info("🔍 RUNNING BASELINE SENSOR EVALUATION")
    logger.info("="*70)
    logger.info("Evaluating individual virtual sensors without transformer processing...")
    logger.info("This provides baseline metrics for comparison with full pipeline results.")
    
    try:
        # Import the baseline evaluator (correct import path for training context)
        from oft.transformer.scripts.evaluate_virtual_sensors_baseline import VirtualSensorBaselineEvaluator
        
        # Use the SAME output directory as training (not a subdirectory)
        baseline_output_dir = output_dir / "baseline_sensors"
        baseline_output_dir.mkdir(exist_ok=True)
        
        # Create temporary config file path for the evaluator
        config_path = "config/pipeline_staged.yaml"  # Use the current config
        
        # Run baseline evaluation (simplified)
        evaluator = VirtualSensorBaselineEvaluator(config_path, str(baseline_output_dir))
        baseline_results = evaluator.run_baseline_evaluation()
        
        # Log summary
        logger.info("\n📊 BASELINE EVALUATION SUMMARY:")
        logger.info("-" * 50)
        
        for sensor_name, metrics in baseline_results.items():
            logger.info(f"\n{sensor_name.upper()}:")
            if metrics:
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")
            else:
                logger.info("  No metrics available")
        
        logger.info(f"\n💾 Baseline results saved to: {baseline_output_dir}")
        logger.info("="*70)
        logger.info("🚀 Starting transformer training...")
        logger.info("="*70)
        
        return baseline_results
        
    except Exception as e:
        logger.error(f"❌ Baseline evaluation failed: {e}")
        logger.info("Continuing with training without baseline metrics...")
        return {}


# Removed: save_training_vs_baseline_comparison - User will derive comparisons themselves


def should_run_baseline_evaluation(cfg: Dict[str, Any]) -> bool:
    """
    Check if baseline evaluation should be run based on configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if baseline evaluation should be run
    """
    # Check if explicitly disabled
    evaluation_cfg = cfg.get('evaluation', {})
    if 'run_baseline_evaluation' in evaluation_cfg:
        return evaluation_cfg['run_baseline_evaluation']
    
    # Check if we're on epoch 0 or fresh training
    training_cfg = cfg.get('training', {})
    resume = training_cfg.get('resume', False)
    
    # Run baseline evaluation for fresh training (not resumed)
    return not resume