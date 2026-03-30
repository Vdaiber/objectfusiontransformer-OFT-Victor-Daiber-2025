#!/usr/bin/env python3
"""
DoE Preprocessing Script for Design of Experiments Campaign.

This script preprocesses all dataset variations required by a Design of Experiments (DoE) plan
and performs baseline evaluations. It reads a CSV file containing multipliers for different 
noise parameters, calculates final parameter values, and ensures all required preprocessed
datasets are available in the cache before any model training or testing begins.

The script is designed to be run once before a DoE campaign to prepare all necessary
data variations. It uses config hashing to avoid recomputing already cached datasets.

Usage:
    python src/oft/transformer/scripts/preprocess_doe_plan.py \
        --doe-plan-csv doe_plans/main_analysis.csv \
        --base-config config/pipeline_staged.yaml \
        --verbose
"""

import argparse
import logging
import sys
import yaml
import pandas as pd
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from oft.transformer.utils.config_hash_utils import generate_unified_config_hash


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration for the preprocessing script.
    
    Args:
        verbose: If True, set logging level to DEBUG for detailed output
        
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('DoE_Preprocessing')


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load the base configuration YAML file.
    
    Args:
        config_path: Path to the base configuration file
        
    Returns:
        Dictionary containing the base configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Base config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def load_doe_plan(csv_path: str) -> pd.DataFrame:
    """Load the DoE plan CSV file and convert column names to full parameter names.
    
    Args:
        csv_path: Path to the DoE plan CSV file
        
    Returns:
        DataFrame containing the DoE plan with experiment configurations
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"DoE plan CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"DoE plan CSV is empty: {csv_path}")
    
    return df


def convert_csv_columns_to_parameter_names(row: pd.Series) -> Dict[str, float]:
    """Convert DOE CSV short column names to full parameter names.
    
    This function maps the short column names used in the DOE CSV file
    (e.g., 'lid_pos', 'cam_clas') to the full parameter names expected
    by the calculate_final_parameters function.
    
    Args:
        row: Single row from the DOE plan DataFrame
        
    Returns:
        Dictionary mapping full parameter names to multiplier values
    """
    # Mapping from CSV columns to full parameter names
    column_mapping = {
        # Position noise multipliers
        'lid_pos': 'lidar_pos_noise_multiplier',
        'cam_pos': 'camera_pos_noise_multiplier', 
        'ra_pos': 'radar_pos_noise_multiplier',
        
        # Dimension noise multipliers
        'lid_dim': 'lidar_dim_noise_multiplier',
        'cam_dim': 'camera_dim_noise_multiplier',
        'ra_dim': 'radar_dim_noise_multiplier',
        
        # Yaw noise multipliers
        'lid_yaw': 'lidar_yaw_noise_multiplier',
        'cam_yaw': 'camera_yaw_noise_multiplier',
        'ra_yaw': 'radar_yaw_noise_multiplier',
        
        # Velocity noise multipliers
        'lid_vel': 'lidar_velocity_noise_multiplier',
        'cam_vel': 'camera_velocity_noise_multiplier',
        'ra_vel': 'radar_velocity_noise_multiplier',
        
        # Dropout odds multipliers (odd scaling)
        'lid_drop': 'lidar_dropout_odds_multiplier',
        'cam_drop': 'camera_dropout_odds_multiplier',
        'ra_drop': 'radar_dropout_odds_multiplier',
        
        # Class accuracy multipliers (bidirectional scaling)
        'lid_clas': 'lidar_class_accuracy_multiplier',
        'cam_clas': 'camera_class_accuracy_multiplier',
        'ra_clas': 'radar_class_accuracy_multiplier',
        
        # Attribute accuracy multipliers (bidirectional scaling)
        'lid_at': 'lidar_attribute_accuracy_multiplier',
        'cam_at': 'camera_attribute_accuracy_multiplier',
        'ra_at': 'radar_attribute_accuracy_multiplier',
        
        # False positive count multiplier
        'FP': 'false_positive_count_multiplier'
    }
    
    # Convert row to parameter dictionary
    multipliers = {}
    for csv_col, param_name in column_mapping.items():
        if csv_col in row:
            multipliers[param_name] = float(row[csv_col])
    
    return multipliers


def calculate_final_parameters(base_config: Dict[str, Any], multipliers: Dict[str, float]) -> Dict[str, Any]:
    """Calculate final parameter values by applying multipliers to base configuration.
    
    This function takes the baseline parameter values from the configuration and
    applies the multipliers from the DoE plan to generate the final parameter values
    for a specific experiment configuration. Supports all 23 DOE parameters with
    proper odd scaling for probability parameters.
    
    Args:
        base_config: Base configuration dictionary with baseline parameter values
        multipliers: Dictionary of multipliers from DoE plan row
        
    Returns:
        Configuration dictionary with final parameter values applied
        
    Note:
        The function creates a deep copy of base_config and modifies virtual sensor
        parameters based on the provided multipliers. Uses odd scaling for 
        probability parameters (dropout_rate, class_accuracy, attribute_accuracy).
    """
    import copy
    final_config = copy.deepcopy(base_config)
    
    # Helper function for odd scaling transformation
    def apply_odd_scaling(baseline_prob: float, multiplier: float) -> float:
        """Apply odd scaling to probability parameters."""
        if baseline_prob <= 0 or baseline_prob >= 1:
            return baseline_prob  # Invalid probability, return as-is
        
        baseline_odds = baseline_prob / (1 - baseline_prob)
        new_odds = baseline_odds * multiplier
        new_prob = new_odds / (1 + new_odds)
        return max(0.0, min(1.0, new_prob))  # Clamp to [0, 1]
    

    
    # Apply multipliers to virtual sensor parameters
    virtual_sensors = final_config.get('dataset', {}).get('virtual_sensors', [])
    
    for sensor in virtual_sensors:
        sensor_name = sensor.get('name', '')
        
        # === POSITION NOISE MULTIPLIERS ===
        for sensor_type in ['lidar', 'camera', 'radar']:
            multiplier_key = f'{sensor_type}_pos_noise_multiplier'
            if multiplier_key in multipliers and sensor_type in sensor_name:
                multiplier = multipliers[multiplier_key]
                if isinstance(sensor.get('pos_noise_std'), list):
                    sensor['pos_noise_std'] = [x * multiplier for x in sensor['pos_noise_std']]
        
        # === DIMENSION NOISE MULTIPLIERS ===
        for sensor_type in ['lidar', 'camera', 'radar']:
            multiplier_key = f'{sensor_type}_dim_noise_multiplier'
            if multiplier_key in multipliers and sensor_type in sensor_name:
                multiplier = multipliers[multiplier_key]
                if 'dim_noise_std' in sensor:
                    sensor['dim_noise_std'] = sensor['dim_noise_std'] * multiplier
        
        # === YAW NOISE MULTIPLIERS ===
        for sensor_type in ['lidar', 'camera', 'radar']:
            multiplier_key = f'{sensor_type}_yaw_noise_multiplier'
            if multiplier_key in multipliers and sensor_type in sensor_name:
                multiplier = multipliers[multiplier_key]
                if 'yaw_noise_std' in sensor:
                    sensor['yaw_noise_std'] = sensor['yaw_noise_std'] * multiplier
        
        # === VELOCITY NOISE MULTIPLIERS ===
        for sensor_type in ['lidar', 'camera', 'radar']:
            multiplier_key = f'{sensor_type}_velocity_noise_multiplier'
            if multiplier_key in multipliers and sensor_type in sensor_name:
                multiplier = multipliers[multiplier_key]
                if 'velocity_noise_std' in sensor:
                    sensor['velocity_noise_std'] = sensor['velocity_noise_std'] * multiplier
        
        # === DROPOUT RATE (ODD SCALING WITH FUSION-VIABILITY RANGE: 0-60%) ===
        for sensor_type in ['lidar', 'camera', 'radar']:
            multiplier_key = f'{sensor_type}_dropout_odds_multiplier'
            if multiplier_key in multipliers and sensor_type in sensor_name:
                multiplier = multipliers[multiplier_key]
                if 'dropout_rate' in sensor:
                    baseline_dropout = sensor['dropout_rate']
                    
                    # Get DOE max multiplier for this sensor type
                    if sensor_type == 'lidar':
                        max_doe_multiplier = 8.0  # From DOE CSV
                    elif sensor_type == 'camera':
                        max_doe_multiplier = 5.0  # From DOE CSV
                    else:  # radar
                        max_doe_multiplier = 7.0  # From DOE CSV
                    
                    # Calculate constrained multiplier that maps DOE range to 0-60%
                    target_max_dropout = 0.60
                    baseline_odds = baseline_dropout / (1 - baseline_dropout)
                    target_max_odds = target_max_dropout / (1 - target_max_dropout)
                    
                    # Handle division by zero when baseline_dropout = 0
                    if baseline_odds == 0:
                        # When baseline dropout is 0, keep it at 0 regardless of multiplier
                        sensor['dropout_rate'] = 0.0
                    else:
                        max_effective_multiplier = target_max_odds / baseline_odds
                        
                        # Scale original multiplier to constrained range
                        effective_multiplier = (multiplier / max_doe_multiplier) * max_effective_multiplier
                        sensor['dropout_rate'] = apply_odd_scaling(baseline_dropout, effective_multiplier)
        
        # === CLASS ACCURACY (DIRECT MULTIPLIER: 0.0=Perfect, 1.0=Baseline, >1.0=Worse) ===
        for sensor_type in ['lidar', 'camera', 'radar']:
            multiplier_key = f'{sensor_type}_class_accuracy_multiplier'
            if multiplier_key in multipliers and sensor_type in sensor_name:
                multiplier = multipliers[multiplier_key]
                if 'class_accuracy' in sensor:
                    baseline_accuracy = sensor['class_accuracy']
                    
                    if multiplier <= 0.0:
                        # Perfect accuracy (100%)
                        sensor['class_accuracy'] = 1.0
                    elif multiplier <= 1.0:
                        # Linear interpolation: Perfect (100%) → Baseline
                        sensor['class_accuracy'] = 1.0 - (1.0 - baseline_accuracy) * multiplier
                    else:
                        # Linear degradation: Baseline → Minimum (5%)
                        # Camera has different range in DOE CSV (2.0-10.0), others use 0.0-2.0
                        if 'camera' in sensor_type:
                            max_multiplier = 10.0  # Camera range: 2.0-10.0 in DOE CSV
                            degradation = (multiplier - 1.0) / (max_multiplier - 1.0)  # 9.0 range
                        else:
                            max_multiplier = 2.0   # LiDAR/Radar range: 0.0-2.0
                            degradation = (multiplier - 1.0) / (max_multiplier - 1.0)  # 1.0 range
                        
                        # Map DOE range to fusion-viable range (baseline → 30%)
                        degradation = min(1.0, degradation)
                        sensor['class_accuracy'] = baseline_accuracy - (baseline_accuracy - 0.30) * degradation
        
        # === ATTRIBUTE ACCURACY (DIRECT MULTIPLIER: 0.0=Perfect, 1.0=Baseline, >1.0=Worse) ===
        for sensor_type in ['lidar', 'camera', 'radar']:
            multiplier_key = f'{sensor_type}_attribute_accuracy_multiplier'
            if multiplier_key in multipliers and sensor_type in sensor_name:
                multiplier = multipliers[multiplier_key]
                if 'attribute_accuracy' in sensor:
                    baseline_accuracy = sensor['attribute_accuracy']
                    
                    if multiplier <= 0.0:
                        # Perfect accuracy (100%)
                        sensor['attribute_accuracy'] = 1.0
                    elif multiplier <= 1.0:
                        # Linear interpolation: Perfect (100%) → Baseline
                        sensor['attribute_accuracy'] = 1.0 - (1.0 - baseline_accuracy) * multiplier
                    else:
                        # Linear degradation: Baseline → Minimum (5%)
                        # All attribute accuracy uses 0.0-2.0 range in DOE CSV
                        max_multiplier = 2.0
                        degradation = (multiplier - 1.0) / (max_multiplier - 1.0)  # 1.0 range
                        # Map DOE range to fusion-viable range (baseline → 30%)
                        degradation = min(1.0, degradation)
                        sensor['attribute_accuracy'] = baseline_accuracy - (baseline_accuracy - 0.30) * degradation
    
    # === GLOBAL SIMULATION PARAMETERS ===
    simulation = final_config.get('dataset', {}).get('simulation', {})
    
    # False Positive Count (discrete parameter)
    if 'false_positive_count' in multipliers:
        base_count = simulation.get('num_fps', 0)
        final_count = int(base_count * multipliers['false_positive_count'])
        final_config['dataset']['simulation']['num_fps'] = max(0, final_count)
    
    return final_config


def check_cache_exists(config_hash: str, cache_root: str) -> bool:
    """Check if preprocessed data and baseline results exist in cache.
    
    Args:
        config_hash: Configuration hash identifying the dataset
        cache_root: Root directory of the cache
        
    Returns:
        True if both preprocessed data and baseline results exist, False otherwise
    """
    cache_dir = Path(cache_root) / f"config_{config_hash}"
    
    # Check for preprocessed data
    data_file = cache_dir / "complete_dataset_val.json"
    
    # Check for baseline evaluation results directory
    baseline_dir = cache_dir / "baseline_evaluation_results"
    
    return data_file.exists() and baseline_dir.exists() and baseline_dir.is_dir()


def run_preprocessing(config_dict: Dict[str, Any], temp_config_path: str, logger: logging.Logger) -> bool:
    """Run standalone preprocessing for validation split.
    
    Args:
        config_dict: Configuration dictionary for this experiment
        temp_config_path: Path to temporary config file
        logger: Logger instance
        
    Returns:
        True if preprocessing successful, False otherwise
    """
    import shutil
    import tempfile
    
    try:
        # Save temporary config for preprocessing
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # SAFE: Use --config parameter to pass temporary config directly
        # This does NOT modify the original config/pipeline_staged.yaml!
        cmd = [
            sys.executable, "src/oft/transformer/scripts/standalone_preprocessing.py",
            "--config", temp_config_path,
            "--force"
        ]
        
        logger.info(f"Running SAFE preprocessing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"Preprocessing failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
        
        logger.info("Preprocessing completed successfully")
        
        # SAVE CONFIG for visual inspection
        config_hash = generate_unified_config_hash(config_dict)
        cache_root = config_dict['preprocessing']['cache_dir']
        config_save_path = Path(cache_root) / f'config_{config_hash}' / 'experiment_config.yaml'
        
        # Ensure directory exists
        config_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the final config with all DOE parameters applied
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"💾 Saved experiment config to: {config_save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return False


def run_baseline_evaluation(config_dict: Dict[str, Any], temp_config_path: str, logger: logging.Logger) -> bool:
    """Run baseline evaluation for virtual sensors.
    
    Args:
        config_dict: Configuration dictionary for this experiment
        temp_config_path: Path to temporary config file
        logger: Logger instance
        
    Returns:
        True if baseline evaluation successful, False otherwise
    """
    try:
        # Save temporary config for baseline evaluation
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Calculate output directory for baseline results
        config_hash = generate_unified_config_hash(config_dict)
        cache_root = config_dict['preprocessing']['cache_dir']
        baseline_output_dir = Path(cache_root) / f'config_{config_hash}' / 'baseline_evaluation_results'
        
        # Run baseline evaluation with DOE config hash
        cmd = [
            sys.executable, "src/oft/transformer/scripts/evaluate_virtual_sensors_baseline.py",
            "--config", temp_config_path,
            "--output-dir", str(baseline_output_dir),
            "--config-hash", config_hash
        ]
        
        logger.info(f"Running baseline evaluation: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"Baseline evaluation failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
        
        logger.info("Baseline evaluation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during baseline evaluation: {e}")
        return False


def main():
    """Main entry point for DoE preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess all dataset variations for DoE campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python src/oft/transformer/scripts/preprocess_doe_plan.py \\
        --doe-plan-csv doe_plans/main_analysis.csv \\
        --base-config config/pipeline_staged.yaml

    # With verbose output
    python src/oft/transformer/scripts/preprocess_doe_plan.py \\
        --doe-plan-csv doe_plans/main_analysis.csv \\
        --base-config config/pipeline_staged.yaml \\
        --verbose
        """
    )
    
    parser.add_argument(
        '--doe-plan-csv',
        type=str,
        required=True,
        help='Path to DoE plan CSV file containing experiment configurations'
    )
    
    parser.add_argument(
        '--base-config',
        type=str,
        required=True,
        help='Path to base configuration YAML file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("🚀 Starting DoE preprocessing campaign...")
    
    try:
        # Load base configuration and DoE plan
        logger.info(f"📋 Loading base config: {args.base_config}")
        base_config = load_base_config(args.base_config)
        
        logger.info(f"📊 Loading DoE plan: {args.doe_plan_csv}")
        doe_plan = load_doe_plan(args.doe_plan_csv)
        
        logger.info(f"🔢 DoE plan contains {len(doe_plan)} experiments")
        
        # FOR DOE: Only process validation split (not training)
        # Use the validation split defined in the config (mini_val for test, val for production)
        if 'preprocessing' not in base_config:
            base_config['preprocessing'] = {}
        
        val_split = base_config.get('training', {}).get('val_split_name', 'val')
        base_config['preprocessing']['splits_to_process'] = [val_split]
        logger.info(f"🎯 DOE Mode: Processing ONLY validation split '{val_split}'")
        
        # Cache configuration
        cache_root = base_config.get('preprocessing', {}).get('cache_dir', '/data/daiber_fent/virtual_sensor_cache')
        logger.info(f"💾 Cache directory: {cache_root}")
        
        # Counters for summary
        total_experiments = len(doe_plan)
        cached_count = 0
        processed_count = 0
        failed_count = 0
        
        # Process each experiment in the DoE plan
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = Path(temp_dir) / "temp_config.yaml"
            
            for idx, row in doe_plan.iterrows():
                experiment_id = row.get('Nr', idx + 1)  # Use 'Nr' from DOE CSV
                logger.info(f"\n--- Processing Experiment {experiment_id} ({idx + 1}/{total_experiments}) ---")
                
                # Convert DOE CSV row to parameter multipliers using proper mapping
                multipliers = convert_csv_columns_to_parameter_names(row)
                logger.debug(f"Multipliers: {multipliers}")
                
                # Calculate final parameters
                final_config = calculate_final_parameters(base_config, multipliers)
                
                # Generate config hash
                config_hash = generate_unified_config_hash(final_config)
                logger.info(f"📝 Config hash: {config_hash}")
                
                # Check if already cached
                if check_cache_exists(config_hash, cache_root):
                    logger.info(f"✅ Already cached, skipping experiment {experiment_id}")
                    cached_count += 1
                    continue
                
                # Run preprocessing
                logger.info(f"🔄 Running preprocessing for experiment {experiment_id}...")
                if not run_preprocessing(final_config, str(temp_config_path), logger):
                    logger.error(f"❌ Preprocessing failed for experiment {experiment_id}")
                    failed_count += 1
                    continue
                
                # Run baseline evaluation
                logger.info(f"📈 Running baseline evaluation for experiment {experiment_id}...")
                if not run_baseline_evaluation(final_config, str(temp_config_path), logger):
                    logger.error(f"❌ Baseline evaluation failed for experiment {experiment_id}")
                    failed_count += 1
                    continue
                
                logger.info(f"✅ Successfully processed experiment {experiment_id}")
                processed_count += 1
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🏁 DoE PREPROCESSING CAMPAIGN SUMMARY")
        logger.info("="*60)
        logger.info(f"📊 Total experiments: {total_experiments}")
        logger.info(f"💾 Already cached: {cached_count}")
        logger.info(f"✅ Newly processed: {processed_count}")
        logger.info(f"❌ Failed: {failed_count}")
        logger.info(f"🎯 Success rate: {(cached_count + processed_count) / total_experiments * 100:.1f}%")
        
        if failed_count > 0:
            logger.warning(f"⚠️  {failed_count} experiments failed. Check logs above for details.")
            sys.exit(1)
        else:
            logger.info("🎉 All experiments successfully prepared for DoE campaign!")
            
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
