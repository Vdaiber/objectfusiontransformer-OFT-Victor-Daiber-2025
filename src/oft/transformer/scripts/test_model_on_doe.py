#!/usr/bin/env python3
"""
DoE Evaluation Orchestrator Script (v5 - Final Corrected).

This script orchestrates the evaluation of pre-trained models against all 
preprocessed dataset variations from a Design of Experiments (DoE) plan.

It follows the robust pattern of calling a Hydra-aware script with overrides:
1.  Loads a base configuration and a DoE plan.
2.  Loops through each model to be tested.
3.  For each experiment in the plan, it calculates the final configuration parameters.
4.  It constructs a list of command-line overrides for these parameters.
5.  It calls the dedicated, Hydra-aware `run_single_evaluation.py` script as a 
    subprocess, passing the overrides.
6.  It captures the NDS score and compiles the results into the correct,
    structured output format with per-experiment folders and a final summary.
"""
import argparse
import logging
import sys
import yaml
import pandas as pd
import subprocess
from pathlib import Path
import re
import json

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from oft.transformer.scripts.preprocess_doe_plan import convert_csv_columns_to_parameter_names, calculate_final_parameters
from oft.transformer.utils.config_hash_utils import generate_unified_config_hash

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('DoE_Orchestrator')

def get_overrides_from_config(config: dict, parent_key: str = '') -> list[str]:
    """Recursively flattens a config dict into a list of Hydra override strings."""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(get_overrides_from_config(v, new_key))
        elif isinstance(v, list):
            # For lists, iterate and handle elements based on their type.
            for i, item in enumerate(v):
                item_key = f"{new_key}.{i}"
                if isinstance(item, dict):
                    # If the list item is a dict, recurse.
                    items.extend(get_overrides_from_config(item, item_key))
                else:
                    # If the list item is a primitive, create the override directly.
                    value_str = str(item)
                    if any(c in value_str for c in ",'[]{}"):
                         value_str = f"'{value_str}'"
                    items.append(f"{item_key}={value_str}")
        else:
            # Escape characters that have special meaning in shell
            value_str = str(v)
            if any(c in value_str for c in ",'[]{}"):
                 value_str = f"'{value_str}'"
            items.append(f"{new_key}={value_str}")
    return items

def main():
    parser = argparse.ArgumentParser(description="Orchestrate a DoE evaluation campaign.")
    parser.add_argument('--model-names', type=str, required=True, nargs='+', help='One or more model identifiers to test.')
    parser.add_argument('--base-config', type=str, required=True, help='Path to the base YAML config file.')
    parser.add_argument('--doe-plan-csv', type=str, required=True, help='Path to the DoE plan CSV file.')
    parser.add_argument('--run-name', type=str, help='Optional name for this evaluation campaign run.')
    parser.add_argument('--results-root', type=str, default="/data/daiber_fent/evaluation_campaign_results", help='Root directory for results.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output.')
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    run_name = args.run_name if args.run_name else Path(args.doe_plan_csv).stem

    logger.info("🚀 Starting DoE evaluation orchestrator.")
    
    try:
        with open(args.base_config, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # CRITICAL: Always use baseline config for data hash calculation
        # Load the baseline config for consistent data hashing across all models
        with open('config/pipeline_staged.yaml', 'r') as f:
            baseline_config_for_data = yaml.safe_load(f)
        
        # Fix Hydra-specific interpolations that cause issues outside Hydra apps
        if 'hydra' in base_config:
            if 'run' in base_config['hydra'] and 'dir' in base_config['hydra']['run']:
                base_config['hydra']['run']['dir'] = '/data/daiber_fent/output/doe_evaluation_runs'
            if 'sweep' in base_config['hydra'] and 'dir' in base_config['hydra']['sweep']:
                base_config['hydra']['sweep']['dir'] = '/data/daiber_fent/output/doe_evaluation_sweeps'
        doe_plan = pd.read_csv(args.doe_plan_csv)
        cache_root = base_config.get('preprocessing', {}).get('cache_dir', '/data/daiber_fent/virtual_sensor_cache')
        results_root = Path(args.results_root)

        for model_name in args.model_names:
            logger.info(f"--- Starting evaluation for model: {model_name} ---")
            campaign_dir = results_root / model_name / run_name
            campaign_dir.mkdir(parents=True, exist_ok=True)
            
            summary_results = []
            
            # Determine starting experiment based on model
            if model_name == "Model_GT":
                start_experiment = 11  # Model_GT failed at experiment 11
            elif model_name == "Model_K_UP":
                start_experiment = 51  # Model_K_UP failed at experiment 51
            else:
                start_experiment = 1   # Default for other models
            
            for idx, row in doe_plan.iterrows():
                experiment_id = row.get('Nr', idx + 1)
                
                # Skip experiments before start_experiment
                if experiment_id < start_experiment:
                    logger.info(f"-- Skipping Exp {experiment_id}/{len(doe_plan)} (before start_experiment={start_experiment}) --")
                    continue
                
                experiment_name = f"experiment_{int(experiment_id):03d}"
                experiment_dir = campaign_dir / experiment_name
                experiment_dir.mkdir(exist_ok=True)
                
                logger.info(f"-- Running Exp {experiment_id}/{len(doe_plan)} for model {model_name} --")

                multipliers = convert_csv_columns_to_parameter_names(row)
                final_config = calculate_final_parameters(base_config, multipliers)
                
                # CRITICAL: Calculate hash from baseline config for consistent data access
                baseline_final_config = calculate_final_parameters(baseline_config_for_data, multipliers)
                config_hash = generate_unified_config_hash(baseline_final_config)
                
                # Remove Hydra-specific keys that cause interpolation errors
                clean_config = final_config.copy()
                if 'hydra' in clean_config:
                    del clean_config['hydra']
                if 'defaults' in clean_config:
                    del clean_config['defaults']
                
                with open(experiment_dir / "1_experiment_config.yaml", 'w') as f:
                    yaml.dump(clean_config, f)

                data_reference = { "config_hash": config_hash, "preprocessed_data_path": str(Path(cache_root) / f"config_{config_hash}")}
                with open(experiment_dir / "2_data_reference.json", 'w') as f:
                    json.dump(data_reference, f, indent=2)
                
                # Use the config hash override system (like in your codebase)
                cmd = [
                    sys.executable, "src/oft/transformer/scripts/run_single_evaluation.py",
                    f"training.model_identifier={model_name}",
                    f"+training.model_archive_dir=/data/daiber_fent/models",
                    f"+_config_hash_override={config_hash}",
                    f"+experiment_output_dir={str(experiment_dir)}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

                nds_score = None
                if result.returncode == 0:
                    match = re.search(r"FINAL_NDS_SCORE:([\d\.]+)", result.stdout)
                    if match:
                        nds_score = float(match.group(1))
                        logger.info(f"✅ Success. NDS: {nds_score:.6f}")
                    else:
                        logger.error(f"❌ Could not parse NDS score. STDOUT: {result.stdout}")
                else:
                    logger.error(f"❌ Subprocess failed. See log in experiment folder.")
                
                log_content = f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
                with open(experiment_dir / "3_evaluation_log.txt", 'w') as f:
                    f.write(log_content)
                
                summary_row = {'experiment_id': experiment_id, 'nds_score': nds_score}
                summary_row.update(row.to_dict())
                summary_results.append(summary_row)

            summary_df = pd.DataFrame(summary_results)
            summary_path = campaign_dir / "_summary_results.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"🎉 Model {model_name} evaluation complete! Summary: {summary_path}")

    except Exception as e:
        logger.error(f"💥 Fatal error in orchestrator: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
