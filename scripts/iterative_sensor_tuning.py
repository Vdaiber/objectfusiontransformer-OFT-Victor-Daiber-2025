#!/usr/bin/env python3
"""
Iterative Virtual Sensor Tuning Script
=====================================

Automatisiert den kompletten Tuning-Prozess:
1. Preprocessing: Erstellt virtuelle Sensordaten mit aktuellen Parametern
2. Baseline-Evaluation: Evaluiert die virtuellen Sensoren
3. Ergebnis-Analyse: Zeigt Performance vs. Targets

Usage:
    python scripts/iterative_sensor_tuning.py
    
Oder für mehrere Iterationen:
    python scripts/iterative_sensor_tuning.py --iterations 3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import json
import yaml
import shutil
from datetime import datetime

def log_message(message: str, level: str = "INFO"):
    """Logging mit Timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def run_command(cmd: str, description: str, show_live_output: bool = True) -> bool:
    """Führt Command aus und loggt Ergebnis"""
    log_message(f"Starting: {description}")
    log_message(f"Command: {cmd}")
    
    try:
        if show_live_output:
            # Live output für lange Prozesse
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/app",
                bufsize=1,
                universal_newlines=True
            )
            
            # Live output anzeigen
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            if process.returncode == 0:
                log_message(f"✅ SUCCESS: {description}")
                return True
            else:
                log_message(f"❌ FAILED: {description} (Exit code: {process.returncode})", "ERROR")
                return False
        else:
            # Stille Ausführung für schnelle Commands
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd="/app"
            )
            log_message(f"✅ SUCCESS: {description}")
            return True
    except subprocess.CalledProcessError as e:
        log_message(f"❌ FAILED: {description}", "ERROR")
        log_message(f"Exit code: {e.returncode}", "ERROR")
        if hasattr(e, 'stdout') and e.stdout:
            log_message(f"STDOUT: {e.stdout}", "ERROR")
        if hasattr(e, 'stderr') and e.stderr:
            log_message(f"STDERR: {e.stderr}", "ERROR")
        return False

def clear_virtual_sensor_cache():
    """Löscht Virtual Sensor Cache für neue Parameter"""
    log_message("🗑️  Clearing virtual sensor cache...")
    return run_command(
        "rm -rf /data/daiber_fent/virtual_sensor_cache/config_*",
        "Clear virtual sensor cache",
        show_live_output=False
    )

def run_preprocessing():
    """Führt Preprocessing für virtuelle Sensordaten durch (VALIDATION-ONLY)"""
    log_message("🔧 Starting preprocessing (validation split only)...")
    
    # Create temporary config with validation-only preprocessing
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
        # Load original config
        with open('config/pipeline_staged.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify to process only validation split
        if 'preprocessing' not in config:
            config['preprocessing'] = {}
        config['preprocessing']['splits_to_process'] = [config['training']['val_split_name']]
        
        # Save temporary config
        yaml.dump(config, temp_config, default_flow_style=False)
        temp_config_path = temp_config.name
    
    cmd = (
        f"python -m src.oft.transformer.scripts.standalone_preprocessing --config {temp_config_path} --force"
    )
    
    try:
        result = run_command(cmd, "Virtual sensor preprocessing (validation-only)")
        # Clean up temporary file
        import os
        os.unlink(temp_config_path)
        return result
    except Exception as e:
        # Clean up temporary file on error
        import os
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        log_message(f"Error during preprocessing: {e}", "ERROR")
        return False

def run_baseline_evaluation():
    """Führt Baseline-Evaluation durch"""
    log_message("📊 Starting baseline evaluation...")
    cmd = (
        "python -m src.oft.transformer.scripts.evaluate_virtual_sensors_baseline "
        "--config config/pipeline_staged.yaml"
    )
    return run_command(cmd, "Baseline evaluation")

def analyze_results():
    """Analysiert die Baseline-Evaluation Ergebnisse"""
    log_message("📈 Analyzing results...")
    
    # Finde neuesten Output-Ordner
    output_dir = Path("/app/output/full_dataset_runs")
    if not output_dir.exists():
        log_message("❌ No output directory found", "ERROR")
        return False
    
    # Finde neuesten Run-Ordner
    run_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        log_message("❌ No run directories found", "ERROR")
        return False
    
    latest_run = run_dirs[0]
    baseline_dir = latest_run / "baseline_sensors"
    
    if not baseline_dir.exists():
        log_message("❌ No baseline_sensors directory found", "ERROR")
        return False
    
    # Analysiere jeden Sensor
    targets = {
        "virtual_lidar": 0.410,
        "virtual_camera": 0.200, 
        "virtual_radar": 0.107
    }
    
    results = {}
    
    for sensor_name, target_nds in targets.items():
        sensor_dir = baseline_dir / f"{sensor_name}_eval"
        metrics_file = sensor_dir / "metrics_summary.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                actual_nds = metrics["all"]["nd_score"]
                results[sensor_name] = {
                    "target_nds": target_nds,
                    "actual_nds": actual_nds,
                    "difference_pct": ((actual_nds - target_nds) / target_nds) * 100,
                    "mAVE": metrics["all"]["tp_errors"]["vel_err"],
                    "mAAE": metrics["all"]["tp_errors"]["attr_err"]
                }
            except Exception as e:
                log_message(f"❌ Error reading {metrics_file}: {e}", "ERROR")
        else:
            log_message(f"❌ Metrics file not found: {metrics_file}", "ERROR")
    
    # Ergebnisse anzeigen
    log_message("=" * 80)
    log_message("🎯 VIRTUAL SENSOR PERFORMANCE ANALYSIS")
    log_message("=" * 80)
    
    for sensor_name, data in results.items():
        log_message(f"\n📊 {sensor_name.upper()}:")
        log_message(f"   Target NDS: {data['target_nds']:.3f}")
        log_message(f"   Actual NDS: {data['actual_nds']:.3f}")
        log_message(f"   Difference: {data['difference_pct']:+.1f}%")
        log_message(f"   mAVE: {data['mAVE']:.3f}")
        log_message(f"   mAAE: {data['mAAE']:.3f}")
        
        if abs(data['difference_pct']) < 5:
            log_message(f"   Status: ✅ PERFECT (within 5%)")
        elif abs(data['difference_pct']) < 10:
            log_message(f"   Status: ✅ GOOD (within 10%)")
        elif abs(data['difference_pct']) < 20:
            log_message(f"   Status: ⚠️  NEEDS TUNING (within 20%)")
        else:
            log_message(f"   Status: ❌ BAD (>20% off target)")
    
    log_message("=" * 80)
    
    return True

def save_iteration_config(iteration_num: int):
    """Speichert die aktuelle Config für diese Iteration"""
    try:
        # Erstelle Iteration-Ordner
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        iteration_dir = Path(f"/app/iterative_baseline_tuning/iteration_{iteration_num}_{timestamp}")
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        # Kopiere aktuelle Config
        config_source = Path("/app/config/pipeline_staged.yaml")
        config_dest = iteration_dir / "config_used.yaml"
        shutil.copy2(config_source, config_dest)
        
        log_message(f"✅ Config saved: {config_dest}")
        return iteration_dir
        
    except Exception as e:
        log_message(f"❌ Failed to save config: {e}", "ERROR")
        return None

def main():
    parser = argparse.ArgumentParser(description="Iterative Virtual Sensor Tuning")
    parser.add_argument("--iterations", type=int, default=1, 
                       help="Number of tuning iterations (default: 1)")
    parser.add_argument("--skip-cache-clear", action="store_true",
                       help="Skip clearing virtual sensor cache")
    
    args = parser.parse_args()
    
    log_message("🚀 Starting Iterative Virtual Sensor Tuning")
    log_message(f"Iterations: {args.iterations}")
    
    for iteration in range(1, args.iterations + 1):
        log_message(f"\n{'='*20} ITERATION {iteration}/{args.iterations} {'='*20}")
        
        # Schritt 0: Config speichern
        iteration_dir = save_iteration_config(iteration)
        if not iteration_dir:
            log_message("❌ Failed to save config. Continuing anyway...", "WARNING")
        
        # Schritt 1: Cache löschen (außer wenn übersprungen)
        #if not args.skip_cache_clear:
        #    if not clear_virtual_sensor_cache():
        #        log_message("❌ Failed to clear cache. Aborting.", "ERROR")
        #        sys.exit(1)
        
        # Schritt 2: Preprocessing
        if not run_preprocessing():
            log_message("❌ Preprocessing failed. Aborting.", "ERROR")  
            sys.exit(1)
        
        # Schritt 3: Baseline-Evaluation
        if not run_baseline_evaluation():
            log_message("❌ Baseline evaluation failed. Aborting.", "ERROR")
            sys.exit(1)
        
        # Schritt 4: Ergebnis-Analyse
        if not analyze_results():
            log_message("❌ Result analysis failed. Continuing...", "WARNING")
        
        if iteration < args.iterations:
            log_message(f"\n⏳ Iteration {iteration} complete. Starting next iteration...")
            time.sleep(2)  # Kurze Pause zwischen Iterationen
    
    log_message("\n🎉 ALL ITERATIONS COMPLETED!")
    log_message("💡 Check the analysis above to see if targets are reached.")
    log_message("💡 If not, adjust parameters in config/pipeline_staged.yaml and run again.")

if __name__ == "__main__":
    main()
