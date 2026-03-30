# Object Fusion Transformer (OFT)

**Master's Thesis: Multi-Modal Sensor Fusion for Autonomous Driving**


## Overview

Object Fusion Transformer (OFT) is an advanced autoregressive transformer architecture designed for multi-modal object detection in autonomous driving scenarios. The system processes camera, LiDAR, and radar sensor data through a sophisticated staged fusion pipeline with lightweight metadata integration.

### Key Innovations

- **Staged Multi-Modal Fusion**: Sequential processing of camera, LiDAR, and radar data through intra-modal encoders
- **Autoregressive Architecture**: Single-frame processing with Hungarian matching for set prediction training (temporal processing prepared for future work)
- **Lightweight Metadata Integration**: Additive combination of environmental metadata (weather, lighting, scene conditions)
- **Future-Ready Components**: Prepared infrastructure for temporal processing, ego-motion compensation, and cross-attention mechanisms (currently disabled)

## Architecture

The OFT system implements a sophisticated staged fusion approach:

```
Input: Multi-Modal Sensor Data (Camera, LiDAR, Radar) + Environmental Metadata
  ↓
1. Intra-Modal Encoding: Independent sensor processing with positional embeddings
  ↓
2. Lightweight Metadata Integration: Additive combination of environmental context
  ↓
3. Inter-Modal Fusion: Cross-sensor feature fusion through transformer layers
  ↓
4. Object Decoding: Multi-head prediction (classification, regression, attributes)
  ↓
Output: Object Detections with Hungarian-matched training

[Future Components - Currently Disabled]:
- Cross-Attention for Metadata
- Temporal Integration with Memory States
- Ego-Motion Compensation
```

## Quick Start

### Prerequisites

- **Operating System**: Linux (Ubuntu 18.04+ recommended)
- **Hardware**: CUDA-compatible GPU (8GB+ VRAM recommended)
- **Python**: 3.9 or higher
- **Docker**: For containerized execution (recommended)

### Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd oft-transformer
   ```

2. **Docker Setup (Recommended)**
   ```bash
   # Build the Docker image
   docker build -f Dockerfile.gpu -t oft_gpu .
   
   # Run the container with GPU support
   ./docker_run_gpu.sh
   
   # Connect to the container
   docker exec -it oft_dev_gpu /bin/bash
   ```

3. **Alternative: Direct Installation**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. **Download TruckScenes Dataset**
   - Download the TruckScenes v1.0-trainval dataset
   - Extract to `/data/` directory structure:
   ```
   /data/
   ├── v1.0-trainval/
   ├── v1.0-test/
   ├── v1.0-mini/
   ├── samples/
   └── sweeps/
   ```

2. **Verify Dataset Structure**
   ```bash
   python -c "from truckscenes import TruckScenes; ts = TruckScenes(version='v1.0-mini', dataroot='/data')"
   ```

## Main Training Pipeline

### Basic Training Execution

The main training pipeline is executed through the autoregressive training script:

```bash
python src/oft/transformer/run_training_autoregressive.py
```

### Configuration Options

The system uses Hydra for configuration management. Key configuration files:

- **`config/pipeline_staged.yaml`**: Main training configuration
- **`config/normalization_stats.yaml`**: Data normalization parameters
- **`config/scene_conditioning_config.yaml`**: Environmental conditioning settings

### How Training Works

The training script uses Hydra configuration management and automatically:

1. **Loads Configuration**: Reads from `config/pipeline_staged.yaml`
2. **Initializes Model**: Creates the autoregressive transformer architecture
3. **Prepares Data**: Sets up TruckScenes dataset with multi-modal sensor data
4. **Executes Training**: Runs the complete training loop with Hungarian matching loss
5. **Saves Results**: Outputs checkpoints and logs to timestamped directories

The system processes sensor data in normalized ego-vehicle coordinates and handles all coordinate transformations internally.

### Training Output

Training generates the following outputs in `/data/daiber_fent/output/`:

```
output/
├── YYYY-MM-DD/HH-MM-SS/           # Timestamped run directory
│   ├── checkpoints/                # Model checkpoints
│   ├── logs/                       # Training logs
│   ├── config.yaml                 # Resolved configuration
│   └── baseline_results/           # Baseline evaluation results
```

## Design of Experiments (DOE) Pipeline

### DOE Preprocessing

The DOE system enables systematic evaluation of model robustness across different noise conditions and sensor configurations.

#### 1. DOE Plan Preparation

Create a DOE plan CSV file (e.g., `DOE_CSV/DOE_D_Optimal.csv`) with experimental conditions:

```csv
Nr,cam_at,lid_drop,lid_pos,ra_clas
1,1.0,1.0,1.0,1.0
2,1.2,0.8,1.1,0.9
3,0.9,1.3,0.8,1.2
...
```

#### 2. Preprocessing Execution

```bash
python src/oft/transformer/scripts/preprocess_doe_plan.py \
  --doe-plan-csv DOE_CSV/DOE_D_Optimal.csv \
  --base-config config/pipeline_staged.yaml \
  --verbose
```

This script:
- Loads the DOE experimental plan
- Calculates final noise parameters for each experiment
- Preprocesses all required dataset variations
- Caches processed data for efficient evaluation

#### 3. Model Evaluation on DOE

```bash
python src/oft/transformer/scripts/test_model_on_doe.py \
  --doe-plan-csv DOE_CSV/DOE_D_Optimal.csv \
  --base-config config/pipeline_staged.yaml \
  --baseline-config config/pipeline_staged.yaml \
  --model-names Model_GT Model_K_UP Model_C_DO \
  --results-root /data/daiber_fent/doe_results \
  --run-name main_analysis
```

This orchestrates:
- Loading pre-trained models for each configuration
- Evaluating each model on all DOE experimental conditions
- Computing NDS (TruckScenes Detection Score) for each experiment
- Generating comprehensive results summary

### DOE Results Structure

```
doe_results/
├── Model_GT/main_analysis/
│   ├── experiment_001/
│   │   ├── evaluation_results.json
│   │   └── detailed_metrics.csv
│   ├── experiment_002/
│   └── summary_results.csv
├── Model_K_UP/main_analysis/
└── Model_C_DO/main_analysis/
```

## Statistical Analysis and Visualization

### Regression Analysis

The project includes comprehensive statistical analysis tools for DOE results:

```bash
python automatic_regression_analysis.py
```

This performs:
- **Multiple Linear Regression**: Cubic polynomial regression with standardized coefficients
- **ANOVA Analysis**: Statistical significance testing
- **Feature Selection**: Automatic predictor selection using AIC/BIC criteria
- **Visualization**: TUM Corporate Design compliant plots

### Analysis Components

1. **`Model_Analyse/comprehensive_regression_analysis.py`**: Complete regression pipeline
2. **`Model_Analyse/tukey_posthoc_analysis.py`**: Post-hoc statistical testing
3. **`Model_Analyse/interaction_plots_analysis.py`**: Factor interaction visualization
4. **`Model_Analyse/mixed_effects_analysis.py`**: Advanced mixed-effects modeling

### Generated Outputs

- **Regression Coefficients**: Standardized and unstandardized coefficients
- **ANOVA Tables**: Statistical significance for each factor
- **Diagnostic Plots**: Residual analysis and model fit visualization
- **Effect Plots**: Factor importance and interaction effects

## Project Structure

```
src/oft/transformer/
├── models/
│   ├── architectures/              # Transformer architectures
│   │   └── autoregressive_architecture.py  # Main OFT model
│   └── components/                 # Model components
│       ├── encoders/               # Sensor encoders
│       │   └── metadata_encoder.py # Metadata additive integration
│       └── inter_modal_fusion.py   # Cross-sensor fusion
├── datasets/
│   ├── loaders/                    # Data loading pipeline
│   └── truckscenes/               # TruckScenes dataset interface
├── training/
│   ├── criteria/                   # Loss functions
│   └── trainers/                   # Training loops
│       └── autoregressive_trainer.py  # Main trainer
├── evaluation/                     # Model evaluation
├── scripts/                        # Utility scripts
│   ├── preprocess_doe_plan.py     # DOE preprocessing
│   └── test_model_on_doe.py       # DOE evaluation
└── utils/                          # Utility functions

config/                             # Configuration files
├── pipeline_staged.yaml           # Main configuration
├── pipeline_staged_Model_*.yaml   # Model-specific configs
├── normalization_stats.yaml       # Data normalization
└── scene_conditioning_config.yaml # Environmental conditioning

Model_Analyse/                      # Statistical analysis
├── comprehensive_regression_analysis.py
├── tukey_posthoc_analysis.py
├── interaction_plots_analysis.py
└── mixed_effects_analysis.py
```


## Development and Deployment

### Docker Environment

The provided Docker setup includes:
- CUDA support for GPU acceleration
- All required dependencies pre-installed
- Volume mounts for data and code directories
- Development tools and debugging utilities

### Container Management

```bash
# Start development container
./docker_run_gpu.sh

# Connect to running container
docker exec -it oft_dev_gpu /bin/bash

# Monitor GPU usage
nvidia-smi
```


## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{oft2025,
  title={Multi-Modal Sensor Fusion for Autonomous Driving using Object Fusion Transformer},
  author={Victor Daiber},
  year={2025},
  school={Technical University of Munich}
}
```

