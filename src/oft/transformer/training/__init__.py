"""
Training Package for Transformer-Based Sensor Fusion Models.

This package provides the complete training infrastructure for transformer-based
sensor fusion models, including loss functions, optimization strategies, and
training utilities. The package is organized into specialized submodules for
different aspects of the training process.

Submodules:
    criteria: Loss function orchestration and Hungarian matching
    losses: Individual loss function implementations (classification, regression, etc.)
    matchers: Object association algorithms (Hungarian matcher)
    trainers: Training loop implementations for different architectures
    utils: Training utilities (checkpointing, metrics, IoU calculations)

The training package supports both autoregressive and staged fusion architectures,
with modular components that can be easily extended for new model variants.
"""

# Import submodules
from . import criteria
from . import losses
from . import matchers
from . import trainers
from . import utils

__all__ = ['criteria', 'losses', 'matchers', 'trainers', 'utils'] 