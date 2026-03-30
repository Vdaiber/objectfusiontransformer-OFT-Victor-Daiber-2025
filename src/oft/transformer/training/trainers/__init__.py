"""
Training Loop Implementations for Transformer-Based Sensor Fusion Models.

This module provides specialized training loop implementations for different
transformer architectures and training strategies. Each trainer class implements
the complete training pipeline including forward passes, loss computation,
optimization, and evaluation.

Classes:
    BaseTrainer: Abstract base class defining the training interface
    AutoregressiveTrainer: Specialized trainer for autoregressive models with
                          sequential prediction and teacher forcing

The trainers follow the template method pattern, where BaseTrainer provides
the overall training structure and subclasses implement specific forward pass
logic for their respective architectures.
"""

# Import key classes for external use
from .autoregressive_trainer import *
from .base_trainer import *

__all__ = [] 