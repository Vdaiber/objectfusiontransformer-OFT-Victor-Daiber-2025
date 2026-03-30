"""
Loss Function Orchestration for Transformer-Based Sensor Fusion Models.

This module provides the main loss function classes that orchestrate multiple
loss components for training transformer-based sensor fusion models. The criteria
classes combine individual loss functions with Hungarian matching to compute
comprehensive training objectives.

Classes:
    SetCriterion: Main loss function orchestrator that combines classification,
                 regression, and geometric losses with optimal object association

The criteria ensure that all loss components are properly weighted and computed
on the correct prediction-ground truth pairs, providing stable and effective
training objectives for complex multi-task learning scenarios.
"""

# Import key classes for external use
from .autoregressive_criterion import *

__all__ = [] 