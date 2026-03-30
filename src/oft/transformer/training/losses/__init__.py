"""
Individual Loss Function Implementations for Object Detection and Tracking.

This module provides specialized loss functions for different aspects of object
detection and tracking tasks. Each loss function is designed to handle specific
prediction types and provides appropriate optimization objectives.

Classes:
    FocalLoss: Classification loss with focal weighting for class imbalance
    UncertaintyWeightedRegressionLoss: Regression loss with uncertainty estimation
    LossGIoUBEV: Geometric IoU loss for bird's eye view bounding boxes
    LossAttributes: Cross-entropy loss for object attribute classification
    LossBoxesL1Offset: L1 regression loss for bounding box offset prediction
    LossVelocities: L1 regression loss for velocity prediction

The loss functions are designed to work together with Hungarian matching to
provide comprehensive training objectives for multi-task learning scenarios.
"""

# Import key functions for external use
from .classification_losses import *
from .regression_losses import *
from .giou_losses import *

__all__ = [] 