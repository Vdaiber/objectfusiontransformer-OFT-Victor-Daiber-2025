"""
Object Association Algorithms for Transformer-Based Sensor Fusion Models.

This module provides algorithms for associating model predictions with ground
truth objects during training. The matching algorithms are crucial for computing
loss functions that require correspondence between predictions and targets.

Classes:
    HungarianMatcher: Optimal bipartite matching using the Hungarian algorithm
                     for assigning predictions to ground truth objects

The matchers ensure that loss functions are computed on the correct prediction-
ground truth pairs, which is essential for effective training of object detection
and tracking models.
"""

# Import key classes for external use
from .hungarian_matcher import *

__all__ = [] 