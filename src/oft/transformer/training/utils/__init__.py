"""
Training Utilities for Transformer-Based Sensor Fusion Models.

This module provides utility functions and classes that support the training
process, including checkpoint management, metric logging, and geometric
calculations for evaluation.

Functions:
    resume_from_checkpoint: Load training state from checkpoint for resuming
    save_checkpoint: Save current training state to checkpoint file

Classes:
    MetricLogger: Utility for logging and tracking training metrics
    SmoothedValue: Container for computing smoothed statistics

Geometric Utilities:
    generalized_box_iou_bev: GIoU calculation for bird's eye view boxes
    cal_iou_bev: Fast IoU calculation for rotated boxes
    get_corners_2d_bev: Convert 3D boxes to 2D BEV corner points

These utilities provide essential functionality for robust training workflows
with proper state management and evaluation capabilities.
"""

# Import key functions for external use
from .checkpoint_utils import *
from .metric_logger import *
from .iou import *

__all__ = [] 