"""
Reproducibility utilities for the transformer-based sensor fusion pipeline.

This module contains functions to ensure reproducible results across different
runs by setting random seeds for Python, NumPy, and PyTorch.
"""
import os
import random
import logging
import numpy as np
import torch
from typing import Optional

def set_seed(seed: Optional[int], logger: Optional[logging.Logger] = None) -> None:
    """
    Set the global random seed for Python, NumPy, and PyTorch to ensure
    reproducibility of experiments.
    
    Args:
        seed: Random seed value to set
        logger: Optional logger for output messages
    """
    if seed is not None:
        if logger:
            logger.info(f"Setting global seed to {seed}")
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if logger:
                logger.info(f"CUDA seed set for {torch.cuda.device_count()} GPU(s).")
    elif logger:
        logger.info("Seed not set in config, proceeding with random initialization.") 