# src/oft/transformer/utils/common.py
"""
Common utility functions for the transformer-based sensor fusion pipeline.

This module contains various general-purpose helper functions that are useful
throughout the project, including data type conversion, reproducibility setup,
and coordinate transformation utilities.
"""
import os
import random
import logging
import numpy as np
import torch
from typing import Any, Optional
from pyquaternion import Quaternion as PyQuaternion

def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize a Python object to be JSON-serializable.
    
    Converts an object that may contain NumPy or PyTorch types into a
    JSON-serializable format (pure Python types).
    
    Args:
        obj: Object to be converted.
        
    Returns:
        JSON-serializable version of the input object.
    """
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                       np.int16, np.int32, np.int64, np.uint8,
                       np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float64, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, (np.ndarray,)):
        return [sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)): 
        return None 
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        return sanitize_for_json(obj.cpu().numpy())
    return obj



