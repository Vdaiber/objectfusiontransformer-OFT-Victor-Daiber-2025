# src/oft/transformer/utils/geometry_utils.py
"""
Geometry utilities for Object Fusion Transformer Pipeline.

This module provides centralized geometry functions including yaw angle conversions
and scene description parsing. This eliminates duplication across multiple files
and provides a single source of truth for geometry operations.

Author: Object Fusion Transformer Team
Year: 2025
"""

import math
from typing import Dict, Tuple
import logging


def yaw_to_sin_cos(yaw: float) -> Tuple[float, float]:
    """Convert yaw angle to sin/cos representation to avoid discontinuity.
    
    Converts a single yaw angle (in radians) to a continuous [sin(yaw), cos(yaw)]
    representation. This avoids the discontinuity at the boundary between -π and π,
    which is crucial for stable training.
    
    Args:
        yaw: Yaw angle in radians.
        
    Returns:
        Tuple of (sin(yaw), cos(yaw)).
    """
    return math.sin(yaw), math.cos(yaw)


def sin_cos_to_yaw(sin_yaw: float, cos_yaw: float) -> float:
    """Convert sin/cos representation back to yaw angle.
    
    Converts a [sin(yaw), cos(yaw)] representation back to a single yaw angle
    using atan2 for robust angle calculation.
    
    Args:
        sin_yaw: Sine of yaw angle.
        cos_yaw: Cosine of yaw angle.
        
    Returns:
        Yaw angle in radians in range [-π, π].
    """
    return math.atan2(sin_yaw, cos_yaw)


def parse_scene_description(description: str) -> Dict[str, str]:
    """Parse structured description string into dictionary format.
    
    Parses semicolon-separated description strings from TruckScenes dataset
    into a structured dictionary format for easier processing.
    
    Args:
        description: Semicolon-separated description string.
                    Example: "weather.rain;area.highway;time.day"
        
    Returns:
        Dictionary mapping scene attributes to their values.
        Example: {"weather": "rain", "area": "highway", "time": "day"}
    """
    if not description:
        return {}
    
    try:
        parts = [p.strip() for p in description.split(';') if p]
        return {p.split('.')[0]: p.split('.')[1] for p in parts if '.' in p}
    except Exception as e:
        logging.warning(f"Could not parse scene description: '{description}'. Error: {e}")
        return {}