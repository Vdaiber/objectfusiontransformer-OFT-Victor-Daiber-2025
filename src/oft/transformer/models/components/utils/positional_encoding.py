# File: src/oft/transformer/models/components/utils/positional_encoding.py
"""
3D sinusoidal positional encoding module for spatial coordinate representation.

This module implements a learnable positional encoding system that transforms 3D spatial coordinates into 
high-dimensional embeddings using sinusoidal functions with multiple frequency bands. The encoding enables 
transformer models to understand spatial relationships, distances, and geometric patterns between objects 
in 3D space by providing explicit spatial context to attention mechanisms.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding3D(nn.Module):
    """
    Generate 3D sinusoidal positional encodings for spatial coordinate representation.
    
    This module implements a learnable positional encoding system that transforms 3D spatial 
    coordinates into high-dimensional embeddings using sinusoidal functions with multiple 
    frequency bands. The encoding enables transformer models to understand spatial relationships, 
    distances, and geometric patterns between objects in 3D space.
    
    Args:
        d_model: The dimensionality of the output positional encoding. Must be divisible 
            by 6 to accommodate sin/cos pairs for 3 coordinates.
        dropout: Dropout probability applied to the final encoding for regularization.
        max_coord_val: Maximum expected coordinate value used for frequency scaling calculations.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_coord_val: float = 150.0):
        """Initialize the 3D positional encoding module with specified parameters.
        
        Args:
            d_model: Output embedding dimensionality, must be divisible by 6 for 
                3D coordinate encoding with sin/cos pairs.
            dropout: Dropout probability for regularization, applied to final encoding.
            max_coord_val: Maximum coordinate value for frequency scaling calculations.
        
        Raises:
            ValueError: If d_model is not divisible by 6 or if d_model//3 is not even.
        """
        super().__init__()
        # Initialize dropout layer for regularization during training
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Validate input parameters for mathematical constraints
        if d_model % 6 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 6 for 3D PE.")
        dim_per_coord = d_model // 3
        if dim_per_coord % 2 != 0:
            raise ValueError(f"d_model // 3 ({dim_per_coord}) must be even for Sin/Cos pairs.")
        
        # Calculate frequency bands using logarithmic scaling for numerical stability
        num_freq_bands_per_coord = dim_per_coord // 2
        
        # Generate logarithmic frequency progression for stable encoding
        log_freq_bands = torch.arange(num_freq_bands_per_coord, dtype=torch.float32)
        
        # Generate frequency bands using standard sinusoidal PE formula
        self.register_buffer('freq_bands',
            torch.exp(log_freq_bands * -(math.log(10000.0) / num_freq_bands_per_coord)))
        

    def forward(self, xyz_coords: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding from 3D spatial coordinates.
        
        Args:
            xyz_coords: Input 3D coordinates tensor with shape (batch_size, num_objects, 3).
                Content: Spatial coordinates where the last dimension contains 
                [x_norm, y_norm, z_norm] values for each object.
        
        Returns:
            torch.Tensor: Positional encoding tensor with shape (batch_size, num_objects, d_model).
                Content: Each object's spatial position encoded into d_model-dimensional 
                representation using sinusoidal functions.
        
        Raises:
            ValueError: If input tensor does not have the expected 3D shape with 3 coordinates.
        """
        # Validate input tensor dimensions and coordinate count
        if xyz_coords.ndim != 3 or xyz_coords.shape[-1] != 3:
            raise ValueError("Input xyz_coords must have shape (batch_size, num_objects, 3).")
        
        # Extract tensor dimensions for processing
        # B: batch size, S: number of objects (detections), 3: coordinate dimensions
        B, S, _ = xyz_coords.shape
        
        # Scale coordinates by frequency bands for multi-scale encoding
        scaled_inputs = xyz_coords.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, -1)
        
        # Generate sin and cos components for each frequency band and coordinate
        encoded_components = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1)
        
        # Reshape to final output format: concatenate all coordinate encodings
        output = encoded_components.view(B, S, self.d_model)
        
        # Apply dropout for regularization and return final encoding
        return self.dropout(output) 