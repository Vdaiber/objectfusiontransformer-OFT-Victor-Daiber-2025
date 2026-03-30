# File: src/oft/transformer/models/components/inter_modal_fusion.py
"""
Inter-modal fusion component for cross-sensor feature integration.

This module implements the inter-modal fusion stage of the transformer pipeline, where features from all
sensor modalities (LiDAR, camera, radar) are fused using self-attention mechanisms. The component enables
cross-sensor information exchange by allowing each detection to attend to all other detections across
different sensors, facilitating duplicate detection and information sharing between modalities.
"""

import torch
import torch.nn as nn


class InterModalFusion(nn.Module):
    """
    Inter-modal fusion using transformer self-attention for cross-sensor feature integration.
    
    This module performs cross-sensor feature fusion by applying self-attention over all sensor
    detections. Each detection can learn from other detections across all sensors, enabling
    information exchange between different modalities.
    
    Args:
        fusion_d_model: Feature dimensionality for fusion (must be divisible by fusion_nhead)
        fusion_nhead: Number of attention heads for parallel processing
        encoder_depth: Number of transformer encoder layers for hierarchical feature learning
        dropout: Dropout probability for regularization during training
    """
    
    def __init__(self, fusion_d_model: int, fusion_nhead: int, 
                 encoder_depth: int, dropout: float):
        """Initialize the InterModalFusion module with fusion-specific transformer encoder architecture.
        
        Args:
            fusion_d_model: Fusion feature dimensionality for input and hidden states
            fusion_nhead: Number of attention heads for parallel processing
            encoder_depth: Number of fusion encoder layers for hierarchical feature learning
            dropout: Dropout probability for regularization during training
        """
        super().__init__()
        
        # Validate input parameters for architectural constraints
        if fusion_d_model % fusion_nhead != 0:
            raise ValueError(f"fusion_d_model ({fusion_d_model}) must be divisible by fusion_nhead ({fusion_nhead})")
        
        # Configure transformer encoder layer with fusion-specific architecture parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_d_model,
            nhead=fusion_nhead,
            dim_feedforward=4 * fusion_d_model,
            dropout=dropout,
            batch_first=True,
            activation='relu',
            norm_first=False  # Post-Norm (standard) for stable training 
        )
        
        # Stack multiple encoder layers to enable hierarchical cross-modal feature learning
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Perform cross-sensor feature fusion using self-attention over all sensor detections.
        
        Args:
            features: Input features with shape [B, N, fusion_d_model] where:
                     B = batch size, N = total number of detections across all sensors
            padding_mask: Boolean mask with shape [B, N] where True indicates
                         padded positions that should be ignored during attention computation
        
        Returns:
            torch.Tensor: Fused features with shape [B, N, fusion_d_model] containing cross-sensor enhanced representations
        """
        # Apply transformer encoding with padding mask to handle variable-length sequences
        return self.encoder(features, src_key_padding_mask=padding_mask) 