# File: src/oft/transformer/models/components/encoders/intra_modal_encoder.py
"""
Intra-modal encoder for single-sensor detection feature processing.

This module processes detection features from individual sensor streams by adding
3D positional encodings and applying self-attention mechanisms within each
sensor modality. The encoder refines detection features to capture spatial relationships
and contextual information within each sensor's detection set before they are passed
to the inter-modal fusion stage.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class IntraModalEncoder(nn.Module):
    """
    Intra-modal encoder for single-sensor detection feature processing.
    
    This module processes detection features from individual sensor streams by
    adding 3D positional encodings and applying self-attention mechanisms.
    The encoder refines detection features within each sensor modality before
    they are passed to the inter-modal fusion stage.
    
    Args:
        cfg: Configuration dictionary containing model parameters:
            - 'model': Model configuration containing:
                - 'intra_modal': Intra-modal specific configuration with d_model, nhead, ff_dim, layers
                - 'dropout': Dropout probability for regularization
                - 'activation': Activation function for feedforward layers
    """
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize the Intra-Modal Encoder module with configurable architecture parameters.
        
        Args:
            cfg: Configuration dictionary containing model parameters.
        """
        super().__init__()
        model_cfg = cfg['model']
        
        # Extract component-specific hyperparameters for intra-modal processing
        d_model = model_cfg['intra_modal']['d_model']
        nhead = model_cfg['intra_modal']['nhead']
        
        # Construct transformer encoder layer with multi-head self-attention mechanism
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=model_cfg['intra_modal']['ff_dim'],
            dropout=model_cfg['dropout'],
            activation=model_cfg['activation'],
            batch_first=True,
            norm_first=False  # Post-Norm (standard) for stable training 
        )
        self.self_attention_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=model_cfg['intra_modal']['layers'],
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, 
                sensor_features: torch.Tensor, 
                sensor_padding_mask: torch.Tensor
               ) -> torch.Tensor:
        """Process single-sensor detection features through intra-modal encoding.
        
        Args:
            sensor_features: Detection features projected to d_model, with positional
                encoding already added. Shape is (batch_size, num_detections, d_model).
            sensor_padding_mask: Boolean mask indicating valid detections
                with shape (batch_size, num_detections) where True indicates padding
                positions that should be masked during attention computation.
        
        Returns:
            torch.Tensor: Refined sensor features ready for inter-modal fusion
                with shape (batch_size, num_detections, d_model).
        """
        # Apply multi-layer transformer encoder with self-attention mechanism
        final_features = self.self_attention_encoder(
            src=sensor_features,
            src_key_padding_mask=sensor_padding_mask
        )

        return final_features 