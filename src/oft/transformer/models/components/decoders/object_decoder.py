# File: src/oft/transformer/models/components/decoders/object_decoder.py
"""
Object decoder for prediction heads with dynamic input support.

Implements a comprehensive object decoder that processes final fused features
from the transformer backbone to generate predictions for classification, attributes, 
bounding box offsets, and velocities. The decoder operates on dynamic input tokens 
corresponding to sensor detections and produces one prediction per input token.
"""

import torch
import torch.nn as nn
import logging
import math

from typing import Dict, Any, Optional

class ObjectDecoder(nn.Module):
    """Neural network module for object prediction with dynamic input support.
    
    This decoder processes final fused features from the transformer backbone to generate
    comprehensive object predictions including classification, attributes, bounding box offsets,
    and velocities. It includes a transformer-based duplicate detection mechanism.
    
    The module accepts dynamic input tokens corresponding to sensor detections and produces
    one prediction per input token. All regression outputs are normalized to [-1, 1] range
    using pre-computed statistics for consistent pipeline processing.
    
    Args:
        cfg: Configuration dictionary containing model parameters:
            - 'model': Dictionary containing model configuration
                - 'd_model': Dimension of the feature representation
                - 'num_classes': Number of object classes (excluding background)
                - 'num_attribute_classes': Number of attribute classes (excluding background)
                - 'output_heads': Component-specific configurations
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize the Object Decoder with prediction heads and normalization parameters.
        
        Args:
            cfg: Configuration dictionary containing model parameters.
                Must include 'model' key with d_model, num_classes, and num_attribute_classes.
        """
        super().__init__()
        model_cfg = cfg['model']
        
        # Base model dimension for input feature processing
        self.d_model = model_cfg['d_model']
        
        decoder_cfg = model_cfg['decoder']
        
        # Component-specific dimensions for flexible architecture
        self.classification_d_model = decoder_cfg['output_heads']['classification']['d_model']
        self.attribute_d_model = decoder_cfg['output_heads']['attributes']['d_model']
        
        # Separate regression head dimensions for different mathematical properties
        self.center_d_model = decoder_cfg['output_heads']['center_offsets']['d_model']
        self.size_d_model = decoder_cfg['output_heads']['size_offsets']['d_model']
        self.yaw_d_model = decoder_cfg['output_heads']['yaw_offsets']['d_model']
        self.velocity_d_model = decoder_cfg['output_heads']['velocity']['d_model']
        
        # Store separate dimensions for each box component
        # No shared bbox dimension - each head uses its specific d_model
        
        # Classification head: Robust 3-layer MLP with dropout regularization
        # Increased complexity and regularization for the challenging task of
        # classification including "no-object" class
        self.class_head = nn.Sequential(
            nn.Linear(self.classification_d_model, self.classification_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.classification_d_model // 2, self.classification_d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.classification_d_model // 4, decoder_cfg['num_classes'] + 1)
        )
        
        # Attribute head: outputs a distribution over object attributes
        self.attribute_head = nn.Sequential(
            nn.Linear(self.attribute_d_model, self.attribute_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.attribute_d_model // 2, decoder_cfg['num_attribute_classes'] + 1)
        )

        # Separate regression heads with direct normalized outputs for better training stability
        # Each component has its own MLP to handle different mathematical properties
        
        # Center offset head: predicts spatial displacement offsets in ego frame
        # Three-layer MLP for spatial refinement with dropout for regularization
        self.center_offset_head = nn.Sequential(
            nn.Linear(self.center_d_model, self.center_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.center_d_model // 2, self.center_d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.center_d_model // 4, 3),
            nn.Tanh()  # Direct [-1,1] normalized output
        )
        
        # Size offset head: predicts logarithmic scaling factors for object dimensions
        # Three-layer MLP for scale refinement with dropout for regularization
        self.size_offset_head = nn.Sequential(
            nn.Linear(self.size_d_model, self.size_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.size_d_model // 2, self.size_d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.size_d_model // 4, 3),
            nn.Tanh()  # Direct [-1,1] normalized output for log-space scaling
        )
        
        # Yaw offset head: predicts sin/cos angular displacement offsets
        # Three-layer MLP with dropout for regularization; output is sin/cos offset pair in [-1,1], scaled downstream
        self.yaw_offset_head = nn.Sequential(
            nn.Linear(self.yaw_d_model, self.yaw_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.yaw_d_model // 2, self.yaw_d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.yaw_d_model // 4, 2),  # Sin/Cos offset pair
            nn.Tanh()  # Normalized output [-1,1] → scaled for trigonometric addition
        )

        # Velocity regression head with direct normalized outputs
        # Three-layer MLP for motion refinement with dropout for regularization
        self.velocity_head_mlp = nn.Sequential(
            nn.Linear(self.velocity_d_model, self.velocity_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.velocity_d_model // 2, self.velocity_d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.velocity_d_model // 4, 2),
            nn.Tanh()  # Direct [-1,1] normalized output
        )

        # Initialize centralized normalization parameters from global statistics
        from oft.transformer.utils.normalization_utils import get_global_normalizer
        self.normalizer = get_global_normalizer()

        # Input projection layers for component-specific dimensions
        self.input_projections = nn.ModuleDict({
            'classification': nn.Linear(self.d_model, self.classification_d_model) if self.classification_d_model != self.d_model else nn.Identity(),
            'attribute': nn.Linear(self.d_model, self.attribute_d_model) if self.attribute_d_model != self.d_model else nn.Identity(),
            'center': nn.Linear(self.d_model, self.center_d_model) if self.center_d_model != self.d_model else nn.Identity(),
            'size': nn.Linear(self.d_model, self.size_d_model) if self.size_d_model != self.d_model else nn.Identity(),
            'yaw': nn.Linear(self.d_model, self.yaw_d_model) if self.yaw_d_model != self.d_model else nn.Identity(),
            'velocity': nn.Linear(self.d_model, self.velocity_d_model) if self.velocity_d_model != self.d_model else nn.Identity(),
        })
        
        # Initialize classification head bias for training stability
        # Following DETR, set an initial prior towards 'no_object' to stabilize early training
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        final_class_layer = self.class_head[-1]
        final_class_layer.bias.data = torch.ones(decoder_cfg['num_classes'] + 1) * bias_value

    def forward(self, final_features: torch.Tensor, logger: Optional[logging.Logger] = None) -> Dict[str, torch.Tensor]:
        """Generate comprehensive object predictions from final fused features.
        
        Args:
            final_features: Final fused features with shape (batch_size, dynamic_num_boxes, d_model)
            logger: Logger instance for debugging information
            
        Returns:
            dict: A comprehensive dictionary containing all predictions:
                - 'pred_class_logits_batch': Classification logits [batch_size, dynamic_num_boxes, num_classes + 1]
                - 'pred_box_offsets_ego_norm_8d_batch': Normalized bounding box offsets [batch_size, dynamic_num_boxes, 8]
                - 'pred_velocity_offsets_ego_norm_2d_batch': Normalized velocity offset predictions [batch_size, dynamic_num_boxes, 2]
                - 'pred_attributes_logits_batch': Attribute classification logits [batch_size, dynamic_num_boxes, num_attribute_classes + 1]
        """
        # Extract input dimensions for tensor allocation and validation
        batch_size, dynamic_num_boxes, d_model = final_features.shape
        
        # Log input dimensions for debugging and monitoring
        if logger is not None:
            logger.debug(f"ObjectDecoder: Input shape - {final_features.shape}")
            logger.debug(f"ObjectDecoder: Processing {dynamic_num_boxes} dynamic boxes per batch")

        # Get projected features for different components
        # Each component uses its specialized projection for optimal performance
        # Input: final_features (latent space, unit-less) from transformer backbone
        # Output: Component-specific features (latent space, unit-less) ready for prediction heads
        classification_features = self.input_projections['classification'](final_features)  # [B, N, classification_d_model] (unit-less)
        attribute_features = self.input_projections['attribute'](final_features)  # [B, N, attribute_d_model] (unit-less)
        center_features = self.input_projections['center'](final_features)  # [B, N, center_d_model] (unit-less)
        size_features = self.input_projections['size'](final_features)  # [B, N, size_d_model] (unit-less)
        yaw_features = self.input_projections['yaw'](final_features)  # [B, N, yaw_d_model] (unit-less)
        velocity_features = self.input_projections['velocity'](final_features)  # [B, N, velocity_d_model] (unit-less)
        
        # Generate classification predictions: raw logits for object classes
        pred_class_logits_batch = self.class_head(classification_features)
        
        # Generate attribute predictions: raw logits for object attributes
        pred_attributes_logits_batch = self.attribute_head(attribute_features)
        
        # Generate bounding box predictions using separate MLPs for each component
        # Each MLP directly outputs normalized values in [-1, 1] range for stable training
        
        # Get normalized predictions directly from separate heads
        # 8D Box Offsets (EGO, normalized, [-1,1]):
        # [0] x_offset_ego_norm      - Center X offset in ego frame, normalized
        # [1] y_offset_ego_norm      - Center Y offset in ego frame, normalized  
        # [2] z_offset_ego_norm      - Center Z offset in ego frame, normalized
        # [3] w_log_offset_ego_norm  - Width log-scale offset, normalized
        # [4] l_log_offset_ego_norm  - Length log-scale offset, normalized
        # [5] h_log_offset_ego_norm  - Height log-scale offset, normalized  
        # [6] sin_yaw_offset_ego_norm - Sin yaw offset, normalized → scaled for trigonometric addition
        # [7] cos_yaw_offset_ego_norm - Cos yaw offset, normalized → scaled for trigonometric addition
        center_offsets_ego_norm_3d_batch = self.center_offset_head(center_features)
        size_offsets_ego_norm_3d_batch = self.size_offset_head(size_features)
        yaw_offsets_ego_norm_2d_batch = self.yaw_offset_head(yaw_features)  # Sin/Cos pair
        
        # Concatenate normalized box components into final box offset tensor
        pred_box_offsets_ego_norm_8d_batch = torch.cat([center_offsets_ego_norm_3d_batch, size_offsets_ego_norm_3d_batch, yaw_offsets_ego_norm_2d_batch], dim=-1)

        # Generate velocity offset predictions using MLP regression head
        pred_velocity_offsets_ego_norm_2d_batch = self.velocity_head_mlp(velocity_features)

        # Validate tensor shapes to ensure consistency across all predictions
        expected_shape = (batch_size, dynamic_num_boxes)
        assert pred_class_logits_batch.shape[:2] == expected_shape, f"Logits shape mismatch: {pred_class_logits_batch.shape[:2]} vs {expected_shape}"
        assert pred_attributes_logits_batch.shape[:2] == expected_shape, f"Attributes shape mismatch: {pred_attributes_logits_batch.shape[:2]} vs {expected_shape}"
        assert pred_box_offsets_ego_norm_8d_batch.shape[:2] == expected_shape, f"Box offsets shape mismatch: {pred_box_offsets_ego_norm_8d_batch.shape[:2]} vs {expected_shape}"
        assert pred_velocity_offsets_ego_norm_2d_batch.shape[:2] == expected_shape, f"Velocity shape mismatch: {pred_velocity_offsets_ego_norm_2d_batch.shape[:2]} vs {expected_shape}"

        # Compile all predictions into comprehensive output dictionary
        predictions = {
            "pred_class_logits_batch": pred_class_logits_batch,  # Classification logits (unit-less) - raw logits before softmax
            "pred_box_offsets_ego_norm_8d_batch": pred_box_offsets_ego_norm_8d_batch,  # Box offsets (EGO, normalized) - [x,y,z,w,l,h,sin_yaw,cos_yaw] in [-1,1]
            "pred_velocity_offsets_ego_norm_2d_batch": pred_velocity_offsets_ego_norm_2d_batch,  # Velocity offsets (EGO, normalized) - [vx,vy] in [-1,1]
            "pred_attributes_logits_batch": pred_attributes_logits_batch,  # Attribute logits (unit-less) - raw logits before softmax
        }

        # Final validation: Raise an error if any prediction contains invalid values.
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    error_msg = f"NaN detected in decoder output for key '{key}'.\n"
                    error_msg += f"This indicates a severe numerical instability inside the model's prediction heads."
                    raise RuntimeError(error_msg)
                if torch.isinf(value).any():
                    error_msg = f"Infinity detected in decoder output for key '{key}'.\n"
                    error_msg += f"This indicates a severe numerical instability, likely exploding gradients."
                    raise RuntimeError(error_msg)

        # Log output shapes for debugging and monitoring
        if logger is not None:
            logger.debug(f"ObjectDecoder: Output shapes:")
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"  - {key}: {value.shape}")
            logger.debug(f"ObjectDecoder: Successfully generated {dynamic_num_boxes} predictions per batch")

        return predictions 

 