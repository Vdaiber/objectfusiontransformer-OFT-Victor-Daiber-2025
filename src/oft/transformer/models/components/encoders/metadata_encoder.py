# File: src/oft/transformer/models/components/encoders/metadata_encoder.py
"""
Metadata encoders for multi-modal sensor fusion.

This module provides specialized encoders for both object-level and scene-level metadata
processing, enabling rich contextual information integration into the transformer model.
The encoders transform categorical sensor identification, continuous confidence scores,
and environmental context (weather, time, location) into high-dimensional embeddings.

Currently uses ACTIVE additive integration for object and scene metadata processing.
Cross-attention mechanisms are PREPARED FOR FUTURE WORK - infrastructure implemented but not activated.
ACTIVATION: Implement cross-attention forward pass in main architecture (lines 147-154).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

class ObjectMetadataEncoder(nn.Module):
    """
    Object-level metadata encoder for sensor-specific information processing.
    
    This module encodes per-object metadata including sensor identification and confidence
    scores into high-dimensional embeddings suitable for transformer-based processing.
    
    Args:
        d_model: Output embedding dimension for integration with transformer model
        object_config: Configuration dictionary containing:
            - 'num_sensors': Number of unique sensor types for embedding lookup
            - 'sensor_embed_dim': Embedding dimension for sensor identification
            - 'confidence_embed_dim': Embedding dimension for confidence scores
    """
    
    def __init__(self, d_model: int, object_config: Dict[str, Any]):
        """Initialize the Object Metadata Encoder with configurable embedding dimensions.
        
        Args:
            d_model: Output embedding dimension for integration with transformer model
            object_config: Configuration dictionary containing embedding parameters
        """
        super().__init__()
        self.d_model = d_model
        
        # Initialize learned embedding layer for categorical sensor identification
        self.sensor_embedding = nn.Embedding(
            num_embeddings=object_config['num_sensors'],
            embedding_dim=object_config['sensor_embed_dim']
        )
        
        # Initialize linear projection layer for continuous confidence scores
        self.confidence_projection = nn.Linear(
            in_features=1,
            out_features=object_config['confidence_embed_dim']
        )
        
        # NEW: Initialize learned embedding layer for unique detection IDs
        self.detection_id_embedding = nn.Embedding(
            num_embeddings=object_config['detection_id']['vocab_size'],
            embedding_dim=object_config['detection_id']['embed_dim']
        )
        
        # Compute total concatenated dimension for combined metadata representation
        object_metadata_dim = (
            object_config['sensor_embed_dim'] +
            object_config['confidence_embed_dim'] +
            object_config['detection_id']['embed_dim']
        )
        
        # Construct multi-layer perceptron for dimensionality reduction to target d_model
        self.output_mlp = nn.Sequential(
            nn.Linear(object_metadata_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        print(f"ObjectMetadataEncoder initialized: Input dim: {object_metadata_dim}, Output dim: {d_model}")

    def forward(self, metadata_features: torch.Tensor) -> torch.Tensor:
        """Encode object-level metadata features into high-dimensional embeddings.
        
        Args:
            metadata_features: Input metadata tensor with shape (batch_size, num_objects, 2)
                where the last dimension contains [confidence_score, sensor_id] for each object detection.
        
        Returns:
            torch.Tensor: Encoded object metadata embeddings with shape (batch_size, num_objects, d_model)
        """
        batch_size, num_objects, meta_dim = metadata_features.shape
        
        # Extract confidence scores, sensor IDs, and detection IDs from input tensor
        # Confidence features are in range [0,1] from sensor detections
        confidence_feature = metadata_features[..., 0:1]
        sensor_id_feature = metadata_features[..., 1].long()
        detection_id_feature = metadata_features[..., 2].long()
        
        # Transform each metadata component through specialized encoding layers
        confidence_embedding = self.confidence_projection(confidence_feature)
        sensor_embedding = self.sensor_embedding(sensor_id_feature)
        detection_id_embedding = self.detection_id_embedding(detection_id_feature)
        
        # Concatenate encoded features along feature dimension and reduce dimensionality
        concatenated_embeddings = torch.cat([
            confidence_embedding, 
            sensor_embedding, 
            detection_id_embedding
        ], dim=-1)
        object_embeddings = self.output_mlp(concatenated_embeddings)
        
        return object_embeddings

class SceneMetadataEncoder(nn.Module):
    """
    Scene-level metadata encoder for environmental context processing.
    
    This module encodes scene metadata (weather, time, location, etc.) into high-dimensional
    embeddings that provide global context for the entire sample.
    
    Args:
        d_model: Output embedding dimension for integration with transformer model
        scene_config: Configuration dictionary containing metadata categories as keys,
            each with 'vocab' (List[str]) and 'embed_dim' (int)
    """
    
    def __init__(self, d_model: int, scene_config: Dict[str, Dict[str, Any]]):
        """Initialize the Scene Metadata Encoder with configurable vocabulary mappings.
        
        Args:
            d_model: Output embedding dimension for integration with transformer model
            scene_config: Configuration dictionary containing metadata categories
        """
        super().__init__()
        self.d_model = d_model
        self.scene_config = scene_config
        
        # Initialize embedding layers and vocabulary mappings for each metadata category
        self.embedding_layers = nn.ModuleDict()
        self.vocab_maps = {}
        
        total_concatenated_dim = 0
        
        # Construct embedding layers and vocabulary mappings for each metadata category
        for key, config in scene_config.items():
            vocab = config['vocab']
            embed_dim = config['embed_dim']
            
            vocab_size = len(vocab)
            self.embedding_layers[key] = nn.Embedding(vocab_size, embed_dim)
            self.vocab_maps[key] = {value: i for i, value in enumerate(vocab)}
            
            total_concatenated_dim += embed_dim
        
        # Construct multi-layer perceptron for dimensionality reduction to target d_model
        self.output_mlp = nn.Sequential(
            nn.Linear(total_concatenated_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        print(f"SceneMetadataEncoder initialized: Input dim: {total_concatenated_dim}, Output dim: {d_model}")

    def _get_index_for_value(self, key: str, value: str) -> int:
        """Find the vocabulary index for a given metadata value with fallback handling.
        
        Args:
            key: Metadata category key corresponding to embedding layer
            value: Metadata value to look up in vocabulary mapping
            
        Returns:
            int: Index for the value in vocabulary, or 0 for unknown values as fallback
        """
        return self.vocab_maps[key].get(value, 0)

    def forward(self, metadata_batch: List[Dict[str, str]]) -> torch.Tensor:
        """Process a batch of metadata dictionaries into scene-level embeddings.
        
        Args:
            metadata_batch: List of dictionaries containing parsed metadata for each sample
                in the batch. Each dictionary maps metadata category keys to string values.
        
        Returns:
            torch.Tensor: Scene-level embeddings with shape (batch_size, d_model)
        """
        batch_embeddings = []
        
        # Process each sample in the batch through metadata encoding pipeline
        for sample_metadata in metadata_batch:
            concatenated_embeddings = []
            
            # Encode each metadata category for the current sample
            for key in self.scene_config.keys():
                # Retrieve metadata value with fallback to first vocabulary item
                value = sample_metadata.get(key, self.scene_config[key]['vocab'][0])
                value_idx = self._get_index_for_value(key, value)
                
                # Convert index to tensor and embed through learned representation
                idx_tensor = torch.tensor([value_idx], device=next(self.parameters()).device)
                embedding = self.embedding_layers[key](idx_tensor)
                concatenated_embeddings.append(embedding.squeeze(0))
            
            # Combine all category embeddings for single sample through concatenation
            single_sample_embedding = torch.cat(concatenated_embeddings)
            batch_embeddings.append(single_sample_embedding)
        
        # Stack all sample embeddings into batch tensor for parallel processing
        batch_tensor = torch.stack(batch_embeddings)
        
        # Reduce dimensionality to target d_model through multi-layer perceptron
        scene_embeddings = self.output_mlp(batch_tensor)
        
        return scene_embeddings

class MetadataEncoder(nn.Module):
    """
    Hybrid metadata encoder combining object-level and scene-level metadata processing.
    
    This module provides a clean separation between object-specific metadata (sensor identification,
    confidence) and scene-level metadata (weather, time, location, etc.). The design
    enables rich contextual information while maintaining clear separation of concerns.
    
    Args:
        d_model: Output embedding dimension matching the transformer model
        metadata_config: Configuration dictionary containing object-level and scene-level parameters
        cfg: Full configuration dictionary containing component-specific parameters
    """
    
    def __init__(self, d_model: int, metadata_config: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None):
        """Initialize the Hybrid Metadata Encoder with configurable object and scene components.
        
        Args:
            d_model: Output embedding dimension matching the transformer model
            metadata_config: Configuration dictionary containing object and scene parameters
            cfg: Full configuration dictionary containing component-specific parameters
        """
        super().__init__()
        
        # Extract component-specific metadata dimensions with fallback to global parameters
        if cfg is not None and 'model' in cfg:
            model_cfg = cfg['model']
            self.metadata_d_model = model_cfg.get('metadata_d_model', d_model)
            self.metadata_nhead = model_cfg.get('metadata_nhead', 12)
        else:
            self.metadata_d_model = d_model
            self.metadata_nhead = 12
        
        self.d_model = d_model
        self.metadata_config = metadata_config
        
        # Extract object-level configuration parameters for sensor and confidence encoding
        object_config = {
            'num_sensors': metadata_config['num_sensors'],
            'sensor_embed_dim': metadata_config['sensor_embed_dim'],
            'confidence_embed_dim': metadata_config['confidence_embed_dim'],
            'detection_id': metadata_config['detection_id']
        }
        
        # Extract scene-level configuration by filtering object-level keys
        scene_config = {}
        object_keys = {'num_sensors', 'sensor_embed_dim', 'confidence_embed_dim',
                      'num_classes', 'class_embed_dim', 'detection_id'}
        
        for key, value in metadata_config.items():
            if key not in object_keys and isinstance(value, dict) and 'vocab' in value:
                scene_config[key] = value
        
        # Initialize object-level metadata encoder for per-object processing
        self.object_encoder = ObjectMetadataEncoder(self.metadata_d_model, object_config)
        
        # Conditionally initialize scene-level metadata encoder based on configuration
        if scene_config:
            self.scene_encoder = SceneMetadataEncoder(self.metadata_d_model, scene_config)
            self.has_scene_encoder = True
        else:
            self.scene_encoder = None
            self.has_scene_encoder = False
        
        # Initialize projection layer to map from metadata_d_model to d_model
        if self.metadata_d_model != self.d_model:
            self.output_projection = nn.Sequential(
                nn.Linear(self.metadata_d_model, self.d_model),
                nn.ReLU()
            )
        else:
            self.output_projection = None
        
        print(f"Hybrid MetadataEncoder initialized:")
        print(f"  Object encoder: ✓")
        print(f"  Scene encoder: {'✓' if self.has_scene_encoder else '✗'}")
        print(f"  Internal dim: {self.metadata_d_model}")
        print(f"  Output dim: {self.d_model}")
        print(f"  Attention heads: {self.metadata_nhead}")

    def forward(self,
                metadata_features: torch.Tensor,
                scene_meta: Optional[List[Dict[str, str]]] = None
                ) -> torch.Tensor:
        """Encode both object-level and scene-level metadata features into unified embeddings.
        
        Args:
            metadata_features: Input metadata tensor with shape (batch_size, num_objects, 3)
                where the last dimension contains [confidence_score, sensor_id, detection_id] for each object detection.
            scene_meta: Optional list of scene metadata dictionaries for each sample in the batch.
                If None, only object-level metadata is processed.
        
        Returns:
            torch.Tensor: Encoded metadata embeddings with shape (batch_size, num_objects, d_model)
        """
        batch_size, num_objects, _ = metadata_features.shape
        
        # Encode object-level metadata through specialized encoder for per-object processing
        object_embeddings = self.object_encoder(metadata_features)
        
        # Conditionally encode scene-level metadata if available and encoder exists
        if scene_meta is not None and self.has_scene_encoder:
            scene_embeddings = self.scene_encoder(scene_meta)
            
            # Expand scene embeddings to match object dimensions through broadcasting
            scene_embeddings_expanded = scene_embeddings.unsqueeze(1).expand(-1, num_objects, -1)
            
            # Combine object and scene embeddings through element-wise addition
            combined_embeddings = object_embeddings + scene_embeddings_expanded
        else:
            # Fallback to object-level metadata only when scene metadata unavailable
            combined_embeddings = object_embeddings
        
        # Project to final output dimension if necessary
        if self.output_projection is not None:
            combined_embeddings = self.output_projection(combined_embeddings)
        
        return combined_embeddings 