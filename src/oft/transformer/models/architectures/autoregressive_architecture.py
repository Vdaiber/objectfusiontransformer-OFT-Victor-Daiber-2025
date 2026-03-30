# File: src/oft/transformer/models/architectures/autoregressive_architecture.py
"""
Autoregressive Staged Fusion Transformer Model.

Implements the autoregressive staged fusion transformer model for multi-modal object detection.
Currently operates in single-frame mode. Temporal memory management and ego-motion compensation
are implemented but disabled - prepared for future work on multi-frame temporal processing.
Supports training and inference modes with coordinate transformations.
"""

import torch
import torch.nn as nn
import logging
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
from pyquaternion import Quaternion as PyQuaternion

from ..components.encoders.intra_modal_encoder import IntraModalEncoder
from ..components.encoders.metadata_encoder import MetadataEncoder
from ..components.decoders.object_decoder import ObjectDecoder
from ..components.utils.positional_encoding import PositionalEncoding3D
from ..components.inter_modal_fusion import InterModalFusion
from ...utils.temporal_utils import (
    compute_ego_motion_transform,
    apply_ego_motion_transform_physical,
    compute_positional_encoding_from_boxes,
    convert_12d_features_to_10d_memory_format,
    denormalize_memory_to_physical,
    renormalize_physical_to_memory,
    extrapolate_memory_with_physics_physical,
    apply_ego_motion_compensation_physical
)







class ObjectFusionTransformerAutoregressive(nn.Module):
    """Object Fusion Transformer for multi-modal object detection.
    
    Advanced transformer architecture that processes multi-modal sensor data through
    a staged fusion pipeline:
    
    1. Intra-modal encoding: Process individual sensor modalities independently
    2. Inter-modal fusion: Cross-attention between sensor modalities  
    3. Object decoding: Generate detection predictions with multiple heads
    
    Currently operates in single-frame mode. Future components (implemented but disabled):
    - Temporal integration: Memory states across sequential frames
    - Ego-motion compensation: Coordinate transforms for temporal consistency
    
    Key Features:
    - Multi-modal sensor fusion (camera, radar, lidar)
    - Hungarian matching-based training with set prediction
    - Lightweight metadata integration (additive combination)
    - Multiple prediction heads (classification, regression, attributes)
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize ASFT model based on configuration.
        
        Args:
            cfg: Configuration dictionary containing model, dataset, and training configs
        """
        # print("DEBUG: ENTERING ObjectFusionTransformerAutoregressive __init__")
        super().__init__()
        
        # ====================================================================
        # STEP 1: CONFIGURATION PARSING AND AUTOREGRESSIVE SETTINGS
        # ====================================================================
        # All values parsed here are architectural hyperparameters; they do not contain
        # sensor data and thus require no coordinate/normalization annotations.
        self.cfg = cfg
        model_cfg = self.cfg['model']
        dataset_cfg = self.cfg['dataset']

        # Configure temporal behavior
        self.autoregressive_cfg = model_cfg['autoregressive']
        self.use_autoregressive = self.autoregressive_cfg['enabled']
        self.use_ego_motion_compensation = self.autoregressive_cfg['use_ego_motion_compensation']
        self.memory_enabled = self.autoregressive_cfg['memory_enabled']

        # print("DEBUG: Autoregressive settings parsed.")
        # Set core architectural dimensions
        self.d_model = model_cfg['d_model']
        self.nhead = model_cfg['nhead']
        
        # print("DEBUG: Core dimensions set.")
        # Set component-specific dimensions
        self.intra_d_model = model_cfg['intra_modal']['d_model']
        self.intra_nhead = model_cfg['intra_modal']['nhead']
        self.fusion_d_model = model_cfg['inter_modal_fusion']['d_model']
        self.fusion_nhead = model_cfg['inter_modal_fusion']['nhead']
        self.metadata_d_model = model_cfg['metadata_encoder']['d_model']
        self.metadata_nhead = model_cfg['metadata_encoder']['nhead']
        
        # print("DEBUG: Component dimensions set.")
        # Identify active sensor modalities
        self.sensor_names = [s['name'] for s in dataset_cfg['virtual_sensors'] if s['enabled']]

        # print("DEBUG: Sensor names identified.")
        # ====================================================================
        # STEP 2: SENSOR-SPECIFIC INPUT PROJECTIONS
        # ====================================================================
        
        self.object_feature_dim = model_cfg['input_feature_dim']
        self.input_proj = nn.ModuleDict({
            s: nn.Linear(self.object_feature_dim, self.intra_d_model) for s in self.sensor_names
        })
        # print("DEBUG: Input projections created.")

        # ====================================================================
        # STEP 3: CORE ENCODER COMPONENTS
        # ====================================================================

        self.metadata_encoder = MetadataEncoder(
            d_model=self.metadata_d_model,
            metadata_config=model_cfg['metadata_encoder']['config'],
            cfg=cfg
        )
        
        # Store scene metadata exclusion setting for metadata encoder (robust with default)
        self.exclude_scene_metadata = cfg.get('metadata', {}).get('exclude_scene_metadata', False)
        if self.exclude_scene_metadata:
            # print("DEBUG: ALL scene metadata will be EXCLUDED from metadata encoder (weather, area, daytime, season, lighting, structure, construction, etc.)")
            # print("DEBUG: Only object-level metadata will be used: confidence, sensor_id, detection_id")
        
        # print("DEBUG: MetadataEncoder created.")
        
        # Each sensor modality gets its own intra-modal encoder to refine its features independently.

        self.intra_modal_encoders = nn.ModuleDict({
            s: IntraModalEncoder(cfg) for s in self.sensor_names
        })
        # print("DEBUG: IntraModalEncoders created.")
        
        # A projection layer is needed to match the metadata feature dimension to the sensor feature dimension.
        self.metadata_projection = nn.ModuleDict({
            s: nn.Linear(self.metadata_d_model, self.intra_d_model) for s in self.sensor_names
        })
        
        # FUTURE WORK - CROSS-ATTENTION FOR METADATA: 
        # Basic metadata encoding is ACTIVELY USED via additive integration (lines 411-427)
        # Cross-attention layers (prepared here) allow advanced querying of global metadata context,
        # enabling more sophisticated adaptive predictions based on environmental conditions.
        # ACTIVATION: Implement cross-attention in forward pass after line 427 (currently uses additive only)
        self.metadata_cross_attention = nn.ModuleDict({
            s: nn.MultiheadAttention(
                embed_dim=self.intra_d_model,
                num_heads=self.intra_nhead,
                dropout=model_cfg['dropout'],
                batch_first=True
            ) for s in self.sensor_names
        })
        # print("DEBUG: Metadata cross-attention created.")

        # ====================================================================
        # STEP 4: INTER-MODAL FUSION
        # ====================================================================
        # This section fuses the information from all sensor modalities together.
        # First, project all intra-modal features to a common fusion dimension.
        self.intra_to_fusion_projection = nn.Linear(self.intra_d_model, self.fusion_d_model)
        # print("DEBUG: Intra-to-fusion projection created.")
        
        fusion_cfg = model_cfg['inter_modal_fusion']
        self.inter_modal_fusion = InterModalFusion(
            fusion_d_model=self.fusion_d_model,
            fusion_nhead=self.fusion_nhead,
            encoder_depth=fusion_cfg['encoder_depth'],
            dropout=fusion_cfg['dropout']
        )
        # print("DEBUG: InterModalFusion created.")

        # ====================================================================
        # STEP 4.5: TEMPORAL CROSS-ATTENTION COMPONENTS - PREPARED FOR FUTURE WORK
        # ====================================================================
        # FUTURE WORK - MULTI-FRAME TEMPORAL PROCESSING:
        # This is the core of the temporal reasoning pipeline for sequential frame processing.
        # Currently DISABLED - operates in single-frame mode only.
        # ACTIVATION: Set autoregressive.enabled=true and memory_enabled=true in config
        temporal_cfg = model_cfg['temporal_cross_attention']
        temporal_d_model = temporal_cfg['d_model']
        temporal_nhead = temporal_cfg['nhead']
        temporal_dropout = temporal_cfg['dropout']
        
        # The temporal cross-attention layer allows the current frame's features (queries) to attend
        # to the ego-compensated memory features from the previous frame (keys/values).
        self.temporal_cross_attention = nn.MultiheadAttention(
            embed_dim=temporal_d_model,
            num_heads=temporal_nhead,
            dropout=temporal_dropout,
            batch_first=True
        )
        # print("DEBUG: Temporal cross-attention created.")
        
        # The temporal fusion gate is a learned mechanism that decides how much information to retain
        # from the current frame versus the historical context from memory, on a per-object basis.
        # It takes the concatenation of current features and temporal context as input.
        self.temporal_fusion_gate = nn.Sequential(
            nn.Linear(temporal_d_model * 2, temporal_d_model),
            nn.ReLU(),
            nn.Linear(temporal_d_model, 1),
            nn.Sigmoid() # Outputs a scalar gate weight in [0, 1].
        )
        # print("DEBUG: Temporal fusion gate created.")
        
        # Memory filtering masks for temporal processing
        # These masks are used to filter the memory, ensuring that only relevant (non-duplicate)
        # objects from the previous frame are used as context. They are set in the forward pass.
        self.memory_keep_mask = None
        self.memory_padding_mask = None

        # print("DEBUG: Initializing ObjectDecoder...")
        # ====================================================================
        # STEP 5: DECODER FOR FINAL PREDICTIONS
        # ====================================================================
        # The object decoder takes the final fused features and generates predictions for all tasks:
        # classification, box regression, attributes, velocity, and duplicate filtering.
        
        # We override the num_classes in the decoder config with the definitive value from the dataset config.
        model_cfg['decoder']['num_classes'] = len(dataset_cfg['class_names'])
        self.decoder = ObjectDecoder(cfg)
        # print("DEBUG: ObjectDecoder initialized successfully.")
        
        # ====================================================================
        # STEP 6: CENTRALIZED NORMALIZATION SETUP
        # ====================================================================
 
        from ...utils.normalization_utils import get_global_normalizer
        self.normalizer = get_global_normalizer()
        # print("DEBUG: Normalizer loaded.")
        
        # The point_cloud_range is essential for normalizing and denormalizing coordinates.
        if self.normalizer.stats and 'metadata' in self.normalizer.stats:
            # Register as buffer to ensure proper device handling
            self.register_buffer(
                'point_cloud_range',
                torch.tensor(
                    self.normalizer.stats['metadata']['point_cloud_range'],
                    dtype=torch.float32
                )
            )
        else:
            raise ValueError(
                "Could not load point_cloud_range from the global normalizer. "
                "Ensure that normalization_stats.yaml is available and correctly loaded."
            )
        # print("DEBUG: Point cloud range registered.")

        # ====================================================================
        # STEP 7: POSITIONAL ENCODING AND WEIGHT INITIALIZATION
        # ====================================================================
        # The 3D positional encoding module converts normalized (x,y,z) coordinates into
        # high-dimensional embeddings that can be added to the feature vectors to provide spatial context.
        # Use intra_d_model since positional encoding is added to intra-modal features
        self.intra_modal_pe = PositionalEncoding3D(self.intra_d_model)
        # print("DEBUG: Intra-modal PE created.")

        # The temporal PE does the same for ego-compensated memory features.
        # It must use fusion_d_model to match the memory feature dimension.
        self.temporal_pe = PositionalEncoding3D(self.fusion_d_model)
        # print("DEBUG: Temporal PE created.")

        # Initialize model weights for stable training.
        self._initialize_weights()
        # print("DEBUG: Weights initialized.")


    def _initialize_weights(self):
        """Initialize model weights for stable autoregressive training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

        # Special initialization for output heads to ensure a "functional pass-through" at the start.
        if hasattr(self, 'decoder'):
            # Initialize classification head with small weights for uncertain initial predictions.
            if hasattr(self.decoder, 'class_head'):
                final_class_layer = self.decoder.class_head[-1]  # Last Linear layer in Sequential
                nn.init.uniform_(final_class_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_class_layer.bias)
            
            # Initialize attribute head similarly.
            if hasattr(self.decoder, 'attribute_head'):
                final_attr_layer = self.decoder.attribute_head[-1]  # Last Linear layer in Sequential
                nn.init.uniform_(final_attr_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_attr_layer.bias)
            
            # Initialize separate box regression heads to predict near-zero offsets initially.
            # Note: All heads now end with Tanh activation, so we target the last Linear layer (index -2)
            
            # Center offset head: predicts spatial displacement offsets
            if hasattr(self.decoder, 'center_offset_head'):
                final_center_layer = self.decoder.center_offset_head[-2]
                nn.init.uniform_(final_center_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_center_layer.bias)
            
            # Size offset head: predicts logarithmic size scaling factors  
            if hasattr(self.decoder, 'size_offset_head'):
                final_size_layer = self.decoder.size_offset_head[-2]  # Last Linear layer before Tanh
                nn.init.uniform_(final_size_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_size_layer.bias)
                
            # Yaw offset head: predicts rotation changes in sin/cos representation
            if hasattr(self.decoder, 'yaw_offset_head'):
                final_yaw_layer = self.decoder.yaw_offset_head[-2]  # Last Linear layer before Tanh
                nn.init.uniform_(final_yaw_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_yaw_layer.bias)
            
            # Initialize velocity head to predict near-zero velocity offsets initially.
            if hasattr(self.decoder, 'velocity_head_mlp'):
                final_velocity_layer = self.decoder.velocity_head_mlp[-2]  # Last Linear layer before Tanh
                nn.init.uniform_(final_velocity_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_velocity_layer.bias)

        # Initialize the temporal fusion gate to output ~0.5, promoting a balanced fusion at the start of training.
        if hasattr(self, 'temporal_fusion_gate'):
            final_gate_layer = self.temporal_fusion_gate[-2] # The last linear layer before the sigmoid.
            nn.init.uniform_(final_gate_layer.weight, -0.1, 0.1)
            nn.init.zeros_(final_gate_layer.bias)

        # print("✓ Model weights initialized with Xavier uniform distribution + functional pass-through for output layers")







    def forward(self,
                sensor_data: Dict[str, Dict[str, torch.Tensor]],
                memory: Optional[torch.Tensor] = None,
                memory_anchor_boxes: Optional[torch.Tensor] = None,
                dt: Optional[torch.Tensor] = None,
                ego_pose_current: Optional[Dict[str, torch.Tensor]] = None,
                ego_pose_previous: Optional[Dict[str, torch.Tensor]] = None,
                scene_meta: Optional[List[Dict[str, str]]] = None,
                logger: Optional[logging.Logger] = None,
        
                ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Perform single frame-level forward pass for training and inference.
        
        Args:
            sensor_data: Data for current frame from dataloader
            memory: Temporal memory features from previous frame (B, N_mem, D) (EGO, normalized)
            memory_anchor_boxes: 12-D anchor vectors for memory features (B, N_mem, 12) (EGO, normalized)
            dt: Time delta from previous frame (seconds)
            ego_pose_current/previous: Ego vehicle poses with 'translation' (3,) and 'rotation' (4,)
            scene_meta: Scene metadata dictionaries (weather, location)
            logger: Logger for diagnostic output
            
        Returns:
            predictions: Dictionary of tensors from ObjectDecoder
            new_memory: Updated memory features for next time step (EGO, normalized)
            new_memory_anchor_boxes: Updated anchor boxes for new memory (EGO, normalized)
        """
        # ====================================================================
        # STEP 1: INTRA-MODAL FEATURE PROCESSING
        # ====================================================================
        processed_sensor_features_list = []
        all_padding_masks_list = []
        sensor_detection_boxes_list = []
        
        # Loop through each active sensor modality (PARALLEL PROCESSING).
        for name in self.sensor_names:
            sd = sensor_data[name]
            obj_feats, meta_feats, pad = sd['features'], sd['metadata'], sd['mask']
            
            # --- Robustness handling ---
            # If a sensor provides no detections, its feature tensor will have a shape of (B, 0, D).
            # The following layers cannot handle this. We skip any sensor that has no detections.
            if obj_feats.shape[1] == 0:
                if logger:
                    logger.debug(f"Sensor '{name}' has no object detections in this frame. Skipping.")
                continue # Skip to the next sensor.
        
            # STEP 1: Input Projection (Features -> d_model)
            proj = self.input_proj[name](obj_feats)
            
            # STEP 2: Positional Encoding 3D (xyz_centers)
            normalized_centers = obj_feats[:, :, :3]  # (EGO, normalized)
            positional_encoding = self.intra_modal_pe(normalized_centers)
            
            # STEP 3: Features + PE (Add Positional Encoding)
            features_with_pe = proj + positional_encoding
        
            # STEP 4: IntraModalEncoder (Self-Attention)
            q = self.intra_modal_encoders[name](
                sensor_features=features_with_pe,
                sensor_padding_mask=pad
            )
            # q is the output of the intra-modal encoder, shape: (B, N, d_model)
            # STEP 5: MetadataEncoder (Conf, SensorID)
            # Exclude all scene metadata if exclude_scene_metadata is enabled (complete ablation)
            if self.exclude_scene_metadata:
                # Pass None to disable all scene metadata encoding (weather, area, daytime, season, lighting, structure, construction, etc.)
                metadata_features = self.metadata_encoder(meta_feats, scene_meta=None)
            else:
                metadata_features = self.metadata_encoder(meta_feats, scene_meta=scene_meta)
            metadata_projected = self.metadata_projection[name](metadata_features)
            
            # STEP 6: ENHANCED ADDITIVE METADATA INTEGRATION (LÖSUNG 3 + GENERAL CONFIDENCE)
            # Scene-Metadata sollte IMMER präsent sein (Sonnig → Kamera besser, Regen → LiDAR besser)

            # 1. Scene + Object Metadata (wie vorher)
            metadata_component = metadata_projected
            
            # Add general detection confidence for regression
            general_detection_confidence = meta_feats[:, :, 0:1]
            confidence_projected = self.metadata_projection[name](general_detection_confidence.expand(-1, -1, metadata_features.shape[-1]))
            
            # 3. ADDITIVE INTEGRATION: Object + Scene-Metadata + General-Confidence
            refined = q + metadata_component + confidence_projected  # Alle Komponenten addieren!
            
            
            processed_sensor_features_list.append(refined)
            all_padding_masks_list.append(pad)
            
            # The original 14-D input features serve as the initial anchor boxes for the decoder.
            # This ensures the regression heads predict offsets in the same (EGO, normalized) space.
            sensor_detection_boxes_list.append(obj_feats)

        # Concatenate features from all sensors
        # --- Robustness handling ---
        # If all sensors were skipped (because none had detections), the lists will be empty.
        # `torch.cat` on an empty list would crash. We handle this by creating empty tensors with the
        # correct shape, allowing the forward pass to complete and return an empty prediction.
        if not processed_sensor_features_list:
            if logger:
                logger.warning("No detections from any sensor. Proceeding with empty tensors.")
            
            batch_size = next(iter(sensor_data.values()))['features'].shape[0]
            # Use model parameter device to select device
            device = next(self.parameters()).device
            
            all_feats = torch.zeros((batch_size, 0, self.intra_d_model), device=device, dtype=torch.float32)
            all_pads = torch.ones((batch_size, 0), device=device, dtype=torch.bool)
            sensor_detection_boxes_cat = torch.zeros((batch_size, 0, self.object_feature_dim), device=device, dtype=torch.float64)
        else:
            all_feats = torch.cat(processed_sensor_features_list, dim=1)
            all_pads = torch.cat(all_padding_masks_list, dim=1)
            sensor_detection_boxes_cat = torch.cat(sensor_detection_boxes_list, dim=1)
        
        # Project features to inter-modal fusion dimension
        all_feats_fusion = self.intra_to_fusion_projection(all_feats)
        
        # ====================================================================
        # STEP 2: INTER-MODAL FUSION
        # ====================================================================
        # Use self-attention across all concatenated detections to create a unified scene representation.
        # This allows information to flow between detections from different sensors.
        current_fused_features = self.inter_modal_fusion(all_feats_fusion, all_pads) # (EGO, normalized latent)

        # The output of inter-modal fusion becomes the primary feature representation for the current frame.
        fused_features = current_fused_features
        fused_anchor_boxes = sensor_detection_boxes_cat
        
        # ====================================================================
        # STEP 2.5: RESIDUAL CONNECTION WITH ANCHOR BOX REFERENCE PRESERVATION
        # ====================================================================
        # CRITICAL ARCHITECTURAL COMPONENT: Anchor-Feature Mapping with Residual Connection
        # 
        # PROBLEM: Pure inter-modal fusion (fused_features) creates rich global context but 
        # loses the direct informational link between features and their original anchor boxes.
        # This creates inconsistency when decoders predict offsets relative to anchors.
        #
        # SOLUTION: Residual connection combining local anchor-specific features with global context
        # Mathematical formulation: final_features = local_anchor_features + global_fusion_context
        # 
        # This preserves:
        # 1. Anchor-specific characteristics (local_features = all_feats_fusion)
        # 2. Multi-modal global context (global_context = fused_features) 
        # 3. Refinement paradigm where decoders predict offsets relative to anchor positions
        #
        # FUTURE WORK: This residual connection can be enhanced with learned gating mechanisms
        # to dynamically balance local vs. global information based on detection confidence.
        
        final_features = all_feats_fusion + fused_features  # RESIDUAL: Anchor-specific + Global context
        
        # ====================================================================
        # STEP 3: TEMPORAL FUSION (AUTOREGRESSIVE) - PHYSICAL COORDINATE PIPELINE
        # ====================================================================
        if self.use_autoregressive and self.memory_enabled and memory is not None:
            if logger: logger.debug("[TEMPORAL] Starting physical coordinate temporal processing...")
            
            # PHASE 3.0: Convert Memory to Standardized Format
            # Convert 12D fused_anchor_boxes to 10D memory format first
            if memory_anchor_boxes.shape[-1] == 12:
                memory_10d_norm = convert_12d_features_to_10d_memory_format(memory_anchor_boxes)
                if logger: logger.debug(f"[TEMPORAL] Converted 12D→10D memory format: {memory_anchor_boxes.shape} → {memory_10d_norm.shape}")
            else:
                memory_10d_norm = memory_anchor_boxes  # Already 10D format
            
            # PHASE 3.1: Denormalize to Physical Space
            # Convert normalized memory to physical coordinates for mathematically correct operations
            memory_10d_phys = denormalize_memory_to_physical(memory_10d_norm, self.point_cloud_range, self.normalizer)
            if logger: logger.debug(f"[TEMPORAL] Denormalized to physical space: norm_range=[{memory_10d_norm.min():.3f}, {memory_10d_norm.max():.3f}] → phys_range=[{memory_10d_phys.min():.3f}, {memory_10d_phys.max():.3f}]")
            
            # PHASE 3.2: Physics-Based State Extrapolation (in Physical Space)
            # Predict where memory objects would be based on their last known velocity
            if dt is not None and dt.item() > 0:
                memory_10d_phys = extrapolate_memory_with_physics_physical(memory_10d_phys, dt)
                if logger: logger.debug(f"[TEMPORAL] Applied physics extrapolation with dt={dt.item():.4f}s")
            
            # PHASE 3.3: Ego-Motion Compensation (in Physical Space)  
            if self.use_ego_motion_compensation and ego_pose_current is not None and ego_pose_previous is not None:
                # Compute the transformation from the previous to the current ego frame
                ego_transform = compute_ego_motion_transform(ego_pose_previous, ego_pose_current)  # (WORLD, physical)
                
                # Apply transformation in physical space (MATHEMATICALLY CORRECT!)
                memory_10d_phys = apply_ego_motion_compensation_physical(memory_10d_phys, ego_transform)
                if logger: logger.debug("[TEMPORAL] Applied ego motion compensation in physical space")
                
                # Generate positional encodings from physical coordinates
                boxes_7d_phys = memory_10d_phys[..., :7]  # Extract [x,y,z,w,l,h,yaw] physical
                normalized_centers = compute_positional_encoding_from_boxes(
                    boxes_7d_phys, self.point_cloud_range.to(boxes_7d_phys.device)
                )  # Physical → normalized [0,1] for PE
                positional_encoding = self.temporal_pe(normalized_centers)
                
                # Add positional encoding to memory features for spatial alignment
                compensated_memory = memory + positional_encoding
                if logger: logger.debug(f"[TEMPORAL] Added positional encoding: PE_shape={positional_encoding.shape}")
                
            else:
                compensated_memory = memory
                if logger: logger.debug("[TEMPORAL] Ego motion compensation disabled")
            
            # PHASE 3.4: Renormalize back to Memory Format
            # Convert physical coordinates back to normalized format for system consistency
            new_memory_anchor_boxes = renormalize_physical_to_memory(memory_10d_phys, self.point_cloud_range, self.normalizer)
            if logger: logger.debug(f"[TEMPORAL] Renormalized back to memory format: phys_range=[{memory_10d_phys.min():.3f}, {memory_10d_phys.max():.3f}] → norm_range=[{new_memory_anchor_boxes.min():.3f}, {new_memory_anchor_boxes.max():.3f}]")

            # PHASE 3.3: STABILIZED Temporal Cross-Attention 
            # Apply stabilization techniques to prevent NaN
            previous_memory_mask = getattr(self, 'memory_padding_mask', None)

            # STEP 1: Validate Memory Quality before Cross-Attention
            memory_stats = {'memory_has_nan': True, 'memory_has_inf': True}
            current_stats = {'current_has_nan': True, 'current_has_inf': True}

            if compensated_memory is not None:
                memory_stats = {
                    'memory_has_nan': torch.isnan(compensated_memory).any().item(),
                    'memory_has_inf': torch.isinf(compensated_memory).any().item()
                }
                current_stats = {
                    'current_has_nan': torch.isnan(final_features).any().item(),
                    'current_has_inf': torch.isinf(final_features).any().item()
                }

            if compensated_memory is not None and not (
                memory_stats['memory_has_nan'] or memory_stats['memory_has_inf'] or 
                current_stats['current_has_nan'] or current_stats['current_has_inf']
            ):
                # STABILIZATION 1: Layer Normalization before Cross-Attention
                current_norm = torch.nn.functional.layer_norm(
                    final_features, final_features.shape[-1:], eps=1e-6
                )
                memory_norm = torch.nn.functional.layer_norm(
                    compensated_memory, compensated_memory.shape[-1:], eps=1e-6
                )
                
                # STABILIZATION 2: Feature scaling to prevent extreme attention weights
                current_scaled = current_norm * 0.1  # Scale down by factor 10
                memory_scaled = memory_norm * 0.1

                # Clip extreme values to prevent attention overflow
                current_scaled = torch.clamp(current_scaled, min=-2.0, max=2.0)
                memory_scaled = torch.clamp(memory_scaled, min=-2.0, max=2.0)
                
                # STABILIZATION 3: Temperature scaling for attention
                with torch.set_grad_enabled(True):  # Ensure gradients for analysis
                    try:
                        temporal_context, attention_weights = self.temporal_cross_attention(
                            query=current_scaled,
                            key=memory_scaled, 
                            value=memory_scaled,
                            key_padding_mask=previous_memory_mask,
                            need_weights=True  # Retrieve attention weights for analysis
                        )
                        
                        # VALIDATION: Check output for NaN/Inf
                        if torch.isnan(temporal_context).any() or torch.isinf(temporal_context).any():
                            temporal_context = final_features * 0.0  # Zero context as fallback
                            
                    except Exception as e:
                        temporal_context = final_features * 0.0  # Zero context as fallback
                        
            else:
                temporal_context = final_features * 0.0  # Zero context as fallback
            
            # PHASE 3.4: STABILIZED Gated Temporal Fusion
            if temporal_context is not None:
                # Concatenate features for gate computation
                features_concat = torch.cat([final_features, temporal_context], dim=-1)
                
                # Apply gate network with gradient clipping
                gate_weights = torch.clamp(
                    self.temporal_fusion_gate(features_concat), 
                    min=0.0, max=1.0  # Ensure valid gate range
                )
                
                # Weighted combination with safety checks
                final_features = gate_weights * final_features + (1 - gate_weights) * temporal_context
                
                # Final validation
                if torch.isnan(final_features).any() or torch.isinf(final_features).any():
                    if logger: logger.error("[TEMPORAL] Final features contain NaN/Inf - reverting to current only")
                    final_features = current_norm  # Revert to normalized current features
                    
                if logger:
                    gate_stats = {
                        'gate_mean': gate_weights.mean().item(),
                        'gate_std': gate_weights.std().item(),
                        'final_mean': final_features.mean().item(),
                        'final_std': final_features.std().item()
                    }
                    logger.debug(f"[TEMPORAL] Gate Stats: {gate_stats}")
            
            # OLD CODE REMOVED:
            # temporal_context, _ = self.temporal_cross_attention(
            
        else:
            # In the non-temporal case, the final_features are the contextualized features from the current step.
            new_memory = final_features
            # Convert current 12D fused_anchor_boxes to 10D memory format for consistency
            if fused_anchor_boxes.shape[-1] == 12:
                new_memory_anchor_boxes = convert_12d_features_to_10d_memory_format(fused_anchor_boxes)
            if logger: logger.debug(f"[TEMPORAL] Non-autoregressive: converted 12D→10D format: {fused_anchor_boxes.shape} → {new_memory_anchor_boxes.shape}")
            else:
                new_memory_anchor_boxes = fused_anchor_boxes
            if logger: logger.debug("Non-autoregressive mode or no memory available.")

        # ====================================================================
        # STEP 4: OBJECT DECODING
        # ====================================================================
        # The decoder generates the final predictions from the fused features.
        predictions = self.decoder(final_features, logger=logger)

        # ====================================================================
        # STEP 5: CENTRALIZED RECONSTRUCTION OF PREDICTIONS (VOLLSTÄNDIG PHYSIKALISCH)
        # ====================================================================

        anchor_boxes = fused_anchor_boxes # The anchors are the initial detections (EGO, normalized)
        pred_offsets = predictions['pred_box_offsets_ego_norm_8d_batch'] # Decoder offsets (EGO, normalized) - 8D with sin/cos yaw offsets
        pred_velocities_offset = predictions['pred_velocity_offsets_ego_norm_2d_batch'] # Decoder velocities (EGO, normalized)

        # Import utilities for physikalische Berechnung
        from ...utils.normalization_utils import denormalize_coordinates, denormalize_dimensions, get_global_normalizer, normalize_coordinates, normalize_dimensions
        from ...utils.geometry_utils import sin_cos_to_yaw, yaw_to_sin_cos

        normalizer = get_global_normalizer()
        point_cloud_range = torch.tensor(normalizer.stats['metadata']['point_cloud_range'], 
                                        device=anchor_boxes.device, dtype=anchor_boxes.dtype)

        # Pre-allocate tensor for reconstructed boxes
        reconstructed_boxes_full = torch.zeros_like(anchor_boxes)
        
        # Centers: fully physical calculation
        anchor_centers_norm = anchor_boxes[..., :3]
        anchor_centers_phys = denormalize_coordinates(anchor_centers_norm, point_cloud_range)
        
        offset_centers_norm = pred_offsets[..., :3]
        offset_centers_phys = offset_centers_norm * normalizer.center_abs_99p.to(pred_offsets.device)
        
        # 3. Physikalische Addition (meters + meters = meters) - MATHEMATISCH KORREKT
        result_centers_phys = anchor_centers_phys + offset_centers_phys  # (EGO, meters)
        
        # 4. Einheitliche Renormalisierung mit point_cloud_range
        result_centers_norm = normalize_coordinates(result_centers_phys, point_cloud_range)  # (EGO, normalized)
        reconstructed_boxes_full[..., :3] = result_centers_norm
        
        # Dimensions: fully physical calculation
        anchor_dims_norm = anchor_boxes[..., 3:6]
        anchor_dims_phys = denormalize_dimensions(anchor_dims_norm)
        
        offset_dims_norm = pred_offsets[..., 3:6]
        offset_dims_log_phys = offset_dims_norm * normalizer.log_size_abs_99p.to(pred_offsets.device)
        
        # 3. Physikalische log-space Addition (exp(log(a) + log_offset) = a * exp(log_offset)) 
        result_dims_phys = anchor_dims_phys * torch.exp(offset_dims_log_phys)  # (EGO, meters)
        
        # 4. Einheitliche Renormalisierung
        result_dims_norm = normalize_dimensions(result_dims_phys)  # (EGO, normalized)
        reconstructed_boxes_full[..., 3:6] = result_dims_norm
        
        # ====================================================================
        # YAW: Simple sin/cos offset scaling + trigonometric addition
        # ====================================================================
        anchor_yaw_sin, anchor_yaw_cos = anchor_boxes[..., 6], anchor_boxes[..., 7]
        
        # 1. Extract normalized sin/cos offsets from decoder predictions
        offset_sin_norm = pred_offsets[..., 6]  # [B, N] sin offset in [-1, 1] range
        offset_cos_norm = pred_offsets[..., 7]  # [B, N] cos offset in [-1, 1] range
        
        # 2. Apply physical scaling to decoder outputs for meaningful angular offsets

        # Mathematical basis: sin(30°) ≈ 0.5, enabling ±30° correction range  
        max_offset_magnitude = 0.500  # Physical scaling: [-1,1] → [±30°] in sin/cos space
        offset_sin_scaled = offset_sin_norm * max_offset_magnitude  # Scaled angular correction (sin)
        offset_cos_scaled = offset_cos_norm * max_offset_magnitude  # Scaled angular correction (cos)
        
        # 3. Trigonometric addition: anchor_yaw + offset_yaw (using scaled offsets)
        # Additionstheorem: sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
        result_yaw_sin = anchor_yaw_sin * (1.0 + offset_cos_scaled) + anchor_yaw_cos * offset_sin_scaled
        # Additionstheorem: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)  
        result_yaw_cos = anchor_yaw_cos * (1.0 + offset_cos_scaled) - anchor_yaw_sin * offset_sin_scaled
        
        reconstructed_boxes_full[..., 6] = result_yaw_sin
        reconstructed_boxes_full[..., 7] = result_yaw_cos
        
        # ====================================================================
        # VELOCITIES: Vollständig physikalische Berechnung mit korrekten Normalization Types
        # ====================================================================
        # 1. Denormalisiere anchor velocities → m/s (ABSOLUTE relative velocities)
        anchor_vel_norm = anchor_boxes[..., 8:10]  # (EGO, normalized)
        anchor_vel_phys = normalizer.denormalize_velocity_absolute(anchor_vel_norm)  # (EGO, m/s) - ABSOLUTE relative velocities
        
        # 2. Denormalisiere offset velocities → m/s (VELOCITY OFFSETS)
        offset_vel_norm = pred_velocities_offset  # From separate velocity head output
        offset_vel_phys = normalizer.denormalize_velocity_offset(offset_vel_norm)
        
        # 3. Physikalische Addition (absolute + offset = absolute) - MATHEMATISCH KORREKT
        result_vel_phys = anchor_vel_phys + offset_vel_phys  # (EGO, m/s) - ABSOLUTE relative velocities
        
        # 4. Einheitliche Renormalisierung (result ist wieder ABSOLUTE relative velocity)
        result_vel_norm = normalizer.normalize_velocity_absolute(result_vel_phys)  # (EGO, normalized) - ABSOLUTE relative velocities
        reconstructed_boxes_full[..., 8:10] = result_vel_norm

        # Add the reconstructed tensors to predictions
        # IMPORTANT: Loss functions expect 10D boxes [x,y,z,w,l,h,sin,cos,vx,vy]
        predictions['pred_boxes_normalized'] = reconstructed_boxes_full  # Full 10D box
        predictions['pred_velocities_normalized'] = reconstructed_boxes_full[..., 8:10] # Velocity part for compatibility
        
        
        # ====================================================================
        # STEP 6: TEMPORAL MEMORY UPDATE
        # ====================================================================
        # Memory for the next frame is derived from current step features and anchors.
        if self.memory_enabled:
            # Use the decoder's class prediction to filter what to keep in memory for the next frame.
            # In the new Hungarian-based approach, no_object predictions should be filtered from memory
            with torch.no_grad():
                if 'pred_class_logits_batch' in predictions:
                    # Get class probabilities and predicted classes
                    class_probs = torch.softmax(predictions['pred_class_logits_batch'], dim=-1)
                    predicted_classes = torch.argmax(class_probs, dim=-1)  # (B, N)
                    no_object_class_id = predictions['pred_class_logits_batch'].shape[-1] - 1  # Last class is no_object
                    
                    # Keep only predictions that are NOT no_object class
                    keep_mask = predicted_classes != no_object_class_id  # Boolean mask of objects to keep
                    
                    # Update the memory features and anchors, zeroing out no_object predictions
                    new_memory = final_features * keep_mask.unsqueeze(-1)
                    # Convert fused_anchor_boxes to 10D memory format if needed, then apply mask
                    if fused_anchor_boxes.shape[-1] == 12:
                        memory_format_boxes = convert_12d_features_to_10d_memory_format(fused_anchor_boxes)
                    else:
                        memory_format_boxes = fused_anchor_boxes
                    new_memory_anchor_boxes = memory_format_boxes * keep_mask.unsqueeze(-1)
                    
                    # Store the masks for the next frame's temporal cross-attention
                    self.memory_keep_mask = keep_mask
                    self.memory_padding_mask = ~keep_mask  # Invert mask for `key_padding_mask`
                else:
                    # If no class predictions are available, keep everything from the current frame
                    new_memory = final_features
                    # Convert fused_anchor_boxes to 10D memory format if needed
                    if fused_anchor_boxes.shape[-1] == 12:
                        new_memory_anchor_boxes = convert_12d_features_to_10d_memory_format(fused_anchor_boxes)
                    else:
                        new_memory_anchor_boxes = fused_anchor_boxes
                    self.memory_keep_mask = None
                    self.memory_padding_mask = None
                    if logger: logger.warning("No class predictions found. Keeping all objects in memory.")
        else:
            # If memory is disabled entirely, the returned memory is just a placeholder and won't be used
            new_memory = final_features
            # Convert fused_anchor_boxes to 10D memory format if needed
            if fused_anchor_boxes.shape[-1] == 12:
                new_memory_anchor_boxes = convert_12d_features_to_10d_memory_format(fused_anchor_boxes)
            else:
                new_memory_anchor_boxes = fused_anchor_boxes
        
        # Add auxiliary data for loss calculation and diagnostics
        predictions["anchor_boxes"] = fused_anchor_boxes
        predictions["fused_padding_mask"] = ~all_pads  # HungarianMatcher expects False for valid items

        return predictions, new_memory, new_memory_anchor_boxes 