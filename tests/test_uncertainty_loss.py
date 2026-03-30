import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

# Add src to path to allow imports
import sys
import os
sys.path.insert(0, './src')

from src.oft.transformer.training.losses.regression_losses import UncertaintyRegressionLoss

def test_uncertainty_loss():
    """
    Unit test for the UncertaintyRegressionLoss module.
    Verifies two core principles:
    1. High uncertainty correctly dampens the loss for large errors.
    2. Low (overconfident) uncertainty correctly penalizes large errors.
    """
    print("\\n" + "="*50)
    print("🧪 UNIT TEST: UncertaintyRegressionLoss")
    print("="*50)

    # --- 1. Setup ---
    cfg = {}  # Dummy config
    loss_fn = UncertaintyRegressionLoss(cfg)
    
    # --- 2. Define Test Data ---
    # We simulate one matched prediction
    indices = [(torch.tensor([0]), torch.tensor([0]))] # 1 match: pred 0 -> gt 0
    num_boxes = torch.tensor(1.0)

    # ANCHOR BOX (normalized space)
    anchor_boxes = torch.tensor([[
        0.5, 0.5, 0.5,  # center
        0.2, 0.2, 0.2,  # size (log-space)
        0.0, 1.0,       # yaw (sin=0, cos=1)
        0.1, 0.1        # velocity
    ]], dtype=torch.float64)

    # GROUND TRUTH (normalized space)
    # The target has a large offset from the anchor, especially in size
    target_boxes = torch.tensor([[
        0.51, 0.51, 0.51, # small center offset
        0.8, 0.8, 0.8,   # HUGE size offset
        0.1, 0.9,        # small yaw offset
        0.15, 0.15       # small velocity offset
    ]], dtype=torch.float64)
    
    targets = {
        'gt_boxes_b_normalized': target_boxes.unsqueeze(0) # Shape [B, N_gt, 10]
    }

    # --- 3. Test Case 1: High Uncertainty (Model is "honest" about its error) ---
    print("\\n--- CASE 1: High Error, High Uncertainty (Correctly Learned) ---")
    
    # Model predicts a bad offset, but also predicts HIGH uncertainty (large positive log_var)
    pred_offsets_case1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    pred_vel_offsets_case1 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    high_log_var = torch.tensor([[5.0]], dtype=torch.float64) # High uncertainty
    
    predictions_case1 = {
        'anchor_boxes': anchor_boxes.unsqueeze(0),
        'pred_box_offsets_ego_norm_8d_batch': pred_offsets_case1.unsqueeze(0),
        'pred_velocity_offsets_ego_norm_2d_batch': pred_vel_offsets_case1.unsqueeze(0),
        'pred_center_log_var': high_log_var.expand(-1, 3),
        'pred_size_log_var': high_log_var.expand(-1, 3),
        'pred_yaw_log_var': high_log_var.expand(-1, 2),
        'pred_velocity_log_var': high_log_var.expand(-1, 2),
    }

    losses_case1 = loss_fn(predictions_case1, targets, indices, num_boxes)
    loss_size_high_uncertainty = losses_case1['loss_size']
    
    print(f"Loss (Size) with HIGH uncertainty: {loss_size_high_uncertainty.item():.4f}")
    assert loss_size_high_uncertainty.item() < 3.0, "High uncertainty should significantly reduce the loss."

    # --- 4. Test Case 2: Low Uncertainty (Model is "overconfident") ---
    print("\\n--- CASE 2: High Error, Low Uncertainty (Overconfident) ---")

    # Model predicts the same bad offset, but is VERY confident (large negative log_var)
    low_log_var = torch.tensor([[-5.0]], dtype=torch.float64) # Very low uncertainty (high confidence)

    predictions_case2 = {
        'anchor_boxes': anchor_boxes.unsqueeze(0),
        'pred_box_offsets_ego_norm_8d_batch': pred_offsets_case1.unsqueeze(0),
        'pred_velocity_offsets_ego_norm_2d_batch': pred_vel_offsets_case1.unsqueeze(0),
        'pred_center_log_var': low_log_var.expand(-1, 3),
        'pred_size_log_var': low_log_var.expand(-1, 3),
        'pred_yaw_log_var': low_log_var.expand(-1, 2),
        'pred_velocity_log_var': low_log_var.expand(-1, 2),
    }

    losses_case2 = loss_fn(predictions_case2, targets, indices, num_boxes)
    loss_size_low_uncertainty = losses_case2['loss_size']

    print(f"Loss (Size) with LOW uncertainty: {loss_size_low_uncertainty.item():.4f}")
    assert loss_size_low_uncertainty.item() > 10.0, "Low uncertainty should heavily penalize the large error."

    print("\\n" + "-"*50)
    print("✅ Unit test passed: Loss function behaves as expected.")
    print("="*50 + "\\n")


if __name__ == '__main__':
    test_uncertainty_loss()
