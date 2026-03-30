# src/oft/transformer/training/utils/diagnostic_logger.py
"""
Diagnostic logging utility for detailed training analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any
import torch
import time
from collections import defaultdict

class DiagnosticLogger:
    """
    Logs detailed batch-level statistics to a structured JSONL file for
    in-depth training analysis of loss contributions and gradient norms.
    """
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, output_dir: str):
        self.model = model
        self.criterion = criterion
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "diagnostics.jsonl"
        # Clear the file at the beginning of a new run
        if self.log_file.exists():
            self.log_file.unlink()

    def _get_gradient_norms(self) -> Dict[str, float]:
        """Calculates the L2 norm of gradients for different model components."""
        grad_norms = defaultdict(float)
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                component_name = 'other'
                if 'decoder.class_head' in name:
                    component_name = 'grad_norm_class_head'
                elif 'decoder.attribute_head' in name:
                    component_name = 'grad_norm_attribute_head'
                elif 'decoder.center_offset_head' in name:
                    component_name = 'grad_norm_center_head'
                elif 'decoder.size_offset_head' in name:
                    component_name = 'grad_norm_size_head'
                elif 'decoder.yaw_offset_head' in name:
                    component_name = 'grad_norm_yaw_head'
                elif 'decoder.velocity_head_mlp' in name:
                    component_name = 'grad_norm_velocity_head'
                elif 'inter_modal_fusion' in name:
                    component_name = 'grad_norm_fusion_encoder'
                elif 'intra_modal_encoders' in name:
                    component_name = 'grad_norm_intra_modal_encoders'

                grad_norm = param.grad.data.norm(2).item()
                grad_norms[component_name] += grad_norm
        return dict(grad_norms)

    def log_batch_stats(
        self,
        epoch: int,
        batch_idx: int,
        loss_dict: Dict[str, torch.Tensor],
        weight_dict: Dict[str, float],
        total_loss: float,
        matcher_costs: Dict[str, torch.Tensor],
    ):
        """Logs statistics for a single training batch."""
        # 1. Calculate weighted losses and contributions
        weighted_losses = {k: v.item() * weight_dict.get(k, 1.0) for k, v in loss_dict.items() if k in weight_dict}
        
        total_loss_from_components = sum(weighted_losses.values())

        if total_loss_from_components > 1e-6:
            loss_contributions = {
                k: v / total_loss_from_components * 100 for k, v in weighted_losses.items()
            }
        else:
            loss_contributions = {k: 0 for k in weighted_losses}

        # 2. Calculate gradient norms
        grad_norms = self._get_gradient_norms()
        
        # 3. Process matcher costs
        matcher_costs_processed = {k: v.item() for k, v in matcher_costs.items()}
        
        # DEBUG: Print cost weights to verify they're loaded correctly
        if hasattr(self.criterion, 'matcher') and hasattr(self.criterion.matcher, 'cost_weights'):
            print(f"[DEBUG] Loaded Cost Weights: {self.criterion.matcher.cost_weights}")
        
        # Calculate total weighted matcher cost for contribution analysis
        # FIX: Use full key names (cost_center, not center) to match cost_weights dict
        total_weighted_matcher_cost = sum(v * self.criterion.matcher.cost_weights.get(k, 1.0) for k, v in matcher_costs_processed.items())

        if total_weighted_matcher_cost > 1e-6:
            matcher_cost_contributions = {
                k: (v * self.criterion.matcher.cost_weights.get(k, 1.0)) / total_weighted_matcher_cost * 100 
                for k, v in matcher_costs_processed.items()
            }
        else:
            matcher_cost_contributions = {k: 0 for k in matcher_costs_processed}

        # 4. Create log entry
        log_entry = {
            "timestamp": time.time(),
            "type": "batch_diag",
            "epoch": epoch,
            "batch": batch_idx,
            "total_loss": total_loss,
            "raw_losses": {k: v.item() for k, v in loss_dict.items()},
            "weighted_losses": weighted_losses,
            "loss_contributions_percent": loss_contributions,
            "gradient_norms": grad_norms,
            "matcher_costs": matcher_costs_processed,
            "matcher_cost_contributions_percent": matcher_cost_contributions,
        }
        
        self._write_log(log_entry)
        self._print_diag_summary(log_entry)

    def _print_diag_summary(self, log_entry: Dict[str, Any]):
        """Prints a formatted summary of the diagnostic log to the console."""
        epoch = log_entry['epoch']
        batch = log_entry['batch']
        total_loss = log_entry['total_loss']
        
        loss_parts = [
            f"{name.replace('loss_', '')}: {percent:.1f}%"
            for name, percent in log_entry['loss_contributions_percent'].items()
        ]
        loss_summary = " | ".join(loss_parts)
        
        matcher_parts = [
            f"{name.replace('cost_', '')}: {percent:.1f}%"
            for name, percent in log_entry['matcher_cost_contributions_percent'].items()
        ]
        matcher_summary = " | ".join(matcher_parts)
        
        grad_parts = [
            f"{name.replace('grad_norm_', '')}: {norm:.4f}"
            for name, norm in log_entry['gradient_norms'].items()
        ]
        grad_summary = " | ".join(grad_parts)

        print(f"\n[DIAGNOSTICS] Epoch {epoch}/{batch} | Total Loss: {total_loss:.4f}")
        print(f"  > Matcher Cost %: [ {matcher_summary} ]")
        print(f"  > Loss %: [ {loss_summary} ]")
        print(f"  > Grad Norms: [ {grad_summary} ]")

    def _write_log(self, data: Dict[str, Any]):
        """Appends a log entry to the JSONL file."""
        sanitized_data = self._sanitize_for_json(data)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(sanitized_data) + '\n')

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively sanitize a Python object to be JSON-serializable."""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(x) for x in obj]
        if hasattr(obj, 'item'):
             return obj.item()
        return obj
