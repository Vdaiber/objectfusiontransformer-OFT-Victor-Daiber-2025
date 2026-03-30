# File: src/oft/transformer/training/callbacks/early_stopping.py
"""
Early Stopping Callback für Overfitting-Prävention.
Stoppt Training bei Validation Loss Anstieg über mehrere Epochen.
"""

import logging
import torch
from typing import Dict, Any, Optional
import numpy as np


class EarlyStopping:
    """
    Early Stopping Implementation für Object Fusion Transformer.
    
    Überwacht Validation Loss und stoppt Training bei ausbleibendem Improvement.
    """
    
    def __init__(
        self, 
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best_weights: bool = True,
        baseline: Optional[float] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Early Stopping.
        
        Args:
            patience: Anzahl Epochen ohne Improvement vor Stopp
            min_delta: Minimum change to qualify as improvement
            mode: 'min' für loss (niedriger = besser), 'max' für accuracy
            restore_best_weights: Whether to restore best model weights
            baseline: Minimum acceptable metric value
            logger: Logger für Status-Updates
        """
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.logger = logger or logging.getLogger(__name__)
        
        # Internal state
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Mode-specific setup
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def __call__(
        self, 
        epoch: int, 
        current_score: float, 
        model: torch.nn.Module = None
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            current_score: Current validation metric value
            model: Model to save weights from (if restore_best_weights=True)
            
        Returns:
            True if training should stop, False otherwise
        """
        # Check baseline if provided
        if self.baseline is not None:
            if self.mode == 'min' and current_score > self.baseline:
                self.logger.warning(f"Metric {current_score:.6f} worse than baseline {self.baseline:.6f}")
            elif self.mode == 'max' and current_score < self.baseline:
                self.logger.warning(f"Metric {current_score:.6f} worse than baseline {self.baseline:.6f}")
        
        # First epoch
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.logger.info(f"Early Stopping initialized: best_score={self.best_score:.6f} at epoch {epoch}")
            return False
        
        # Check for improvement
        if self.monitor_op(current_score, self.best_score + self.min_delta):
            # Improvement found
            improvement = abs(current_score - self.best_score)
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            self.logger.info(f"🏆 NEW BEST: score={self.best_score:.6f} (improvement: {improvement:.6f}) at epoch {epoch}")
            return False
        else:
            # No improvement
            self.wait += 1
            patience_remaining = self.patience - self.wait
            self.logger.info(f"No improvement: score={current_score:.6f} vs best={self.best_score:.6f} "
                           f"(patience: {patience_remaining}/{self.patience})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.logger.warning(f"EARLY STOPPING: No improvement for {self.patience} epochs")
                self.logger.info(f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
                return True
            
            return False
    
    def restore_best_model(self, model: torch.nn.Module) -> None:
        """Restore model to best weights."""
        if self.best_weights is not None:
            # Move weights back to model's device
            device = next(model.parameters()).device
            best_weights_on_device = {k: v.to(device) for k, v in self.best_weights.items()}
            model.load_state_dict(best_weights_on_device)
            self.logger.info(f"Restored model to best weights from epoch {self.best_epoch}")
        else:
            self.logger.warning("No best weights available to restore")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of early stopping results."""
        return {
            'early_stopped': self.stopped_epoch > 0,
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'patience_used': self.wait,
            'patience_limit': self.patience
        }


class ValidationLossEarlyStopping(EarlyStopping):
    """
    Specialized Early Stopping für Validation Loss.
    Optimiert für Object Detection Training.
    """
    
    def __init__(
        self, 
        patience: int = 3,          # Weniger Geduld für Loss
        min_delta: float = 0.005,   # Minimum 0.5% Improvement
        restore_best_weights: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(
            patience=patience,
            min_delta=min_delta,
            mode='min',  # Lower loss is better
            restore_best_weights=restore_best_weights,
            baseline=None,
            logger=logger
        )
        
        # Additional tracking for validation loss
        self.loss_history = []
        self.improvement_history = []
    
    def __call__(
        self, 
        epoch: int, 
        val_loss: float, 
        model: torch.nn.Module = None
    ) -> bool:
        """
        Check validation loss for early stopping.
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
            model: Model for weight saving
            
        Returns:
            True if training should stop
        """
        self.loss_history.append(val_loss)
        
        # Calculate improvement from last epoch
        if len(self.loss_history) > 1:
            improvement = self.loss_history[-2] - self.loss_history[-1]  # Positive = improvement
            self.improvement_history.append(improvement)
        
        should_stop = super().__call__(epoch, val_loss, model)
        
        # Additional logging für validation loss
        if len(self.improvement_history) > 0:
            recent_improvement = self.improvement_history[-1]
            if recent_improvement > 0:
                self.logger.info(f"Validation Loss improved by {recent_improvement:.6f}")
            else:
                self.logger.info(f"Validation Loss worsened by {abs(recent_improvement):.6f}")
        
        return should_stop
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get detailed validation loss summary."""
        summary = self.get_summary()
        
        if self.loss_history:
            summary.update({
                'initial_val_loss': self.loss_history[0],
                'final_val_loss': self.loss_history[-1],
                'best_val_loss': self.best_score,
                'total_improvement': self.loss_history[0] - self.best_score if self.best_score else 0,
                'loss_trajectory': self.loss_history[-5:] if len(self.loss_history) >= 5 else self.loss_history
            })
        
        return summary
