# src/oft/transformer/training/utils/gradient_monitor.py
"""
Gradient monitoring utilities for training diagnostics.

This module provides comprehensive gradient monitoring capabilities to track
gradient sizes for different prediction heads during training, helping to
diagnose learning issues and ensure proper gradient flow.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import json
import time
from datetime import datetime


class GradientMonitor:
    """Comprehensive gradient monitoring for training diagnostics."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 config: Dict[str, Any],
                 output_dir: str = "output/gradient_monitoring",
                 log_frequency: int = 10,
                 max_history: int = 1000):
        """
        Initialize gradient monitor.
        
        Args:
            model: PyTorch model to monitor
            config: Configuration dictionary
            output_dir: Output directory for gradient logs and plots
            log_frequency: How often to log gradients (every N batches)
            max_history: Maximum number of gradient measurements to keep in memory
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_frequency = log_frequency
        self.max_history = max_history
        
        # Gradient tracking
        self.gradient_history = defaultdict(lambda: deque(maxlen=max_history))
        self.parameter_groups = self._identify_parameter_groups()
        
        # Statistics tracking
        self.stats = {
            'total_batches': 0,
            'logged_batches': 0,
            'gradient_explosions': 0,
            'gradient_vanishing': 0,
            'nan_gradients': 0,
            'inf_gradients': 0,
        }
        
        # Alert thresholds
        self.thresholds = {
            'explosion_threshold': config.get('gradient_monitoring', {}).get('explosion_threshold', 1000.0),
            'vanishing_threshold': config.get('gradient_monitoring', {}).get('vanishing_threshold', 1e-6),
            'nan_threshold': 0,  # Any NaN is a problem
            'inf_threshold': 0,  # Any Inf is a problem
        }
        
        print(f"📊 Initialized GradientMonitor:")
        print(f"   - Output directory: {self.output_dir}")
        print(f"   - Log frequency: every {log_frequency} batches")
        print(f"   - Parameter groups: {list(self.parameter_groups.keys())}")
        print(f"   - Explosion threshold: {self.thresholds['explosion_threshold']}")
        print(f"   - Vanishing threshold: {self.thresholds['vanishing_threshold']}")
        
    def _identify_parameter_groups(self) -> Dict[str, List[str]]:
        """Identify parameter groups for monitoring."""
        parameter_groups = {
            'box_regression': [],
            'classification': [],
            'velocity_regression': [],
            'attribute_classification': [],
            'transformer': [],
            'encoder': [],
            'decoder': [],
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'box' in name.lower() or 'offset' in name.lower():
                    parameter_groups['box_regression'].append(name)
                elif 'logit' in name.lower() or 'class' in name.lower():
                    parameter_groups['classification'].append(name)
                elif 'velocity' in name.lower():
                    parameter_groups['velocity_regression'].append(name)
                elif 'attribute' in name.lower():
                    parameter_groups['attribute_classification'].append(name)
                elif 'transformer' in name.lower():
                    parameter_groups['transformer'].append(name)
                elif 'encoder' in name.lower():
                    parameter_groups['encoder'].append(name)
                elif 'decoder' in name.lower():
                    parameter_groups['decoder'].append(name)
                else:
                    # Default to transformer for unmatched parameters
                    parameter_groups['transformer'].append(name)
                    
        # Remove empty groups
        parameter_groups = {k: v for k, v in parameter_groups.items() if v}
        
        return parameter_groups
        
    def monitor_gradients(self, 
                         epoch: int, 
                         batch_idx: int, 
                         losses: Dict[str, torch.Tensor],
                         force_log: bool = False) -> Dict[str, Any]:
        """
        Monitor gradients after backward pass.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            losses: Computed losses
            force_log: Force logging regardless of frequency
            
        Returns:
            Dictionary containing gradient statistics and alerts
        """
        self.stats['total_batches'] += 1
        
        # Check if we should log this batch
        should_log = (batch_idx % self.log_frequency == 0) or force_log
        
        if not should_log:
            return {}
            
        self.stats['logged_batches'] += 1
        
        # Collect gradient statistics
        gradient_stats = self._collect_gradient_stats()
        
        # Check for issues
        alerts = self._check_gradient_issues(gradient_stats)
        
        # Store in history
        timestamp = time.time()
        for group_name, stats in gradient_stats.items():
            self.gradient_history[group_name].append({
                'epoch': epoch,
                'batch': batch_idx,
                'timestamp': timestamp,
                **stats
            })
            
        # Log to file
        self._log_gradient_stats(epoch, batch_idx, gradient_stats, alerts, losses)
        
        # Print summary if there are alerts
        if alerts:
            self._print_alerts(epoch, batch_idx, alerts)
            
        return {
            'gradient_stats': gradient_stats,
            'alerts': alerts,
            'should_log': should_log
        }
        
    def _collect_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """Collect gradient statistics for all parameter groups."""
        gradient_stats = {}
        
        for group_name, param_names in self.parameter_groups.items():
            group_norms = []
            group_gradients = []
            
            for param_name in param_names:
                param = dict(self.model.named_parameters())[param_name]
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    group_norms.append(grad_norm)
                    group_gradients.append(param.grad.data.clone())
                    
            if group_norms:
                # Calculate statistics
                group_norms = np.array(group_norms)
                gradient_stats[group_name] = {
                    'mean_norm': float(np.mean(group_norms)),
                    'std_norm': float(np.std(group_norms)),
                    'min_norm': float(np.min(group_norms)),
                    'max_norm': float(np.max(group_norms)),
                    'total_norm': float(np.sum(group_norms)),
                    'num_params': len(group_norms),
                    'has_nan': bool(np.isnan(group_norms).any()),
                    'has_inf': bool(np.isinf(group_norms).any()),
                }
                
                # Calculate gradient magnitude distribution
                if group_gradients:
                    all_gradients = torch.cat([g.flatten() for g in group_gradients])
                    gradient_stats[group_name].update({
                        'gradient_mean': float(all_gradients.mean().item()),
                        'gradient_std': float(all_gradients.std().item()),
                        'gradient_min': float(all_gradients.min().item()),
                        'gradient_max': float(all_gradients.max().item()),
                    })
            else:
                gradient_stats[group_name] = {
                    'mean_norm': 0.0,
                    'std_norm': 0.0,
                    'min_norm': 0.0,
                    'max_norm': 0.0,
                    'total_norm': 0.0,
                    'num_params': 0,
                    'has_nan': False,
                    'has_inf': False,
                    'gradient_mean': 0.0,
                    'gradient_std': 0.0,
                    'gradient_min': 0.0,
                    'gradient_max': 0.0,
                }
                
        return gradient_stats
        
    def _check_gradient_issues(self, gradient_stats: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Check for gradient issues and return alerts."""
        alerts = defaultdict(list)
        
        for group_name, stats in gradient_stats.items():
            # Check for gradient explosion
            if stats['max_norm'] > self.thresholds['explosion_threshold']:
                alerts['explosion'].append(f"{group_name}: max_norm={stats['max_norm']:.2f}")
                self.stats['gradient_explosions'] += 1
                
            # Check for gradient vanishing
            if stats['max_norm'] < self.thresholds['vanishing_threshold']:
                alerts['vanishing'].append(f"{group_name}: max_norm={stats['max_norm']:.2e}")
                self.stats['gradient_vanishing'] += 1
                
            # Check for NaN gradients
            if stats['has_nan']:
                alerts['nan'].append(f"{group_name}: NaN gradients detected")
                self.stats['nan_gradients'] += 1
                
            # Check for Inf gradients
            if stats['has_inf']:
                alerts['inf'].append(f"{group_name}: Inf gradients detected")
                self.stats['inf_gradients'] += 1
                
        return dict(alerts)
        
    def _log_gradient_stats(self, epoch: int, batch_idx: int, gradient_stats: Dict[str, Dict[str, float]], 
                           alerts: Dict[str, List[str]], losses: Dict[str, torch.Tensor]):
        """Log gradient statistics to file."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'batch': batch_idx,
            'gradient_stats': gradient_stats,
            'alerts': alerts,
            'losses': {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        }
        
        # Append to log file
        log_file = self.output_dir / 'gradient_log.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def _print_alerts(self, epoch: int, batch_idx: int, alerts: Dict[str, List[str]]):
        """Print gradient alerts."""
        print(f"\n⚠️  GRADIENT ALERTS - Epoch {epoch}, Batch {batch_idx}:")
        
        for alert_type, messages in alerts.items():
            print(f"   {alert_type.upper()}:")
            for message in messages:
                print(f"     - {message}")
                
    def create_gradient_plots(self, epoch: int, save_plots: bool = True) -> Dict[str, str]:
        """
        Create comprehensive gradient plots.
        
        Args:
            epoch: Current epoch
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        print(f"\n📊 Creating gradient plots for epoch {epoch}...")
        
        plot_paths = {}
        
        # 1. Gradient norm evolution over time
        plot_paths['norm_evolution'] = self._plot_gradient_norm_evolution(epoch, save_plots)
        
        # 2. Gradient distribution by parameter group
        plot_paths['distribution'] = self._plot_gradient_distribution(epoch, save_plots)
        
        # 3. Gradient statistics summary
        plot_paths['statistics'] = self._plot_gradient_statistics(epoch, save_plots)
        
        # 4. Alert frequency over time
        plot_paths['alerts'] = self._plot_alert_frequency(epoch, save_plots)
        
        return plot_paths
        
    def _plot_gradient_norm_evolution(self, epoch: int, save_plots: bool) -> Optional[str]:
        """Plot gradient norm evolution over time."""
        if not any(self.gradient_history.values()):
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Gradient Norm Evolution - Epoch {epoch}', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        for i, (group_name, history) in enumerate(self.gradient_history.items()):
            if i >= len(axes):
                break
                
            if not history:
                continue
                
            ax = axes[i]
            
            # Extract data
            batches = [entry['batch'] for entry in history]
            mean_norms = [entry['mean_norm'] for entry in history]
            max_norms = [entry['max_norm'] for entry in history]
            
            # Plot
            ax.plot(batches, mean_norms, label='Mean Norm', alpha=0.7)
            ax.plot(batches, max_norms, label='Max Norm', alpha=0.7)
            
            # Add thresholds
            ax.axhline(y=self.thresholds['explosion_threshold'], color='red', linestyle='--', 
                      alpha=0.5, label='Explosion Threshold')
            ax.axhline(y=self.thresholds['vanishing_threshold'], color='orange', linestyle='--', 
                      alpha=0.5, label='Vanishing Threshold')
            
            ax.set_title(f'{group_name.replace("_", " ").title()}')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Gradient Norm')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f'gradient_norm_evolution_epoch_{epoch}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
            
    def _plot_gradient_distribution(self, epoch: int, save_plots: bool) -> Optional[str]:
        """Plot gradient distribution by parameter group."""
        if not any(self.gradient_history.values()):
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Gradient Distribution - Epoch {epoch}', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        for i, (group_name, history) in enumerate(self.gradient_history.items()):
            if i >= len(axes):
                break
                
            if not history:
                continue
                
            ax = axes[i]
            
            # Extract gradient means
            gradient_means = [entry.get('gradient_mean', 0) for entry in history]
            
            # Plot histogram
            ax.hist(gradient_means, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{group_name.replace("_", " ").title()} - Gradient Distribution')
            ax.set_xlabel('Gradient Mean')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f'gradient_distribution_epoch_{epoch}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
            
    def _plot_gradient_statistics(self, epoch: int, save_plots: bool) -> Optional[str]:
        """Plot gradient statistics summary."""
        if not any(self.gradient_history.values()):
            return None
            
        # Get latest statistics for each group
        latest_stats = {}
        for group_name, history in self.gradient_history.items():
            if history:
                latest_stats[group_name] = history[-1]
                
        if not latest_stats:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Gradient Statistics Summary - Epoch {epoch}', fontsize=16)
        
        # 1. Mean gradient norms by group
        ax1 = axes[0, 0]
        groups = list(latest_stats.keys())
        mean_norms = [latest_stats[g]['mean_norm'] for g in groups]
        
        bars1 = ax1.bar(groups, mean_norms, alpha=0.7)
        ax1.set_title('Mean Gradient Norms by Group')
        ax1.set_ylabel('Mean Norm')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_yscale('log')
        
        # 2. Max gradient norms by group
        ax2 = axes[0, 1]
        max_norms = [latest_stats[g]['max_norm'] for g in groups]
        
        bars2 = ax2.bar(groups, max_norms, alpha=0.7, color='orange')
        ax2.set_title('Max Gradient Norms by Group')
        ax2.set_ylabel('Max Norm')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')
        
        # 3. Number of parameters by group
        ax3 = axes[1, 0]
        num_params = [latest_stats[g]['num_params'] for g in groups]
        
        bars3 = ax3.bar(groups, num_params, alpha=0.7, color='green')
        ax3.set_title('Number of Parameters by Group')
        ax3.set_ylabel('Number of Parameters')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Gradient standard deviations by group
        ax4 = axes[1, 1]
        std_norms = [latest_stats[g]['std_norm'] for g in groups]
        
        bars4 = ax4.bar(groups, std_norms, alpha=0.7, color='red')
        ax4.set_title('Gradient Standard Deviations by Group')
        ax4.set_ylabel('Standard Deviation')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f'gradient_statistics_epoch_{epoch}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
            
    def _plot_alert_frequency(self, epoch: int, save_plots: bool) -> Optional[str]:
        """Plot alert frequency over time."""
        # This would require parsing the log file to get alert history
        # For now, create a simple summary plot
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        alert_types = ['explosion', 'vanishing', 'nan', 'inf']
        alert_counts = [
            self.stats['gradient_explosions'],
            self.stats['gradient_vanishing'],
            self.stats['nan_gradients'],
            self.stats['inf_gradients']
        ]
        
        bars = ax.bar(alert_types, alert_counts, alpha=0.7, color=['red', 'orange', 'purple', 'brown'])
        ax.set_title(f'Gradient Alert Summary - Epoch {epoch}')
        ax.set_ylabel('Number of Alerts')
        ax.set_xlabel('Alert Type')
        
        # Add value labels on bars
        for bar, count in zip(bars, alert_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f'gradient_alerts_epoch_{epoch}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
            
    def get_gradient_summary(self) -> Dict[str, Any]:
        """Get comprehensive gradient summary."""
        summary = {
            'statistics': self.stats.copy(),
            'parameter_groups': {k: len(v) for k, v in self.parameter_groups.items()},
            'thresholds': self.thresholds.copy(),
            'history_lengths': {k: len(v) for k, v in self.gradient_history.items()},
        }
        
        # Add latest gradient statistics
        latest_stats = {}
        for group_name, history in self.gradient_history.items():
            if history:
                latest_stats[group_name] = history[-1]
        summary['latest_gradient_stats'] = latest_stats
        
        return summary
        
    def print_gradient_summary(self):
        """Print comprehensive gradient summary."""
        summary = self.get_gradient_summary()
        
        print("\n" + "=" * 80)
        print("📊 GRADIENT MONITORING SUMMARY")
        print("=" * 80)
        
        # Statistics
        stats = summary['statistics']
        print(f"Total batches processed: {stats.get('total_batches', 0)}")
        print(f"Batches logged: {stats.get('logged_batches', 0)}")
        print(f"Gradient explosions: {stats.get('gradient_explosions', 0)}")
        print(f"Gradient vanishing: {stats.get('gradient_vanishing', 0)}")
        print(f"NaN gradients: {stats.get('nan_gradients', 0)}")
        print(f"Inf gradients: {stats.get('inf_gradients', 0)}")
        
        # Parameter groups
        print(f"\nParameter groups monitored:")
        for group_name, num_params in summary['parameter_groups'].items():
            print(f"  - {group_name}: {num_params} parameters")
            
        # Latest statistics
        if summary['latest_gradient_stats']:
            print(f"\nLatest gradient statistics:")
            for group_name, stats in summary['latest_gradient_stats'].items():
                print(f"  - {group_name}:")
                print(f"    Mean norm: {stats['mean_norm']:.4f}")
                print(f"    Max norm: {stats['max_norm']:.4f}")
                print(f"    Std norm: {stats['std_norm']:.4f}")
                
        # Alerts - use get() with default values to avoid KeyError
        total_alerts = (stats.get('gradient_explosions', 0) + stats.get('gradient_vanishing', 0) + 
                       stats.get('nan_gradients', 0) + stats.get('inf_gradients', 0))
        
        if total_alerts > 0:
            print(f"\n⚠️  {total_alerts} gradient issues detected!")
            if stats.get('gradient_explosions', 0) > 0:
                print(f"   - Consider reducing learning rate or adding gradient clipping")
            if stats.get('gradient_vanishing', 0) > 0:
                print(f"   - Consider increasing learning rate or checking loss scaling")
            if stats.get('nan_gradients', 0) > 0 or stats.get('inf_gradients', 0) > 0:
                print(f"   - Check for numerical instability in loss computation")
        else:
            print(f"\n✅ No gradient issues detected!")
            
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)


def create_gradient_monitor(model: torch.nn.Module, config: Dict[str, Any]) -> GradientMonitor:
    """
    Create a gradient monitor for training.
    
    Args:
        model: PyTorch model to monitor
        config: Configuration dictionary
        
    Returns:
        GradientMonitor instance
    """
    return GradientMonitor(model, config)


# Example usage
if __name__ == "__main__":
    # Example configuration
    example_config = {
        'gradient_monitoring': {
            'explosion_threshold': 1000.0,
            'vanishing_threshold': 1e-6,
        }
    }
    
    # Create a dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.box_head = torch.nn.Linear(256, 8)
            self.class_head = torch.nn.Linear(256, 12)
            self.velocity_head = torch.nn.Linear(256, 2)
            
        def forward(self, x):
            return {
                'pred_box_offsets': self.box_head(x),
                'pred_logits': self.class_head(x),
                'pred_velocities': self.velocity_head(x),
            }
    
    dummy_model = DummyModel()
    
    # Create gradient monitor
    monitor = GradientMonitor(dummy_model, example_config)
    
    print("📊 GradientMonitor created successfully!")
    print("Use this in your training loop for comprehensive gradient monitoring.") 