# File: src/oft/transformer/training/utils/checkpoint_utils.py
"""
Checkpoint utilities for training state management.

This module provides utility functions for saving and loading training checkpoints,
enabling training to be resumed from a specific point in case of interruption.
The checkpoint system ensures that training progress is not lost due to system
failures or manual interruption.

The checkpoint files contain all necessary state information including model
parameters, optimizer state, learning rate scheduler state, and current epoch
number, allowing for seamless training resumption.

Author: Object Fusion Transformer Team
Year: 2025
"""
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional


def resume_from_checkpoint(model, optimizer, lr_scheduler, output_dir, device, logger):
    """
    Resume training from the latest checkpoint if available.
    
    This function loads model state, optimizer state, learning rate scheduler
    state, and epoch information from the latest checkpoint file. It enables
    training to resume from a specific point in case of interruption.
    
    The function checks for the existence of a checkpoint file and loads
    the training state if found, updating the start_epoch accordingly.
    
    Args:
        model: The neural network model to restore state for.
        optimizer: The optimization algorithm to restore state for.
        lr_scheduler: The learning rate scheduler to restore state for.
        output_dir: Directory containing checkpoints (Path or string).
        device: Computational device (CPU/GPU) for loading tensors.
        logger: Logger instance for progress reporting.
        
    Returns:
        start_epoch: Epoch number to start training from (0 if no checkpoint found).
        
    Note:
        If no checkpoint is found, training will start from epoch 0.
        The checkpoint file is expected to be located at 'output_dir/checkpoints/latest.pth'.
    """
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / 'checkpoints' / 'latest.pth'
    start_epoch = 0
    
    if checkpoint_path.exists():
        # Load checkpoint data from file
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Restore model and optimizer states
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        # Set starting epoch for resuming training
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Checkpoint loaded. Training will continue from epoch {start_epoch}.")
    
    return start_epoch


def save_checkpoint(model, optimizer, lr_scheduler, epoch, output_dir):
    """
    Save the current training state to a checkpoint file.
    
    This function creates a checkpoint file containing the current model state,
    optimizer state, learning rate scheduler state, and epoch information.
    The checkpoint enables training to be resumed from this point if needed.
    
    The checkpoint is saved as 'latest.pth' in the checkpoints subdirectory,
    overwriting any previous checkpoint to maintain a single latest state.
    
    Args:
        model: The neural network model to save state for.
        optimizer: The optimization algorithm to save state for.
        lr_scheduler: The learning rate scheduler to save state for.
        epoch: Current epoch number to save in the checkpoint.
        output_dir: Directory to save the checkpoint (Path or string).
        
    Note:
        The checkpoint directory structure is created automatically if it doesn't exist.
        The checkpoint file contains all necessary information for training resumption.
    """
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / 'checkpoints' / 'latest.pth'
    
    # Create checkpoint dictionary with all necessary state information
    checkpoint_data = {
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(), 
        'epoch': epoch
    }
    
    # Save checkpoint to file
    torch.save(checkpoint_data, checkpoint_path)


def save_best_nds_checkpoint(model, optimizer, lr_scheduler, epoch, nds_score, output_dir, logger):
    """Save model checkpoint if it achieves the best NDS score so far.
    
    This function maintains the best performing model based on NDS (nuScenes Detection Score)
    throughout training. It only saves a new checkpoint if the current NDS score is better
    than the previously recorded best, ensuring we always have access to the best model.
    
    The function creates two files:
    - best_nds.pth: Contains the complete model state, optimizer, scheduler, epoch, and NDS score
    - best_nds_info.json: Contains metadata about the best checkpoint for easy inspection
    
    Args:
        model: The neural network model to save state for
        optimizer: The optimization algorithm to save state for 
        lr_scheduler: The learning rate scheduler to save state for
        epoch: Current epoch number
        nds_score: Current NDS score from evaluation
        output_dir: Directory to save the checkpoint (Path or string)
        logger: Logger instance for progress reporting
        
    Returns:
        bool: True if a new best checkpoint was saved, False otherwise
        
    Note:
        This function is designed to work alongside the regular checkpoint saving
        and provides an easy way to track the best performing model during training.
    """
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    best_checkpoint_path = checkpoints_dir / 'best_nds.pth'
    best_info_path = checkpoints_dir / 'best_nds_info.json'
    
    # Load previous best NDS score if it exists
    best_nds = float('-inf')
    best_epoch = -1
    
    if best_info_path.exists():
        try:
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
                best_nds = best_info.get('best_nds', float('-inf'))
                best_epoch = best_info.get('best_epoch', -1)
        except Exception as e:
            logger.warning(f"Could not load best NDS info: {e}. Starting fresh.")
    
    # Check if current NDS is better than the previous best
    if nds_score > best_nds:
        # Save new best checkpoint
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'nds_score': nds_score
        }
        
        torch.save(checkpoint_data, best_checkpoint_path)
        
        # Save metadata
        best_info = {
            'best_nds': nds_score,
            'best_epoch': epoch,
            'timestamp': epoch  # Using epoch as timestamp for simplicity
        }
        
        with open(best_info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        
        # Log improvement
        improvement = nds_score - best_nds if best_nds != float('-inf') else nds_score
        logger.info(f"🏆 NEW BEST NDS: {nds_score:.6f} (improvement: {improvement:.6f}) at epoch {epoch}")
        logger.info(f"💾 Best checkpoint saved to: {best_checkpoint_path}")
        
        return True
    else:
        logger.info(f"NDS: {nds_score:.6f} vs best: {best_nds:.6f} (epoch {best_epoch})")
        return False


def load_best_nds_checkpoint(model, optimizer, lr_scheduler, output_dir, device, logger):
    """Load the best NDS checkpoint if it exists.
    
    This function loads the model state from the best NDS checkpoint, restoring
    the model to its best performing state during training. This is useful for
    inference or for resuming training from the best checkpoint rather than the latest.
    
    Args:
        model: The neural network model to restore state for
        optimizer: The optimization algorithm to restore state for (optional)
        lr_scheduler: The learning rate scheduler to restore state for (optional)
        output_dir: Directory containing checkpoints (Path or string)
        device: Computational device (CPU/GPU) for loading tensors
        logger: Logger instance for progress reporting
        
    Returns:
        dict: Checkpoint information if loaded successfully, None otherwise
        
    Note:
        If the best NDS checkpoint doesn't exist, the function returns None
        and logs a warning. The model state remains unchanged.
    """
    output_dir = Path(output_dir)
    best_checkpoint_path = output_dir / 'checkpoints' / 'best_nds.pth'
    
    if not best_checkpoint_path.exists():
        logger.warning(f"Best NDS checkpoint not found at: {best_checkpoint_path}")
        return None
    
    try:
        # Load checkpoint data
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        
        # Restore model state
        model.load_state_dict(checkpoint['model'])
        
        # Optionally restore optimizer and scheduler state
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        # Extract checkpoint information
        epoch = checkpoint.get('epoch', -1)
        nds_score = checkpoint.get('nds_score', -1)
        
        logger.info(f"🏆 Best NDS checkpoint loaded: NDS={nds_score:.6f}, Epoch={epoch}")
        
        return {
            'epoch': epoch,
            'nds_score': nds_score
        }
        
    except Exception as e:
        logger.error(f"Failed to load best NDS checkpoint: {e}")
        return None 