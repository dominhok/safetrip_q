"""
Optimizer utilities for multi-task learning.
"""

import torch
from torch.optim import AdamW


def create_multi_task_optimizer(model, criterion, base_lr=1e-3, depth_lr_scale=2.0, weight_decay=1e-4):
    """
    Create optimizer with task-specific learning rates.
    
    Args:
        model: The model
        criterion: The loss function (with learnable parameters)
        base_lr: Base learning rate
        depth_lr_scale: Scale factor for depth-related parameters
        weight_decay: Weight decay
        
    Returns:
        Optimizer with parameter groups
    """
    # Separate parameters by task
    seg_params = []
    depth_params = []
    shared_params = []
    loss_params = []
    
    # Model parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'segm' in name or 'seg_head' in name:
            seg_params.append(param)
        elif 'depth' in name or 'depth_head' in name:
            depth_params.append(param)
        else:
            # Encoder and shared layers
            shared_params.append(param)
    
    # Loss parameters (uncertainty weights)
    for param in criterion.parameters():
        if param.requires_grad:
            loss_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': shared_params, 'lr': base_lr, 'name': 'shared'},
        {'params': seg_params, 'lr': base_lr, 'name': 'segmentation'},
        {'params': depth_params, 'lr': base_lr * depth_lr_scale, 'name': 'depth'},  # Higher LR for depth
        {'params': loss_params, 'lr': base_lr * 0.1, 'name': 'loss_weights'}  # Lower LR for stability
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    # Create optimizer
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    
    # Print parameter group info
    print("\nOptimizer parameter groups:")
    for group in param_groups:
        print(f"  {group['name']}: {len(group['params'])} params, lr={group['lr']}")
    
    return optimizer