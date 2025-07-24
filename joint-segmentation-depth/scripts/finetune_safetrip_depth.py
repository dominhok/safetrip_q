"""
Fine-tuning script for SafeTrip depth data.
After pre-training on KITTI + SafeTrip Seg, fine-tune on SafeTrip depth samples.
"""

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
from datetime import datetime
import wandb
from dotenv import load_dotenv

from model.hydranet_safetrip import HydranetSafeTrip
from dataset_safetrip import SafeTripDataset
from losses import AsymmetricMultiTaskLoss
from utils.metrics import SegmentationMetrics, DepthMetrics

# Load environment variables
load_dotenv('../../.env')


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(cfg):
    """Create Hydranet model."""
    model = HydranetSafeTrip(cfg)
    print(f"Created HydranetSafeTrip model with {cfg['model']['num_classes']} classes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_dataloaders(cfg):
    """Create dataloaders for SafeTrip depth-only fine-tuning."""
    
    # Create SafeTrip dataset with depth_only=True
    train_dataset = SafeTripDataset(
        data_root=cfg['dataset']['root'],
        split='train',
        img_size=tuple(cfg['dataset']['img_size']),
        augment=cfg['training']['augment'],
        depth_only=True  # Only load depth samples
    )
    
    val_dataset = SafeTripDataset(
        data_root=cfg['dataset']['root'],
        split='val',
        img_size=tuple(cfg['dataset']['img_size']),
        augment=False,
        depth_only=True  # Only load depth samples
    )
    
    print(f"\nDepth-only datasets created:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['evaluation']['eval_batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def validate_depth_only(model, val_loader, criterion, cfg):
    """Validate on depth-only samples."""
    model.eval()
    
    # Initialize metrics
    depth_metrics = DepthMetrics(
        max_depth=cfg['dataset']['max_depth'],
        min_depth=cfg['dataset']['min_depth']
    )
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move to device
            images = batch['image'].to(cfg['model']['device'])
            
            # Forward pass
            pred_seg, pred_depth = model(images)
            
            # Move batch data to device
            for key in ['segmentation', 'depth', 'depth_valid', 'has_segmentation', 'has_depth']:
                batch[key] = batch[key].to(cfg['model']['device'])
            
            # Compute loss
            loss, loss_dict = criterion(pred_seg, pred_depth, batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Update depth metrics
            if batch['has_depth'].any():
                depth_indices = batch['has_depth'].nonzero(as_tuple=True)[0]
                pred_depth_valid = pred_depth[depth_indices].squeeze(1)
                gt_depth_valid = batch['depth'][depth_indices]
                valid_mask = batch['depth_valid'][depth_indices]
                depth_metrics.update(pred_depth_valid, gt_depth_valid, valid_mask)
    
    # Compute final metrics
    val_metrics = {
        'total_loss': total_loss / num_batches,
        'depth_rmse': depth_metrics.get_rmse(),
        'depth_mae': depth_metrics.get_mae(),
        'depth_abs_rel': depth_metrics.get_abs_rel(),
        'depth_delta_1': depth_metrics.get_delta_accuracy(1.25)
    }
    
    return val_metrics


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Hydranet on SafeTrip depth data')
    parser.add_argument('--config', type=str, default='../configs/cfg_safetrip.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pre-trained checkpoint (from KITTI+Seg training)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for fine-tuning (default: 1e-4)')
    parser.add_argument('--project', type=str, default='safetrip-finetune',
                        help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder weights during fine-tuning')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override config with command line args
    cfg['training']['epochs'] = args.epochs
    cfg['training']['learning_rate'] = args.lr
    
    # Initialize wandb
    run_name = args.run_name or f"finetune_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.project,
        name=run_name,
        config=cfg,
        tags=['finetune', 'depth-only', 'safetrip']
    )
    
    # Update config with wandb
    cfg = wandb.config.as_dict() if wandb.config else cfg
    
    # Set device
    device = torch.device(cfg['model']['device'] if torch.cuda.is_available() else 'cpu')
    cfg['model']['device'] = device
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(cfg).to(device)
    
    # Load pre-trained checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_metrics' in checkpoint:
            print(f"Pre-trained metrics: mIoU={checkpoint['val_metrics'].get('seg_miou', 0):.4f}, "
                  f"Depth RMSE={checkpoint['val_metrics'].get('depth_rmse', 0):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    # Optionally freeze encoder
    if args.freeze_encoder:
        print("\nFreezing encoder weights...")
        for name, param in model.named_parameters():
            if 'encoder' in name or 'backbone' in name:
                param.requires_grad = False
        
        # Count trainable parameters after freezing
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters after freezing: {trainable_params:,}")
    
    # Create dataloaders
    print("\nCreating depth-only dataloaders...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Load class weights (still needed for loss function structure)
    class_weights = None
    if os.path.exists(cfg['dataset']['class_weights_path']):
        class_weights = torch.from_numpy(
            np.load(cfg['dataset']['class_weights_path'])
        ).float().to(device)
    
    # Create loss function
    criterion = AsymmetricMultiTaskLoss(
        num_classes=cfg['model']['num_classes'],
        class_weights=class_weights,
        use_uncertainty_weighting=cfg['training']['use_uncertainty_weighting'],
        lambda_seg=0.1,  # Lower weight for segmentation during depth fine-tuning
        lambda_depth=0.9  # Higher weight for depth
    ).to(device)
    
    # Create optimizer (only for trainable parameters)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    all_params = list(trainable_params) + list(criterion.parameters())
    optimizer = AdamW(all_params, lr=cfg['training']['learning_rate'], 
                      weight_decay=cfg['training']['weight_decay'])
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs'],
        eta_min=cfg['training']['learning_rate'] * 0.01
    )
    
    # Create gradient scaler
    scaler = GradScaler()
    
    # Training loop
    print("\nStarting fine-tuning...")
    best_rmse = float('inf')
    
    for epoch in range(cfg['training']['epochs']):
        # Training
        model.train()
        total_loss = 0
        depth_loss_sum = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        
        for i, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                pred_seg, pred_depth = model(images)
                
                # Move batch data to device
                for key in ['segmentation', 'depth', 'depth_valid', 'has_segmentation', 'has_depth']:
                    batch[key] = batch[key].to(device)
                
                # Compute loss
                loss, loss_dict = criterion(pred_seg, pred_depth, batch)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            if 'depth_loss' in loss_dict:
                depth_loss_sum += loss_dict['depth_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'depth': f'{loss_dict.get("depth_loss", 0):.4f}'
            })
            
            # Log to wandb
            if i % cfg['training']['log_freq'] == 0:
                wandb.log({
                    'train/total_loss': loss.item(),
                    'train/depth_loss': loss_dict.get('depth_loss', 0),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch + 1,
                }, step=epoch * num_batches + i)
        
        # Validation
        val_metrics = validate_depth_only(model, val_loader, criterion, cfg)
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch metrics
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")
        print(f"Train Loss: {total_loss / num_batches:.4f}")
        print(f"Val Depth RMSE: {val_metrics['depth_rmse']:.4f}")
        print(f"Val Depth MAE: {val_metrics['depth_mae']:.4f}")
        
        # Log validation metrics
        wandb.log({
            'val/total_loss': val_metrics['total_loss'],
            'val/depth_rmse': val_metrics['depth_rmse'],
            'val/depth_mae': val_metrics['depth_mae'],
            'val/depth_abs_rel': val_metrics['depth_abs_rel'],
            'val/depth_delta_1.25': val_metrics['depth_delta_1'],
            'epoch': epoch + 1
        }, step=epoch + 1)
        
        # Save best model
        if val_metrics['depth_rmse'] < best_rmse:
            best_rmse = val_metrics['depth_rmse']
            
            checkpoint_dir = os.path.join('..', 'checkpoints', 'finetune')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'best_rmse': best_rmse,
                'config': cfg
            }
            
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with RMSE: {best_rmse:.4f}")
            
            # Save to wandb
            wandb.save(best_path)
            wandb.run.summary["best_depth_rmse"] = best_rmse
            wandb.run.summary["best_epoch"] = epoch + 1
    
    # Finish wandb run
    wandb.finish()
    print(f"\nFine-tuning completed! Best Depth RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()