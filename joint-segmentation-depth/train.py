#!/usr/bin/env python
"""
Unified training script for SafeTrip-Q Hydranet
Supports KITTI + SafeTrip mixed training
"""
import os
import sys
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
from datetime import datetime
import wandb

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.hydranet_safetrip import HydranetSafeTrip
from dataset_safetrip import SafeTripDataset
from losses import InverseHuberLoss
from utils.metrics import SegmentationMetrics, DepthMetrics


def train_epoch(model, dataloader, seg_criterion, depth_criterion, optimizer, scaler, device, epoch=0):
    """Train for one epoch."""
    model.train()
    metrics = {'loss': 0, 'seg_loss': 0, 'depth_loss': 0, 'count': 0}
    
    pbar = tqdm(dataloader, desc='Training')
    for i, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        seg_gt = batch['segmentation'].to(device)
        depth_gt = batch['depth'].to(device)
        depth_valid = batch['depth_valid'].to(device)
        has_seg = batch['has_segmentation'].to(device)
        has_depth = batch['has_depth'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            # Forward pass
            seg_pred, depth_pred = model(images)
            
            # Initialize losses
            seg_loss = torch.tensor(0.0, device=device)
            depth_loss = torch.tensor(0.0, device=device)
            
            # Segmentation loss (only for samples with segmentation)
            if has_seg.any():
                seg_indices = has_seg.nonzero(as_tuple=True)[0]
                if len(seg_indices) > 0:
                    seg_pred_batch = seg_pred[seg_indices]
                    seg_gt_batch = seg_gt[seg_indices]
                    seg_loss = seg_criterion(seg_pred_batch, seg_gt_batch)
            
            # Depth loss (only for samples with depth)
            if has_depth.any():
                depth_indices = has_depth.nonzero(as_tuple=True)[0]
                if len(depth_indices) > 0:
                    depth_pred_batch = depth_pred[depth_indices].squeeze(1)
                    depth_gt_batch = depth_gt[depth_indices]
                    valid_mask = depth_valid[depth_indices]
                    
                    # Only compute loss on valid depth pixels
                    if valid_mask.any():
                        valid_pred = depth_pred_batch[valid_mask]
                        valid_gt = depth_gt_batch[valid_mask]
                        
                        # Avoid NaN by checking for valid values
                        if valid_gt.numel() > 0 and not torch.isnan(valid_gt).any():
                            depth_loss = depth_criterion(valid_pred, valid_gt)
            
            # Total loss with NaN check
            total_loss = torch.tensor(0.0, device=device)
            if not torch.isnan(seg_loss):
                total_loss = total_loss + 0.5 * seg_loss
            if not torch.isnan(depth_loss):
                total_loss = total_loss + 0.5 * depth_loss
            
            # Skip batch if loss is NaN
            if torch.isnan(total_loss):
                continue
        
        # Backward pass
        scaler.scale(total_loss).backward()
        
        # Gradient clipping to prevent explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        metrics['loss'] += total_loss.item()
        metrics['seg_loss'] += seg_loss.item() if not torch.isnan(seg_loss) else 0
        metrics['depth_loss'] += depth_loss.item() if not torch.isnan(depth_loss) else 0
        metrics['count'] += 1
        
        # Log to wandb every 50 steps
        global_step = epoch * len(dataloader) + i
        if i % 50 == 0:
            wandb.log({
                'train/loss': total_loss.item(),
                'train/seg_loss': seg_loss.item() if not torch.isnan(seg_loss) else 0,
                'train/depth_loss': depth_loss.item() if not torch.isnan(depth_loss) else 0,
                'train/global_step': global_step
            })
        
        # Update progress bar
        if metrics['count'] > 0:
            pbar.set_postfix({
                'loss': f"{metrics['loss']/metrics['count']:.4f}",
                'seg': f"{metrics['seg_loss']/metrics['count']:.4f}",
                'depth': f"{metrics['depth_loss']/metrics['count']:.4f}"
            })
    
    # Return average metrics
    return {
        'loss': metrics['loss'] / max(metrics['count'], 1),
        'seg_loss': metrics['seg_loss'] / max(metrics['count'], 1),
        'depth_loss': metrics['depth_loss'] / max(metrics['count'], 1)
    }


def validate(model, dataloader, seg_criterion, depth_criterion, device, num_classes):
    """Validate the model."""
    model.eval()
    
    metrics = {'loss': 0, 'seg_loss': 0, 'depth_loss': 0, 'count': 0}
    seg_metrics = SegmentationMetrics(num_classes)
    depth_metrics = DepthMetrics()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move to device
            images = batch['image'].to(device)
            seg_gt = batch['segmentation'].to(device)
            depth_gt = batch['depth'].to(device)
            depth_valid = batch['depth_valid'].to(device)
            has_seg = batch['has_segmentation'].to(device)
            has_depth = batch['has_depth'].to(device)
            
            with autocast():
                # Forward pass
                seg_pred, depth_pred = model(images)
                
                # Initialize losses
                seg_loss = torch.tensor(0.0, device=device)
                depth_loss = torch.tensor(0.0, device=device)
                
                # Segmentation
                if has_seg.any():
                    seg_indices = has_seg.nonzero(as_tuple=True)[0]
                    if len(seg_indices) > 0:
                        seg_pred_batch = seg_pred[seg_indices]
                        seg_gt_batch = seg_gt[seg_indices]
                        
                        # Loss
                        seg_loss = seg_criterion(seg_pred_batch, seg_gt_batch)
                        
                        # Metrics
                        seg_pred_argmax = seg_pred_batch.argmax(dim=1)
                        seg_metrics.update(seg_pred_argmax, seg_gt_batch)
                
                # Depth
                if has_depth.any():
                    depth_indices = has_depth.nonzero(as_tuple=True)[0]
                    if len(depth_indices) > 0:
                        depth_pred_batch = depth_pred[depth_indices].squeeze(1)
                        depth_gt_batch = depth_gt[depth_indices]
                        valid_mask = depth_valid[depth_indices]
                        
                        if valid_mask.any():
                            valid_pred = depth_pred_batch[valid_mask]
                            valid_gt = depth_gt_batch[valid_mask]
                            
                            if valid_gt.numel() > 0 and not torch.isnan(valid_gt).any():
                                # Loss
                                depth_loss = depth_criterion(valid_pred, valid_gt)
                                
                                # Metrics
                                depth_metrics.update(depth_pred_batch, depth_gt_batch, valid_mask)
                
                # Total loss
                total_loss = torch.tensor(0.0, device=device)
                if not torch.isnan(seg_loss):
                    total_loss = total_loss + 0.5 * seg_loss
                if not torch.isnan(depth_loss):
                    total_loss = total_loss + 0.5 * depth_loss
                
                if not torch.isnan(total_loss):
                    metrics['loss'] += total_loss.item()
                    metrics['seg_loss'] += seg_loss.item() if not torch.isnan(seg_loss) else 0
                    metrics['depth_loss'] += depth_loss.item() if not torch.isnan(depth_loss) else 0
                    metrics['count'] += 1
    
    # Compute final metrics
    val_metrics = {
        'loss': metrics['loss'] / max(metrics['count'], 1),
        'seg_loss': metrics['seg_loss'] / max(metrics['count'], 1),
        'depth_loss': metrics['depth_loss'] / max(metrics['count'], 1),
        'seg_miou': seg_metrics.get_miou(),
        'seg_acc': seg_metrics.get_pixel_accuracy(),
        'depth_rmse': depth_metrics.get_rmse(),
        'depth_mae': depth_metrics.get_mae(),
        'depth_abs_rel': depth_metrics.get_abs_rel(),
        'depth_delta_1': depth_metrics.get_delta_accuracy(1.25)
    }
    
    # Replace NaN with 0
    for key, value in val_metrics.items():
        if np.isnan(value):
            val_metrics[key] = 0.0
    
    per_class_iou = seg_metrics.get_iou_per_class()
    
    return val_metrics, per_class_iou


def main():
    parser = argparse.ArgumentParser(description='Train Hydranet on SafeTrip-Q')
    parser.add_argument('--config', type=str, default='configs/cfg_safetrip.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--project', type=str, default='safetrip-hydranet')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--tags', nargs='+', default=['hydranet', 'safetrip'])
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Initialize wandb
    run_name = args.run_name or f"hydranet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.project,
        name=run_name,
        config=cfg,
        tags=args.tags
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = HydranetSafeTrip(cfg).to(device)
    print(f"Model created with {cfg['model']['num_classes']} classes")
    
    # Datasets
    print("\nCreating datasets...")
    
    # SafeTrip dataset with pseudo labels for depth samples
    train_dataset = SafeTripDataset(
        data_root=cfg['dataset']['root'],
        split='train',
        img_size=tuple(cfg['dataset']['img_size']),
        augment=cfg['training']['augment'],
        segmentation_only=False,  # Use both segmentation and depth
        pseudo_labels_path='data/depth_pseudo_labels.pkl'  # Use pseudo labels for depth samples
    )
    print(f"SafeTrip train samples: {len(train_dataset)}")
    
    # Validation dataset
    val_dataset = SafeTripDataset(
        data_root=cfg['dataset']['root'],
        split='val',
        img_size=tuple(cfg['dataset']['img_size']),
        augment=False,
        segmentation_only=False,  # Validate on both tasks
        pseudo_labels_path='data/depth_pseudo_labels.pkl'  # Use pseudo labels for depth samples
    )
    print(f"Validation (segmentation only): {len(val_dataset)} samples")
    
    # Dataloaders
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
    
    # Loss functions
    class_weights = None
    if os.path.exists(cfg['dataset']['class_weights_path']):
        class_weights = torch.from_numpy(
            np.load(cfg['dataset']['class_weights_path'])
        ).float().to(device)
        print("Loaded class weights")
    
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    # Reduced delta for normalized depth values [0, 1]
    depth_criterion = InverseHuberLoss(delta=0.1)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0)
    
    # Load class names
    import json
    with open(cfg['dataset']['class_info_path'], 'r') as f:
        class_info = json.load(f)
    class_names = class_info['class_names']
    
    # Training loop
    print("\nStarting training...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, seg_criterion, depth_criterion,
            optimizer, scaler, device, epoch
        )
        
        # Validate
        val_metrics, per_class_iou = validate(
            model, val_loader, seg_criterion, depth_criterion,
            device, cfg['model']['num_classes']
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Seg: {train_metrics['seg_loss']:.4f}, "
              f"Depth: {train_metrics['depth_loss']:.4f}")
        
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Seg: {val_metrics['seg_loss']:.4f}, "
              f"Depth: {val_metrics['depth_loss']:.4f}")
        
        print(f"\nSegmentation - mIoU: {val_metrics['seg_miou']:.4f}, "
              f"Acc: {val_metrics['seg_acc']:.4f}")
        
        print(f"Depth - RMSE: {val_metrics['depth_rmse']:.4f}, "
              f"MAE: {val_metrics['depth_mae']:.4f}, "
              f"δ<1.25: {val_metrics['depth_delta_1']:.4f}")
        
        # Print per-class IoU (only non-zero)
        print("\nPer-Class IoU:")
        for name, iou in zip(class_names, per_class_iou):
            if iou > 0:
                print(f"  {name}: {iou:.4f}")
        
        # Log to wandb
        log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/seg_loss': train_metrics['seg_loss'],
                'train/depth_loss': train_metrics['depth_loss'],
                'val/loss': val_metrics['loss'],
                'val/seg_loss': val_metrics['seg_loss'],
                'val/depth_loss': val_metrics['depth_loss'],
                'val/seg_miou': val_metrics['seg_miou'],
                'val/seg_acc': val_metrics['seg_acc'],
                'val/depth_rmse': val_metrics['depth_rmse'],
                'val/depth_mae': val_metrics['depth_mae'],
                'val/depth_delta_1': val_metrics['depth_delta_1'],
                'lr': scheduler.get_last_lr()[0]
            }
        wandb.log(log_dict)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
        
        # Save best model
        if val_metrics['seg_miou'] > best_miou:
            best_miou = val_metrics['seg_miou']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            print(f"\n✓ Saved best model (mIoU: {best_miou:.4f})")
    
    print("\nTraining completed!")
    wandb.finish()


if __name__ == '__main__':
    main()