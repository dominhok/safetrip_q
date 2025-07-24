#!/usr/bin/env python
"""
Train teacher model for knowledge distillation.
Supports both SegFormer (Cityscapes pre-trained) and DeepLabV3+ (ImageNet pre-trained).
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_safetrip import SafeTripDataset
from utils.metrics import SegmentationMetrics

# Import transformers for Cityscapes models
try:
    from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using smp only")

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Please install segmentation_models_pytorch: pip install segmentation-models-pytorch")
    sys.exit(1)


def create_cityscapes_model(model_type='segformer', num_classes=24):
    """Create model from Cityscapes pre-trained weights."""
    
    if model_type == 'segformer' and TRANSFORMERS_AVAILABLE:
        # Use SegFormer pre-trained on Cityscapes
        print("Loading SegFormer-B2 pre-trained on Cityscapes...")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Wrap for consistent interface
        class SegFormerWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                outputs = self.model(pixel_values=x)
                # Upsample to input resolution
                logits = outputs.logits
                upsampled = nn.functional.interpolate(
                    logits, 
                    size=x.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                return upsampled
        
        return SegFormerWrapper(model)
    
    else:
        # Use DeepLabV3+ with ImageNet pre-training
        # (Cityscapes pre-trained weights not directly available in smp)
        print("Loading DeepLabV3+ with ImageNet pre-training...")
        print("Note: For better results, consider using SegFormer with Cityscapes weights")
        
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )
        
        return model


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, num_classes):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    seg_metrics = SegmentationMetrics(num_classes)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            masks = batch['segmentation'].to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            num_batches += 1
            
            preds = outputs.argmax(dim=1)
            seg_metrics.update(preds, masks)
    
    metrics = {
        'loss': total_loss / num_batches,
        'miou': seg_metrics.get_miou(),
        'acc': seg_metrics.get_pixel_accuracy()
    }
    
    per_class_iou = seg_metrics.get_iou_per_class()
    
    return metrics, per_class_iou


def main():
    parser = argparse.ArgumentParser(description='Train teacher from Cityscapes')
    parser.add_argument('--config', type=str, default='configs/cfg_safetrip.yaml')
    parser.add_argument('--model', type=str, default='segformer', 
                        choices=['segformer', 'deeplabv3plus'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)  # Lower LR for fine-tuning
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder and only train decoder')
    parser.add_argument('--project', type=str, default='safetrip-teacher-cityscapes')
    parser.add_argument('--run-name', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['training']['batch_size'] = args.batch_size
    
    # Initialize wandb
    run_name = args.run_name or f"{args.model}_cityscapes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.project,
        name=run_name,
        config={
            'model': args.model,
            'pretrained': 'cityscapes',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'num_classes': cfg['model']['num_classes'],
            'freeze_encoder': args.freeze_encoder
        }
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_cityscapes_model(
        model_type=args.model,
        num_classes=cfg['model']['num_classes']
    )
    model = model.to(device)
    
    # Freeze encoder if requested
    if args.freeze_encoder:
        print("Freezing encoder layers...")
        frozen_params = 0
        for name, param in model.named_parameters():
            if 'encoder' in name or 'backbone' in name:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"Frozen {frozen_params:,} parameters")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SafeTripDataset(
        data_root=cfg['dataset']['root'],
        split='train',
        img_size=tuple(cfg['dataset']['img_size']),
        augment=True,
        segmentation_only=True
    )
    
    val_dataset = SafeTripDataset(
        data_root=cfg['dataset']['root'],
        split='val',
        img_size=tuple(cfg['dataset']['img_size']),
        augment=False,
        segmentation_only=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
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
    
    # Loss function
    class_weights = None
    if os.path.exists(cfg['dataset']['class_weights_path']):
        class_weights = torch.from_numpy(
            np.load(cfg['dataset']['class_weights_path'])
        ).float().to(device)
        print("Loaded class weights")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    # Optimizer - different LR for encoder and decoder
    if args.freeze_encoder:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.0001
        )
    else:
        # Different learning rates for encoder and decoder
        encoder_params = []
        decoder_params = []
        
        for name, param in model.named_parameters():
            if 'encoder' in name or 'backbone' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        optimizer = AdamW([
            {'params': encoder_params, 'lr': args.lr * 0.1},  # Lower LR for encoder
            {'params': decoder_params, 'lr': args.lr}
        ], weight_decay=0.0001)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # Load class names
    import json
    with open(cfg['dataset']['class_info_path'], 'r') as f:
        class_info = json.load(f)
    class_names = class_info['class_names']
    
    os.makedirs('checkpoints/teacher', exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    best_miou = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        val_metrics, per_class_iou = validate(
            model, val_loader, criterion, device, cfg['model']['num_classes']
        )
        
        scheduler.step()
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val mIoU: {val_metrics['miou']:.4f}")
        print(f"Val Acc: {val_metrics['acc']:.4f}")
        
        # Print top 5 classes
        print("\nTop 5 Classes by IoU:")
        class_ious = [(name, iou) for name, iou in zip(class_names, per_class_iou) if iou > 0]
        class_ious.sort(key=lambda x: x[1], reverse=True)
        for name, iou in class_ious[:5]:
            print(f"  {name}: {iou:.4f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'val/loss': val_metrics['loss'],
            'val/miou': val_metrics['miou'],
            'val/acc': val_metrics['acc'],
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save checkpoint
        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'val_metrics': val_metrics,
                'config': {
                    'model': args.model,
                    'num_classes': cfg['model']['num_classes'],
                    'pretrained': 'cityscapes'
                }
            }
            torch.save(checkpoint, 'checkpoints/teacher/best_cityscapes_teacher.pth')
            print(f"\n✓ Saved best model (mIoU: {best_miou:.4f})")
    
    print("\nTraining completed!")
    print(f"Best mIoU: {best_miou:.4f}")
    wandb.finish()


if __name__ == '__main__':
    main()