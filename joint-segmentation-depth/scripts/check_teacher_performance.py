#!/usr/bin/env python
"""Check teacher model performance from checkpoint."""
import os
import sys
import torch
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    checkpoint_path = 'checkpoints/teacher/best_cityscapes_teacher.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "="*60)
    print("TEACHER MODEL PERFORMANCE")
    print("="*60)
    
    # Model info
    config = checkpoint.get('config', {})
    print(f"\nModel Information:")
    print(f"  Model Type: {config.get('model', 'N/A')}")
    print(f"  Pre-trained: {config.get('pretrained', 'N/A')}")
    print(f"  Number of Classes: {config.get('num_classes', 'N/A')}")
    
    # Training info
    print(f"\nTraining Information:")
    print(f"  Best Epoch: {checkpoint.get('epoch', 'N/A') + 1}")
    
    # Performance metrics
    val_metrics = checkpoint.get('val_metrics', {})
    best_miou = checkpoint.get('best_miou', val_metrics.get('miou', 'N/A'))
    
    print(f"\nValidation Performance:")
    print(f"  mIoU: {best_miou:.4f}" if isinstance(best_miou, float) else f"  mIoU: {best_miou}")
    print(f"  Pixel Accuracy: {val_metrics.get('acc', 'N/A'):.4f}" if 'acc' in val_metrics else "  Pixel Accuracy: N/A")
    print(f"  Loss: {val_metrics.get('loss', 'N/A'):.4f}" if 'loss' in val_metrics else "  Loss: N/A")
    
    # Load class names
    try:
        with open('data/class_info.json', 'r') as f:
            class_info = json.load(f)
        class_names = class_info['class_names']
        
        # Per-class IoU if available
        per_class_iou = checkpoint.get('per_class_iou', None)
        if per_class_iou is not None:
            print(f"\nPer-Class IoU (Top 10):")
            class_ious = [(name, iou) for name, iou in zip(class_names, per_class_iou) if iou > 0]
            class_ious.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, iou) in enumerate(class_ious[:10]):
                print(f"  {i+1:2d}. {name:20s}: {iou:.4f}")
                
            # Classes with low performance
            print(f"\nClasses with Low IoU (<0.1):")
            low_perf = [name for name, iou in class_ious if iou < 0.1]
            if low_perf:
                print(f"  {', '.join(low_perf)}")
            else:
                print("  None")
    except Exception as e:
        print(f"\nCould not load class information: {e}")
    
    # File size
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"\nCheckpoint Size: {file_size:.1f} MB")
    
    print("\n" + "="*60)
    
    # Recommendations
    print("\nRecommendations:")
    if isinstance(best_miou, float):
        if best_miou > 0.5:
            print("✓ Good performance! Ready for pseudo-label generation.")
        elif best_miou > 0.3:
            print("⚡ Decent performance. Consider more training epochs.")
        else:
            print("⚠️  Low performance. Check data or training settings.")
    
    print("\nNext Steps:")
    print("1. Generate pseudo labels for KITTI:")
    print("   python scripts/generate_pseudo_labels.py \\")
    print("     --teacher-checkpoint checkpoints/teacher/best_cityscapes_teacher.pth \\")
    print("     --output-path data/kitti_pseudo_labels.pkl")
    print("\n2. Train Hydranet with distillation:")
    print("   python train.py --config configs/cfg_safetrip.yaml")

if __name__ == '__main__':
    main()