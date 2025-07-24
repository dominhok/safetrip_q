#!/usr/bin/env python
"""
Debug pseudo label generation to understand low confidence issue.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_kitti import KITTIDepthDataset
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load teacher model
    checkpoint_path = 'checkpoints/teacher/best_cityscapes_teacher.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Model type: {checkpoint['config']['model']}")
    print(f"Number of classes: {checkpoint['config']['num_classes']}")
    print(f"Best mIoU: {checkpoint['best_miou']:.4f}")
    
    # Create model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_labels=24,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Wrap model
    class SegFormerWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            outputs = self.model(pixel_values=x)
            logits = outputs.logits
            upsampled = nn.functional.interpolate(
                logits, 
                size=x.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            return upsampled
    
    model = SegFormerWrapper(model).to(device)
    model.eval()
    
    # Load one KITTI sample
    dataset = KITTIDepthDataset(
        root_dir='../data/data_depth_annotated',
        split='train',
        img_size=(512, 512),
        preprocess_depth=False
    )
    
    # Get first sample
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)
    
    print(f"\nImage shape: {image.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, labels = torch.max(probs, dim=1)
    
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"Mean confidence: {confidence.mean():.3f}")
    
    # Check class distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"\nPredicted classes: {unique_labels.cpu().numpy()}")
    print(f"Class counts: {counts.cpu().numpy()}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    img_vis = image[0].cpu()
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = img_vis * std + mean
    img_vis = img_vis.clamp(0, 1)
    
    axes[0].imshow(img_vis.permute(1, 2, 0))
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(labels[0].cpu().numpy())
    axes[1].set_title(f'Predictions (classes: {unique_labels.cpu().numpy()})')
    axes[1].axis('off')
    
    # Confidence
    conf_vis = axes[2].imshow(confidence[0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes[2].set_title(f'Confidence (mean: {confidence.mean():.3f})')
    axes[2].axis('off')
    plt.colorbar(conf_vis, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('debug_pseudo_labels.png')
    print("\nSaved visualization to debug_pseudo_labels.png")
    
    # Check if model outputs are reasonable
    print("\n=== Diagnosis ===")
    if confidence.mean() < 0.1:
        print("❌ Very low confidence - model might be:")
        print("   1. Not properly loaded")
        print("   2. Input normalization mismatch") 
        print("   3. KITTI images too different from training data")
        
        # Check if outputs are uniform
        if outputs.std() < 0.1:
            print("   4. Model outputs are nearly uniform - likely initialization issue")


if __name__ == '__main__':
    main()