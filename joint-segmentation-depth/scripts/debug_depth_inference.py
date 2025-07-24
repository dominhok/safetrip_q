#!/usr/bin/env python
"""
Debug inference on SafeTrip Depth images to understand low confidence issue.
"""
import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
import json


def load_and_preprocess_image(img_path, img_size=(512, 512)):
    """Load and preprocess image exactly as in training."""
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    
    # Resize
    img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and normalize
    img_tensor = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)
    
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_normalized = normalize(img_tensor)
    
    return original_img, img, img_tensor, img_normalized


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
    
    # Load checkpoint weights
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
    
    # Test on Depth images only
    test_images = [
        '../data/Depth/Depth_001/ZED1_KSC_001032_L.png',
        '../data/Depth/Depth_001/ZED1_KSC_001033_L.png',
        '../data/Depth/Depth_002/ZED1_KSC_001251_L.png',
        '../data/Depth/Depth_002/ZED1_KSC_001533_L.png',
    ]
    
    # Filter existing files
    test_images = [img for img in test_images if os.path.exists(img)]
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"\nTesting on {len(test_images)} images...")
    
    # Load class names
    try:
        with open('../data/class_info.json', 'r') as f:
            class_info = json.load(f)
        class_names = class_info['class_names']
    except:
        # Fallback class names
        class_names = [f"Class_{i}" for i in range(24)]
    
    # Create visualization
    fig, axes = plt.subplots(len(test_images), 4, figsize=(16, 4*len(test_images)))
    if len(test_images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(test_images):
        print(f"\n{'='*60}")
        print(f"Testing: {img_path}")
        print(f"Image type: {'Depth' if 'Depth' in img_path else 'Surface' if 'Surface' in img_path else 'Polygon'}")
        
        try:
            # Load and preprocess
            original_img, resized_img, img_tensor, img_normalized = load_and_preprocess_image(img_path)
            
            print(f"Original shape: {original_img.shape}")
            print(f"Resized shape: {resized_img.shape}")
            print(f"Tensor range before norm: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
            print(f"Tensor range after norm: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
            
            # Get predictions
            with torch.no_grad():
                input_batch = img_normalized.unsqueeze(0).to(device)
                outputs = model(input_batch)
                probs = torch.softmax(outputs, dim=1)
                confidence, labels = torch.max(probs, dim=1)
            
            # Analyze results
            conf_map = confidence[0].cpu().numpy()
            label_map = labels[0].cpu().numpy()
            
            print(f"\nPrediction statistics:")
            print(f"  Confidence range: [{conf_map.min():.3f}, {conf_map.max():.3f}]")
            print(f"  Mean confidence: {conf_map.mean():.3f}")
            
            # Class distribution
            unique_labels, counts = np.unique(label_map, return_counts=True)
            print(f"  Predicted classes: {len(unique_labels)}")
            print(f"  Top 3 classes:")
            sorted_idx = np.argsort(counts)[::-1][:3]
            for i in sorted_idx:
                class_id = unique_labels[i]
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                percentage = counts[i] / label_map.size * 100
                print(f"    {class_name}: {percentage:.1f}%")
            
            # Visualize
            axes[idx, 0].imshow(resized_img)
            axes[idx, 0].set_title(f"{os.path.basename(img_path)[:20]}...")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(label_map)
            axes[idx, 1].set_title(f"Predictions")
            axes[idx, 1].axis('off')
            
            conf_im = axes[idx, 2].imshow(conf_map, cmap='jet', vmin=0, vmax=1)
            axes[idx, 2].set_title(f"Confidence (mean: {conf_map.mean():.3f})")
            axes[idx, 2].axis('off')
            
            # Show logits distribution
            logits_std = outputs[0].std(dim=0).cpu().numpy()
            axes[idx, 3].imshow(logits_std, cmap='viridis')
            axes[idx, 3].set_title(f"Logits Std (mean: {logits_std.mean():.3f})")
            axes[idx, 3].axis('off')
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            for j in range(4):
                axes[idx, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_depth_inference.png', dpi=150)
    print(f"\n✓ Saved visualization to debug_depth_inference.png")
    
    # Additional diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    if conf_map.mean() < 0.1:
        print("⚠️  Extremely low confidence detected!")
        print("\nPossible causes:")
        print("1. Domain gap: Depth images look very different from Surface/Polygon images")
        print("2. Model initialization: The model might not be properly loaded")
        print("3. Input preprocessing: Check if Depth images need different preprocessing")
        print("\nRecommendations:")
        print("1. Use Surface/Polygon images only (they already have segmentation)")
        print("2. Fine-tune teacher model on a few Depth images with manual labels")
        print("3. Use a pre-trained model on a more similar domain")


if __name__ == '__main__':
    main()