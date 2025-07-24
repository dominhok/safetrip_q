#!/usr/bin/env python
"""
Generate pseudo segmentation labels for SafeTrip Depth samples using trained teacher model.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import pickle
from datetime import datetime
import cv2
from PIL import Image
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DepthOnlyDataset(Dataset):
    """Dataset for SafeTrip Depth-only samples."""
    
    def __init__(self, data_root, img_size=(512, 512)):
        self.data_root = data_root
        self.img_size = img_size
        self.samples = []
        
        # Image normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Collect all depth samples
        depth_dir = os.path.join(data_root, 'Depth')
        if os.path.exists(depth_dir):
            print(f"Scanning Depth directory: {depth_dir}")
            
            # Get all Depth_XXX folders
            depth_folders = sorted([d for d in os.listdir(depth_dir) 
                                  if d.startswith('Depth_') and 
                                  os.path.isdir(os.path.join(depth_dir, d))])
            
            for folder in tqdm(depth_folders, desc="Collecting Depth samples"):
                folder_path = os.path.join(depth_dir, folder)
                
                # Find RGB images (left images)
                # Priority: *_L.png > *_left.png
                left_images = glob.glob(os.path.join(folder_path, '*_L.png'))
                if not left_images:
                    left_images = glob.glob(os.path.join(folder_path, '*_left.png'))
                
                for img_path in left_images:
                    self.samples.append({
                        'image_path': img_path,
                        'folder': folder
                    })
        
        print(f"Found {len(self.samples)} depth-only samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load image
            img = cv2.imread(sample['image_path'])
            if img is None:
                print(f"\nWarning: Failed to load image: {sample['image_path']}")
                # Return a placeholder image
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), 
                            interpolation=cv2.INTER_LINEAR)
            
            # Convert to tensor and normalize
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = self.normalize(img)
            
            return {
                'image': img,
                'path': sample['image_path'],
                'valid': img is not None
            }
        except Exception as e:
            print(f"\nError loading {sample['image_path']}: {e}")
            # Return placeholder
            img = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            return {
                'image': img,
                'path': sample['image_path'],
                'valid': False
            }


def generate_pseudo_labels(model, dataloader, device, confidence_threshold=0.7):
    """Generate pseudo labels for depth-only samples."""
    model.eval()
    pseudo_labels = {}
    skipped_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating pseudo labels'):
            images = batch['image'].to(device)
            paths = batch['path']
            valid_flags = batch.get('valid', [True] * len(paths))
            
            # Get model predictions
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Get pseudo labels and confidence
            confidence, labels = torch.max(probs, dim=1)
            
            # Process each sample
            for i in range(len(paths)):
                # Skip invalid images
                if not valid_flags[i]:
                    skipped_count += 1
                    continue
                    
                path = paths[i]
                
                # Store pseudo label and confidence
                label = labels[i].cpu().numpy()
                conf = confidence[i].cpu().numpy()
                
                # Create mask for high-confidence predictions
                high_conf_mask = conf > confidence_threshold
                
                # Store results
                pseudo_labels[path] = {
                    'pseudo_label': label,
                    'confidence': conf,
                    'high_conf_mask': high_conf_mask,
                    'mean_confidence': float(conf.mean()),
                    'high_conf_ratio': float(high_conf_mask.mean())
                }
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} invalid images")
    
    return pseudo_labels


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo labels for SafeTrip Depth samples')
    parser.add_argument('--teacher-checkpoint', type=str, required=True,
                        help='Path to trained teacher model checkpoint')
    parser.add_argument('--data-root', type=str, default='../data',
                        help='Path to SafeTrip data')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Minimum confidence for pseudo labels')
    parser.add_argument('--output-path', type=str, default='data/depth_pseudo_labels.pkl',
                        help='Path to save pseudo labels')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load teacher checkpoint
    print(f"\nLoading teacher model from {args.teacher_checkpoint}")
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    
    # Check if checkpoint needs fixing
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    if state_dict_keys[0].startswith('model.'):
        print("Fixing checkpoint keys (removing 'model.' prefix)...")
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            new_key = key[6:] if key.startswith('model.') else key
            new_state_dict[new_key] = value
        checkpoint['model_state_dict'] = new_state_dict
    
    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})
    model_type = model_config.get('model', 'segformer')
    num_classes = model_config.get('num_classes', 24)
    
    print(f"Teacher model: {model_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Best mIoU: {checkpoint.get('best_miou', 'N/A'):.4f}")
    
    # Create model
    if model_type == 'segformer' and TRANSFORMERS_AVAILABLE:
        # Create SegFormer model
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Wrap for consistent interface
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
        
        model = SegFormerWrapper(model)
    else:
        print("SegFormer not available, using checkpoint model type")
        import segmentation_models_pytorch as smp
        encoder_name = model_config.get('encoder', 'resnet101')
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Create dataset
    print(f"\nLoading Depth-only samples from {args.data_root}")
    dataset = DepthOnlyDataset(
        data_root=args.data_root,
        img_size=tuple(args.img_size)
    )
    
    if len(dataset) == 0:
        print("No depth samples found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Generate pseudo labels
    print("\nGenerating pseudo labels...")
    pseudo_labels = generate_pseudo_labels(
        model, dataloader, device, args.confidence_threshold
    )
    
    # Print statistics
    if pseudo_labels:
        confidences = [v['mean_confidence'] for v in pseudo_labels.values()]
        high_conf_ratios = [v['high_conf_ratio'] for v in pseudo_labels.values()]
        
        print(f"\nStatistics:")
        print(f"  Total samples: {len(pseudo_labels)}")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Min confidence: {np.min(confidences):.3f}")
        print(f"  Max confidence: {np.max(confidences):.3f}")
        print(f"  Average high-confidence ratio: {np.mean(high_conf_ratios):.3f}")
        
        # Show confidence distribution
        conf_bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(confidences, bins=conf_bins)
        print("\nConfidence distribution:")
        for i in range(len(conf_bins)-1):
            print(f"  [{conf_bins[i]:.1f}, {conf_bins[i+1]:.1f}): {hist[i]} samples")
    
    # Save pseudo labels
    print(f"\nSaving pseudo labels to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'wb') as f:
        pickle.dump({
            'pseudo_labels': pseudo_labels,
            'metadata': {
                'teacher_checkpoint': args.teacher_checkpoint,
                'confidence_threshold': args.confidence_threshold,
                'img_size': args.img_size,
                'num_classes': num_classes,
                'timestamp': datetime.now().isoformat()
            }
        }, f)
    
    print(f"✓ Saved {len(pseudo_labels)} pseudo labels")
    print("\nDone!")


if __name__ == '__main__':
    main()