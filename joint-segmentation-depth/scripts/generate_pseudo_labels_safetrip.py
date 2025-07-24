#!/usr/bin/env python
"""
Generate pseudo segmentation labels for SafeTrip depth samples using trained teacher model.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_safetrip import SafeTripDataset

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Please install segmentation_models_pytorch: pip install segmentation-models-pytorch")
    sys.exit(1)

try:
    from transformers import SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def generate_pseudo_labels(model, dataloader, device, confidence_threshold=0.7):
    """
    Generate pseudo labels for SafeTrip depth samples.
    
    Returns:
        dict: Mapping from sample path to (pseudo_label, confidence)
    """
    model.eval()
    pseudo_labels = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Generating pseudo labels')):
            # Skip if already has segmentation
            if batch['has_segmentation'].all():
                continue
                
            images = batch['image'].to(device)
            
            # Get model predictions
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Get pseudo labels and confidence
            confidence, labels = torch.max(probs, dim=1)
            
            # Process each sample in batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Only process samples without segmentation (i.e., depth-only samples)
                if batch['has_segmentation'][i]:
                    continue
                    
                # Get sample index
                sample_idx = batch_idx * dataloader.batch_size + i
                if sample_idx >= len(dataloader.dataset):
                    break
                    
                # Get sample info
                sample_info = dataloader.dataset.samples[sample_idx]
                
                # Create sample key based on path
                if 'depth_path' in sample_info:
                    sample_key = sample_info['depth_path']
                elif 'image_path' in sample_info:
                    sample_key = sample_info['image_path']
                else:
                    sample_key = f"sample_{sample_idx}"
                
                # Store pseudo label and confidence
                label = labels[i].cpu().numpy()
                conf = confidence[i].cpu().numpy()
                
                # Create mask for high-confidence predictions
                high_conf_mask = conf > confidence_threshold
                
                # Store results
                pseudo_labels[sample_key] = {
                    'pseudo_label': label,
                    'confidence': conf,
                    'high_conf_mask': high_conf_mask,
                    'mean_confidence': float(conf.mean()),
                    'high_conf_ratio': float(high_conf_mask.mean())
                }
    
    return pseudo_labels


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo labels for SafeTrip depth samples')
    parser.add_argument('--teacher-checkpoint', type=str, required=True,
                        help='Path to trained teacher model checkpoint')
    parser.add_argument('--data-root', type=str, default='../data',
                        help='Path to SafeTrip data')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Minimum confidence for pseudo labels')
    parser.add_argument('--output-path', type=str, default='data/safetrip_pseudo_labels.pkl',
                        help='Path to save pseudo labels')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load teacher checkpoint
    print(f"Loading teacher model from {args.teacher_checkpoint}")
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    
    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})
    model_type = model_config.get('model', 'deeplabv3plus')
    num_classes = model_config.get('num_classes', 24)
    
    print(f"Teacher model: {model_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Best mIoU: {checkpoint.get('best_miou', 'N/A'):.4f}")
    
    # Create model based on type
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
        # DeepLabV3+
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
    
    # Create SafeTrip dataset - load ALL samples to find depth-only ones
    print(f"\nLoading SafeTrip dataset from {args.data_root}")
    
    # Process both train and val splits
    all_pseudo_labels = {}
    
    for split in ['train', 'val']:
        print(f"\nProcessing {split} split...")
        
        # Load dataset without segmentation_only flag to get all samples
        dataset = SafeTripDataset(
            data_root=args.data_root,
            split=split,
            img_size=tuple(args.img_size),
            augment=False,
            segmentation_only=False  # Get all samples including depth-only
        )
        
        print(f"Found {len(dataset)} total samples")
        
        # Count depth-only samples
        depth_only_count = 0
        for i in range(min(100, len(dataset))):  # Check first 100 samples
            sample = dataset[i]
            if sample['has_depth'] and not sample['has_segmentation']:
                depth_only_count += 1
        
        estimated_depth_only = int(depth_only_count * len(dataset) / min(100, len(dataset)))
        print(f"Estimated depth-only samples: ~{estimated_depth_only}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Generate pseudo labels
        pseudo_labels = generate_pseudo_labels(
            model, dataloader, device, args.confidence_threshold
        )
        
        print(f"Generated {len(pseudo_labels)} pseudo labels for depth-only samples")
        
        # Add to all labels with split prefix
        for key, value in pseudo_labels.items():
            all_pseudo_labels[f"{split}/{key}"] = value
        
        # Print statistics
        if pseudo_labels:
            confidences = [v['mean_confidence'] for v in pseudo_labels.values()]
            high_conf_ratios = [v['high_conf_ratio'] for v in pseudo_labels.values()]
            
            print(f"\n{split} statistics:")
            print(f"  Average confidence: {np.mean(confidences):.3f}")
            print(f"  Min confidence: {np.min(confidences):.3f}")
            print(f"  Max confidence: {np.max(confidences):.3f}")
            print(f"  Average high-confidence ratio: {np.mean(high_conf_ratios):.3f}")
    
    # Save pseudo labels
    if all_pseudo_labels:
        print(f"\nSaving pseudo labels to {args.output_path}")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        with open(args.output_path, 'wb') as f:
            pickle.dump({
                'pseudo_labels': all_pseudo_labels,
                'metadata': {
                    'teacher_checkpoint': args.teacher_checkpoint,
                    'confidence_threshold': args.confidence_threshold,
                    'img_size': args.img_size,
                    'num_classes': num_classes,
                    'timestamp': datetime.now().isoformat()
                }
            }, f)
        
        print(f"✓ Saved {len(all_pseudo_labels)} pseudo labels")
    else:
        print("\n⚠️  No depth-only samples found in SafeTrip dataset")
        print("All samples already have segmentation labels")
    
    print("\nDone!")


if __name__ == '__main__':
    main()