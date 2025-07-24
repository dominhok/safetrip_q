#!/usr/bin/env python
"""
Check if model weights are properly loaded.
"""
import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import SegformerForSemanticSegmentation
import torch.nn as nn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/teacher/best_cityscapes_teacher.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print("="*60)
    print("CHECKPOINT ANALYSIS")
    print("="*60)
    
    # Check checkpoint contents
    print("\nCheckpoint keys:", checkpoint.keys())
    print(f"Model type: {checkpoint['config']['model']}")
    print(f"Number of classes: {checkpoint['config']['num_classes']}")
    print(f"Best mIoU: {checkpoint['best_miou']:.4f}")
    
    # Check model state dict
    state_dict = checkpoint['model_state_dict']
    print(f"\nTotal parameters in checkpoint: {len(state_dict)}")
    
    # Show some key layers
    print("\nSome key layers in checkpoint:")
    for i, (key, value) in enumerate(state_dict.items()):
        if i < 5 or 'classifier' in key or 'decode_head' in key:
            print(f"  {key}: {value.shape}")
    
    # Create fresh SegFormer model
    print("\n" + "="*60)
    print("CREATING FRESH MODEL")
    print("="*60)
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_labels=24,
        ignore_mismatched_sizes=True
    )
    
    # Check model structure
    print("\nModel structure (decode head):")
    print(model.decode_head)
    
    # Try to load weights
    print("\n" + "="*60)
    print("LOADING WEIGHTS")
    print("="*60)
    
    # Check what's missing/unexpected
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"\nMissing keys: {len(missing_keys)}")
    if missing_keys:
        print("First 10 missing keys:")
        for key in missing_keys[:10]:
            print(f"  - {key}")
    
    print(f"\nUnexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print("First 10 unexpected keys:")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")
    
    # Check if classifier weights were loaded
    print("\n" + "="*60)
    print("CHECKING CLASSIFIER WEIGHTS")
    print("="*60)
    
    classifier_weight = model.decode_head.classifier.weight
    print(f"Classifier weight shape: {classifier_weight.shape}")
    print(f"Classifier weight mean: {classifier_weight.mean().item():.6f}")
    print(f"Classifier weight std: {classifier_weight.std().item():.6f}")
    print(f"Classifier weight min: {classifier_weight.min().item():.6f}")
    print(f"Classifier weight max: {classifier_weight.max().item():.6f}")
    
    # Check if weights are initialized (not trained)
    if classifier_weight.std().item() < 0.01:
        print("\n⚠️  WARNING: Classifier weights appear to be randomly initialized!")
        print("The model is NOT using the trained weights!")
    
    # Test inference
    print("\n" + "="*60)
    print("TESTING INFERENCE")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    # Create random input
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)
        logits = outputs.logits
        
    print(f"Output shape: {logits.shape}")
    print(f"Output mean: {logits.mean().item():.6f}")
    print(f"Output std: {logits.std().item():.6f}")
    
    # Check class predictions
    probs = torch.softmax(logits, dim=1)
    max_probs = probs.max(dim=1)[0]
    print(f"Max probability mean: {max_probs.mean().item():.6f}")
    
    if max_probs.mean().item() < 0.1:
        print("\n⚠️  WARNING: Model outputs are nearly uniform!")
        print("This suggests the model is not properly initialized.")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if checkpoint['best_miou'] > 0.7:
        print("✓ Checkpoint shows good training performance (mIoU > 0.7)")
    
    if len(missing_keys) > 0:
        print("❌ Many keys are missing when loading the model")
        print("   This means the checkpoint and model architecture don't match!")
    
    print("\nPROBLEM: The SegFormer model architecture doesn't match the checkpoint!")
    print("The checkpoint was saved with a different model wrapper/structure.")


if __name__ == '__main__':
    main()