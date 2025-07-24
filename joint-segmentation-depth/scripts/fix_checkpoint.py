#!/usr/bin/env python
"""
Fix checkpoint key names to match model architecture.
"""
import torch
import os


def fix_checkpoint(checkpoint_path, output_path):
    """Fix checkpoint keys by removing 'model.' prefix."""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get old state dict
    old_state_dict = checkpoint['model_state_dict']
    
    # Create new state dict with fixed keys
    new_state_dict = {}
    for key, value in old_state_dict.items():
        # Remove 'model.' prefix if it exists
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.'
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Update checkpoint
    checkpoint['model_state_dict'] = new_state_dict
    
    # Save fixed checkpoint
    torch.save(checkpoint, output_path)
    print(f"Saved fixed checkpoint to {output_path}")
    
    # Show some examples
    print("\nExample key mappings:")
    for i, (old_key, new_key) in enumerate(zip(list(old_state_dict.keys())[:5], 
                                               list(new_state_dict.keys())[:5])):
        print(f"  {old_key} -> {new_key}")


if __name__ == '__main__':
    fix_checkpoint(
        'checkpoints/teacher/best_cityscapes_teacher.pth',
        'checkpoints/teacher/best_cityscapes_teacher_fixed.pth'
    )