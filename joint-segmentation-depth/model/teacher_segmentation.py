"""
DeepLabV3+ teacher model for knowledge distillation.
"""
import torch
import torch.nn as nn
from typing import Optional
import warnings

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("Please install segmentation_models_pytorch: pip install segmentation-models-pytorch")


class DeepLabV3PlusTeacher(nn.Module):
    """DeepLabV3+ model for generating pseudo labels."""
    
    def __init__(self, 
                 num_classes: int = 24,
                 encoder_name: str = "resnet101",
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        
        # Create DeepLabV3+ model
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
            activation=None,  # Raw logits
            encoder_output_stride=16,
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Forward pass returning logits."""
        return self.model(x)
    
    def get_pseudo_labels(self, x, temperature=1.0, return_soft=False):
        """
        Generate pseudo labels with optional temperature scaling.
        
        Args:
            x: Input images (B, 3, H, W)
            temperature: Temperature for softmax (higher = softer labels)
            return_soft: If True, also return soft labels
            
        Returns:
            pseudo_labels: Hard labels (B, H, W)
            confidence: Confidence scores (B, H, W)
            soft_labels: Soft labels (B, num_classes, H, W) - only if return_soft=True
        """
        with torch.no_grad():
            logits = self.forward(x)
            
            # Apply temperature scaling
            logits_scaled = logits / temperature
            
            # Get soft labels (probabilities)
            soft_labels = torch.softmax(logits_scaled, dim=1)
            
            # Get hard labels and confidence
            confidence, pseudo_labels = torch.max(soft_labels, dim=1)
        
        if return_soft:
            return pseudo_labels, confidence, soft_labels
        else:
            return pseudo_labels, confidence


def load_teacher_model(checkpoint_path: str, 
                      device: str = 'cuda',
                      eval_mode: bool = True) -> DeepLabV3PlusTeacher:
    """
    Load a trained teacher model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        eval_mode: If True, set model to eval mode
        
    Returns:
        Loaded teacher model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})
    encoder_name = model_config.get('encoder', 'resnet101')
    num_classes = model_config.get('num_classes', 24)
    
    print(f"Loading DeepLabV3+ with {encoder_name} encoder, {num_classes} classes")
    
    # Create model
    model = DeepLabV3PlusTeacher(
        num_classes=num_classes,
        encoder_name=encoder_name,
        pretrained=False  # Will load weights from checkpoint
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    if eval_mode:
        model.eval()
    
    # Print checkpoint info if available
    if 'epoch' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    if 'best_miou' in checkpoint:
        print(f"Best mIoU: {checkpoint['best_miou']:.4f}")
    
    return model


# Quick test
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test model creation
    print("Creating DeepLabV3+ teacher model...")
    teacher = DeepLabV3PlusTeacher(
        num_classes=24,
        encoder_name="resnet101",
        pretrained=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in teacher.parameters())
    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test inference
    x = torch.randn(2, 3, 512, 512).to(device)
    
    with torch.no_grad():
        # Test forward pass
        logits = teacher(x)
        print(f"\nLogits shape: {logits.shape}")
        
        # Test pseudo label generation
        labels, confidence = teacher.get_pseudo_labels(x)
        print(f"Labels shape: {labels.shape}")
        print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
        
        # Test with soft labels
        labels, confidence, soft = teacher.get_pseudo_labels(x, return_soft=True)
        print(f"Soft labels shape: {soft.shape}")
    
    print("\n✓ All tests passed!")