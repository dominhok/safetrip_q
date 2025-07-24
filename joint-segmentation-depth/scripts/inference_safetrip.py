import torch
import numpy as np
import cv2
import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import json
from torchvision import transforms

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.hydranet_safetrip import HydranetSafeTrip
from utils.config import load_config


def load_model(checkpoint_path, cfg):
    """Load the trained model from checkpoint."""
    device = torch.device(cfg['model']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = HydranetSafeTrip(cfg).to(device)
    
    # Load checkpoint
    print(f"Loading model from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    return model, device


def preprocess_image(image_path, cfg):
    """Preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize to model input size
    img_size = tuple(cfg['dataset']['img_size'])
    image = image.resize((img_size[1], img_size[0]), Image.BILINEAR)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg['preprocessing']['img_mean'],
            std=cfg['preprocessing']['img_std']
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, original_size


def postprocess_outputs(pred_seg, pred_depth, original_size, cfg):
    """Postprocess model outputs."""
    # Get segmentation prediction
    seg_pred = pred_seg.argmax(dim=1).squeeze().cpu().numpy()
    
    # Get depth prediction
    depth_pred = pred_depth.squeeze().cpu().numpy()
    
    # Resize back to original size
    seg_pred = cv2.resize(seg_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    depth_pred = cv2.resize(depth_pred, original_size, interpolation=cv2.INTER_LINEAR)
    
    return seg_pred, depth_pred


def visualize_results(image_path, seg_pred, depth_pred, class_names, colormap, save_path=None):
    """Visualize segmentation and depth results."""
    # Load original image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Segmentation prediction
    seg_colored = colormap[seg_pred]
    axes[0, 1].imshow(seg_colored)
    axes[0, 1].set_title('Segmentation Prediction')
    axes[0, 1].axis('off')
    
    # Segmentation overlay
    overlay = original_img.copy()
    mask = seg_pred > 0  # Non-background pixels
    overlay[mask] = overlay[mask] * 0.5 + seg_colored[mask] * 0.5
    axes[1, 0].imshow(overlay.astype(np.uint8))
    axes[1, 0].set_title('Segmentation Overlay')
    axes[1, 0].axis('off')
    
    # Depth prediction
    depth_viz = axes[1, 1].imshow(depth_pred, cmap='plasma')
    axes[1, 1].set_title('Depth Prediction')
    axes[1, 1].axis('off')
    plt.colorbar(depth_viz, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()
    
    # Print class statistics
    print("\nDetected classes:")
    unique_classes, counts = np.unique(seg_pred, return_counts=True)
    total_pixels = seg_pred.size
    for cls_id, count in zip(unique_classes, counts):
        if cls_id < len(class_names):
            percentage = (count / total_pixels) * 100
            print(f"  {class_names[cls_id]}: {percentage:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Run inference on SafeTrip-Q dataset')
    parser.add_argument('--checkpoint', type=str, 
                        default='../checkpoints/safetrip/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='config/cfg_safetrip.yaml',
                        help='Path to configuration file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='../outputs/inference',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Load class information
    with open(cfg['dataset']['class_info_path'], 'r') as f:
        class_info = json.load(f)
    class_names = class_info['class_names']
    
    # Load colormap
    colormap = np.load(cfg['general']['cmap'])
    
    # Load model
    model, device = load_model(args.checkpoint, cfg)
    
    # Preprocess image
    print(f"\nProcessing image: {args.image}")
    image_tensor, original_size = preprocess_image(args.image, cfg)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        pred_seg, pred_depth = model(image_tensor)
    
    # Postprocess outputs
    seg_pred, depth_pred = postprocess_outputs(pred_seg, pred_depth, original_size, cfg)
    
    # Save raw predictions
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    np.save(os.path.join(args.output_dir, f'{image_name}_seg.npy'), seg_pred)
    np.save(os.path.join(args.output_dir, f'{image_name}_depth.npy'), depth_pred)
    
    # Visualize results
    viz_path = os.path.join(args.output_dir, f'{image_name}_results.png')
    visualize_results(args.image, seg_pred, depth_pred, class_names, colormap, viz_path)
    
    # Print depth statistics
    print(f"\nDepth statistics:")
    print(f"  Min depth: {depth_pred.min():.2f}m")
    print(f"  Max depth: {depth_pred.max():.2f}m")
    print(f"  Mean depth: {depth_pred.mean():.2f}m")
    print(f"  Std depth: {depth_pred.std():.2f}m")


if __name__ == "__main__":
    main()