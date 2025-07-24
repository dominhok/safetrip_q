import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import cv2
from PIL import Image
import json

from model.hydranet import Hydranet
from dataset_safetrip import SafeTripDataset
from utils.metrics import SegmentationMetrics, DepthMetrics


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(cfg, checkpoint_path):
    """Create and load trained model."""
    model = Hydranet(cfg)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=cfg['model']['device'])
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model


def visualize_predictions(image, pred_seg, pred_depth, gt_seg, gt_depth, 
                         class_names, colormap, save_path=None):
    """
    Visualize predictions and ground truth.
    
    Args:
        image: Input RGB image (H, W, 3)
        pred_seg: Predicted segmentation (H, W)
        pred_depth: Predicted depth (H, W)
        gt_seg: Ground truth segmentation (H, W)
        gt_depth: Ground truth depth (H, W)
        class_names: List of class names
        colormap: Colormap for segmentation
        save_path: Path to save visualization
    """
    fig_height = 10
    fig_width = 15
    
    # Create figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height))
    
    # Input image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Predicted segmentation
    pred_seg_color = colormap[pred_seg]
    axes[0, 1].imshow(pred_seg_color)
    axes[0, 1].set_title('Predicted Segmentation')
    axes[0, 1].axis('off')
    
    # Predicted depth
    axes[0, 2].imshow(pred_depth, cmap='viridis')
    axes[0, 2].set_title('Predicted Depth')
    axes[0, 2].axis('off')
    
    # Ground truth segmentation
    gt_seg_color = np.zeros_like(image)
    valid_mask = gt_seg != 255
    gt_seg_color[valid_mask] = colormap[gt_seg[valid_mask]]
    axes[1, 1].imshow(gt_seg_color)
    axes[1, 1].set_title('Ground Truth Segmentation')
    axes[1, 1].axis('off')
    
    # Ground truth depth
    axes[1, 2].imshow(gt_depth, cmap='viridis')
    axes[1, 2].set_title('Ground Truth Depth')
    axes[1, 2].axis('off')
    
    # Hide empty subplot
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_model(model, dataloader, cfg, save_predictions=False):
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        cfg: Configuration dictionary
        save_predictions: Whether to save prediction visualizations
        
    Returns:
        Dictionary of evaluation metrics
    """
    device = cfg['model']['device']
    model = model.to(device)
    
    # Initialize metrics
    seg_metrics = SegmentationMetrics(cfg['model']['num_classes'])
    depth_metrics = DepthMetrics()
    
    # Load colormap and class info
    colormap = np.load(cfg['general']['cmap'])
    with open(cfg['dataset']['class_info_path'], 'r') as f:
        class_info = json.load(f)
    class_names = class_info['class_names']
    
    # Create output directories
    if save_predictions:
        pred_dir = cfg['evaluation']['prediction_dir']
        os.makedirs(pred_dir, exist_ok=True)
        seg_dir = os.path.join(pred_dir, 'segmentation')
        depth_dir = os.path.join(pred_dir, 'depth')
        viz_dir = os.path.join(pred_dir, 'visualizations')
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
    
    # Evaluation loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            # Move to device
            images = batch['image'].to(device)
            
            # Forward pass
            with autocast():
                pred_seg, pred_depth = model(images)
                
            # Move predictions to CPU
            pred_seg = pred_seg.cpu()
            pred_depth = pred_depth.cpu()
            
            # Process each sample in batch
            for j in range(images.shape[0]):
                # Get predictions for this sample
                seg_logits = pred_seg[j]  # (C, H, W)
                seg_pred = seg_logits.argmax(dim=0).numpy()  # (H, W)
                depth_pred = pred_depth[j, 0].numpy()  # (H, W)
                
                # Get ground truth
                has_seg = batch['has_segmentation'][j].item()
                has_depth = batch['has_depth'][j].item()
                
                # Update segmentation metrics
                if has_seg:
                    gt_seg = batch['segmentation'][j].numpy()
                    seg_metrics.update(
                        torch.tensor(seg_pred).unsqueeze(0),
                        torch.tensor(gt_seg).unsqueeze(0)
                    )
                    
                # Update depth metrics
                if has_depth:
                    gt_depth = batch['depth'][j].numpy()
                    depth_valid = batch['depth_valid'][j].numpy()
                    depth_metrics.update(
                        torch.tensor(depth_pred).unsqueeze(0),
                        torch.tensor(gt_depth).unsqueeze(0),
                        torch.tensor(depth_valid).unsqueeze(0)
                    )
                    
                # Save predictions if requested
                if save_predictions and i < 50:  # Save first 50 batches
                    sample_idx = i * dataloader.batch_size + j
                    
                    # Get original image
                    img = images[j].cpu().numpy().transpose(1, 2, 0)
                    img = (img * np.array(cfg['preprocessing']['img_std']) + 
                          np.array(cfg['preprocessing']['img_mean']))
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    
                    # Save segmentation
                    if has_seg:
                        seg_path = os.path.join(seg_dir, f'seg_{sample_idx:06d}.png')
                        seg_color = colormap[seg_pred]
                        Image.fromarray(seg_color).save(seg_path)
                        
                    # Save depth
                    if has_depth:
                        depth_path = os.path.join(depth_dir, f'depth_{sample_idx:06d}.npy')
                        np.save(depth_path, depth_pred)
                        
                        # Also save as normalized image
                        depth_img = (depth_pred / depth_pred.max() * 255).astype(np.uint8)
                        depth_img_path = os.path.join(depth_dir, f'depth_{sample_idx:06d}.png')
                        Image.fromarray(depth_img).save(depth_img_path)
                        
                    # Save visualization
                    if (has_seg or has_depth) and i < 20:  # Save first 20 visualizations
                        viz_path = os.path.join(viz_dir, f'viz_{sample_idx:06d}.png')
                        gt_seg = batch['segmentation'][j].numpy() if has_seg else np.zeros_like(seg_pred)
                        gt_depth = batch['depth'][j].numpy() if has_depth else np.zeros_like(depth_pred)
                        
                        visualize_predictions(
                            img, seg_pred, depth_pred,
                            gt_seg, gt_depth,
                            class_names, colormap,
                            save_path=viz_path
                        )
    
    # Compute final metrics
    results = {}
    
    # Segmentation metrics
    results['segmentation'] = {
        'mIoU': seg_metrics.get_miou(),
        'pixel_accuracy': seg_metrics.get_pixel_accuracy(),
        'mean_class_accuracy': seg_metrics.get_mean_class_accuracy(),
        'per_class_iou': seg_metrics.get_iou_per_class().tolist(),
        'per_class_accuracy': seg_metrics.get_class_accuracy().tolist()
    }
    
    # Depth metrics
    depth_all_metrics = depth_metrics.get_all_metrics()
    results['depth'] = depth_all_metrics
    
    return results


def print_results(results, class_names):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Segmentation results
    print("\nSEGMENTATION METRICS:")
    print(f"  mIoU: {results['segmentation']['mIoU']:.4f}")
    print(f"  Pixel Accuracy: {results['segmentation']['pixel_accuracy']:.4f}")
    print(f"  Mean Class Accuracy: {results['segmentation']['mean_class_accuracy']:.4f}")
    
    print("\n  Per-Class IoU:")
    for i, (name, iou) in enumerate(zip(class_names, results['segmentation']['per_class_iou'])):
        if iou > 0:  # Only show classes that appear in validation set
            print(f"    {i:2d} {name:30s}: {iou:.4f}")
            
    # Depth results
    print("\nDEPTH METRICS:")
    print(f"  RMSE: {results['depth']['rmse']:.4f}")
    print(f"  MAE: {results['depth']['mae']:.4f}")
    print(f"  Abs Rel: {results['depth']['abs_rel']:.4f}")
    print(f"  Sq Rel: {results['depth']['sq_rel']:.4f}")
    print(f"  δ < 1.25: {results['depth']['delta_1.25']:.4f}")
    print(f"  δ < 1.25²: {results['depth']['delta_1.25^2']:.4f}")
    print(f"  δ < 1.25³: {results['depth']['delta_1.25^3']:.4f}")
    
    print("="*60 + "\n")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate Hydranet on SafeTrip-Q dataset')
    parser.add_argument('--config', type=str, default='config/cfg_safetrip.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction visualizations')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override batch size if specified
    if args.batch_size is not None:
        cfg['evaluation']['eval_batch_size'] = args.batch_size
        
    # Set device
    device = torch.device(cfg['model']['device'] if torch.cuda.is_available() else 'cpu')
    cfg['model']['device'] = device
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(cfg, args.checkpoint)
    
    # Create validation dataloader
    val_dataset = SafeTripDataset(
        data_root=cfg['dataset']['root'],
        split='val',
        img_size=tuple(cfg['dataset']['img_size']),
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['evaluation']['eval_batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Evaluate model
    results = evaluate_model(
        model, val_loader, cfg, 
        save_predictions=args.save_predictions
    )
    
    # Load class names
    with open(cfg['dataset']['class_info_path'], 'r') as f:
        class_info = json.load(f)
    
    # Print results
    print_results(results, class_info['class_names'])
    
    # Save results to file
    results_path = os.path.join(
        os.path.dirname(args.checkpoint),
        'evaluation_results.json'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    

if __name__ == "__main__":
    main()