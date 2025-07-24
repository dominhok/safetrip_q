import numpy as np
import torch
from typing import Optional


class SegmentationMetrics:
    """Metrics for semantic segmentation evaluation."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        """
        Initialize segmentation metrics.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in evaluation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update confusion matrix with predictions and targets.
        
        Args:
            pred: Predicted labels (B, H, W)
            target: Ground truth labels (B, H, W)
        """
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        # Filter out ignored pixels
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        for t, p in zip(target.flatten(), pred.flatten()):
            if t < self.num_classes and p < self.num_classes:
                self.confusion_matrix[t, p] += 1
                
    def get_iou_per_class(self):
        """Get IoU for each class."""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                self.confusion_matrix.sum(axis=0) - intersection)
        
        # Avoid division by zero
        iou = np.zeros(self.num_classes)
        valid_classes = union > 0
        iou[valid_classes] = intersection[valid_classes] / union[valid_classes]
        
        return iou
        
    def get_miou(self):
        """Get mean IoU across all classes."""
        iou = self.get_iou_per_class()
        # Only consider classes that appear in the data
        valid_classes = self.confusion_matrix.sum(axis=1) > 0
        if valid_classes.any():
            return iou[valid_classes].mean()
        else:
            return 0.0
            
    def get_pixel_accuracy(self):
        """Get overall pixel accuracy."""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        if total > 0:
            return correct / total
        else:
            return 0.0
            
    def get_class_accuracy(self):
        """Get per-class accuracy."""
        class_correct = np.diag(self.confusion_matrix)
        class_total = self.confusion_matrix.sum(axis=1)
        
        acc = np.zeros(self.num_classes)
        valid_classes = class_total > 0
        acc[valid_classes] = class_correct[valid_classes] / class_total[valid_classes]
        
        return acc
        
    def get_mean_class_accuracy(self):
        """Get mean accuracy across all classes."""
        acc = self.get_class_accuracy()
        valid_classes = self.confusion_matrix.sum(axis=1) > 0
        if valid_classes.any():
            return acc[valid_classes].mean()
        else:
            return 0.0


class DepthMetrics:
    """Metrics for depth estimation evaluation."""
    
    def __init__(self):
        """Initialize depth metrics."""
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.rmse_sum = 0.0
        self.mae_sum = 0.0
        self.abs_rel_sum = 0.0
        self.sq_rel_sum = 0.0
        self.log_rmse_sum = 0.0
        
        self.delta_1_count = 0
        self.delta_2_count = 0
        self.delta_3_count = 0
        
        self.total_pixels = 0
        
    def update(self, pred: torch.Tensor, target: torch.Tensor, 
               valid_mask: Optional[torch.Tensor] = None):
        """
        Update metrics with predictions and targets.
        
        Args:
            pred: Predicted depth (B, H, W)
            target: Ground truth depth (B, H, W)
            valid_mask: Valid pixel mask (B, H, W)
        """
        # Apply valid mask if provided
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
            
        if len(pred) == 0:
            return
            
        # Ensure positive depth values
        pred = torch.clamp(pred, min=1e-3)
        target = torch.clamp(target, min=1e-3)
        
        # Compute errors
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # RMSE
        self.rmse_sum += torch.sum(diff ** 2).item()
        
        # MAE
        self.mae_sum += torch.sum(abs_diff).item()
        
        # Absolute relative error
        self.abs_rel_sum += torch.sum(abs_diff / target).item()
        
        # Squared relative error
        self.sq_rel_sum += torch.sum((diff ** 2) / target).item()
        
        # Log RMSE
        log_diff = torch.log(pred) - torch.log(target)
        self.log_rmse_sum += torch.sum(log_diff ** 2).item()
        
        # Delta accuracies
        ratio = torch.max(pred / target, target / pred)
        self.delta_1_count += torch.sum(ratio < 1.25).item()
        self.delta_2_count += torch.sum(ratio < 1.25 ** 2).item()
        self.delta_3_count += torch.sum(ratio < 1.25 ** 3).item()
        
        self.total_pixels += len(pred)
        
    def get_rmse(self):
        """Get Root Mean Squared Error."""
        if self.total_pixels > 0:
            return np.sqrt(self.rmse_sum / self.total_pixels)
        else:
            return float('inf')
            
    def get_mae(self):
        """Get Mean Absolute Error."""
        if self.total_pixels > 0:
            return self.mae_sum / self.total_pixels
        else:
            return float('inf')
            
    def get_abs_rel(self):
        """Get Absolute Relative Error."""
        if self.total_pixels > 0:
            return self.abs_rel_sum / self.total_pixels
        else:
            return float('inf')
            
    def get_sq_rel(self):
        """Get Squared Relative Error."""
        if self.total_pixels > 0:
            return self.sq_rel_sum / self.total_pixels
        else:
            return float('inf')
            
    def get_log_rmse(self):
        """Get Log Root Mean Squared Error."""
        if self.total_pixels > 0:
            return np.sqrt(self.log_rmse_sum / self.total_pixels)
        else:
            return float('inf')
            
    def get_delta_accuracy(self, threshold: float):
        """
        Get delta accuracy (percentage of pixels with relative error < threshold).
        
        Args:
            threshold: Threshold value (typically 1.25, 1.25^2, or 1.25^3)
        """
        if self.total_pixels > 0:
            if threshold == 1.25:
                return self.delta_1_count / self.total_pixels
            elif threshold == 1.25 ** 2:
                return self.delta_2_count / self.total_pixels
            elif threshold == 1.25 ** 3:
                return self.delta_3_count / self.total_pixels
            else:
                raise ValueError(f"Unsupported threshold: {threshold}")
        else:
            return 0.0
            
    def get_all_metrics(self):
        """Get all metrics as a dictionary."""
        return {
            'rmse': self.get_rmse(),
            'mae': self.get_mae(),
            'abs_rel': self.get_abs_rel(),
            'sq_rel': self.get_sq_rel(),
            'log_rmse': self.get_log_rmse(),
            'delta_1.25': self.get_delta_accuracy(1.25),
            'delta_1.25^2': self.get_delta_accuracy(1.25 ** 2),
            'delta_1.25^3': self.get_delta_accuracy(1.25 ** 3)
        }