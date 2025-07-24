import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InverseHuberLoss(nn.Module):
    """
    Inverse Huber Loss for depth estimation.
    More robust to outliers than L1/L2 loss.
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Initialize Inverse Huber Loss.
        
        Args:
            delta: Threshold parameter for switching between L1 and L2
        """
        super().__init__()
        self.delta = delta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Inverse Huber Loss.
        
        Args:
            pred: Predicted depth values
            target: Ground truth depth values
            valid_mask: Boolean mask for valid depth pixels
            
        Returns:
            Loss value
        """
        # Compute absolute difference
        diff = torch.abs(pred - target)
        
        # Inverse Huber formulation
        # For small errors: 0.5 * x^2 / delta
        # For large errors: |x| - 0.5 * delta
        loss = torch.where(
            diff <= self.delta,
            0.5 * diff.pow(2) / self.delta,
            diff - 0.5 * self.delta
        )
        
        # Apply valid mask if provided
        if valid_mask is not None:
            loss = loss * valid_mask.float()
            # Average over valid pixels only
            return loss.sum() / (valid_mask.sum() + 1e-7)
        else:
            return loss.mean()


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018).
    Automatically learns task weights during training.
    """
    
    def __init__(self, num_tasks: int = 2):
        """
        Initialize uncertainty-weighted loss.
        
        Args:
            num_tasks: Number of tasks (2 for seg + depth)
        """
        super().__init__()
        # Initialize log variance parameters (learnable)
        # Initialize with small negative values for better initial balance
        self.log_vars = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        
    def forward(self, *losses) -> Tuple[torch.Tensor, dict]:
        """
        Compute weighted multi-task loss.
        
        Args:
            *losses: Variable number of task losses
            
        Returns:
            Tuple of (total_loss, weight_dict)
        """
        assert len(losses) == len(self.log_vars), \
            f"Number of losses ({len(losses)}) must match number of tasks ({len(self.log_vars)})"
            
        # Compute precision (inverse variance) for each task
        precisions = torch.exp(-self.log_vars)
        
        # Weighted sum of losses
        total_loss = 0
        weights = {}
        
        for i, (loss, log_var) in enumerate(zip(losses, self.log_vars)):
            # L = (1/2σ²)L_task + log(σ)
            weighted_loss = precisions[i] * loss + log_var
            total_loss += weighted_loss
            
            # Store weights for logging
            weights[f'weight_task_{i}'] = precisions[i].item()
            weights[f'log_var_task_{i}'] = log_var.item()
            
        return total_loss, weights


class AsymmetricMultiTaskLoss(nn.Module):
    """
    Multi-task loss for asymmetric annotations.
    Handles samples with missing segmentation or depth annotations.
    """
    
    def __init__(self, 
                 num_classes: int,
                 class_weights: Optional[torch.Tensor] = None,
                 seg_loss_type: str = 'ce',
                 depth_loss_type: str = 'inverse_huber',
                 use_uncertainty_weighting: bool = True,
                 lambda_seg: float = 0.5,
                 lambda_depth: float = 0.5):
        """
        Initialize asymmetric multi-task loss.
        
        Args:
            num_classes: Number of segmentation classes
            class_weights: Class weights for segmentation
            seg_loss_type: Type of segmentation loss ('ce' or 'focal')
            depth_loss_type: Type of depth loss ('inverse_huber' or 'l1')
            use_uncertainty_weighting: Whether to use dynamic weighting
            lambda_seg: Fixed weight for segmentation (if not using uncertainty)
            lambda_depth: Fixed weight for depth (if not using uncertainty)
        """
        super().__init__()
        
        # Segmentation loss
        if seg_loss_type == 'ce':
            self.seg_criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=255  # Ignore unannotated pixels
            )
        else:
            raise ValueError(f"Unknown segmentation loss: {seg_loss_type}")
            
        # Depth loss
        if depth_loss_type == 'inverse_huber':
            self.depth_criterion = InverseHuberLoss()
        elif depth_loss_type == 'l1':
            self.depth_criterion = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown depth loss: {depth_loss_type}")
            
        # Loss weighting
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            self.loss_weights = UncertaintyWeightedLoss(num_tasks=2)
        else:
            self.lambda_seg = lambda_seg
            self.lambda_depth = lambda_depth
            
        # Running averages for loss normalization
        self.register_buffer('seg_loss_avg', torch.tensor(1.0))
        self.register_buffer('depth_loss_avg', torch.tensor(1.0))
        self.momentum = 0.99
            
    def forward(self, 
                pred_seg: torch.Tensor,
                pred_depth: torch.Tensor,
                batch: dict) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss with asymmetric annotations.
        
        Args:
            pred_seg: Predicted segmentation logits (B, C, H, W)
            pred_depth: Predicted depth map (B, 1, H, W)
            batch: Batch dictionary with ground truth and masks
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = 0
        loss_dict = {}
        
        # Get ground truth and masks
        gt_seg = batch['segmentation']  # (B, H, W)
        gt_depth = batch['depth']  # (B, H, W)
        depth_valid = batch['depth_valid']  # (B, H, W)
        has_seg = batch['has_segmentation']  # (B,)
        has_depth = batch['has_depth']  # (B,)
        
        # Remove channel dimension from depth prediction
        pred_depth = pred_depth.squeeze(1)  # (B, H, W)
        
        # Segmentation loss
        seg_loss = torch.tensor(0.0, device=pred_seg.device)
        if has_seg.any():
            # Get indices of samples with segmentation
            seg_indices = has_seg.nonzero(as_tuple=True)[0]
            
            if len(seg_indices) > 0:
                # Select samples with segmentation
                pred_seg_valid = pred_seg[seg_indices]
                gt_seg_valid = gt_seg[seg_indices]
                
                # Compute segmentation loss
                seg_loss = self.seg_criterion(pred_seg_valid, gt_seg_valid)
                loss_dict['seg_loss'] = seg_loss.item()
                loss_dict['num_seg_samples'] = len(seg_indices)
                
        # Depth loss
        depth_loss = torch.tensor(0.0, device=pred_depth.device)
        if has_depth.any():
            # Get indices of samples with depth
            depth_indices = has_depth.nonzero(as_tuple=True)[0]
            
            if len(depth_indices) > 0:
                # Select samples with depth
                pred_depth_valid = pred_depth[depth_indices]
                gt_depth_valid = gt_depth[depth_indices]
                depth_valid_mask = depth_valid[depth_indices]
                
                # Count valid pixels
                valid_pixels = depth_valid_mask.sum()
                loss_dict['valid_depth_pixels'] = valid_pixels.item()
                loss_dict['valid_depth_ratio'] = (valid_pixels.float() / depth_valid_mask.numel()).item()
                
                # Check if this is sparse data (likely KITTI)
                is_sparse = loss_dict['valid_depth_ratio'] < 0.1  # Less than 10% valid pixels
                
                # Compute depth loss
                if isinstance(self.depth_criterion, InverseHuberLoss):
                    depth_loss = self.depth_criterion(
                        pred_depth_valid, gt_depth_valid, depth_valid_mask
                    )
                else:
                    # L1 loss
                    depth_diff = self.depth_criterion(pred_depth_valid, gt_depth_valid)
                    depth_loss = (depth_diff * depth_valid_mask.float()).sum() / (
                        depth_valid_mask.sum() + 1e-7
                    )
                
                # Scale up sparse depth loss to match dense loss magnitude
                if is_sparse and valid_pixels > 0:
                    # Scale by inverse of valid ratio to compensate for sparsity
                    scale_factor = min(10.0, 0.5 / loss_dict['valid_depth_ratio'])
                    depth_loss = depth_loss * scale_factor
                    loss_dict['depth_scale_factor'] = scale_factor
                    
                loss_dict['depth_loss'] = depth_loss.item()
                loss_dict['num_depth_samples'] = len(depth_indices)
                
        # Update running averages for normalization
        with torch.no_grad():
            if seg_loss > 0:
                self.seg_loss_avg = self.momentum * self.seg_loss_avg + (1 - self.momentum) * seg_loss
            if depth_loss > 0:
                self.depth_loss_avg = self.momentum * self.depth_loss_avg + (1 - self.momentum) * depth_loss
                
        # Normalize losses to similar scales
        if seg_loss > 0:
            seg_loss_normalized = seg_loss / (self.seg_loss_avg + 1e-8)
        else:
            seg_loss_normalized = seg_loss
            
        if depth_loss > 0:
            depth_loss_normalized = depth_loss / (self.depth_loss_avg + 1e-8)
        else:
            depth_loss_normalized = depth_loss
            
        loss_dict['seg_loss_normalized'] = seg_loss_normalized.item() if seg_loss > 0 else 0
        loss_dict['depth_loss_normalized'] = depth_loss_normalized.item() if depth_loss > 0 else 0
        loss_dict['seg_loss_avg'] = self.seg_loss_avg.item()
        loss_dict['depth_loss_avg'] = self.depth_loss_avg.item()
        
        # Combine losses
        if self.use_uncertainty_weighting:
            # Dynamic weighting with normalized losses
            if has_seg.any() and has_depth.any():
                total_loss, weights = self.loss_weights(seg_loss_normalized, depth_loss_normalized)
                loss_dict.update(weights)
            elif has_seg.any():
                total_loss = seg_loss_normalized
            elif has_depth.any():
                total_loss = depth_loss_normalized
            else:
                # No valid annotations in batch
                total_loss = torch.tensor(0.0, device=pred_seg.device)
        else:
            # Fixed weighting with normalized losses
            total_loss = 0
            if has_seg.any():
                total_loss = total_loss + self.lambda_seg * seg_loss_normalized
            if has_depth.any():
                total_loss = total_loss + self.lambda_depth * depth_loss_normalized
                
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.
    Optional alternative to weighted cross-entropy.
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 ignore_index: int = 255):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights
            gamma: Focusing parameter
            ignore_index: Index to ignore
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
            
        Returns:
            Loss value
        """
        # Flatten tensors
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, pred.size(1))
        target = target.view(-1)
        
        # Filter out ignored indices
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if len(target) == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # Compute cross-entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Compute focal term
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
            
        return focal_loss.mean()