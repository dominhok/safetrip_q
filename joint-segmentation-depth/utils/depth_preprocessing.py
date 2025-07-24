"""
Depth preprocessing utilities for KITTI sparse depth data.
Implements efficient morphological operations for depth completion.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_dilation
import torch


class KITTIDepthPreprocessor:
    """
    Preprocesses KITTI sparse depth maps using morphological operations.
    Based on IP-Basic approach with optimizations for deep learning.
    """
    
    def __init__(self, max_depth=80.0):
        self.max_depth = max_depth
        
        # Define kernels for different depth ranges
        self.kernel_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_far = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        # Full kernels for closing operations
        self.kernel_close_5 = np.ones((5, 5), np.uint8)
        self.kernel_close_7 = np.ones((7, 7), np.uint8)
    
    def preprocess(self, depth_map, method='fast'):
        """
        Preprocess sparse depth map.
        
        Args:
            depth_map: Sparse depth map (H, W) with 0 for missing values
            method: 'fast', 'balanced', or 'accurate'
            
        Returns:
            Preprocessed depth map with filled values
        """
        if method == 'fast':
            return self._fast_completion(depth_map)
        elif method == 'balanced':
            return self._balanced_completion(depth_map)
        elif method == 'accurate':
            return self._accurate_completion(depth_map)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _fast_completion(self, depth_map):
        """Fast depth completion using simple dilation."""
        # Create mask of valid depth values
        valid_mask = depth_map > 0
        
        # Apply dilation to expand depth values
        dilated = cv2.dilate(depth_map, self.kernel_med, iterations=2)
        
        # Fill holes with dilated values
        filled = np.where(valid_mask, depth_map, dilated)
        
        # Apply closing to smooth
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, self.kernel_close_5)
        
        # Clip to max depth
        filled = np.clip(filled, 0, self.max_depth)
        
        return filled
    
    def _balanced_completion(self, depth_map):
        """Balanced approach with multi-scale processing."""
        # Create masks for different depth ranges
        near_mask = (depth_map > 0.1) & (depth_map < 15.0)
        med_mask = (depth_map >= 15.0) & (depth_map < 30.0)
        far_mask = (depth_map >= 30.0) & (depth_map <= self.max_depth)
        
        # Process each range separately
        depth_near = np.zeros_like(depth_map)
        depth_med = np.zeros_like(depth_map)
        depth_far = np.zeros_like(depth_map)
        
        if near_mask.any():
            depth_near[near_mask] = depth_map[near_mask]
            depth_near = cv2.dilate(depth_near, self.kernel_near, iterations=1)
        
        if med_mask.any():
            depth_med[med_mask] = depth_map[med_mask]
            depth_med = cv2.dilate(depth_med, self.kernel_med, iterations=2)
        
        if far_mask.any():
            depth_far[far_mask] = depth_map[far_mask]
            depth_far = cv2.dilate(depth_far, self.kernel_far, iterations=3)
        
        # Combine results
        filled = np.maximum.reduce([depth_near, depth_med, depth_far])
        
        # Fill remaining holes
        valid_mask = depth_map > 0
        filled = np.where(valid_mask, depth_map, filled)
        
        # Apply bilateral filter for smoothing
        filled = cv2.bilateralFilter(filled.astype(np.float32), 5, 80, 80)
        
        return filled
    
    def _accurate_completion(self, depth_map):
        """
        Accurate completion with iterative refinement.
        Best for training, slower for inference.
        """
        # Initial fast completion
        filled = self._balanced_completion(depth_map)
        
        # Iterative refinement
        for i in range(3):
            # Create confidence mask based on original sparse points
            confidence = np.zeros_like(depth_map)
            confidence[depth_map > 0] = 1.0
            
            # Dilate confidence
            confidence = cv2.dilate(confidence, self.kernel_near, iterations=i+1)
            confidence = cv2.GaussianBlur(confidence, (5, 5), 1.0)
            
            # Weighted average between filled and smoothed
            smoothed = cv2.bilateralFilter(filled.astype(np.float32), 9, 150, 150)
            filled = confidence * filled + (1 - confidence) * smoothed
        
        # Ensure original sparse points are preserved
        valid_mask = depth_map > 0
        filled = np.where(valid_mask, depth_map, filled)
        
        return filled
    
    def create_validity_mask(self, depth_map, preprocessed_depth):
        """
        Create a validity/confidence mask for the preprocessed depth.
        
        Args:
            depth_map: Original sparse depth
            preprocessed_depth: Preprocessed dense depth
            
        Returns:
            Confidence mask (0-1) indicating reliability of depth values
        """
        # Start with original valid points having confidence 1
        confidence = np.zeros_like(depth_map, dtype=np.float32)
        confidence[depth_map > 0] = 1.0
        
        # Decay confidence based on distance from original points
        for i in range(5):
            dilated = cv2.dilate(confidence, self.kernel_near, iterations=1)
            confidence = np.maximum(confidence, dilated * (0.9 ** (i + 1)))
        
        # Further reduce confidence for extrapolated regions
        confidence[preprocessed_depth > 50.0] *= 0.8  # Far regions less reliable
        
        return confidence


def preprocess_kitti_batch(depth_batch, method='balanced'):
    """
    Preprocess a batch of KITTI depth maps.
    
    Args:
        depth_batch: Tensor of shape (B, H, W) or (B, 1, H, W)
        method: Preprocessing method
        
    Returns:
        Preprocessed depth batch and confidence masks
    """
    if isinstance(depth_batch, torch.Tensor):
        depth_batch_np = depth_batch.cpu().numpy()
        device = depth_batch.device
    else:
        depth_batch_np = depth_batch
        device = None
    
    # Handle different input shapes
    if depth_batch_np.ndim == 4:
        depth_batch_np = depth_batch_np.squeeze(1)
    
    batch_size = depth_batch_np.shape[0]
    preprocessor = KITTIDepthPreprocessor()
    
    processed_depths = []
    confidence_masks = []
    
    for i in range(batch_size):
        # Preprocess
        processed = preprocessor.preprocess(depth_batch_np[i], method=method)
        
        # Create confidence mask
        confidence = preprocessor.create_validity_mask(depth_batch_np[i], processed)
        
        processed_depths.append(processed)
        confidence_masks.append(confidence)
    
    # Stack results
    processed_depths = np.stack(processed_depths)
    confidence_masks = np.stack(confidence_masks)
    
    # Convert back to torch if needed
    if device is not None:
        processed_depths = torch.from_numpy(processed_depths).float().to(device)
        confidence_masks = torch.from_numpy(confidence_masks).float().to(device)
        
        # Add channel dimension back if needed
        if depth_batch.ndim == 4:
            processed_depths = processed_depths.unsqueeze(1)
            confidence_masks = confidence_masks.unsqueeze(1)
    
    return processed_depths, confidence_masks


# Example usage for training
if __name__ == "__main__":
    # Test with random sparse depth
    sparse_depth = np.zeros((512, 512), dtype=np.float32)
    # Simulate sparse LiDAR points (5% density)
    num_points = int(0.05 * 512 * 512)
    indices = np.random.choice(512*512, num_points, replace=False)
    sparse_depth.flat[indices] = np.random.uniform(0.5, 80.0, num_points)
    
    # Test different methods
    preprocessor = KITTIDepthPreprocessor()
    
    fast_result = preprocessor.preprocess(sparse_depth, method='fast')
    balanced_result = preprocessor.preprocess(sparse_depth, method='balanced')
    accurate_result = preprocessor.preprocess(sparse_depth, method='accurate')
    
    print(f"Original sparsity: {(sparse_depth > 0).sum() / sparse_depth.size:.2%}")
    print(f"Fast completion: {(fast_result > 0).sum() / fast_result.size:.2%}")
    print(f"Balanced completion: {(balanced_result > 0).sum() / balanced_result.size:.2%}")
    print(f"Accurate completion: {(accurate_result > 0).sum() / accurate_result.size:.2%}")