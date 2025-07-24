import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms
from tqdm import tqdm
from utils.depth_preprocessing import KITTIDepthPreprocessor
import pickle


class KITTIDepthDataset(Dataset):
    """KITTI depth dataset loader for data_depth_annotated."""
    
    def __init__(self, root_dir, split='train', img_size=(512, 512), 
                 preprocess_depth=True, preprocess_method='balanced',
                 pseudo_labels_path=None, confidence_threshold=0.7):
        self.root = os.path.join(root_dir, split)
        self.img_size = img_size
        self.split = split
        self.preprocess_depth = preprocess_depth
        self.preprocess_method = preprocess_method
        self.confidence_threshold = confidence_threshold
        
        # Initialize depth preprocessor if needed
        if self.preprocess_depth:
            self.depth_preprocessor = KITTIDepthPreprocessor(max_depth=80.0)
        
        # Load pseudo labels if provided
        self.pseudo_labels = None
        if pseudo_labels_path and os.path.exists(pseudo_labels_path):
            print(f"Loading pseudo labels from {pseudo_labels_path}")
            with open(pseudo_labels_path, 'rb') as f:
                data = pickle.load(f)
                self.pseudo_labels = data['pseudo_labels']
                print(f"Loaded {len(self.pseudo_labels)} pseudo labels")
        
        # Image normalization (ImageNet standards)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Collect all samples
        self.samples = self._collect_samples()
        print(f"Found {len(self.samples)} KITTI {split} samples")
        
    def _collect_samples(self):
        """Collect all depth samples from KITTI structure."""
        samples = []
        
        # Get all drive sequences
        drives = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        print(f"\n[KITTI] Scanning {len(drives)} drive sequences...")
        
        # Iterate through all drive sequences
        for drive in tqdm(drives, desc=f"[KITTI {self.split}]"):
            if not os.path.isdir(os.path.join(self.root, drive)):
                continue
                
            depth_dir = os.path.join(self.root, drive, 'proj_depth', 'groundtruth', 'image_02')
            
            if not os.path.exists(depth_dir):
                continue
                
            # Get all depth files
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
            
            for depth_file in depth_files:
                sample = {
                    'drive': drive,
                    'frame_id': depth_file[:-4],  # Remove .png
                    'depth_path': os.path.join(depth_dir, depth_file)
                }
                samples.append(sample)
                
        return samples
    
    def _load_depth(self, depth_path):
        """Load depth map from KITTI format (uint16 PNG, depth in mm)."""
        # Read depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth from {depth_path}")
            
        # Convert from mm to meters
        depth = depth.astype(np.float32) / 1000.0
        
        # KITTI specific: Clip to reasonable range for driving scenes
        # This helps normalize with SafeTrip's closer range depths
        depth = np.clip(depth, 0, 80.0)  # Max 80m for KITTI
        
        # Optional: Apply log-scale normalization to compress range
        # This helps balance KITTI's long-range with SafeTrip's short-range
        # depth = np.log1p(depth) / np.log1p(80.0) * 10.0  # Normalize to ~0-10m range
        
        # KITTI depth is sparse, but we keep all values (0 = invalid)
        return depth
    
    def _load_rgb(self, sample):
        """Load corresponding RGB image from KITTI raw data (if available)."""
        # For now, create a placeholder RGB image
        # In full implementation, you would load from KITTI raw dataset
        # Path would be something like: kitti_raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/
        
        # Create gray placeholder that matches depth resolution
        depth = self._load_depth(sample['depth_path'])
        h, w = depth.shape
        
        # Create a simple gradient image as placeholder
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[np.newaxis, :]
        rgb[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, np.newaxis]
        rgb[:, :, 2] = 128
        
        return rgb
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load depth
        depth = self._load_depth(sample['depth_path'])
        original_size = depth.shape
        
        # Load RGB (placeholder for now)
        rgb = self._load_rgb(sample)
        
        # Resize to target size
        rgb = cv2.resize(rgb, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convert RGB to tensor and normalize
        rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        rgb = self.normalize(rgb)
        
        # Store original sparse depth for validation
        sparse_depth = depth.copy()
        
        # Apply preprocessing if enabled
        if self.preprocess_depth:
            # Preprocess depth
            processed_depth = self.depth_preprocessor.preprocess(
                depth, method=self.preprocess_method
            )
            
            # Create confidence mask
            confidence_mask = self.depth_preprocessor.create_validity_mask(
                sparse_depth, processed_depth
            )
            
            # Use processed depth
            depth = processed_depth
            
            # Convert to tensors
            depth = torch.from_numpy(depth).float()
            depth_valid = torch.from_numpy(confidence_mask).float()
        else:
            # Convert depth to tensor
            depth = torch.from_numpy(depth).unsqueeze(0)
            depth = depth.squeeze(0)
            
            # Create simple depth validity mask (non-zero pixels)
            depth_valid = (depth > 0).float()
        
        # Get pseudo segmentation label if available
        segmentation = torch.ones(self.img_size, dtype=torch.long) * 255  # Default: no segmentation
        has_segmentation = False
        
        if self.pseudo_labels is not None:
            # Create sample key matching the format used in generate_pseudo_labels.py
            sample_key = f"{self.split}/{sample['drive']}_{sample['frame_id']}"
            
            if sample_key in self.pseudo_labels:
                pseudo_data = self.pseudo_labels[sample_key]
                pseudo_label = pseudo_data['pseudo_label']
                confidence = pseudo_data['confidence']
                
                # Resize pseudo label to target size
                pseudo_label = cv2.resize(
                    pseudo_label.astype(np.int32), 
                    (self.img_size[1], self.img_size[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
                confidence = cv2.resize(
                    confidence, 
                    (self.img_size[1], self.img_size[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Apply confidence threshold
                mask = confidence > self.confidence_threshold
                segmentation = torch.from_numpy(pseudo_label).long()
                segmentation[~mask] = 255  # Ignore low confidence pixels
                
                has_segmentation = mask.any()
        
        # Create output similar to SafeTrip format
        return {
            'image': rgb,
            'segmentation': segmentation,
            'depth': depth,
            'depth_valid': depth_valid if isinstance(depth_valid, torch.Tensor) else depth_valid.squeeze(0),
            'has_segmentation': torch.tensor(has_segmentation, dtype=torch.bool),
            'has_depth': torch.tensor(True, dtype=torch.bool)
        }


class MixedDataset(Dataset):
    """Combined dataset for KITTI depth + SafeTrip segmentation pre-training."""
    
    def __init__(self, safetrip_dataset, kitti_dataset, kitti_ratio=0.5):
        self.safetrip = safetrip_dataset
        self.kitti = kitti_dataset
        self.kitti_ratio = kitti_ratio
        
        # Calculate total length
        self.total_length = len(self.safetrip) + len(self.kitti)
        
        print(f"Mixed dataset: {len(self.safetrip)} SafeTrip + {len(self.kitti)} KITTI = {self.total_length} total")
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Randomly choose dataset based on ratio
        if np.random.random() < self.kitti_ratio:
            # Sample from KITTI
            kitti_idx = np.random.randint(0, len(self.kitti))
            return self.kitti[kitti_idx]
        else:
            # Sample from SafeTrip
            safetrip_idx = np.random.randint(0, len(self.safetrip))
            return self.safetrip[safetrip_idx]