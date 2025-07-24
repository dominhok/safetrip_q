import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from typing import Dict, List, Tuple, Optional
from torchvision import transforms
from utils.xml_parser import CVATXMLParser
from utils.depth_converter import DepthConverter
from tqdm import tqdm


class SafeTripDataset(Dataset):
    """
    SafeTrip-Q dataset for joint semantic segmentation and depth estimation.
    Handles asymmetric annotations where samples have either segmentation or depth, but not both.
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 img_size: Tuple[int, int] = (512, 512),
                 train_ratio: float = 0.8,
                 augment: bool = True,
                 depth_only: bool = False,
                 segmentation_only: bool = False,
                 pseudo_labels_path: Optional[str] = None):
        """
        Initialize SafeTrip-Q dataset.
        
        Args:
            data_root: Root directory containing Surface/, Polygon/, and Depth/ folders
            split: 'train' or 'val'
            img_size: Target image size (height, width)
            train_ratio: Ratio of training data
            augment: Whether to apply data augmentation
            depth_only: If True, only load depth samples (for fine-tuning)
            segmentation_only: If True, only load segmentation samples (no depth)
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.augment = augment and (split == 'train')
        self.depth_only = depth_only
        self.segmentation_only = segmentation_only
        
        # Initialize parsers
        self.class_mapping = CVATXMLParser.get_unified_class_mapping()
        self.xml_parser = CVATXMLParser(self.class_mapping)
        self.depth_converter = DepthConverter()
        
        # Load pseudo labels if provided
        self.pseudo_labels = None
        if pseudo_labels_path and os.path.exists(pseudo_labels_path):
            import pickle
            print(f"Loading pseudo labels from {pseudo_labels_path}")
            with open(pseudo_labels_path, 'rb') as f:
                pseudo_data = pickle.load(f)
                self.pseudo_labels = pseudo_data.get('pseudo_labels', {})
        
        # Get removed classes for filtering
        self.removed_classes = set(CVATXMLParser.get_removed_classes())
        
        # Image normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Collect all samples
        self.samples = []
        self._load_all_samples()
        
        # Split data
        print(f"\nTotal samples collected: {len(self.samples)}")
        self._split_data()
        
        print(f"\nFinal dataset: {len(self.samples)} {split} samples from SafeTrip-Q dataset")
        print(f"  - Surface: {sum(1 for s in self.samples if s['type'] == 'surface')} samples")
        print(f"  - Polygon: {sum(1 for s in self.samples if s['type'] == 'polygon')} samples")
        print(f"  - Depth: {sum(1 for s in self.samples if s['type'] == 'depth')} samples")
        
    def _load_all_samples(self):
        """Load all samples from Surface, Polygon, and Depth folders."""
        if self.depth_only:
            # Only load depth samples for fine-tuning
            self._load_depth_samples()
        elif self.segmentation_only:
            # Only load segmentation samples (no depth)
            self._load_surface_samples()
            self._load_polygon_samples()
        else:
            # Load all sample types
            # Load Surface samples
            self._load_surface_samples()
            
            # Load Polygon samples  
            self._load_polygon_samples()
            
            # Load Depth samples
            self._load_depth_samples()
        
    def _load_surface_samples(self):
        """Load Surface segmentation samples."""
        surface_dir = os.path.join(self.data_root, 'Surface')
        if not os.path.exists(surface_dir):
            print(f"Surface directory not found: {surface_dir}")
            return
            
        # Find all Surface_XXX folders
        surface_folders = sorted(glob.glob(os.path.join(surface_dir, 'Surface_*')))
        print(f"\nLoading Surface samples from {len(surface_folders)} folders...")
        
        for folder in tqdm(surface_folders, desc="Surface folders"):
            if not os.path.isdir(folder):
                continue
                
            # Find XML file in the folder
            xml_files = glob.glob(os.path.join(folder, '*.xml'))
            if not xml_files:
                print(f"No XML found in {folder}")
                continue
                
            # Parse XML to get all annotations
            xml_path = xml_files[0]
            annotations = self.xml_parser.parse_xml(xml_path)
            
            # Find all images in the folder
            image_files = glob.glob(os.path.join(folder, '*.jpg'))
            image_files.extend(glob.glob(os.path.join(folder, '*.png')))
            
            for img_path in image_files:
                img_name = os.path.basename(img_path)
                
                # Check if this image has annotation
                if img_name in annotations:
                    self.samples.append({
                        'type': 'surface',
                        'image_path': img_path,
                        'annotations': annotations,
                        'image_name': img_name,
                        'has_segmentation': True,
                        'has_depth': False
                    })
                    
    def _load_polygon_samples(self):
        """Load Polygon (obstacle) segmentation samples."""
        polygon_dir = os.path.join(self.data_root, 'Polygon')
        if not os.path.exists(polygon_dir):
            print(f"Polygon directory not found: {polygon_dir}")
            return
            
        # Find all Polygon_XXXX folders
        polygon_folders = sorted(glob.glob(os.path.join(polygon_dir, 'Polygon_*')))
        print(f"\nLoading Polygon samples from {len(polygon_folders)} folders...")
        
        for folder in tqdm(polygon_folders, desc="Polygon folders"):
            if not os.path.isdir(folder):
                continue
                
            # Find XML file in the folder
            xml_files = glob.glob(os.path.join(folder, '*.xml'))
            if not xml_files:
                print(f"No XML found in {folder}")
                continue
                
            # Parse XML to get all annotations
            xml_path = xml_files[0]
            annotations = self.xml_parser.parse_xml(xml_path)
            
            # Find all images in the folder
            image_files = glob.glob(os.path.join(folder, '*.jpg'))
            image_files.extend(glob.glob(os.path.join(folder, '*.png')))
            
            for img_path in image_files:
                img_name = os.path.basename(img_path)
                
                # Check if this image has annotation
                if img_name in annotations:
                    self.samples.append({
                        'type': 'polygon',
                        'image_path': img_path,
                        'annotations': annotations,
                        'image_name': img_name,
                        'has_segmentation': True,
                        'has_depth': False
                    })
                    
    def _load_depth_samples(self):
        """Load Depth estimation samples."""
        depth_dir = os.path.join(self.data_root, 'Depth')
        if not os.path.exists(depth_dir):
            print(f"Depth directory not found: {depth_dir}")
            return
            
        # Find all Depth_XXX folders
        depth_folders = sorted(glob.glob(os.path.join(depth_dir, 'Depth_*')))
        print(f"\nLoading Depth samples from {len(depth_folders)} folders...")
        
        for folder in tqdm(depth_folders, desc="Depth folders"):
            if not os.path.isdir(folder):
                continue
                
            # Load calibration for this folder
            folder_name = os.path.basename(folder)
            calib_path = os.path.join(folder, f"{folder_name}.conf")
            
            if not os.path.exists(calib_path):
                print(f"Calibration not found: {calib_path}")
                continue
                
            calibration = self.depth_converter.parse_calibration(calib_path)
            
            # Get all image sets in this folder
            image_sets = self.depth_converter.get_depth_image_sets(folder)
            
            for base_name in image_sets:
                # Check if we have pseudo label for this depth sample
                rgb_path = os.path.join(folder, f"{base_name}_L.png")
                has_pseudo_label = False
                if self.pseudo_labels and rgb_path in self.pseudo_labels:
                    has_pseudo_label = True
                    
                self.samples.append({
                    'type': 'depth',
                    'depth_folder': folder,
                    'base_name': base_name,
                    'calibration': calibration,
                    'has_segmentation': has_pseudo_label,
                    'has_depth': True,
                    'rgb_path': rgb_path
                })
                
    def _split_data(self):
        """Split data into train/val sets."""
        # Group samples by type
        surface_samples = [s for s in self.samples if s['type'] == 'surface']
        polygon_samples = [s for s in self.samples if s['type'] == 'polygon']
        depth_samples = [s for s in self.samples if s['type'] == 'depth']
        
        # Split each type
        def split_list(lst, train_ratio):
            n_train = int(len(lst) * train_ratio)
            if self.split == 'train':
                return lst[:n_train]
            else:
                return lst[n_train:]
                
        # Combine splits
        self.samples = (
            split_list(surface_samples, self.train_ratio) +
            split_list(polygon_samples, self.train_ratio) +
            split_list(depth_samples, self.train_ratio)
        )
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample_info = self.samples[idx]
        
        if sample_info['type'] in ['surface', 'polygon']:
            return self._get_segmentation_sample(sample_info)
        else:
            return self._get_depth_sample(sample_info)
            
    def _get_segmentation_sample(self, sample_info):
        """Load a segmentation sample (Surface or Polygon)."""
        # Load image
        image = cv2.imread(sample_info['image_path'])
        if image is None:
            raise ValueError(f"Could not read image: {sample_info['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get segmentation mask
        mask = self.xml_parser.get_mask_for_image(
            sample_info['annotations'],
            sample_info['image_name'],
            target_shape=self.img_size
        )
        
        if mask is None:
            # Create dummy mask with ignore index
            mask = np.full(self.img_size, 255, dtype=np.uint8)
            
        # Resize image
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
            
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        
        mask = torch.from_numpy(mask).long()
        
        # Create dummy depth data
        depth = torch.zeros(self.img_size, dtype=torch.float32)
        depth_valid = torch.zeros(self.img_size, dtype=torch.bool)
        
        return {
            'image': image,
            'segmentation': mask,
            'depth': depth,
            'depth_valid': depth_valid,
            'has_segmentation': torch.tensor(True, dtype=torch.bool),
            'has_depth': torch.tensor(False, dtype=torch.bool)
        }
        
    def _get_depth_sample(self, sample_info):
        """Load a depth sample."""
        # Load depth data
        depth_data = self.depth_converter.load_depth_sample(
            sample_info['depth_folder'],
            sample_info['base_name'],
            sample_info['calibration']
        )
        
        if depth_data is None:
            # Return dummy sample
            return self._get_dummy_sample()
            
        image = depth_data['rgb_image']
        depth_map = depth_data['depth_map']
        valid_mask = depth_data['valid_mask']
        
        # Resize all
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        depth_map = cv2.resize(depth_map, (self.img_size[1], self.img_size[0]))
        valid_mask = cv2.resize(valid_mask.astype(np.uint8), 
                               (self.img_size[1], self.img_size[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Apply augmentation if enabled
        if self.augment:
            image, depth_map, valid_mask = self._apply_depth_augmentation(
                image, depth_map, valid_mask
            )
            
        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        
        # Normalize depth to [0, 1] range for better training stability
        # Using 80m as max depth for normalization (most relevant range)
        max_depth = 80.0
        depth = torch.from_numpy(depth_map).float()
        depth = depth / max_depth  # Normalize to [0, 1]
        depth = torch.clamp(depth, 0, 1)  # Ensure values are in [0, 1]
        depth_valid = torch.from_numpy(valid_mask)
        
        # Get pseudo label if available
        has_segmentation = sample_info.get('has_segmentation', False)
        if has_segmentation and self.pseudo_labels:
            rgb_path = sample_info.get('rgb_path')
            if rgb_path and rgb_path in self.pseudo_labels:
                pseudo_data = self.pseudo_labels[rgb_path]
                segmentation = torch.from_numpy(pseudo_data['pseudo_label']).long()
                # Resize pseudo label to match target size
                segmentation_np = segmentation.numpy()
                segmentation_np = cv2.resize(segmentation_np.astype(np.int32), 
                                           (self.img_size[1], self.img_size[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                segmentation = torch.from_numpy(segmentation_np).long()
                
                # Apply confidence threshold - set low confidence pixels to ignore
                confidence = pseudo_data['confidence']
                confidence = cv2.resize(confidence, 
                                      (self.img_size[1], self.img_size[0]), 
                                      interpolation=cv2.INTER_LINEAR)
                low_conf_mask = confidence < 0.7
                segmentation[low_conf_mask] = 255  # Ignore low confidence pixels
            else:
                # No pseudo label found
                segmentation = torch.full(self.img_size, 255, dtype=torch.long)
                has_segmentation = False
        else:
            # No pseudo labels
            segmentation = torch.full(self.img_size, 255, dtype=torch.long)
        
        return {
            'image': image,
            'segmentation': segmentation,
            'depth': depth,
            'depth_valid': depth_valid,
            'has_segmentation': torch.tensor(has_segmentation, dtype=torch.bool),
            'has_depth': torch.tensor(True, dtype=torch.bool)
        }
        
    def _apply_augmentation(self, image, mask):
        """Apply augmentation to image and mask."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            
        # Random brightness/contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-20, 20)    # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
        return image, mask
        
    def _apply_depth_augmentation(self, image, depth, valid_mask):
        """Apply augmentation to depth data."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            depth = cv2.flip(depth, 1)
            valid_mask = cv2.flip(valid_mask.astype(np.uint8), 1).astype(bool)
            
        # Random brightness/contrast (image only)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
        # Random depth scale (simulate different distances)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            depth = depth * scale
            
        return image, depth, valid_mask
        
    def _get_dummy_sample(self):
        """Get a dummy sample when loading fails."""
        return {
            'image': torch.zeros(3, *self.img_size),
            'segmentation': torch.full(self.img_size, 255, dtype=torch.long),
            'depth': torch.zeros(self.img_size),
            'depth_valid': torch.zeros(self.img_size, dtype=torch.bool),
            'has_segmentation': torch.tensor(False, dtype=torch.bool),
            'has_depth': torch.tensor(False, dtype=torch.bool)
        }