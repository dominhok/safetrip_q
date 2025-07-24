# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the SafeTrip-Q Hydranet implementation.

## Project Overview

This repository implements a multi-task learning system for training on the SafeTrip-Q sidewalk dataset, specifically designed for joint semantic segmentation and depth estimation using the Hydranet architecture. The dataset provides asymmetric annotations across three modalities:

- **Surface**: Road surface segmentation (6 classes) - segmentation only
- **Polygon**: Obstacle segmentation (18 classes after filtering) - segmentation only  
- **Depth**: Depth estimation data - depth only

Total: 24 unified segmentation classes (reduced from original 35 by removing rare and non-obstacle classes).

## Data Structure

```
data/
├── Depth/        # Depth estimation data
│   ├── Depth_XXX/  (3-digit folders)
│   │   ├── Depth_XXX.conf  # Calibration parameters
│   │   └── [8 image types per sample]:
│   │       - *_L.png (Raw_Left, 1920x1080) - RGB input
│   │       - *_R.png (Raw_Right, 1920x1080)
│   │       - *_left.png (Crop_Left, 1920x592)
│   │       - *_right.png (Crop_Right, 1920x592)
│   │       - *_disp16.png (Disparity16) - depth GT source
│   │       - *_disp.png (Disparity visualization)
│   │       - *_confidence_save.png (0-255 confidence)
│   │       - *_confidence.png (Binary mask)
├── Polygon/      # Obstacle segmentation
│   ├── Polygon_XXXX/  (4-digit folders)
│   │   ├── *.jpg images
│   │   └── *.xml (CVAT format annotations)
└── Surface/      # Surface segmentation
    ├── Surface_XXX/  (3-digit folders)
        ├── *.jpg images
        └── *.xml (CVAT format annotations)
```

## Key Implementation Tasks

### 1. Data Loader Development
- Parse XML annotations for Surface/Polygon segmentation
- Convert polygon coordinates to segmentation masks
- Load depth data from Disparity16 files
- Apply confidence masks for reliable depth values
- Handle asymmetric annotations (some images have only segmentation, others only depth)

### 2. Depth Conversion
Each Depth folder has unique calibration parameters. Use:
```
depth = baseline × focal_length / disparity
```
Where baseline and focal_length come from the .conf files.

### 3. Class Mapping

**Surface Classes (6)**:
- sidewalk (with attributes: blocks, cement, urethane, asphalt, soil_stone, damaged, other)
- braille_guide_blocks (normal, damaged)
- roadway (normal, crosswalk)
- alley (normal, crosswalk, speed_bump, damaged)
- bike_lane
- caution_zone (stairs, manhole, grating, repair_zone, tree_zone)

**Polygon Classes - Moving Objects (8)**:
- person, car, bus, truck, bicycle, motorcycle, stroller, scooter

**Polygon Classes - Fixed Objects (10)**:
- tree_trunk, potted_plant, pole, bench, bollard, barricade, fire_hydrant, kiosk, power_controller, traffic_light_controller

**Removed Classes (from original 35)**:
- Rare (<100 annotations): parking_meter, cat, dog, wheelchair
- Non-obstacles: traffic_light, traffic_sign, stop, movable_signage
- Ambiguous: carrier, chair, table

### 4. Training Strategy
The Hydranet architecture already supports asymmetric annotations. Implement:
- Loss masking: Only compute losses for available annotations
- Mixed batch sampling from all three data types
- Multi-task loss balancing

### 5. Loss Functions and Balancing

#### Segmentation Loss
- **CrossEntropyLoss** with ignore_index=255 for missing annotations
- **Class Weights**: Based on class frequency to handle imbalance
  - Calculated from actual dataset statistics
  - 24 class weights for unified Surface + Polygon classes

#### Depth Loss  
- **Inverse Huber Loss**: Robust to outliers in depth estimation
  - For small errors: 0.5 * x² / δ
  - For large errors: |x| - 0.5 * δ
  - δ = 1.0 (tunable parameter)

#### Dynamic Loss Balancing
- **Uncertainty Weighting** (Kendall et al., CVPR 2018)
  - L_total = (1/2σ₁²)L_seg + log(σ₁) + (1/2σ₂²)L_depth + log(σ₂)
  - σ₁, σ₂ are learnable parameters representing task uncertainty
  - Automatically balances between segmentation and depth tasks

#### Training Configuration
```yaml
training:
  batch_size: 8
  learning_rate: 0.001
  optimizer: AdamW
  weight_decay: 0.0001
  lr_scheduler: cosine  # Cosine annealing
  warmup_epochs: 5
  lambda_seg: 0.5  # Initial weight (adjusted dynamically)
  lambda_depth: 0.5  # Initial weight (adjusted dynamically)
```

## Common Commands

```bash
# Install dependencies
pip install torch torchvision opencv-python pillow lxml numpy wandb pyyaml

# Prepare data (creates colormap, class info, weights)
cd joint-segmentation-depth/scripts
python prepare_safetrip_data.py

# Train with wandb logging
python train_safetrip_wandb.py --config ../configs/cfg_safetrip.yaml --project safetrip-hydranet --run-name experiment-name

# Inference on single image
python inference_safetrip.py --image path/to/image.jpg --checkpoint path/to/model.pth

# Evaluate model
python evaluate.py --config ../configs/cfg_safetrip.yaml --checkpoint path/to/model.pth
```

## Architecture Notes

When implementing, ensure:
- Input resolution: 1920x1080 (native) or resize as needed
- Output heads: 24 classes for unified segmentation (6 Surface + 18 Polygon) + 1 depth channel
- Loss functions: Weighted Cross-Entropy for segmentation, Inverse Huber for depth
- Handle missing annotations gracefully in the loss computation

### Asymmetric Data Handling Details

#### Batch Composition
- Each batch contains samples from all three data types (Surface, Polygon, Depth)
- DataLoader with custom collate_fn to handle heterogeneous samples
- Batch tensors include `has_segmentation` and `has_depth` boolean masks

#### Loss Computation Flow
```python
# Pseudo-code for asymmetric loss computation
if batch['has_segmentation'].any():
    seg_indices = batch['has_segmentation'].nonzero()
    seg_loss = CrossEntropyLoss(pred_seg[seg_indices], gt_seg[seg_indices])
    
if batch['has_depth'].any():
    depth_indices = batch['has_depth'].nonzero()
    depth_loss = InverseHuberLoss(pred_depth[depth_indices], gt_depth[depth_indices])
    
total_loss = loss_balancer(seg_loss, depth_loss)  # Dynamic balancing
```

#### Data Augmentation Strategy
- Consistent augmentation across RGB image and annotations
- Geometric: Random crop, flip, rotation (applied to both image and masks)
- Photometric: Color jitter, brightness (applied to RGB only)
- Depth-specific: Scale augmentation to simulate distance variations

## Important Considerations

1. **Memory efficiency**: Images are high-resolution (1920x1080), consider downsampling
2. **Class imbalance**: Use weighted losses based on class frequencies
3. **Depth reliability**: Always apply confidence masks when using depth data
4. **XML parsing**: Handle z_order for overlapping polygons in annotation files

## Performance Optimization Strategies

### Memory Management
- **Input Size**: Downsample to 512x512 for training (from 1920x1080)
- **Gradient Accumulation**: Effective batch size through accumulation steps
- **Mixed Precision Training**: FP16 with automatic mixed precision (AMP)

### Computational Efficiency  
- **Backbone Freezing**: Option to freeze early MobileNet-v2 layers
- **Cached Preprocessing**: Pre-compute and cache normalized images
- **Multi-worker DataLoader**: Parallel data loading (num_workers=4)

### Jetson Deployment Optimization
- **TensorRT Conversion**: INT8/FP16 quantization for inference
- **Dynamic Batch Size**: Adjust based on available memory
- **Memory Pooling**: Reuse allocated tensors to reduce fragmentation

## Hydranet Model Analysis

### Architecture Overview
The Hydranet model from the GitHub repository uses a Light-Weight RefineNet architecture built on MobileNet-v2:

1. **Encoder (MobileNet-v2)**:
   - Initial conv: 3→32 channels, stride 2
   - 7 layers of Inverted Residual Blocks with expansion factors
   - Progressive downsampling to 1/32 of input size
   - Channel progression: 16→24→32→64→96→160→320

2. **Decoder (Light-Weight RefineNet)**:
   - Uses Chained Residual Pooling (CRP) blocks
   - Skip connections from encoder layers
   - Progressive upsampling with bilinear interpolation
   - All intermediate features normalized to 256 channels

3. **Task-Specific Heads**:
   - Segmentation: 256→256 (depthwise)→num_classes (3×3 conv)
   - Depth: 256→256 (depthwise)→1 (3×3 conv)
   - Both use ReLU6 activation

### Key Implementation Details

1. **Model Configuration** (from config/cfg.yaml):
   ```yaml
   model:
     num_classes: 6  # Need to change to 24 for SafeTrip-Q
     device: cuda
     encoder_config: # [expansion, out_channels, repeats, stride]
       - [1, 16, 1, 1]
       - [6, 24, 2, 2]
       - [6, 32, 3, 2]
       - [6, 64, 4, 2]
       - [6, 96, 3, 1]
       - [6, 160, 3, 2]
       - [6, 320, 1, 1]
   ```

2. **Loss Functions**:
   - Segmentation: Softmax Cross-Entropy (implicit in the paper)
   - Depth: Inverse Huber Loss
   - Combined loss: λ × seg_loss + (1-λ) × depth_loss (λ=0.5)

3. **Preprocessing**:
   - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Input normalized to [0,1] then standardized

### Modifying Existing Hydranet for SafeTrip-Q

Instead of creating new code from scratch, modify the existing implementation:

#### 1. Modify Configuration (joint-segmentation-depth/config/cfg.yaml)
```yaml
general:
  environment: safetrip  # Change from 'residential'
  input_dir: ../data/    # Point to SafeTrip-Q data
  output_dir: ../output/
  cmap: ../data/cmap_safetrip_24.npy  # Create new colormap for 24 classes

model:
  num_classes: 24  # Change from 6 to 24 (6 Surface + 18 Polygon)
  device: cuda
  weights: ../data/weights/hydranet_safetrip.ckpt  # New weights path
```

#### 2. Create SafeTrip-Q Dataset Loader
Create `joint-segmentation-depth/dataset_safetrip.py`:
```python
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

class SafeTripDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=(512, 512)):
        self.data_root = data_root
        self.img_size = img_size
        self.samples = []
        
        # Collect all samples from three data types
        self._load_depth_samples()
        self._load_surface_samples()
        self._load_polygon_samples()
        
    def _parse_cvat_xml(self, xml_path):
        # Parse CVAT format XML for Surface/Polygon annotations
        tree = ET.parse(xml_path)
        # Implementation details...
        
    def _load_depth_from_disparity(self, disp_path, conf_path, calib):
        # Convert disparity to depth using calibration
        # depth = baseline * focal_length / disparity
```

#### 3. Modify Training Script
Create `joint-segmentation-depth/train_safetrip.py` based on main.py:
```python
import torch
import torch.nn as nn
from model import Hydranet
from dataset_safetrip import SafeTripDataset
from torch.utils.data import DataLoader

def compute_loss(pred_seg, pred_depth, gt_seg, gt_depth, has_seg, has_depth):
    """Asymmetric loss for handling missing annotations"""
    seg_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    depth_criterion = InverseHuberLoss()
    
    total_loss = 0
    if has_seg and gt_seg is not None:
        seg_loss = seg_criterion(pred_seg, gt_seg)
        total_loss += 0.5 * seg_loss
        
    if has_depth and gt_depth is not None:
        valid_mask = gt_depth > 0  # Use confidence mask
        if valid_mask.any():
            depth_loss = depth_criterion(pred_depth[valid_mask], gt_depth[valid_mask])
            total_loss += 0.5 * depth_loss
            
    return total_loss

def train():
    cfg = load_config('config/cfg.yaml')
    cfg['model']['num_classes'] = 35  # Override for SafeTrip-Q
    
    # Initialize model with modified config
    model = Hydranet(cfg)
    # Remove pretrained weights loading for training from scratch
    model.load_weights = lambda x: None
    
    # Create dataset and dataloader
    dataset = SafeTripDataset('../data/', split='train')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Training loop...
```

#### 4. Minimal Required Modifications

**In joint-segmentation-depth/model/hydranet.py:**
- Line 19: `self.num_classes = self.cfg['model']['num_classes']` already supports changing classes
- Line 65: `self.segm = conv3x3(256, self.num_classes, bias=True)` automatically adapts

**Create these new files in joint-segmentation-depth/:**
1. `dataset_safetrip.py` - SafeTrip-Q specific data loader
2. `train_safetrip.py` - Training script with asymmetric loss
3. `utils/xml_parser.py` - CVAT XML parsing utilities
4. `utils/depth_converter.py` - Disparity to depth conversion

**Create data preparation script:**
```python
# prepare_safetrip_data.py
def create_class_mapping():
    """Create unified class mapping for Surface + Polygon classes"""
    surface_classes = ['sidewalk', 'braille_guide_blocks', 'roadway', 
                      'alley', 'bike_lane', 'caution_zone']
    polygon_classes = ['person', 'car', 'bus', ...] # all 18 classes
    
    class_to_id = {cls: idx for idx, cls in enumerate(surface_classes + polygon_classes)}
    return class_to_id

def create_colormap():
    """Generate colormap for 35 classes"""
    cmap = create_colormap(24)  # Create distinct colors
    np.save('../data/cmap_safetrip_24.npy', cmap)
```

#### 5. Quick Start Commands
```bash
cd joint-segmentation-depth

# Prepare data
python prepare_safetrip_data.py

# Modify config
vim config/cfg.yaml  # Update num_classes to 24

# Train model
python train_safetrip.py --epochs 100 --batch_size 8

# Test inference
python main.py  # Will use modified config automatically
```

This approach reuses 90% of the existing code and only adds SafeTrip-Q specific components!

## Architecture & Methodology

### Based on Papers
1. **"Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations"** (ICRA 2019)
   - Asymmetric annotation handling
   - Single model for both segmentation and depth

2. **"Light-Weight RefineNet for Real-Time Semantic Segmentation"** (BMVC 2018)
   - Lightweight version of RefineNet
   - CRP (Chained Residual Pooling) blocks

3. **"MobileNetV2: Inverted Residuals and Linear Bottlenecks"** (CVPR 2018)
   - Inverted Residual Block structure
   - Linear Bottleneck design

### Network Structure
```
MobileNet-v2 Encoder → Light-Weight RefineNet Decoder → Multi-task Heads
```

- **Parameters**: ~3.06M (optimized for edge deployment)
- **FLOPs**: ~4.18G @ 512×512
- **Backbone**: ImageNet pretrained MobileNet-v2

## Training Configuration Details

### Loss Balancing
- Initial weights: λ_seg = 0.5, λ_depth = 0.5
- Dynamic adjustment using learnable uncertainty parameters
- Class weights calculated from dataset frequency

### Data Split
- Total samples: ~40,203
- Train: 80% (32,162 samples)
- Validation: 20% (8,041 samples)

### Evaluation Metrics
- **Segmentation**: mIoU, pixel accuracy, per-class IoU
- **Depth**: RMSE, MAE, δ < 1.25 accuracy

## Important Implementation Notes

1. **Asymmetric Handling**: The Hydranet architecture is specifically designed for asymmetric annotations. No additional modifications needed.

2. **Depth Conversion**: Each Depth folder has unique calibration:
   ```python
   depth = baseline × focal_length / disparity
   ```

3. **XML Parsing**: Handle z_order for overlapping polygons, convert polygon points to masks using PIL ImageDraw

4. **Class Balancing**: Use weighted cross-entropy based on class frequency to handle extreme imbalance

5. **Memory Efficiency**: Original 1920x1080 images are downsampled to 512x512 for training