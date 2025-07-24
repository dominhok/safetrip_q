# KITTI Depth Data Integration Guide for SafeTrip-Q

## Overview

The KITTI depth dataset provides high-quality depth annotations that can significantly enhance your SafeTrip-Q model's depth estimation capabilities. Here's how to integrate it effectively.

## KITTI Data Structure

```
data_depth_annotated/
├── train/
│   └── 2011_xx_xx_drive_xxxx_sync/
│       └── proj_depth/
│           ├── groundtruth/
│           │   ├── image_02/  # Left camera depth (main)
│           │   │   └── *.png  # uint16 depth maps
│           │   └── image_03/  # Right camera depth
│           └── velodyne_raw/
│               ├── image_02/  # Sparse LiDAR projections
│               └── image_03/
└── val/
    └── (same structure as train)
```

## Integration Strategy

### 1. **Mixed Dataset Training**
Since KITTI only has depth annotations (no segmentation), it perfectly complements SafeTrip-Q:
- SafeTrip-Q: Segmentation (Surface/Polygon) + Some depth
- KITTI: High-quality depth only

```python
# Dataset composition per batch
batch = {
    'safetrip_surface': 2,    # Segmentation only
    'safetrip_polygon': 2,    # Segmentation only  
    'safetrip_depth': 2,      # Depth only
    'kitti_depth': 2          # Depth only (higher quality)
}
```

### 2. **Data Format Differences**

#### SafeTrip-Q Depth:
- Source: Stereo disparity (GA-Net)
- Format: Disparity16 → depth conversion
- Resolution: 1920×1080
- Calibration: Per-folder

#### KITTI Depth:
- Source: LiDAR projection
- Format: uint16 PNG (depth in mm)
- Resolution: 1242×375 (typical)
- Calibration: Per-sequence

### 3. **Implementation Steps**

#### Step 1: Create KITTI Dataset Loader
```python
class KITTIDepthDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root = os.path.join(root_dir, split)
        self.samples = self._load_file_list()
        
    def _load_depth(self, depth_path):
        # KITTI depth is in mm, stored as uint16
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        return depth
```

#### Step 2: Modify Training Pipeline
```python
# In dataset_safetrip.py, add KITTI support
class CombinedDataset(Dataset):
    def __init__(self, safetrip_root, kitti_root, split='train'):
        self.safetrip_dataset = SafeTripDataset(safetrip_root, split)
        self.kitti_dataset = KITTIDepthDataset(kitti_root, split)
        
        # Calculate sampling weights
        # More KITTI samples since SafeTrip depth is limited
        self.dataset_weights = [0.7, 0.3]  # 70% SafeTrip, 30% KITTI
```

#### Step 3: Handle Resolution Differences
```python
def preprocess_kitti(self, image, depth):
    # KITTI: 1242×375 → 512×512 (match SafeTrip training size)
    # Option 1: Resize directly
    image = F.interpolate(image, size=(512, 512), mode='bilinear')
    depth = F.interpolate(depth, size=(512, 512), mode='nearest')
    
    # Option 2: Crop and resize (preserve aspect ratio)
    # Crop to 1216×352 (standard KITTI crop)
    # Then resize to 512×512
```

### 4. **Training Benefits**

1. **Better Depth Generalization**: KITTI's outdoor driving scenes complement SafeTrip's sidewalk scenarios
2. **Higher Quality Depth**: LiDAR-based ground truth vs stereo matching
3. **More Depth Samples**: 86k additional training samples
4. **Domain Adaptation**: Learn features that transfer between datasets

### 5. **Loss Weighting Strategy**

```python
# Adaptive weighting based on data source
if source == 'kitti':
    # KITTI has more reliable depth, weight it higher
    depth_weight = 1.2
elif source == 'safetrip_depth':
    depth_weight = 1.0
```

### 6. **Validation Strategy**

- Keep SafeTrip-Q validation separate for sidewalk performance
- Add KITTI validation for general depth performance
- Report metrics separately:
  ```
  SafeTrip mIoU: XX.X%
  SafeTrip Depth RMSE: X.XX
  KITTI Depth RMSE: X.XX
  ```

## Quick Implementation

1. **Download KITTI Depth Dataset**:
   ```bash
   # Download data_depth_annotated.zip from KITTI website
   # Extract to data/kitti/
   ```

2. **Update Configuration**:
   ```yaml
   dataset:
     safetrip_root: ../data/
     kitti_root: ../data/kitti/data_depth_annotated/
     use_kitti: true
     kitti_weight: 0.3  # 30% of batches from KITTI
   ```

3. **Modify Dataset Loader**:
   - Add KITTI dataset class
   - Implement mixed batch sampling
   - Handle different resolutions/formats

## Expected Improvements

- **Depth RMSE**: 15-25% improvement expected
- **Depth Generalization**: Better performance on unseen scenarios
- **Training Stability**: More diverse depth samples
- **No Segmentation Impact**: KITTI doesn't affect segmentation task

## Notes

- KITTI uses different camera intrinsics - consider normalizing depth
- KITTI depth is sparse (5% density) but high quality
- Can use KITTI RGB images for self-supervised learning later
- Consider depth completion techniques for sparse KITTI data