# SafeTrip-Q + KITTI Depth Training Guide

## Overview

This project implements Hydranet for multi-task learning with:
- **Segmentation**: SafeTrip-Q dataset (Surface + Polygon classes)
- **Depth**: KITTI depth dataset (with preprocessing for sparse LiDAR data)

## Key Features

### 1. KITTI Sparse Depth Preprocessing

KITTI LiDAR data is extremely sparse (~5% valid pixels). We implement three preprocessing methods:

- **`fast`**: Simple morphological dilation (real-time, ~30ms)
- **`balanced`**: Multi-scale processing with bilateral filtering (recommended, ~100ms)
- **`accurate`**: Iterative refinement with confidence weighting (best quality, ~300ms)

### 2. Confidence-Aware Training

Instead of binary validity masks, we use confidence maps that indicate reliability:
- Original LiDAR points: confidence = 1.0
- Nearby interpolated: confidence = 0.8-0.9
- Far extrapolated: confidence = 0.5-0.7

## Quick Start

```bash
# Prepare data
python scripts/prepare_safetrip_data.py

# Train with KITTI depth preprocessing
python train.py \
    --config configs/cfg_safetrip.yaml \
    --project safetrip-kitti \
    --run-name balanced-preprocessing
```

## Configuration Options

### In `dataset_kitti.py`:
```python
KITTIDepthDataset(
    preprocess_depth=True,      # Enable preprocessing
    preprocess_method='balanced' # 'fast', 'balanced', or 'accurate'
)
```

### Training Strategy:
1. **Stage 1**: Train with `balanced` preprocessing (good speed/quality trade-off)
2. **Stage 2**: Fine-tune with `accurate` preprocessing (if needed)

## Results

With preprocessing, expect:
- **Depth RMSE**: ~20% improvement over raw sparse data
- **Training stability**: Reduced NaN losses
- **Convergence speed**: 2-3x faster

## Technical Details

### Morphological Operations
- **Near range (0.1-15m)**: 5×5 elliptical kernel
- **Medium range (15-30m)**: 7×7 elliptical kernel  
- **Far range (30-80m)**: 9×9 elliptical kernel

### Why This Works
1. KITTI sparse points are accurate but incomplete
2. Morphological operations preserve edges while filling gaps
3. Multi-scale approach handles depth-dependent sparsity
4. Confidence weighting prevents overconfident predictions

## Monitoring

Key metrics to watch:
- `val/depth_rmse`: Should decrease steadily
- `val/depth_delta_1`: Should increase (% within 1.25x of GT)
- Loss should not have NaN values

## Tips

1. Start with `balanced` method for most use cases
2. Use `fast` for quick experiments
3. Use `accurate` for final model training
4. Monitor confidence maps in wandb to ensure quality

## Common Issues

1. **Still getting NaN losses?**
   - Check max depth clipping (80m for KITTI)
   - Ensure gradient clipping is enabled

2. **Depth not improving?**
   - Try increasing KITTI ratio in mixed dataset
   - Check if preprocessing is actually enabled

3. **Training too slow?**
   - Switch to `fast` preprocessing
   - Reduce image size temporarily