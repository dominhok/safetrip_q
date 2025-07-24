# Hydranet for SafeTrip-Q Dataset

Implementation of multi-task learning model for the SafeTrip-Q sidewalk dataset, supporting joint semantic segmentation and depth estimation with asymmetric annotations.

Based on the paper [Real Time Joint Semantic Segmentation & Depth Estimation Using Asymmetric Annotations](https://arxiv.org/pdf/1809.04766.pdf).

![](assets/architecture.png)

## 🚀 New Features

- **Knowledge Distillation**: SegFormer teacher model for pseudo-labeling depth samples
- **Normalized Depth**: Depth values normalized to [0, 1] range with sigmoid activation
- **Improved Training**: Real-time WandB logging and better loss balancing

## Overview

The SafeTrip-Q dataset provides three types of asymmetric annotations:
- **Surface**: Road surface segmentation (6 classes) - segmentation only
- **Polygon**: Obstacle segmentation (18 classes after filtering) - segmentation only  
- **Depth**: Depth estimation data - depth only

Total: 24 classes for unified segmentation task.

## Architecture

The model uses a Light-Weight RefineNet architecture built on MobileNet-v2:
- **Encoder**: MobileNet-v2 (pretrained on ImageNet)
- **Decoder**: Light-Weight RefineNet with Chained Residual Pooling (CRP) blocks
- **Task Heads**: Separate branches for segmentation (24 classes) and depth (1 channel)
- **Parameters**: ~3.06M (optimized for edge deployment)

## Project Structure

```
joint-segmentation-depth/
├── configs/              # Configuration files
│   └── cfg_safetrip.yaml
├── model/                # Model architecture
│   ├── hydranet.py       # Original Hydranet
│   └── hydranet_safetrip.py  # SafeTrip-Q adapted version
├── utils/                # Utility functions
│   ├── config.py         # Configuration loader
│   ├── depth_converter.py    # Disparity to depth conversion
│   ├── metrics.py        # Evaluation metrics
│   └── xml_parser.py     # CVAT XML annotation parser
├── scripts/              # Training and data scripts
│   ├── prepare_safetrip_data.py     # Data preparation
│   ├── generate_pseudo_labels_depth.py  # Generate pseudo labels
│   └── fix_checkpoint.py            # Fix checkpoint keys
├── train.py              # Main training script with pseudo labels
├── train_teacher.py      # Train SegFormer teacher model
├── dataset_safetrip.py   # SafeTrip-Q dataset loader
├── losses.py             # Loss functions
└── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n hydranet python=3.8
conda activate hydranet

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
cd scripts
python prepare_safetrip_data.py
```

This creates:
- `data/cmap_safetrip_24.npy`: Colormap for visualization
- `data/class_weights_24.npy`: Class weights for balanced training
- `data/class_info.json`: Class names and mappings

### 3. Knowledge Distillation (Optional but Recommended)

Train a teacher model and generate pseudo labels for depth samples:

```bash
# Step 1: Train SegFormer teacher model
python train_teacher.py --config configs/cfg_teacher.yaml --epochs 20

# Step 2: Generate pseudo labels for depth samples  
cd scripts
python generate_pseudo_labels_depth.py \
    --checkpoint ../checkpoints/segformer_best.pth \
    --data-root ../data \
    --output-path ../data/depth_pseudo_labels.pkl
```

### 4. Train Hydranet

```bash
python train.py \
    --config configs/cfg_safetrip.yaml \
    --project safetrip-hydranet \
    --run-name my-experiment \
    --epochs 100
```

Key features:
- Asymmetric annotation handling
- Normalized depth prediction [0, 1]
- Dynamic loss balancing
- Mixed precision training
- Real-time WandB logging
- Pseudo label integration

### 5. Inference

Run inference on a single image:
```bash
python inference.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --config configs/cfg_safetrip.yaml
```

### 6. Evaluate

```bash
python evaluate.py \
    --config configs/cfg_safetrip.yaml \
    --checkpoint checkpoints/best_model.pth \
    --save-predictions
```

## Configuration

Key configuration options in `configs/cfg_safetrip.yaml`:
- `num_classes`: 24 (filtered from original 35)
- `batch_size`: 8
- `learning_rate`: 0.001
- `epochs`: 100
- `use_uncertainty_weighting`: true

## Class List (24 classes)

**Surface (0-5)**:
- sidewalk, braille_guide_blocks, roadway, alley, bike_lane, caution_zone

**Polygon - Moving objects (6-13)**:
- person, car, bus, truck, bicycle, motorcycle, stroller, scooter

**Polygon - Fixed objects (14-23)**:
- tree_trunk, potted_plant, pole, bench, bollard, barricade, fire_hydrant, kiosk, power_controller, traffic_light_controller

## Loss Functions

- **Segmentation**: Weighted Cross-Entropy with class balancing
- **Depth**: Inverse Huber Loss (δ=0.1 for normalized depth)
- **Multi-task**: Fixed weights (0.5 each) or dynamic uncertainty weighting

For uncertainty weighting (Kendall et al., 2018):
Total loss: L = (1/2σ₁²)L_seg + log(σ₁) + (1/2σ₂²)L_depth + log(σ₂)

## Performance

- **Input**: 512×512 RGB (downsampled from 1920×1080)
- **Inference Time**: ~50ms (PyTorch), ~30ms (TensorRT FP16)
- **Memory Usage**: ~600MB GPU memory
- **Target**: 20-30 FPS on Jetson Orin Nano

## Implementation Details

### Depth Normalization
- Depth values are normalized to [0, 1] range (80m max depth)
- Model uses sigmoid activation for depth output
- InverseHuberLoss with δ=0.1 for normalized values

### Pseudo-Labeling Pipeline
1. Train SegFormer (mit-b2) on Surface + Polygon data
2. Generate pseudo segmentation labels for Depth samples
3. Filter predictions by confidence (threshold: 0.7)
4. Use pseudo labels during Hydranet training

### Data Handling
- Classes with <100 annotations were removed (parking_meter, cat, dog, wheelchair)
- Non-obstacle classes removed (traffic_light, traffic_sign, stop, movable_signage)
- Ambiguous classes removed (carrier, chair, table)
- Each Depth folder has unique calibration parameters for depth conversion

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{nekrasov2019hydranet,
  title={Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations},
  author={Nekrasov, Vladimir and others},
  booktitle={ICRA},
  year={2019}
}
```