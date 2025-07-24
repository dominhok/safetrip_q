import logging
import torch
from torch import nn
from torchvision import models

from model.utils import (CRPBlock, conv1x1, conv3x3)


class HydranetSafeTrip(nn.Module):
    """
    Hydranet model adapted for SafeTrip-Q dataset.
    Uses pretrained MobileNet-v2 backbone from torchvision.
    """
    
    def __init__(self, cfg):        
        super().__init__()
        self.cfg = cfg
        self.num_classes = self.cfg['model']['num_classes']
        
        # Load pretrained MobileNet-v2
        self.define_mobilenet_pretrained()
        
        # Define Light-Weight RefineNet decoder
        self.define_lightweight_refinenet()
        
        logging.info(f'HydranetSafeTrip initialized with {self.num_classes} classes')
        
    def define_mobilenet_pretrained(self):
        """Load pretrained MobileNet-v2 and extract layers."""
        # Load pretrained MobileNet-v2 using new API
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Extract features from MobileNet-v2
        features = mobilenet.features
        
        # Map MobileNet-v2 layers to Hydranet layer structure
        # Layer 1: Initial conv (channels: 3 -> 32, stride 2)
        self.layer1 = nn.Sequential(features[0])  # ConvBNReLU
        
        # Layer 2: First inverted residual block (channels: 32 -> 16, stride 1)
        self.layer2 = nn.Sequential(features[1])
        
        # Layer 3: Inverted residual blocks (channels: 16 -> 24, stride 2)
        self.layer3 = nn.Sequential(*features[2:4])
        
        # Layer 4: Inverted residual blocks (channels: 24 -> 32, stride 2)
        self.layer4 = nn.Sequential(*features[4:7])
        
        # Layer 5: Inverted residual blocks (channels: 32 -> 64, stride 2)
        self.layer5 = nn.Sequential(*features[7:11])
        
        # Layer 6: Inverted residual blocks (channels: 64 -> 96, stride 1)
        self.layer6 = nn.Sequential(*features[11:14])
        
        # Layer 7: Inverted residual blocks (channels: 96 -> 160, stride 2)
        self.layer7 = nn.Sequential(*features[14:17])
        
        # Layer 8: Last inverted residual block (channels: 160 -> 320, stride 1)
        # Note: torchvision MobileNet-v2 has an extra Conv2d layer (features[18]) that outputs 1280 channels
        # We only need up to the 320 channel output from features[17]
        self.layer8 = nn.Sequential(features[17])
        
        print("Loaded pretrained MobileNet-v2 backbone from torchvision")
        
    def make_crp(self, in_planes, out_planes, stages, groups=False):
        """Create Chained Residual Pooling block."""
        layers = [CRPBlock(in_planes, out_planes, stages, groups=groups)]
        return nn.Sequential(*layers)
        
    def define_lightweight_refinenet(self):
        """Define Light-Weight RefineNet decoder."""
        # Channel adaptation layers
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        
        # CRP blocks
        self.crp4 = self.make_crp(256, 256, 4, groups=False)
        self.crp3 = self.make_crp(256, 256, 4, groups=False)
        self.crp2 = self.make_crp(256, 256, 4, groups=False)
        self.crp1 = self.make_crp(256, 256, 4, groups=True)
        
        # Adaptation convolutions
        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)
        
        # Task-specific heads
        # Segmentation head
        self.pre_segm = conv1x1(256, 256, groups=256, bias=False)
        self.segm = conv3x3(256, self.num_classes, bias=True)
        
        # Depth head
        self.pre_depth = conv1x1(256, 256, groups=256, bias=False)
        self.depth = conv3x3(256, 1, bias=True)
        
        # Activation
        self.relu = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        """Forward pass through the network."""
        # MobileNet-v2 encoder
        x = self.layer1(x)      # 32 channels, x/2
        x = self.layer2(x)      # 16 channels, x/2
        l3 = self.layer3(x)     # 24 channels, x/4
        l4 = self.layer4(l3)    # 32 channels, x/8
        l5 = self.layer5(l4)    # 64 channels, x/16
        l6 = self.layer6(l5)    # 96 channels, x/16
        l7 = self.layer7(l6)    # 160 channels, x/32
        l8 = self.layer8(l7)    # 320 channels, x/32
        
        # Light-Weight RefineNet decoder
        # Process deepest layer
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.functional.interpolate(l7, size=l6.size()[2:], mode='bilinear', align_corners=False)
        
        # Merge with l6 and l5
        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.functional.interpolate(l5, size=l4.size()[2:], mode='bilinear', align_corners=False)
        
        # Merge with l4
        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.functional.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=False)
        
        # Merge with l3
        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)
        
        # Task-specific heads
        # Segmentation output
        out_segm = self.pre_segm(l3)
        out_segm = self.relu(out_segm)
        out_segm = self.segm(out_segm)
        
        # Depth output
        out_depth = self.pre_depth(l3)
        out_depth = self.relu(out_depth)
        out_depth = self.depth(out_depth)
        # Apply sigmoid to ensure depth output is in [0, 1] range
        out_depth = torch.sigmoid(out_depth)
        
        # Upsample outputs to input size
        # l3 is at 1/4 resolution, so we need to upsample by 4x
        out_segm = nn.functional.interpolate(out_segm, scale_factor=4, mode='bilinear', align_corners=False)
        out_depth = nn.functional.interpolate(out_depth, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out_segm, out_depth