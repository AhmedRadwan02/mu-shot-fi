import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import torch
import torch.nn as nn

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



class CNN2DBase(nn.Module):
    
    #ResNet-50 backbone adapted for CSI data.
    #Input:  (B,T,F)      -> reshaped to (B,1,T,F)
    #        (B,T,F,C<=2) -> reshaped to (B,C,T,F)
    #Output: (B,2048) feature vectors

    def __init__(self, var_x_shape=None, input_channels: int = 1, pretrained: bool = True):
        super(CNN2DBase, self).__init__()
        self.input_channels = input_channels

        # Load ResNet-50 backbone
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = torchvision.models.resnet18(pretrained=pretrained)

        # Adapt first conv layer to match input channels
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(input_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=False)
        with torch.no_grad():
            if pretrained and old_conv.weight.shape[1] == 3:
                # Convert RGB weights to grayscale
                w_rgb = old_conv.weight.data  # [64,3,7,7]
                alpha = torch.tensor([0.2989, 0.5870, 0.1140],
                                     device=w_rgb.device,
                                     dtype=w_rgb.dtype).view(1,3,1,1)
                w_gray = (w_rgb * alpha).sum(dim=1, keepdim=True)  # [64,1,7,7]
                if input_channels == 1:
                    new_conv.weight.copy_(w_gray)
                else:
                    new_conv.weight.copy_(w_gray.repeat(1, input_channels, 1, 1) / float(input_channels))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        backbone.conv1 = new_conv

        # Remove classifier head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # until global avgpool
        self.in_features = backbone.fc.in_features  # 2048 for ResNet-50

    def forward(self, var_input: torch.Tensor):
        # Handle input reshaping
        if var_input.dim() == 3:   # (B,T,F)
            var_input = var_input.unsqueeze(1)  # (B,1,T,F)
        elif var_input.dim() == 4: # (B,T,F,C)
            if var_input.size(-1) <= 2:
                var_input = var_input.permute(0, 3, 1, 2).contiguous()  # (B,C,T,F)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {var_input.dim()}D")

        if var_input.size(1) != self.input_channels:
            raise ValueError(f"Input has {var_input.size(1)} channels, expected {self.input_channels}")

        # Pass through ResNet backbone
        features = self.backbone(var_input)  # (B,2048,1,1)
        features = torch.flatten(features, 1)  # (B,2048)
        return features
        
"""
class CNN2DBase(nn.Module):
    def __init__(self, var_x_shape, input_channels=1):
        super(CNN2DBase, self).__init__()
        
        # Store input channels
        self.input_channels = input_channels
        
        # Print debug info about input shape
        print(f"CNN2DBase initialized with input_shape: {var_x_shape}, input_channels: {input_channels}")
        
        # Normalization layers - first layer uses input_channels
        self.layer_norm_0 = nn.BatchNorm2d(input_channels)
        self.layer_norm_1 = nn.BatchNorm2d(32)
        self.layer_norm_2 = nn.BatchNorm2d(64)
        self.layer_norm_3 = nn.BatchNorm2d(128)
        
        # Convolutional layers with adjusted kernel sizes for WiDAR data
        # Input will be (B, C, T, F) where T=10000, F=540
        self.layer_cnn_2d_0 = nn.Conv2d(in_channels=input_channels,
                                        out_channels=32,
                                        kernel_size=(5, 5),  # Smaller kernels
                                        stride=(2, 2),      # Smaller strides
                                        padding=(2, 2))     # Add padding
        
        self.layer_cnn_2d_1 = nn.Conv2d(in_channels=32, 
                                        out_channels=64,
                                        kernel_size=(3, 3),  # Smaller kernels
                                        stride=(2, 2),      # Smaller strides
                                        padding=(1, 1))     # Add padding
        
        self.layer_cnn_2d_2 = nn.Conv2d(in_channels=64, 
                                        out_channels=128,
                                        kernel_size=(3, 3),  # Smaller kernels
                                        stride=(2, 2),      # Smaller strides  
                                        padding=(1, 1))     # Add padding
        # Activation and regularization
        self.layer_leakyrelu = nn.LeakyReLU()
        self.layer_dropout = nn.Dropout(0.2)
        
        # Feature dimension after global average pooling
        self.in_features = 128
        
        # Apply initialization
        self.apply(init_weights)

    def forward(self, var_input):
        var_t = var_input
        
        # Handle different input shapes
        if var_t.dim() == 3:  # (B, T, F) - amplitude only
            var_t = torch.unsqueeze(var_t, dim=1)  # (B, 1, T, F)
        elif var_t.dim() == 4:  # (B, T, F, C) - amplitude + phase
            if var_t.size(-1) <= 2:  # Channel dimension
                var_t = var_t.permute(0, 3, 1, 2)  # (B, C, T, F)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {var_t.dim()}D")
        
        if var_t.size(1) != self.input_channels:
            raise ValueError(f"Input has {var_t.size(1)} channels, but model expects {self.input_channels}")
        
        var_t = self.layer_norm_0(var_t)
        var_t = self.layer_cnn_2d_0(var_t)
        # print(f"After conv1: {var_t.shape}")  # REMOVE
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)
        
        var_t = self.layer_norm_1(var_t)
        var_t = self.layer_cnn_2d_1(var_t)
        # print(f"After conv2: {var_t.shape}")  # REMOVE
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)
        
        var_t = self.layer_norm_2(var_t)
        var_t = self.layer_cnn_2d_2(var_t)
        # print(f"After conv3: {var_t.shape}")  # REMOVE
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)
        
        var_t = self.layer_norm_3(var_t)
        var_t = torch.mean(var_t, dim=(-2, -1))
        # print(f"After GAP: {var_t.shape}")  # REMOVE
        
        return var_t
"""
class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256):
        super(feat_bottleneck, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        
    def forward(self, x):
        x = self.bottleneck(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256):
        super(feat_classifier, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x

class CNN_2D(nn.Module):
    def __init__(self, var_x_shape, var_y_shape, bottleneck_dim=256, input_channels=1):
        super(CNN_2D, self).__init__()
        
        var_dim_output = var_y_shape[-1]
        
        # Initialize components with input_channels support
        self.base = CNN2DBase(var_x_shape, input_channels=input_channels)
        self.bottleneck = feat_bottleneck(self.base.in_features, bottleneck_dim)
        self.classifier = feat_classifier(var_dim_output, bottleneck_dim)
        
    def forward(self, var_input):
        # Base feature extraction
        features = self.base(var_input)
        
        # Bottleneck
        bottleneck_features = self.bottleneck(features)
        
        # Classification
        output = self.classifier(bottleneck_features)
        
        return output