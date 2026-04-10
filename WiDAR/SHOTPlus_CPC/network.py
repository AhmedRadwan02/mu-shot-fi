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



class SelfAttention2D(nn.Module):
    """
    Self-attention module for 2D feature maps.
    Operates on spatial dimensions (H, W) of feature maps.
    """
    def __init__(self, in_channels, reduction=8):
        super(SelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)
        
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Input: x of shape (B, C, H, W)
        Output: attention-refined features of shape (B, C, H, W)
        """
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        # Query: (B, C', H, W) -> (B, C', H*W) -> (B, H*W, C')
        query = self.query_conv(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        
        # Key: (B, C', H, W) -> (B, C', H*W)
        key = self.key_conv(x).view(batch_size, self.inter_channels, -1)
        
        # Value: (B, C, H, W) -> (B, C, H*W)
        value = self.value_conv(x).view(batch_size, C, -1)
        
        # Attention map: (B, H*W, C') x (B, C', H*W) -> (B, H*W, H*W)
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # Apply attention to value: (B, C, H*W) x (B, H*W, H*W) -> (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        
        # Reshape back: (B, C, H*W) -> (B, C, H, W)
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class CNN2DBase(nn.Module):
    """
    ResNet-18 backbone adapted for CSI data with optional self-attention.
    Input:  (B,T,F)      -> reshaped to (B,1,T,F)
            (B,T,F,C<=2) -> reshaped to (B,C,T,F)
    Output: (B,512) feature vectors (ResNet-18 has 512 features, not 2048)
    """
    def __init__(self, var_x_shape=None, input_channels: int = 1, 
                 pretrained: bool = True, use_attention: bool = False):
        super(CNN2DBase, self).__init__()
        self.input_channels = input_channels
        self.use_attention = use_attention

        # Load ResNet-18 backbone
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

        # Extract feature extractor (before avgpool and fc)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # Stop before avgpool
        
        # Get the output channels from the last layer
        # For ResNet-18, this is 512
        self.in_features = backbone.fc.in_features  # 512 for ResNet-18
        
        # Add self-attention before global pooling (optional)
        if self.use_attention:
            self.self_attention = SelfAttention2D(self.in_features, reduction=8)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, var_input: torch.Tensor, return_attention_map: bool = False):
        
        if var_input.dim() == 3:   # (B,T,F)
            var_input = var_input.unsqueeze(1)  # (B,1,T,F)
            var_input = var_input.permute(0, 1, 3, 2).contiguous()  # (B,1,F,T)
            
        elif var_input.dim() == 4: # (B,T,F,C)
            if var_input.size(-1) <= 2:
                var_input = var_input.permute(0, 3, 2, 1).contiguous()  # (B,C,F,T) - ORIGINAL
                
        else:
            raise ValueError(f"Expected 3D or 4D input, got {var_input.dim()}D")
        
        if var_input.size(1) != self.input_channels:
            raise ValueError(f"Input has {var_input.size(1)} channels, expected {self.input_channels}")
        
        # Pass through feature extractor (before global pooling)
        features = self.feature_extractor(var_input)  # (B, 512, H, W)
        
        # Apply self-attention if enabled
        if self.use_attention:
            features_before_attn = features
            features = self.self_attention(features)  # (B, 512, H, W)
            
            if return_attention_map:
                # Can optionally return attention weights for visualization
                attention_weights = self.self_attention.gamma.item()
        
        # Global average pooling
        features = self.global_pool(features)  # (B, 512, 1, 1)
        features = torch.flatten(features, 1)  # (B, 512)
        
        return features

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256):
        super(feat_bottleneck, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        
    def forward(self, x):
        x = self.bottleneck(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256,type=None):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
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