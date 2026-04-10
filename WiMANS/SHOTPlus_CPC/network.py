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
    def __init__(self, var_x_shape):
        super(CNN2DBase, self).__init__()

        # Normalization layers
        self.layer_norm_0 = nn.BatchNorm2d(1)
        self.layer_norm_1 = nn.BatchNorm2d(32)
        self.layer_norm_2 = nn.BatchNorm2d(64)
        self.layer_norm_3 = nn.BatchNorm2d(128)

        # Convolutional layers
        self.layer_cnn_2d_0 = nn.Conv2d(in_channels=1,
                                        out_channels=32,
                                        kernel_size=(27, 27),
                                        stride=(7, 7))

        self.layer_cnn_2d_1 = nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=(15, 15),
                                        stride=(3, 3))

        self.layer_cnn_2d_2 = nn.Conv2d(in_channels=64,
                                        out_channels=128,
                                        kernel_size=(7, 7),
                                        stride=(1, 1))

        # Activation and regularization
        self.layer_leakyrelu = nn.LeakyReLU()
        self.layer_dropout = nn.Dropout(0.2)

        # Feature dimension after global average pooling
        self.in_features = 128

        # Apply initialization
        self.apply(init_weights)
    
    def forward(self, var_input):
        var_t = var_input
        var_t = torch.unsqueeze(var_t, dim=1)

        # First conv block
        var_t = self.layer_norm_0(var_t)
        var_t = self.layer_cnn_2d_0(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)

        # Second conv block
        var_t = self.layer_norm_1(var_t)
        var_t = self.layer_cnn_2d_1(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)

        # Third conv block
        var_t = self.layer_norm_2(var_t)
        var_t = self.layer_cnn_2d_2(var_t)
        var_t = self.layer_leakyrelu(var_t)
        var_t = self.layer_dropout(var_t)

        # Global average pooling
        var_t = self.layer_norm_3(var_t)
        var_t = torch.mean(var_t, dim=(-2, -1))

        return var_t


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
    def __init__(self, var_x_shape, var_y_shape, bottleneck_dim=256):
        super(CNN_2D, self).__init__()

        var_dim_output = var_y_shape[-1]

        # Initialize components
        self.base = CNN2DBase(var_x_shape)
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