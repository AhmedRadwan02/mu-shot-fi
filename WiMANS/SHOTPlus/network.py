import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

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

# ============================================================================
# FIXED CNN2DBase - MATCHES ORIGINAL EXACTLY
# ============================================================================
class CNN2DBase(nn.Module):
    def __init__(self, var_x_shape, use_transformer=False):
        super(CNN2DBase, self).__init__()
        
        self.use_transformer = use_transformer
        
        # Normalization layers
        self.layer_norm_0 = nn.BatchNorm2d(1)
        self.layer_norm_1 = nn.BatchNorm2d(32)
        self.layer_norm_2 = nn.BatchNorm2d(64)
        self.layer_norm_3 = nn.BatchNorm2d(128)
        
        # Convolutional layers - EXACTLY AS ORIGINAL (no padding!)
        self.layer_cnn_2d_0 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(27, 27),
            stride=(7, 7)
            # NO PADDING - this was causing the issue
        )
        self.layer_cnn_2d_1 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(15, 15),
            stride=(3, 3)
            # NO PADDING
        )
        self.layer_cnn_2d_2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(7, 7),
            stride=(1, 1)
            # NO PADDING
        )
        
        # Activation and regularization
        self.layer_leakyrelu = nn.LeakyReLU()
        self.layer_dropout = nn.Dropout(0.2)
        
        # Feature dimension after global average pooling
        self.in_features = 128
        
        # Apply initialization
        self.apply(init_weights)
        
        print(f"CNN2DBase initialized (use_transformer={use_transformer}):")
        print(f"  Conv1: kernel=(27,27), stride=(7,7), padding=0")
        print(f"  Conv2: kernel=(15,15), stride=(3,3), padding=0")
        print(f"  Conv3: kernel=(7,7), stride=(1,1), padding=0")
    
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
        
        var_t = self.layer_norm_3(var_t)
        
        # Different outputs based on use_transformer
        if self.use_transformer:
            # DETR: Keep spatial structure for cross-attention
            # Output: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = var_t.shape
            var_t = var_t.flatten(2).transpose(1, 2)  # [B, H*W, C]
        else:
            # Normal CNN: Global average pooling
            # Output: [B, C]
            var_t = torch.mean(var_t, dim=(-2, -1))  # [B, 128]
        
        return var_t

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, use_transformer=False):
        super(feat_bottleneck, self).__init__()
        self.use_transformer = use_transformer
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        
    def forward(self, x):
        """
        Args:
            x: [B, feature_dim] for normal CNN
               [B, seq_len, feature_dim] for DETR
        Returns:
            [B, bottleneck_dim] for normal CNN
            [B, seq_len, bottleneck_dim] for DETR
        """
        x = self.bottleneck(x)
        return x

class feat_classifier(nn.Module):
    """
    Simple linear classifier (for backward compatibility)
    """
    def __init__(self, class_num, bottleneck_dim=256):
        super(feat_classifier, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x

# ============================================================================
# TRANSFORMER MODULES (UNCHANGED)
# ============================================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return output

class TransformerClassifier(nn.Module):
    def __init__(self, num_queries=6, num_classes=10, d_model=256, nhead=4, 
                 num_decoder_layers=6, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.d_model = d_model
        
        self.query_embed = nn.Embedding(num_queries, d_model)
        nn.init.xavier_uniform_(self.query_embed.weight)
        
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.class_head = nn.Linear(d_model, num_classes)
        self.apply(init_weights)
    
    def forward(self, encoder_output):
        B = encoder_output.shape[0]
        
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        decoder_output = self.decoder(queries, encoder_output)
        logits = self.class_head(decoder_output)
        
        return logits

# ============================================================================
# MAIN MODEL
# ============================================================================
class CNN_2D(nn.Module):
    def __init__(self, var_x_shape, var_y_shape, bottleneck_dim=256, 
                 use_transformer=False, num_queries=6, num_classes=10,
                 num_decoder_layers=6, nhead=4):
        super(CNN_2D, self).__init__()
        
        self.use_transformer = use_transformer
        
        self.base = CNN2DBase(var_x_shape, use_transformer=use_transformer)
        self.bottleneck = feat_bottleneck(self.base.in_features, bottleneck_dim, 
                                         use_transformer=use_transformer)
        
        if use_transformer:
            self.classifier = TransformerClassifier(
                num_queries=num_queries,
                num_classes=num_classes,
                d_model=bottleneck_dim,
                nhead=nhead,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=bottleneck_dim * 2,
                dropout=0.1
            )
            print(f"✓ Using DETR-style Transformer Classifier")
        else:
            var_dim_output = var_y_shape[-1]
            self.classifier = feat_classifier(var_dim_output, bottleneck_dim)
            print(f"✓ Using Normal Linear Classifier")
    
    def forward(self, var_input):
        features = self.base(var_input)
        bottleneck_features = self.bottleneck(features)
        output = self.classifier(bottleneck_features)
        return output

# ============================================================================
# QUERY ADAPTER MODULES (UNCHANGED)
# ============================================================================
class QueryAdapter(nn.Module):
    def __init__(self, num_queries, d_model, nhead=4, num_decoder_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        
        self.query_embed = nn.Embedding(num_queries, d_model)
        nn.init.xavier_uniform_(self.query_embed.weight)
        
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout
        )
        
        self.apply(init_weights)
    
    def forward(self, bottleneck_features):
        B = bottleneck_features.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query_features = self.decoder(queries, bottleneck_features)
        return query_features

class HybridQueryAdapter(nn.Module):
    def __init__(self, num_queries, num_classes, bottleneck_dim, 
                 adapter_num_layers=2, adapter_nhead=4):
        super().__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        
        self.query_adapter = QueryAdapter(
            num_queries=num_queries,
            d_model=bottleneck_dim,
            nhead=adapter_nhead,
            num_decoder_layers=adapter_num_layers,
            dropout=0.1
        )
        
        self.per_query_classifier = feat_classifier(
            class_num=num_classes + 1,
            bottleneck_dim=bottleneck_dim
        )
        
        self.register_buffer('initial_classifier_weight', None)
        self.register_buffer('initial_classifier_bias', None)
    
    def initialize_from_source_linear(self, source_linear_classifier):
        with torch.no_grad():
            source_weight = source_linear_classifier.fc.weight.data
            source_bias = source_linear_classifier.fc.bias.data
            
            source_weight_reshaped = source_weight.view(
                self.num_queries, self.num_classes + 1, self.bottleneck_dim
            )
            source_bias_reshaped = source_bias.view(
                self.num_queries, self.num_classes + 1
            )
            
            init_weight = source_weight_reshaped.mean(dim=0)
            init_bias = source_bias_reshaped.mean(dim=0)
            
            self.per_query_classifier.fc.weight.copy_(init_weight)
            self.per_query_classifier.fc.bias.copy_(init_bias)
            
            self.initial_classifier_weight = init_weight.clone()
            self.initial_classifier_bias = init_bias.clone()
            
            print(f"✓ Initialized per-query classifier from source linear classifier")
    
    def forward(self, bottleneck_features):
        B = bottleneck_features.shape[0]
        query_features = self.query_adapter(bottleneck_features)
        
        query_features_flat = query_features.view(B * self.num_queries, self.bottleneck_dim)
        logits_flat = self.per_query_classifier(query_features_flat)
        logits = logits_flat.view(B, self.num_queries, self.num_classes + 1)
        
        return logits
    
    def get_regularization_loss(self):
        if self.initial_classifier_weight is None:
            return torch.tensor(0.0, device=self.per_query_classifier.fc.weight.device)
        
        weight_diff = torch.norm(
            self.per_query_classifier.fc.weight - self.initial_classifier_weight
        )
        bias_diff = torch.norm(
            self.per_query_classifier.fc.bias - self.initial_classifier_bias
        )
        
        return weight_diff + bias_diff