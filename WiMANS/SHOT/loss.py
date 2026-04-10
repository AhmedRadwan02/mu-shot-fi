import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class BCELabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(BCELabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets_one_hot = torch.zeros(inputs.size())
            targets_one_hot.scatter_(1, targets.unsqueeze(1).cpu(), 1)
            if self.use_gpu: 
                targets_one_hot = targets_one_hot.cuda()
        else:
            targets_one_hot = targets.float()
        
        # Apply label smoothing
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / 2
        
        # Calculate BCE loss manually for smoothed targets
        loss = -(targets_smooth * torch.log(probs + 1e-8) + 
                (1 - targets_smooth) * torch.log(1 - probs + 1e-8))
        
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum(1)
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets_one_hot = torch.zeros(inputs.size())
            targets_one_hot.scatter_(1, targets.unsqueeze(1).cpu(), 1)
            if inputs.is_cuda:
                targets_one_hot = targets_one_hot.cuda()
        else:
            targets_one_hot = targets.float()
        
        # Calculate focal loss
        pt = probs * targets_one_hot + (1 - probs) * (1 - targets_one_hot)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        bce_loss = -(targets_one_hot * torch.log(probs + 1e-8) + 
                    (1 - targets_one_hot) * torch.log(1 - probs + 1e-8))
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

class MultiLabelCrossEntropy(nn.Module):
    def __init__(self):
        super(MultiLabelCrossEntropy, self).__init__()
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_users * num_classes) - model outputs
            targets: (batch_size, num_users * num_classes) - ground truth
        """
        # Apply standard BCE with logits loss
        loss = F.binary_cross_entropy_with_logits(inputs, targets.float())
        return loss

class MultiUserActivityLoss(nn.Module):
    """
    Specialized loss for multi-user activity classification.
    Handles the reshaping and structured loss calculation.
    """
    def __init__(self, num_users, num_classes, base_loss='bce', **kwargs):
        super(MultiUserActivityLoss, self).__init__()
        self.num_users = num_users
        self.num_classes = num_classes
        
        # Choose base loss function
        if base_loss == 'bce':
            self.loss_fn = MultiLabelCrossEntropy(**kwargs)
        elif base_loss == 'focal':
            self.loss_fn = FocalLoss(**kwargs)
        elif base_loss == 'asymmetric':
            self.loss_fn = AsymmetricLoss(**kwargs)
        elif base_loss == 'smooth_bce':
            self.loss_fn = BCELabelSmooth(num_classes=num_users * num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown base_loss: {base_loss}")
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_users * num_classes) - raw logits
            targets: (batch_size, num_users * num_classes) - ground truth labels
        """
        # Ensure targets are properly shaped and typed
        if targets.dim() == 3:  # (batch_size, num_users, num_classes)
            targets = targets.view(targets.size(0), -1)
        
        # Apply the base loss function
        return self.loss_fn(inputs, targets)

class WeightedMultiUserLoss(nn.Module):
    """
    Weighted loss that can give different importance to different users or activities.
    """
    def __init__(self, num_users, num_classes, user_weights=None, activity_weights=None, base_loss='bce'):
        super(WeightedMultiUserLoss, self).__init__()
        self.num_users = num_users
        self.num_classes = num_classes
        
        # Set default weights if not provided
        if user_weights is None:
            user_weights = torch.ones(num_users)
        if activity_weights is None:
            activity_weights = torch.ones(num_classes)
            
        # Create weight matrix: (num_users, num_classes)
        weight_matrix = user_weights.unsqueeze(1) * activity_weights.unsqueeze(0)
        # Flatten to match the output format
        self.register_buffer('weights', weight_matrix.view(-1))
        
        # Base loss function
        if base_loss == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif base_loss == 'focal':
            self.loss_fn = FocalLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_users * num_classes)
            targets: (batch_size, num_users * num_classes)
        """
        # Calculate base loss
        if isinstance(self.loss_fn, FocalLoss):
            loss = self.loss_fn(inputs, targets.float())
            if loss.dim() > 1:
                loss = loss.mean(dim=0)
        else:
            loss = self.loss_fn(inputs, targets.float())
        
        # Apply weights
        if loss.dim() == 2:  # (batch_size, num_features)
            weighted_loss = loss * self.weights.unsqueeze(0)
            return weighted_loss.mean()
        else:  # Already reduced
            return loss