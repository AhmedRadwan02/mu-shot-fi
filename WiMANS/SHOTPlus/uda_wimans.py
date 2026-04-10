import os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pickle
import json
from datetime import datetime
from sklearn.metrics import hamming_loss, f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Project modules
import network

from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y
import torch.nn.functional as F
import pandas as pd
from augmentations import AugMixDataset
import rotation
from scipy.optimize import linear_sum_assignment

class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index  # Return index as well

    def __len__(self):
        return len(self.dataset)
        
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularization"""
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction logits, shape (batch_size, num_classes)
            targets: ground truth labels, shape (batch_size,) as indices
        Returns:
            loss value
        """
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets_one_hot = targets_one_hot.cuda()
        targets_smoothed = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets_smoothed * log_probs).mean(0).sum()
        else:
            loss = (- targets_smoothed * log_probs).sum(1)
        return loss
"""
def compute_temporal_consistency_loss(inputs, netF, netB, netC, num_users, num_classes):
    \"""
    Temporal window consistency for multi-user SFUDA.
    
    Consecutive windows should predict same users doing same activities.
    Hungarian matches predictions between windows to respect slot permutation.
    
    Args:
        inputs: [B, T, F] - CSI batch (T=3000, F=270 for WiMANS)
        netF, netB, netC: Encoder, Bottleneck, Classifier
        num_users: M=6 slots
        num_classes: K=9 activities (excluding no-person)
    
    Returns:
        Scalar consistency loss
    \"""
    B, T, feature_dim = inputs.shape
    mid = T // 2
    
    # Split temporally: [B, T, F] -> two [B, T/2, F] windows
    window_1 = inputs[:, :mid, :]       # First half
    window_2 = inputs[:, mid:2*mid, :]  # Second half
    
    # Forward pass (CNN handles variable length due to global avg pool)
    logits_1 = netC(netB(netF(window_1))).view(B, num_users, num_classes + 1)
    logits_2 = netC(netB(netF(window_2))).view(B, num_users, num_classes + 1)
    
    # Softmax probabilities
    prob_1 = F.softmax(logits_1, dim=-1)  # [B, M, K+1]
    prob_2 = F.softmax(logits_2, dim=-1)  # [B, M, K+1]
    
    total_loss = 0.0
    
    for b in range(B):
        p1, p2 = prob_1[b], prob_2[b]  # [M, K+1] each
        
        # Hungarian matching: find optimal slot alignment between windows
        cost_matrix = -torch.mm(p1, p2.t())  # [M, M] - negative similarity
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        
        # Reorder p2 to match p1's slot order
        col_idx = torch.tensor(col_ind, device=inputs.device, dtype=torch.long)
        p2_matched = p2[col_idx]
        
        # Symmetric soft cross-entropy (KL in both directions)
        loss_1to2 = -(p1.detach() * torch.log(p2_matched + 1e-8)).sum(dim=-1).mean()
        loss_2to1 = -(p2_matched.detach() * torch.log(p1 + 1e-8)).sum(dim=-1).mean()
        
        total_loss += (loss_1to2 + loss_2to1) / 2
    
    return total_loss / B
"""

def compute_temporal_consistency_loss(inputs, netF, netB, netC, num_users, num_classes):
    """
    Occupancy-weighted augmentation consistency.
    Focus on ACTIVE slots, not empty ones.
    """
    B, T, Feature_dim = inputs.shape
    
    # Augmented views
    noise_std = 0.15
    view_1 = inputs + torch.randn_like(inputs) * noise_std
    view_2 = inputs + torch.randn_like(inputs) * noise_std
    
    # Forward pass
    logits_1 = netC(netB(netF(view_1))).view(B, num_users, num_classes + 1)
    logits_2 = netC(netB(netF(view_2))).view(B, num_users, num_classes + 1)
    
    prob_1 = F.softmax(logits_1, dim=-1)  # [B, M, K+1]
    prob_2 = F.softmax(logits_2, dim=-1)
    
    total_loss = 0.0
    total_weight = 0.0
    
    for b in range(B):
        p1, p2 = prob_1[b], prob_2[b]  # [M, K+1]
        
        # Hungarian matching
        cost_matrix = -torch.mm(p1, p2.t())
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        col_idx = torch.tensor(col_ind, device=inputs.device, dtype=torch.long)
        p2_matched = p2[col_idx]
        
        # Occupancy weight: p(occupied) = 1 - p(no-person)
        # Use AVERAGE of both views for stability
        occ_weight = ((1 - p1[:, -1]) + (1 - p2_matched[:, -1])) / 2  # [M]
        
        # Per-slot consistency loss
        kl_1to2 = -(p1.detach() * torch.log(p2_matched + 1e-8)).sum(dim=-1)  # [M]
        kl_2to1 = -(p2_matched.detach() * torch.log(p1 + 1e-8)).sum(dim=-1)  # [M]
        per_slot_loss = (kl_1to2 + kl_2to1) / 2  # [M]
        
        # Weight by occupancy
        weighted_loss = (occ_weight * per_slot_loss).sum()
        total_loss += weighted_loss
        total_weight += occ_weight.sum()
    
    # Normalize by total occupancy weight
    return total_loss / (total_weight + 1e-8)
    
def convert_onehot_to_indices(data_y, num_classes):
    """
    Convert one-hot encoded labels [N, 6, num_classes] to class indices [N, 6].
    If all zeros (no person), assign index = num_classes (the "no_person" class).

    Args:
        data_y: numpy array of shape [N, 6, num_classes] with one-hot encoding
        num_classes: number of activity/location classes (9 for activity, 5 for location)

    Returns:
        indices: numpy array of shape [N, 6] with class indices (0 to num_classes)
                 num_classes represents "no_person"
    """
    N, num_users = data_y.shape[0], data_y.shape[1]
    indices = np.zeros((N, num_users), dtype=np.int64)

    for i in range(N):
        for j in range(num_users):
            user_label = data_y[i, j]  # [num_classes]
            if user_label.sum() == 0:
                # No person in this slot
                indices[i, j] = num_classes
            else:
                # Get the activity/location class
                indices[i, j] = np.argmax(user_label)

    return indices


def hungarian_matching_batch(pred_logits, gt_indices, num_classes):
    """
    Perform Hungarian matching between predicted slots and ground truth slots.
    
    Args:
        pred_logits: [B, M, num_classes+1] predicted logits for each slot
        gt_indices: [B, M] ground truth class indices (0 to num_classes)
        num_classes: number of classes excluding "no_person" (e.g., 9 for activity)
    
    Returns:
        matched_pred_indices: [B, M] reordered prediction slot indices
        matched_gt_indices: [B, M] reordered ground truth slot indices
    """
    B, M, K = pred_logits.shape
    
    matched_pred_indices = []
    matched_gt_indices = []
    
    pred_probs = F.softmax(pred_logits, dim=-1)  # [B, M, num_classes+1]
    
    for b in range(B):
        # Build cost matrix: [M, M] negative log probability
        cost_matrix = np.zeros((M, M))
        
        for i in range(M):  # pred slots
            for j in range(M):  # gt slots
                gt_class = gt_indices[b, j].item()
                # Cost = negative log probability of predicting the GT class
                cost_matrix[i, j] = -torch.log(pred_probs[b, i, gt_class] + 1e-8).item()
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_pred_indices.append(row_ind)
        matched_gt_indices.append(col_ind)
    
    matched_pred_indices = np.array(matched_pred_indices)  # [B, M]
    matched_gt_indices = np.array(matched_gt_indices)  # [B, M]
    
    return matched_pred_indices, matched_gt_indices

def debug_data_availability():
    """Debug what data is available for different user counts"""
    print("Debugging data availability...")
    
    # Load all data without filtering
    data_pd_y_all = pd.read_csv(preset["path"]["data_y"], dtype=str)
    
    print(f"Total samples in dataset: {len(data_pd_y_all)}")
    print("\nSamples by number of users:")
    user_counts = data_pd_y_all["number_of_users"].value_counts().sort_index()
    print(user_counts)
    
    print("\nSamples by environment:")
    env_counts = data_pd_y_all["environment"].value_counts()
    print(env_counts)
    
    print("\nSamples by WiFi band:")
    wifi_counts = data_pd_y_all["wifi_band"].value_counts()
    print(wifi_counts)
    
    # Test your specific filtering
    print(f"\nTesting source filter:")
    source_data = load_data_y(
        preset["path"]["data_y"],
        var_environment=preset["source_data"]["environment"],
        var_wifi_band=preset["source_data"]["wifi_band"], 
        var_num_users=preset["source_data"]["num_users"]
    )
    print(f"Source samples found: {len(source_data)}")
    
    print(f"\nTesting target filter:")
    target_data = load_data_y(
        preset["path"]["data_y"],
        var_environment=preset["target_data"]["environment"],
        var_wifi_band=preset["target_data"]["wifi_band"],
        var_num_users=preset["target_data"]["num_users"] 
    )
    print(f"Target samples found: {len(target_data)}")

def create_models(var_x_shape, num_users, num_classes, bottleneck_dim):
    """
    Create netF, netB, netC based on configuration.
    
    Returns:
        netF, netB, netC (all on CUDA, NOT wrapped in DataParallel yet)
    """
    # ============================================================
    # CHECK ARCHITECTURE FLAG FIRST
    # ============================================================
    use_transformer = preset["training"].get("use_transformer", False)
    
    # ============================================================
    # Encoder - NOW PASSES use_transformer FLAG
    # ============================================================
    netF = network.CNN2DBase(var_x_shape, use_transformer=use_transformer).cuda()
    
    # ============================================================
    # Bottleneck - NOW PASSES use_transformer FLAG
    # ============================================================
    netB = network.feat_bottleneck(
        feature_dim=netF.in_features,
        bottleneck_dim=bottleneck_dim,
        use_transformer=use_transformer  # ← ADD THIS
    ).cuda()
    
    # ============================================================
    # Classifier (depends on architecture flag)
    # ============================================================
    if use_transformer:
        print("\n" + "="*60)
        print("USING TRANSFORMER CLASSIFIER WITH LEARNABLE QUERIES")
        print("="*60)
        netC = network.TransformerClassifier(
            num_queries=num_users,
            num_classes=num_classes + 1,  # +1 for "no_person"
            d_model=bottleneck_dim,
            nhead=preset["training"].get("nhead", 4),
            num_decoder_layers=preset["training"].get("num_decoder_layers", 4),
            dim_feedforward=preset["training"].get("dim_feedforward", bottleneck_dim * 2),
            dropout=preset["training"].get("transformer_dropout", 0.1)
        ).cuda()
        print(f"  Queries: {num_users}")
        print(f"  Classes per query: {num_classes + 1}")
        print(f"  Decoder layers: {preset['training'].get('num_decoder_layers', 4)}")
        print(f"  Attention heads: {preset['training'].get('nhead', 4)}")
        print(f"  Model dimension: {bottleneck_dim}")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("USING LINEAR CLASSIFIER (Legacy)")
        print("="*60)
        classifier_output_size = num_users * (num_classes + 1)
        netC = network.feat_classifier(
            class_num=classifier_output_size,
            bottleneck_dim=bottleneck_dim
        ).cuda()
        print(f"  Output size: {classifier_output_size}")
        print("="*60 + "\n")
    
    return netF, netB, netC
    
def create_dataset(data_x, data_y, config=None, batch_size=None, shuffle=True, is_training=True):
    """Create PyTorch dataset and dataloader with optional AugMix"""
    if batch_size is None:
        batch_size = preset["training"]["batch_size"]
    
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        batch_size = batch_size * gpu_count
    
    # Adjust batch size if dataset is smaller
    dataset_size = len(data_x)
    if dataset_size < batch_size:
        batch_size = max(1, dataset_size // 2)
        print(f"Warning: Dataset size ({dataset_size}) smaller than batch size. Adjusted to {batch_size}")
    
    # Check if AugMix should be used - ONLY for training
    use_augmix = config.get("use_augmix", False) if config is not None else False
    use_augmix = use_augmix and is_training  # ← KEY: Only augment during training
    
    if use_augmix:
        dataset = AugMixDataset(data_x, data_y, config)
    else:
        dataset = TensorDataset(torch.FloatTensor(data_x), torch.LongTensor(data_y))
    
    # WRAP DATASET TO RETURN INDICES
    dataset = IndexDataset(dataset)  

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=False, num_workers=12, pin_memory=True)
    return dataloader
    
def dset_loaders(config_key="source_data", random_state=39):
    """Load data based on configuration"""
    config = preset[config_key]
    data_pd_y = load_data_y(
        preset["path"]["data_y"],
        var_environment=config["environment"], 
        var_wifi_band=config["wifi_band"], 
        var_num_users=config["num_users"]
    )
    
    var_label_list = data_pd_y["label"].to_list()
    
    # Load data with normalization settings from config
    normalize_method = config.get("normalize", None)
    data_type = config.get("data_type", "amp")
    
    data_x, valid_indices = load_data_x(
        preset["path"]["data_x"], 
        var_label_list,
        normalize=normalize_method,
        data_type=data_type
    )
    

    data_pd_y_filtered = data_pd_y.iloc[valid_indices].reset_index(drop=True)
    
    task = preset["source_task"] if config_key == "source_data" else preset["target_task"]
    data_y = encode_data_y(data_pd_y_filtered, task)
    
    from sklearn.model_selection import train_test_split
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(
        data_x, data_y, test_size=0.2, shuffle=True, random_state=random_state
    )
    
    # Split test set into validation and test (50/50)
    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(
        data_test_x, data_test_y, test_size=0.5, shuffle=True, random_state=random_state
    )
    
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    
    var_x_shape = data_train_x[0].shape
    
    # ============================================================
    # INFER num_users FROM CONFIGURATION (NOT HARD-CODED)
    # ============================================================
    # Get the maximum number of users from the filter
    num_users = max([int(x) for x in config["num_users"]]) + 1
    
    # Alternative: Infer from actual data shape
    # num_users = data_train_y.shape[1]  # Shape is [N, num_users, num_classes]
    
    print(f"\n{'='*50}")
    print(f"Inferred num_users = {num_users} from config: {config['num_users']}")
    print(f"{'='*50}\n")
    
    task_encoding = preset["encoding"][task]
    num_task_classes = len(list(task_encoding.values())[0])
    
    # NEW: Convert one-hot [N, num_users, num_task_classes] to indices [N, num_users]
    data_train_y = convert_onehot_to_indices(data_train_y, num_task_classes)
    data_valid_y = convert_onehot_to_indices(data_valid_y, num_task_classes)
    data_test_y = convert_onehot_to_indices(data_test_y, num_task_classes)
    print(data_test_y[0:10])
    
    # Classifier output size
    if preset["training"].get("use_transformer", False):
        classifier_output_size = None  
    else:
        classifier_output_size = num_users * (num_task_classes + 1)
    
    # Create dataloaders
    train_loader = create_dataset(data_train_x, data_train_y, config=config, shuffle=True, is_training=True)
    valid_loader = create_dataset(data_valid_x, data_valid_y, config=config, shuffle=False, is_training=False)
    test_loader = create_dataset(data_test_x, data_test_y, config=config, shuffle=False, is_training=False)
    print(f"\nData Configuration:")
    print(f"  Filtering for user counts: {config['num_users']}")
    print(f"  Model num_users (slots): {num_users}")
    print(f"  Expected occupancy: {int(config['num_users'][0])}-{int(config['num_users'][-1])} people")
    print(f"  Expected empty slots: {num_users - int(config['num_users'][-1])} to {num_users - int(config['num_users'][0])}")

    return train_loader, valid_loader, test_loader, classifier_output_size, var_x_shape, num_users, num_task_classes

def cal_acc(loader, netF, netB, netC, num_users, num_classes):
    """
    Calculate comprehensive metrics with slot-based predictions and Hungarian matching.

    Args:
        loader: DataLoader with labels as [B, 6] indices
        netF, netB, netC: Networks
        num_users: 6
        num_classes: number of task classes (9 for activity, 5 for location)

    Returns:
        Dictionary containing all metrics:
        - slot_wise_accuracy: Percentage of correctly predicted slots over all slots (main metric)
                             Computed as: (# correct slots) / (total slots) * 100
                             where each sample has 6 slots, and predictions are matched via Hungarian algorithm
        - f1_micro: Micro-averaged F1 (across all classes including no_person)
        - f1_macro: Macro-averaged F1 (across all classes including no_person)
        - per_activity_f1: F1 score for each of the 9 activities (excluding no_person)
        - activity_macro_f1: Macro-average F1 across only the 9 activities
        - occupancy_mae: Mean absolute error of occupancy count
        - occupancy_exact_match: Percentage of samples with exact occupancy match
        - exact_match_accuracy: Percentage of samples where all 6 slots are correct
        - hamming_loss: Error rate (1 - slot_wise_accuracy)
        - classification_report: Detailed sklearn classification report (string)
        - classification_report_dict: Classification report as dictionary for JSON saving
    """
    netF.eval()
    netB.eval()
    netC.eval()

    all_preds = []
    all_labels = []
    all_sample_preds = []  # Store predictions per sample for exact match calculation
    all_sample_labels = []  # Store labels per sample for exact match calculation

    with torch.no_grad():
        for inputs, targets, _ in loader: # Unpack 3 values
            inputs = inputs.cuda()
            targets = targets.cuda()  # [B, 6] indices

            # Handle NaN values (safety check)
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)

            
            outputs = forward_with_reshape(netF, netB, netC, inputs, num_users, num_classes)
            B = outputs.shape[0]
            

            # Hungarian matching
            matched_pred_indices, matched_gt_indices = hungarian_matching_batch(
                outputs, targets, num_classes
            )

            # Extract matched predictions
            for b in range(B):
                pred_slots = outputs[b]  # [6, num_classes+1]
                gt_slots = targets[b]    # [6]

                # Reorder based on matching
                pred_indices = matched_pred_indices[b]
                gt_indices = matched_gt_indices[b]

                sample_preds = []
                sample_labels = []

                for i in range(6):
                    pred_slot_idx = pred_indices[i]
                    gt_slot_idx = gt_indices[i]

                    pred_class = torch.argmax(pred_slots[pred_slot_idx]).item()
                    gt_class = gt_slots[gt_slot_idx].item()

                    all_preds.append(pred_class)
                    all_labels.append(gt_class)
                    sample_preds.append(pred_class)
                    sample_labels.append(gt_class)

                all_sample_preds.append(sample_preds)
                all_sample_labels.append(sample_labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_sample_preds = np.array(all_sample_preds)  # [num_samples, 6]
    all_sample_labels = np.array(all_sample_labels)  # [num_samples, 6]

    # ========================================================================
    # SLOT-WISE ACCURACY (Main Metric)
    # ========================================================================
    # Percentage of correctly predicted slots over all slots (including no_person)
    # Computation: (number of correct slots) / (total number of slots) * 100
    # After Hungarian matching, we compare each matched slot
    slot_wise_accuracy = accuracy_score(all_labels, all_preds) * 100

    # ========================================================================
    # EXISTING METRICS (kept for compatibility)
    # ========================================================================
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    hamming = (1 - slot_wise_accuracy / 100) * 100

    # Classification report (all classes including no_person)
    report = classification_report(all_labels, all_preds, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)

    # ========================================================================
    # PER-ACTIVITY F1-SCORE (9 activities, excluding no_person)
    # ========================================================================
    # Filter out "no_person" class (class index = num_classes)
    activity_mask = all_labels < num_classes
    activity_preds = all_preds[activity_mask]
    activity_labels = all_labels[activity_mask]

    per_activity_f1 = {}
    if len(activity_labels) > 0:
        # Compute F1 for each activity class (0 to num_classes-1)
        for class_idx in range(num_classes):
            # Binary classification: current class vs rest
            binary_labels = (activity_labels == class_idx).astype(int)
            binary_preds = (activity_preds == class_idx).astype(int)

            if binary_labels.sum() > 0:  # Only compute if class exists in labels
                f1 = f1_score(binary_labels, binary_preds, zero_division=0) * 100
            else:
                f1 = 0.0

            per_activity_f1[f'activity_{class_idx}'] = f1

        # Macro-average F1 across the 9 activities (equal weight per activity)
        activity_macro_f1 = np.mean(list(per_activity_f1.values()))
    else:
        # No activity predictions (all "no_person")
        per_activity_f1 = {f'activity_{i}': 0.0 for i in range(num_classes)}
        activity_macro_f1 = 0.0

    # ========================================================================
    # OCCUPANCY ESTIMATION ACCURACY
    # ========================================================================
    # For each sample, count occupied slots (not "no_person")
    true_occupancy = (all_sample_labels < num_classes).sum(axis=1)  # [num_samples]
    pred_occupancy = (all_sample_preds < num_classes).sum(axis=1)  # [num_samples]

    # Mean Absolute Error of occupancy count
    occupancy_mae = np.mean(np.abs(true_occupancy - pred_occupancy))

    # Percentage of samples with exact occupancy match
    occupancy_exact_match = (true_occupancy == pred_occupancy).mean() * 100

    # ========================================================================
    # EXACT MATCH ACCURACY
    # ========================================================================
    # Sample is correct only if ALL 6 slots are predicted correctly
    exact_matches = (all_sample_preds == all_sample_labels).all(axis=1)  # [num_samples]
    exact_match_accuracy = exact_matches.mean() * 100

    # ========================================================================
    # RETURN COMPREHENSIVE METRICS
    # ========================================================================
    metrics = {
        # Main metrics
        'slot_wise_accuracy': float(slot_wise_accuracy),
        'exact_match_accuracy': float(exact_match_accuracy),

        # Existing metrics (kept for compatibility)
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'hamming_loss': float(hamming),

        # Per-activity F1 (9 activities)
        'per_activity_f1': {k: float(v) for k, v in per_activity_f1.items()},
        'activity_macro_f1': float(activity_macro_f1),

        # Occupancy metrics
        'occupancy_mae': float(occupancy_mae),
        'occupancy_exact_match': float(occupancy_exact_match),

        # Classification reports
        'classification_report': report,
        'classification_report_dict': report_dict
    }

    return metrics

def lr_scheduler(optimizer, epoch, max_epochs, gamma=10, power=0.75):
    decay = (1 + gamma * epoch / max_epochs) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def get_model_device_safe(model):
    return model.module if hasattr(model, 'module') else model

def save_run_results(output_dir, run_results):
    """Save comprehensive results for a single run"""
    results_file = osp.join(output_dir, "run_complete_results.json")

    # Add timestamp and run info
    run_results['timestamp'] = datetime.now().isoformat()
    run_results['run_directory'] = output_dir

    with open(results_file, 'w') as f:
        json.dump(run_results, f, indent=2)

    # Helper function to format metric values safely
    def format_metric(value, default='N/A'):
        """Format a metric value, handling None and non-numeric values"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return f"{value:.2f}%"
        return str(value)

    # Also save a human-readable summary
    summary_file = osp.join(output_dir, "run_results_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPLETE RUN RESULTS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Run Directory: {output_dir}\n")
        f.write(f"Timestamp: {run_results['timestamp']}\n")
        f.write(f"Random Seed: {run_results['seed']}\n\n")

        # Source domain results
        if 'source_domain' in run_results and run_results['source_domain']:
            f.write("SOURCE DOMAIN RESULTS:\n")
            f.write("-" * 30 + "\n")
            source_results = run_results['source_domain']
            f.write(f"Best Training Slot-wise Accuracy: {format_metric(source_results.get('best_train_slot_wise_accuracy'))}\n")
            f.write(f"Best Valid Slot-wise Accuracy: {format_metric(source_results.get('best_valid_slot_wise_accuracy'))}\n")
            f.write(f"Test Slot-wise Accuracy: {format_metric(source_results.get('test_slot_wise_accuracy'))}\n")
            f.write(f"Test Exact Match Accuracy: {format_metric(source_results.get('test_exact_match_accuracy'))}\n")
            f.write(f"Test Activity Macro-F1: {format_metric(source_results.get('test_activity_macro_f1'))}\n")
            f.write(f"Test F1-Micro: {format_metric(source_results.get('test_f1_micro'))}\n")
            f.write(f"Test F1-Macro: {format_metric(source_results.get('test_f1_macro'))}\n\n")

        # Target domain baseline (source model on target)
        if 'target_domain_baseline' in run_results:
            f.write("TARGET DOMAIN BASELINE (Source Model on Target):\n")
            f.write("-" * 30 + "\n")
            target_baseline = run_results['target_domain_baseline']
            f.write(f"Slot-wise Accuracy: {format_metric(target_baseline.get('slot_wise_accuracy'))}\n")
            f.write(f"Exact Match Accuracy: {format_metric(target_baseline.get('exact_match_accuracy'))}\n")
            f.write(f"Activity Macro-F1: {format_metric(target_baseline.get('activity_macro_f1'))}\n")
            f.write(f"F1-Micro: {format_metric(target_baseline.get('f1_micro'))}\n")
            f.write(f"F1-Macro: {format_metric(target_baseline.get('f1_macro'))}\n\n")

        # Target domain SHOT adaptation
        if 'target_domain_adapted' in run_results:
            f.write("TARGET DOMAIN RESULTS (SHOT Adaptation):\n")
            f.write("-" * 30 + "\n")
            target_shot = run_results['target_domain_adapted']
            f.write(f"Best Valid Slot-wise Accuracy: {format_metric(target_shot.get('best_valid_slot_wise_accuracy'))}\n")
            f.write(f"Test Slot-wise Accuracy: {format_metric(target_shot.get('test_slot_wise_accuracy'))}\n")
            f.write(f"Test Exact Match Accuracy: {format_metric(target_shot.get('test_exact_match_accuracy'))}\n")
            f.write(f"Test Activity Macro-F1: {format_metric(target_shot.get('test_activity_macro_f1'))}\n")
            f.write(f"Test F1-Micro: {format_metric(target_shot.get('test_f1_micro'))}\n")
            f.write(f"Test F1-Macro: {format_metric(target_shot.get('test_f1_macro'))}\n")
            f.write(f"Best Epoch: {target_shot.get('best_epoch', 'N/A')}\n\n")

        # Performance comparisons
        if 'target_domain_baseline' in run_results and 'target_domain_adapted' in run_results:
            f.write("PERFORMANCE IMPROVEMENT (SHOT vs Source Baseline):\n")
            f.write("-" * 30 + "\n")
            baseline_acc = run_results['target_domain_baseline'].get('slot_wise_accuracy', 0)
            shot_acc = run_results['target_domain_adapted'].get('test_slot_wise_accuracy', 0)
            if isinstance(baseline_acc, (int, float)) and isinstance(shot_acc, (int, float)):
                acc_improvement = shot_acc - baseline_acc
                f.write(f"Slot-wise Accuracy Improvement: {acc_improvement:+.2f}%\n\n")

        f.write("="*60 + "\n")

    print(f"Run results saved to: {results_file}")
    print(f"Run summary saved to: {summary_file}")

def train_source(output_dir, out_file, random_state):
    """Train source domain model"""
    print("Loading source data...")
    train_loader, valid_loader, test_loader, classifier_output_size, var_x_shape, num_users, num_classes = dset_loaders("source_data", random_state=random_state)
    
    # Model setup
    netF, netB, netC = create_models(
        var_x_shape=var_x_shape,
        num_users=num_users,
        num_classes=num_classes,
        bottleneck_dim=preset["training"]["bottleneck"]
    )
    
    # ============================================================
    # Handle output reshaping based on architecture
    # ============================================================
    use_transformer = preset["training"].get("use_transformer", False)
    

    if torch.cuda.device_count() > 1:
        netF, netB, netC = nn.DataParallel(netF), nn.DataParallel(netB), nn.DataParallel(netC)

    # Optimizer setup
    param_group = []
    for v in get_model_device_safe(netF).parameters(): param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netB).parameters(): param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netC).parameters(): param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["lr"], weight_decay=1e-4))
    
    max_epochs = preset["training"]["max_epoch"]
    best_acc = 0
    best_train_acc = 0
    best_valid_acc = 0
    best_test_results = {}

    # Use Label Smoothing for better generalization
    smooth_epsilon = preset["training"].get("smooth", 0.1)
    criterion = CrossEntropyLabelSmooth(
        num_classes=num_classes + 1,  # +1 for "no_person" class
        epsilon=smooth_epsilon,
        use_gpu=torch.cuda.is_available()
    )
    # Store training history
    training_history = []

    print("Starting source training...")
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.train()

        # Training loop
        epoch_loss = 0
        num_batches = 0
        for inputs_source, labels_source, _ in train_loader: # Unpack and ignore index
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            # labels_source: [B, 6] indices

            outputs_source = forward_with_reshape(netF, netB, netC, inputs_source, num_users, num_classes)
            if torch.isnan(outputs_source).any() or torch.isinf(outputs_source).any():
                print(f"WARNING: Model outputs contain NaN/Inf!")
                print(f"  NaN: {torch.isnan(outputs_source).sum().item()}")
                print(f"  Inf: {torch.isinf(outputs_source).sum().item()}")
                print(f"  Input stats: min={inputs_source.min()}, max={inputs_source.max()}")
            B = outputs_source.shape[0]
            # Hungarian matching
            matched_pred_indices, matched_gt_indices = hungarian_matching_batch(
                outputs_source, labels_source, num_classes
            )

            # Compute CE loss on matched pairs
            total_loss = 0
            for b in range(B):
                pred_slots = outputs_source[b]  # [6, num_classes+1]
                gt_slots = labels_source[b]      # [6]

                # Get matched pairs
                pred_indices = matched_pred_indices[b]
                gt_indices = matched_gt_indices[b]

                for i in range(6):
                    pred_slot_idx = pred_indices[i]
                    gt_slot_idx = gt_indices[i]

                    pred_logits = pred_slots[pred_slot_idx]  # [num_classes+1]
                    gt_class = gt_slots[gt_slot_idx]          # scalar

                    # CE loss for this matched pair
                    total_loss += criterion(pred_logits.unsqueeze(0), gt_class.unsqueeze(0))

            # Average over batch and 6 slots
            classifier_loss = total_loss / (B * 6)

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
            epoch_loss += classifier_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation
        netF.eval(); netB.eval(); netC.eval()

        # Training metrics
        train_metrics = cal_acc(train_loader, netF, netB, netC, num_users, num_classes)

        # Validation metrics
        valid_metrics = cal_acc(valid_loader, netF, netB, netC, num_users, num_classes)

        # Store epoch results with ALL metrics
        epoch_results = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            # Training metrics
            'train_slot_wise_accuracy': train_metrics['slot_wise_accuracy'],
            'train_exact_match_accuracy': train_metrics['exact_match_accuracy'],
            'train_f1_micro': train_metrics['f1_micro'],
            'train_f1_macro': train_metrics['f1_macro'],
            'train_activity_macro_f1': train_metrics['activity_macro_f1'],
            'train_occupancy_mae': train_metrics['occupancy_mae'],
            'train_occupancy_exact_match': train_metrics['occupancy_exact_match'],
            'train_hamming_loss': train_metrics['hamming_loss'],
            'train_per_activity_f1': train_metrics['per_activity_f1'],
            # Validation metrics
            'valid_slot_wise_accuracy': valid_metrics['slot_wise_accuracy'],
            'valid_exact_match_accuracy': valid_metrics['exact_match_accuracy'],
            'valid_f1_micro': valid_metrics['f1_micro'],
            'valid_f1_macro': valid_metrics['f1_macro'],
            'valid_activity_macro_f1': valid_metrics['activity_macro_f1'],
            'valid_occupancy_mae': valid_metrics['occupancy_mae'],
            'valid_occupancy_exact_match': valid_metrics['occupancy_exact_match'],
            'valid_hamming_loss': valid_metrics['hamming_loss'],
            'valid_per_activity_f1': valid_metrics['per_activity_f1'],
        }
        training_history.append(epoch_results)

        log_str = f'Source Training - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Train Slot-Acc: {train_metrics["slot_wise_accuracy"]:.2f}%; Valid Slot-Acc: {valid_metrics["slot_wise_accuracy"]:.2f}%'
        out_file.write(log_str + '\n'); out_file.flush(); print(log_str)

        if valid_metrics['slot_wise_accuracy'] >= best_acc:
            best_acc = valid_metrics['slot_wise_accuracy']
            best_train_acc = train_metrics['slot_wise_accuracy']
            best_valid_acc = valid_metrics['slot_wise_accuracy']
            best_netF = copy.deepcopy(get_model_device_safe(netF).state_dict())
            best_netB = copy.deepcopy(get_model_device_safe(netB).state_dict())
            best_netC = copy.deepcopy(get_model_device_safe(netC).state_dict())

            # Save best model's detailed report (VALIDATION)
            best_report_file = osp.join(output_dir, "source_best_validation_report.txt")
            with open(best_report_file, 'w') as f:
                f.write(f"Best Source Model Performance on Validation Set (Epoch {epoch+1})\n")
                f.write("="*60 + "\n\n")
                f.write(f"Slot-wise Accuracy: {valid_metrics['slot_wise_accuracy']:.2f}%\n")
                f.write(f"Exact Match Accuracy: {valid_metrics['exact_match_accuracy']:.2f}%\n")
                f.write(f"F1-Micro (all classes): {valid_metrics['f1_micro']:.2f}%\n")
                f.write(f"F1-Macro (all classes): {valid_metrics['f1_macro']:.2f}%\n")
                f.write(f"Activity Macro-F1 (9 activities): {valid_metrics['activity_macro_f1']:.2f}%\n")
                f.write(f"Occupancy MAE: {valid_metrics['occupancy_mae']:.4f}\n")
                f.write(f"Occupancy Exact Match: {valid_metrics['occupancy_exact_match']:.2f}%\n")
                f.write(f"Hamming Loss: {valid_metrics['hamming_loss']:.2f}%\n\n")
                f.write("Per-Activity F1 Scores:\n")
                f.write("-"*40 + "\n")
                for act_key, f1_val in valid_metrics['per_activity_f1'].items():
                    f.write(f"  {act_key}: {f1_val:.2f}%\n")
                f.write("\n" + "="*60 + "\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(valid_metrics['classification_report'])
    

    # After training completes, evaluate best model on TEST set
    print("\nEvaluating best model on TEST set...")

    # Load best model states into the base models (NOT DataParallel wrapped)
    base_netF = get_model_device_safe(netF) if hasattr(netF, 'module') else netF
    base_netB = get_model_device_safe(netB) if hasattr(netB, 'module') else netB
    base_netC = get_model_device_safe(netC) if hasattr(netC, 'module') else netC

    base_netF.load_state_dict(best_netF)
    base_netB.load_state_dict(best_netB)
    base_netC.load_state_dict(best_netC)

    # Now wrap in DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        netF = nn.DataParallel(base_netF)
        netB = nn.DataParallel(base_netB)
        netC = nn.DataParallel(base_netC)
    else:
        netF = base_netF
        netB = base_netB
        netC = base_netC

    netF.eval(); netB.eval(); netC.eval()
    test_metrics = cal_acc(test_loader, netF, netB, netC, num_users, num_classes)

    # Save TEST set results
    test_report_file = osp.join(output_dir, "source_best_test_report.txt")
    with open(test_report_file, 'w') as f:
        f.write(f"Best Source Model Performance on TEST Set\n")
        f.write("="*60 + "\n\n")
        f.write(f"Slot-wise Accuracy: {test_metrics['slot_wise_accuracy']:.2f}%\n")
        f.write(f"Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.2f}%\n")
        f.write(f"F1-Micro (all classes): {test_metrics['f1_micro']:.2f}%\n")
        f.write(f"F1-Macro (all classes): {test_metrics['f1_macro']:.2f}%\n")
        f.write(f"Activity Macro-F1 (9 activities): {test_metrics['activity_macro_f1']:.2f}%\n")
        f.write(f"Occupancy MAE: {test_metrics['occupancy_mae']:.4f}\n")
        f.write(f"Occupancy Exact Match: {test_metrics['occupancy_exact_match']:.2f}%\n")
        f.write(f"Hamming Loss: {test_metrics['hamming_loss']:.2f}%\n\n")
        f.write("Per-Activity F1 Scores:\n")
        f.write("-"*40 + "\n")
        for act_key, f1_val in test_metrics['per_activity_f1'].items():
            f.write(f"  {act_key}: {f1_val:.2f}%\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(test_metrics['classification_report'])

    # Also save test metrics as JSON
    test_metrics_file = osp.join(output_dir, "source_best_test_metrics.json")
    test_metrics_for_json = {k: v for k, v in test_metrics.items() if k != 'classification_report'}
    with open(test_metrics_file, 'w') as f:
        json.dump(test_metrics_for_json, f, indent=2)

    log_str = f'\nFinal Source Model - Valid Slot-Acc: {best_valid_acc:.2f}%, Test Slot-Acc: {test_metrics["slot_wise_accuracy"]:.2f}%'
    out_file.write(log_str + '\n'); out_file.flush(); print(log_str)

    # Save models (use base models to avoid module. prefix issues)
    torch.save(base_netF.state_dict(), osp.join(output_dir, "source_F.pt"))
    torch.save(base_netB.state_dict(), osp.join(output_dir, "source_B.pt"))
    torch.save(base_netC.state_dict(), osp.join(output_dir, "source_C.pt"))


    # Save training history
    history_file = osp.join(output_dir, "source_training_history.json")
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"Source training completed. Best test slot-wise accuracy: {test_metrics['slot_wise_accuracy']:.2f}%")

    # Return source domain results with ALL metrics
    return {
        'best_train_slot_wise_accuracy': best_train_acc,
        'best_valid_slot_wise_accuracy': best_valid_acc,
        'test_slot_wise_accuracy': test_metrics['slot_wise_accuracy'],
        'test_exact_match_accuracy': test_metrics['exact_match_accuracy'],
        'test_f1_micro': test_metrics['f1_micro'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_activity_macro_f1': test_metrics['activity_macro_f1'],
        'test_occupancy_mae': test_metrics['occupancy_mae'],
        'test_occupancy_exact_match': test_metrics['occupancy_exact_match'],
        'test_hamming_loss': test_metrics['hamming_loss'],
        'test_per_activity_f1': test_metrics['per_activity_f1'],
        'training_history': training_history
    }


def test_target_baseline(source_dir, output_dir, out_file, random_state):
    """Test on target domain using the trained source model"""
    print("Loading target data for baseline testing...")
    _, _, test_loader, classifier_output_size, var_x_shape, num_users, num_classes = dset_loaders("target_data", random_state=random_state)

    netF, netB, netC = create_models(
        var_x_shape=var_x_shape,
        num_users=num_users,
        num_classes=num_classes,
        bottleneck_dim=preset["training"]["bottleneck"]
    )
    
    # Load saved weights
    netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(source_dir, "source_B.pt")))
    netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt")))
    

    if torch.cuda.device_count() > 1:
        netF, netB, netC = nn.DataParallel(netF), nn.DataParallel(netB), nn.DataParallel(netC)

    netF.eval(); netB.eval(); netC.eval()

    test_metrics = cal_acc(test_loader, netF, netB, netC, num_users, num_classes)

    # Save baseline results
    baseline_report_file = osp.join(output_dir, "target_baseline_classification_report.txt")
    with open(baseline_report_file, 'w') as f:
        f.write("Target Domain Baseline Performance (Source Model)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Slot-wise Accuracy: {test_metrics['slot_wise_accuracy']:.2f}%\n")
        f.write(f"Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.2f}%\n")
        f.write(f"F1-Micro (all classes): {test_metrics['f1_micro']:.2f}%\n")
        f.write(f"F1-Macro (all classes): {test_metrics['f1_macro']:.2f}%\n")
        f.write(f"Activity Macro-F1 (9 activities): {test_metrics['activity_macro_f1']:.2f}%\n")
        f.write(f"Occupancy MAE: {test_metrics['occupancy_mae']:.4f}\n")
        f.write(f"Occupancy Exact Match: {test_metrics['occupancy_exact_match']:.2f}%\n")
        f.write(f"Hamming Loss: {test_metrics['hamming_loss']:.2f}%\n\n")
        f.write("Per-Activity F1 Scores:\n")
        f.write("-"*40 + "\n")
        for act_key, f1_val in test_metrics['per_activity_f1'].items():
            f.write(f"  {act_key}: {f1_val:.2f}%\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(test_metrics['classification_report'])

    # Save baseline metrics as JSON
    baseline_metrics_file = osp.join(output_dir, "target_baseline_metrics.json")
    baseline_metrics_for_json = {k: v for k, v in test_metrics.items() if k != 'classification_report'}
    with open(baseline_metrics_file, 'w') as f:
        json.dump(baseline_metrics_for_json, f, indent=2)

    log_str = f'Target Test (Source Model) - Slot-Acc: {test_metrics["slot_wise_accuracy"]:.2f}%, Exact-Match: {test_metrics["exact_match_accuracy"]:.2f}%, Activity-F1: {test_metrics["activity_macro_f1"]:.2f}%'
    out_file.write(log_str + '\n'); out_file.flush(); print(log_str)

    # Return all metrics
    return {
        'slot_wise_accuracy': test_metrics['slot_wise_accuracy'],
        'exact_match_accuracy': test_metrics['exact_match_accuracy'],
        'f1_micro': test_metrics['f1_micro'],
        'f1_macro': test_metrics['f1_macro'],
        'activity_macro_f1': test_metrics['activity_macro_f1'],
        'occupancy_mae': test_metrics['occupancy_mae'],
        'occupancy_exact_match': test_metrics['occupancy_exact_match'],
        'hamming_loss': test_metrics['hamming_loss'],
        'per_activity_f1': test_metrics['per_activity_f1'],
        'classification_report': test_metrics['classification_report'],
        'classification_report_dict': test_metrics['classification_report_dict']
    }

def train_target_rot(output_dir, out_file, train_loader, netF, netB, num_users, num_classes):
    """
    Pre-train rotation classifier on target domain with comprehensive tracking.
    
    NOTE: This function is NOW architecture-aware!
    For DETR mode, we pool spatial features before rotation classification.
    
    Returns:
        best_netR: Best rotation classifier state dict
        ssl_metrics: Dictionary containing SSL training history and final metrics
    """
    # Initialize rotation classifier (2 classes: 0° and 180°)
    bottleneck_dim = preset["training"]["bottleneck"]
    netR = network.feat_classifier(class_num=2, bottleneck_dim=2*bottleneck_dim).cuda()
    
    # Check if we're using transformer (DETR mode)
    use_transformer = preset["training"].get("use_transformer", False)
    
    # Freeze feature extractor and bottleneck
    netF.eval()
    for k, v in get_model_device_safe(netF).named_parameters():
        v.requires_grad = False
    
    netB.eval()
    for k, v in get_model_device_safe(netB).named_parameters():
        v.requires_grad = False
    
    if torch.cuda.device_count() > 1:
        netR = nn.DataParallel(netR)
    
    # Setup optimizer - only train netR
    param_group = [{'params': v, 'lr': preset["training"]["target_lr"]}
                   for v in get_model_device_safe(netR).parameters()]
    netR.train()
    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["target_lr"], weight_decay=1e-4))
    
    max_epochs = preset["training"]["ssl_max_epoch"]
    interval_epochs = max(1, max_epochs // 10)
    best_rot_acc = 0
    best_netR = None
    best_epoch = 0
    
    # Track SSL training history
    ssl_history = []
    
    print("Pre-training rotation classifier...")
    for epoch in range(max_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for inputs_target, _, target_indices in train_loader: 
            inputs_target = inputs_target.cuda()
            target_indices = target_indices.cuda() # Indices of the current batch samples
            
            # Generate random rotations (binary: 0 or 180 degrees)
            r_labels = torch.randint(0, 2, (inputs_target.shape[0],), dtype=torch.long, device=inputs_target.device)
            r_inputs = rotation.rotate_batch_with_labels(inputs_target, r_labels)
            
            # ========================================
            # Forward pass - extract bottleneck features
            # ========================================
            f_outputs = netB(netF(inputs_target))      # [B, bottleneck_dim] or [B, seq_len, bottleneck_dim]
            f_r_outputs = netB(netF(r_inputs))         # [B, bottleneck_dim] or [B, seq_len, bottleneck_dim]
            
            # ========================================
            # HANDLE DETR MODE: Pool spatial features
            # ========================================
            if use_transformer:
                # DETR mode: features are [B, seq_len, bottleneck_dim]
                # Need to pool to [B, bottleneck_dim]
                f_outputs = f_outputs.mean(dim=1)      # Global average pooling: [B, seq_len, D] → [B, D]
                f_r_outputs = f_r_outputs.mean(dim=1)  # Global average pooling: [B, seq_len, D] → [B, D]
            
            # Now both are [B, bottleneck_dim] regardless of architecture
            
            # Concatenate and classify rotation
            r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))  # [B, 2*bottleneck_dim] → [B, 2]
            rotation_loss = nn.CrossEntropyLoss()(r_outputs, r_labels)
            
            optimizer.zero_grad()
            rotation_loss.backward()
            optimizer.step()
            
            epoch_loss += rotation_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation
        if (epoch + 1) % interval_epochs == 0 or epoch == max_epochs - 1:
            netR.eval()
            acc_rot = cal_acc_rot(train_loader, netF, netB, netR)
            
            # Store epoch metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'rotation_loss': float(avg_loss),
                'rotation_accuracy': float(acc_rot)
            }
            ssl_history.append(epoch_metrics)
            
            log_str = f'SSL Rotation - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Rotation Accuracy = {acc_rot:.2f}%'
            out_file.write(log_str + '\n')
            out_file.flush()
            print(log_str)
            
            netR.train()
            
            if acc_rot > best_rot_acc:
                best_rot_acc = acc_rot
                best_epoch = epoch + 1
                best_netR = copy.deepcopy(get_model_device_safe(netR).state_dict())
    
    log_str = f'Best Rotation Accuracy = {best_rot_acc:.2f}% (Epoch {best_epoch})'
    out_file.write(log_str + '\n')
    out_file.flush()
    print(log_str)
    
    # Compile SSL metrics
    ssl_metrics = {
        'best_rotation_accuracy': float(best_rot_acc),
        'best_epoch': best_epoch,
        'training_history': ssl_history
    }
    
    # Save SSL training history to separate file
    ssl_history_file = osp.join(output_dir, "ssl_rotation_history.json")
    with open(ssl_history_file, 'w') as f:
        json.dump(ssl_metrics, f, indent=2)
    
    print(f"SSL rotation history saved to: {ssl_history_file}")
    
    return best_netR, ssl_metrics


def cal_acc_rot(loader, netF, netB, netR):
    """Calculate rotation prediction accuracy - now architecture-aware"""
    use_transformer = preset["training"].get("use_transformer", False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.cuda()
            
            # Generate random rotations
            r_labels = torch.randint(0, 2, (inputs.shape[0],), dtype=torch.long, device=inputs.device)
            r_inputs = rotation.rotate_batch_with_labels(inputs, r_labels)
            
            # Forward pass - extract bottleneck features
            f_outputs = netB(netF(inputs))
            f_r_outputs = netB(netF(r_inputs))
            
            # ========================================
            # HANDLE DETR MODE: Pool spatial features
            # ========================================
            if use_transformer:
                # DETR mode: features are [B, seq_len, bottleneck_dim]
                # Pool to [B, bottleneck_dim]
                f_outputs = f_outputs.mean(dim=1)
                f_r_outputs = f_r_outputs.mean(dim=1)
            
            # Classify rotation
            r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))
            _, preds = torch.max(r_outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(r_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100
    
    return accuracy
def obtain_pseudo_labels(loader, netF, netB, netC, num_users, num_classes):
    """
    Generate pseudo-labels from model predictions.

    Args:
        loader: Non-shuffled dataloader for target training data
        netF, netB, netC: Networks
        num_users: Number of slots (6)
        num_classes: Number of activity/location classes

    Returns:
        pseudo_labels: numpy array [N, 6] with class indices
        confidences: numpy array [N, 6] with prediction confidences
    """
    netF.eval()
    netB.eval()
    netC.eval()

    all_pseudo_labels = []
    all_confidences = []

    with torch.no_grad():
        for inputs, _ ,_ in loader:
            inputs = inputs.cuda()

            # Forward pass
            outputs = forward_with_reshape(netF, netB, netC, inputs, num_users, num_classes)
            B = outputs.shape[0]
            
            # Softmax to get probabilities
            probs = F.softmax(outputs, dim=-1)  # [B, 6, num_classes+1]

            # Get pseudo-labels (argmax) and confidences (max probability)
            confidences, pseudo_labels = torch.max(probs, dim=-1)  # Both: [B, 6]

            all_pseudo_labels.append(pseudo_labels.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())

    # Concatenate all batches
    pseudo_labels = np.concatenate(all_pseudo_labels, axis=0)  # [N, 6]
    confidences = np.concatenate(all_confidences, axis=0)      # [N, 6]

    return pseudo_labels, confidences


def build_target_centroids(loader, netF, netB, netC, num_users, num_classes, conf_thresh=0.5):
    """
    Build per-class centroids from target data.
    Only use samples where at least one slot:
      - is predicted as a real class (not no_person)
      - has max prob >= conf_thresh
    Centroids live in the bottleneck feature space.

    Args:
        loader: Non-shuffled dataloader for target training data
        netF, netB, netC: Networks
        num_users: Number of slots (6)
        num_classes: Number of activity/location classes
        conf_thresh: Confidence threshold for including samples

    Returns:
        centroids: numpy array [num_classes, bottleneck_dim] with per-class centroids
    """
    netF.eval()
    netB.eval()
    netC.eval()

    all_features = []
    all_outputs = []

    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.cuda()

            features, outputs = get_features_and_outputs(netF, netB, netC, inputs, num_users, num_classes)
            
            all_features.append(features.cpu())
            all_outputs.append(outputs.cpu())
    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)  # [N, bottleneck_dim]
    all_outputs = torch.cat(all_outputs, dim=0)    # [N, 6, num_classes+1]

    # Normalize features (add bias term and L2 normalize, like WiDAR)
    all_features = torch.cat((all_features, torch.ones(all_features.size(0), 1)), 1)  # [N, bottleneck_dim+1]
    all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()  # L2 normalize
    all_features_np = all_features.float().cpu().numpy()

    # Compute softmax probabilities
    all_probs = F.softmax(all_outputs, dim=-1)  # [N, 6, num_classes+1]

    # Build centroids for each class (excluding no_person)
    centroids = []
    for c in range(num_classes):
        # Find samples where this class appears with high confidence in any slot
        class_probs = all_probs[:, :, c]  # [N, 6] - prob of class c in each slot
        max_prob_per_sample = class_probs.max(dim=1)[0]  # [N] - max prob across 6 slots

        # Select samples with confidence >= threshold
        confident_mask = max_prob_per_sample >= conf_thresh

        if confident_mask.sum() > 0:
            # Weight each sample by its max probability for this class
            weights = max_prob_per_sample[confident_mask].cpu().numpy()
            features_for_class = all_features_np[confident_mask.cpu().numpy()]

            # Compute weighted centroid
            centroid = (weights[:, None] * features_for_class).sum(axis=0) / (weights.sum() + 1e-8)
        else:
            # If no confident samples, use mean of all features
            centroid = all_features_np.mean(axis=0)

        centroids.append(centroid)

    centroids = np.array(centroids)  # [num_classes, bottleneck_dim+1]
    print("\n" + "="*50)
    print("CENTROID BUILD SUMMARY")
    print("="*50)
    total_confident_samples = 0
    for c in range(num_classes):
        class_probs = all_probs[:, :, c]
        max_prob_per_sample = class_probs.max(dim=1)[0]
        confident_mask = max_prob_per_sample >= conf_thresh
        n_samples = confident_mask.sum().item()
        total_confident_samples += n_samples
        avg_conf = max_prob_per_sample[confident_mask].mean().item() if n_samples > 0 else 0
        print(f"  Activity {c}: {n_samples:4d} samples (avg conf: {avg_conf:.3f})")
    print(f"\nTotal confident samples: {total_confident_samples}")
    print(f"Total dataset size: {len(all_probs)}")
    print(f"Coverage: {total_confident_samples/(len(all_probs)*num_classes)*100:.1f}%")
    print("="*50 + "\n")
    return centroids


def obtain_pseudo_labels_with_centroids(loader, netF, netB, netC, centroids, num_users, num_classes, out_file):
    """
    Generate pseudo-labels using centroid-based refinement.

    Args:
        loader: Non-shuffled dataloader for target training data
        netF, netB, netC: Networks
        centroids: numpy array [num_classes, bottleneck_dim+1] with per-class centroids
        num_users: Number of slots (6)
        num_classes: Number of activity/location classes
        out_file: Log file

    Returns:
        pseudo_labels: numpy array [N, 6] with class indices
        confidences: numpy array [N, 6] with prediction confidences
    """
    netF.eval()
    netB.eval()
    netC.eval()

    all_features = []
    all_outputs = []

    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.cuda()

            # Extract bottleneck features
            features = netB(netF(inputs))  # [B, bottleneck_dim]

            # Get predictions
            outputs = netC(features)  # [B, 6*(num_classes+1)]
            B = outputs.shape[0]
            outputs = outputs.view(B, num_users, num_classes + 1)  # [B, 6, num_classes+1]

            all_features.append(features.cpu())
            all_outputs.append(outputs.cpu())

    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)  # [N, bottleneck_dim]
    all_outputs = torch.cat(all_outputs, dim=0)    # [N, 6, num_classes+1]

    # Normalize features (same as centroids)
    all_features = torch.cat((all_features, torch.ones(all_features.size(0), 1)), 1)
    all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
    all_features_np = all_features.float().cpu().numpy()

    # Compute initial pseudo-labels from model predictions
    all_probs = F.softmax(all_outputs, dim=-1)  # [N, 6, num_classes+1]
    initial_confidences, initial_pseudo_labels = torch.max(all_probs, dim=-1)  # Both: [N, 6]

    # Compute distances to centroids
    distances = cdist(all_features_np, centroids, 'cosine')  # [N, num_classes]
    nearest_centroid = distances.argmin(axis=1)  # [N] - nearest class for each sample

    # Refine pseudo-labels using centroid information
    refined_pseudo_labels = []
    refined_confidences = []

    for i in range(all_probs.shape[0]):  # For each sample
        sample_probs = all_probs[i]  # [6, num_classes+1]
        sample_nearest_class = nearest_centroid[i]

        # For each slot
        slot_labels = []
        slot_confs = []
        for slot_idx in range(num_users):
            slot_prob = sample_probs[slot_idx]  # [num_classes+1]

            # Get initial prediction
            max_prob, max_class = slot_prob.max(dim=0)

            # If the nearest centroid class has reasonable probability, boost it
            centroid_class_prob = slot_prob[sample_nearest_class].item()
            no_person_prob = slot_prob[num_classes].item()  # Last class is no_person

            # If no_person is dominant, keep it
            if no_person_prob > 0.5:
                final_class = num_classes
                final_conf = no_person_prob
            # If centroid class has good probability, use it
            elif centroid_class_prob > 0.3:
                final_class = sample_nearest_class
                final_conf = centroid_class_prob
            # Otherwise use model's prediction
            else:
                final_class = max_class.item()
                final_conf = max_prob.item()

            slot_labels.append(final_class)
            slot_confs.append(final_conf)

        refined_pseudo_labels.append(slot_labels)
        refined_confidences.append(slot_confs)

    refined_pseudo_labels = np.array(refined_pseudo_labels)  # [N, 6]
    refined_confidences = np.array(refined_confidences)      # [N, 6]

    # Compare with initial predictions
    initial_pseudo_labels_np = initial_pseudo_labels.cpu().numpy()
    agreement = (refined_pseudo_labels == initial_pseudo_labels_np).mean()

    log_str = f'Centroid refinement: {agreement*100:.2f}% agreement with initial predictions'
    out_file.write(log_str + '\n')
    out_file.flush()
    print(log_str)

    avg_confidence = refined_confidences.mean()
    log_str = f'Average confidence: {avg_confidence:.4f}'
    out_file.write(log_str + '\n')
    out_file.flush()
    print(log_str)

    return refined_pseudo_labels, refined_confidences

def feature_coverage_regularization(features):
    """
    Ensure model uses ALL feature dimensions, not just a subset.
    Based on NEST-CM (Neuron Structure Coverage Maximization).
    
    Args:
        features: [B*6, bottleneck_dim] - bottleneck features
    
    Returns:
        loss that encourages using all feature dimensions (negative for maximization)
    """
    # Compute feature covariance matrix
    features_centered = features - features.mean(dim=0)  # Center: [B*6, D]
    cov = (features_centered.T @ features_centered) / features.shape[0]  # [D, D]
    
    # Eigenvalue decomposition
    eigenvalues = torch.linalg.eigvalsh(cov)  # [D]
    eigenvalues = torch.abs(eigenvalues)  # Ensure positive
    
    # Normalize eigenvalues
    eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-8)
    
    # Identify under-utilized dimensions (small eigenvalues)
    dim = features.shape[1]
    threshold = 1.0 / dim  # Adaptive threshold (uniform baseline)
    
    # Find eigenvalues below threshold
    small_mask = eigenvalues < threshold
    small_eigenvalues = eigenvalues[small_mask]
    
    if small_eigenvalues.numel() == 0:
        # No under-utilized dimensions, return zero loss
        return torch.tensor(0.0, device=features.device)
    
    # MAXIMIZE small eigenvalues (force model to use all dimensions)
    # Return negative because we minimize loss in optimizer
    coverage_loss = -small_eigenvalues.sum()
    
    return coverage_loss


def slot_diversity_regularization(outputs_target, num_users, num_classes):
    """
    Encourage diversity across slots WITHIN each sample.
    Prevents all slots from predicting the same thing.
    
    Args:
        outputs_target: [B, num_users, num_classes+1]
        num_users: M (e.g., 6)
        num_classes: K (e.g., 9)
    
    Returns:
        loss that penalizes slot similarity within samples (positive, to minimize)
    """
    B = outputs_target.shape[0]
    
    # Softmax per slot
    slot_probs = F.softmax(outputs_target, dim=2)  # [B, 6, K+1]
    
    total_diversity_loss = 0
    
    for b in range(B):
        # Get slot predictions for this sample
        sample_slots = slot_probs[b]  # [6, K+1]
        
        # Compute pairwise cosine similarity between slots
        # Normalize slots first for cosine similarity
        sample_slots_norm = F.normalize(sample_slots, p=2, dim=1)  # [6, K+1]
        similarities = torch.mm(sample_slots_norm, sample_slots_norm.t())  # [6, 6]
        
        # We want LOW similarity between different slots
        # (diagonal is self-similarity = 1, ignore it)
        mask = 1 - torch.eye(num_users, device=similarities.device)
        off_diagonal_sim = similarities * mask
        
        # Average off-diagonal similarity
        avg_similarity = off_diagonal_sim.sum() / (num_users * (num_users - 1))
        
        total_diversity_loss += avg_similarity
    
    # Average over batch
    diversity_loss = total_diversity_loss / B
    
    return diversity_loss  # Positive value to minimize
def hierarchical_gent(softmax_out, num_classes):
    """
    Two-level diversity: 
    1. Diverse occupancy (empty vs. occupied)
    2. Diverse activities (among occupied slots)
    """
    # Level 1: Occupancy diversity
    occupancy_probs = softmax_out[:, :num_classes].sum(dim=1)  # P(occupied)
    empty_probs = softmax_out[:, num_classes]  # P(no person)
    
    # Want balanced occupancy distribution
    occupancy_marginal = torch.stack([
        occupancy_probs.mean(),
        empty_probs.mean()
    ])
    occupancy_entropy = -(occupancy_marginal * torch.log(occupancy_marginal + 1e-5)).sum()
    
    # Level 2: Activity diversity (among occupied slots)
    activity_probs = softmax_out[:, :num_classes]  # [B*6, K]
    activity_weights = occupancy_probs.unsqueeze(1)  # [B*6, 1]
    weighted_activities = activity_probs * activity_weights
    activity_marginal = weighted_activities.mean(dim=0)
    activity_marginal = activity_marginal / (activity_marginal.sum() + 1e-8)
    activity_entropy = -(activity_marginal * torch.log(activity_marginal + 1e-5)).sum()
    
    # Maximize both
    return -(occupancy_entropy + activity_entropy)  

def occupancy_conditioned_entropy(outputs_target, num_classes):
    """
    Two-stage entropy minimization:
    1. Occupancy (occupied vs empty) - minimize uncertainty
    2. Activity (only for occupied slots) - minimize uncertainty conditionally
    """
    B, M, K_plus_1 = outputs_target.shape  # [B, 6, num_classes+1]
    
    probs = F.softmax(outputs_target, dim=2)  # [B, 6, num_classes+1]
    
    # Stage 1: Occupancy entropy (binary: occupied vs empty)
    activity_prob = probs[:, :, :num_classes].sum(dim=2)  # [B, 6] - P(any activity)
    empty_prob = probs[:, :, num_classes]  # [B, 6] - P(no person)
    
    # Binary distribution [B, 6, 2]
    occupancy_dist = torch.stack([activity_prob, empty_prob], dim=2)
    occupancy_entropy = -(occupancy_dist * torch.log(occupancy_dist + 1e-5)).sum(dim=2)  # [B, 6]
    occupancy_loss = occupancy_entropy.mean()
    
    # Stage 2: Activity entropy (only for occupied slots)
    occupied_mask = (activity_prob > 0.5).float()  # [B, 6]
    
    # Normalize activity distribution (exclude "no_person")
    activity_probs = probs[:, :, :num_classes]  # [B, 6, num_classes]
    activity_probs_norm = activity_probs / (activity_probs.sum(dim=2, keepdim=True) + 1e-8)
    
    # Activity entropy
    activity_entropy = -(activity_probs_norm * torch.log(activity_probs_norm + 1e-5)).sum(dim=2)  # [B, 6]
    activity_loss = (activity_entropy * occupied_mask).sum() / (occupied_mask.sum() + 1e-8)
    
    # Combine (weight occupancy more - it's the critical decision)
    total_loss = 1.0 * occupancy_loss + 0.5 * activity_loss
    
    return total_loss

def occupancy_count_consistency(outputs_target, num_classes):
    """
    Force occupancy count to be close to integer (0, 1, 2, 3, 4, 5, 6)
    """
    B, M, K_plus_1 = outputs_target.shape
    probs = F.softmax(outputs_target, dim=2)
    activity_prob = probs[:, :, :num_classes].sum(dim=2)  # [B, 6]
    
    # Expected occupancy count per sample
    expected_count = activity_prob.sum(dim=1)  # [B]
    
    # Distance to nearest integer
    nearest_int = torch.round(expected_count)
    count_loss = F.mse_loss(expected_count, nearest_int)
    
    return count_loss
#
def forward_with_reshape(netF, netB, netC, inputs, num_users, num_classes):
    """
    Forward pass with automatic reshaping based on architecture.
    
    Args:
        netF, netB, netC: Network modules
        inputs: [B, T, F] input tensor
        num_users: Number of slots (6)
        num_classes: Number of activity classes (9)
    
    Returns:
        outputs: [B, num_users, num_classes+1] - always this shape
    """
    # Forward pass through all networks
    outputs = netC(netB(netF(inputs)))
    B = outputs.shape[0]
    
    # Check architecture flag
    use_transformer = preset["training"].get("use_transformer", False)
    
    if use_transformer:
        # Transformer already outputs [B, num_users, num_classes+1]
        assert outputs.shape == (B, num_users, num_classes + 1), \
            f"Transformer output shape mismatch: expected {(B, num_users, num_classes + 1)}, got {outputs.shape}"
        return outputs
    else:
        # Linear classifier outputs [B, num_users*(num_classes+1)]
        # Need to reshape to [B, num_users, num_classes+1]
        assert outputs.shape == (B, num_users * (num_classes + 1)), \
            f"Linear output shape mismatch: expected {(B, num_users * (num_classes + 1))}, got {outputs.shape}"
        return outputs.view(B, num_users, num_classes + 1)
        
def get_features_and_outputs(netF, netB, netC, inputs, num_users, num_classes):
    """
    Get both bottleneck features and reshaped outputs.
    Useful for pseudo-labeling and centroid building.
    
    Returns:
        features: [B, bottleneck_dim]
        outputs: [B, num_users, num_classes+1]
    """
    features = netB(netF(inputs))
    outputs = netC(features)
    B = outputs.shape[0]
    
    use_transformer = preset["training"].get("use_transformer", False)
    if not use_transformer:
        outputs = outputs.view(B, num_users, num_classes + 1)
    
    return features, outputs

def slot_occupancy_diversity_loss(logits, num_classes):
    """
    Slot-aware diversity loss for transformer queries in SFUDA.
    Replaces occupancy-weighted GENT when using learnable queries.
    
    Encourages:
    1. Diverse slot activations (not all predicting no_person)
    2. Diverse activity predictions across active slots
    
    Args:
        logits: [B, num_queries, num_classes+1]
        num_classes: Number of activity classes (excluding no_person)
    
    Returns:
        Diversity loss (to be SUBTRACTED like GENT, or use negative)
    """
    B, Q, C = logits.shape
    no_person_class = num_classes
    
    # Get probabilities
    probs = F.softmax(logits, dim=-1)  # [B, Q, num_classes+1]
    
    # Separate activity vs no_person probabilities
    activity_probs = probs[:, :, :num_classes]  # [B, Q, num_classes]
    no_person_probs = probs[:, :, no_person_class]  # [B, Q]
    
    # Occupancy probability per slot (sum of all activities)
    occupancy_probs = activity_probs.sum(dim=-1)  # [B, Q]
    
    # Weight activity predictions by how "occupied" each slot is
    weighted_activities = activity_probs * occupancy_probs.unsqueeze(-1)  # [B, Q, num_classes]
    
    # Average across batch and slots to get marginal distribution
    marginal = weighted_activities.sum(dim=(0, 1))  # [num_classes]
    marginal = marginal / (marginal.sum() + 1e-8)
    
    # Compute entropy of marginal (high = diverse)
    diversity = -torch.sum(marginal * torch.log(marginal + 1e-8))
    
    return diversity
    
def train_target_shot(source_dir, output_dir, out_file, random_state):
    """Domain adaptation training on target"""
    
    print("Loading target data for adaptation...")
    train_loader, valid_loader, test_loader, classifier_output_size, var_x_shape, num_users, num_classes = dset_loaders("target_data", random_state=random_state)
    
    # ============================================================
    # LOAD SOURCE DATA FOR UDA (if enabled)
    # ============================================================
    use_uda = preset["training"].get("use_uda", False)
    source_train_loader = None
    
    if use_uda:
        print("\n" + "="*50)
        print("UDA MODE: Loading source domain data")
        print("="*50)
        source_train_loader, _, _, _, _, _, _ = dset_loaders("source_data", random_state=random_state)
        print(f"Source batches: {len(source_train_loader)}")
        print(f"Target batches: {len(train_loader)}")
    
    # Create non-shuffled loader for pseudo-labeling

    train_loader_no_shuffle, _, _, _, _, _, _ = dset_loaders("target_data", random_state=random_state)
    # Override with non-shuffled version
    # Unwrap the IndexDataset to get to the underlying TensorDataset
    if hasattr(train_loader.dataset, 'dataset'):
        inner_dataset = train_loader.dataset.dataset
    else:
        inner_dataset = train_loader.dataset

    if hasattr(inner_dataset, 'tensors'):
        train_data_x = inner_dataset.tensors[0].cpu().numpy()
        train_data_y = inner_dataset.tensors[1].cpu().numpy()
    elif hasattr(inner_dataset, 'data'): # For AugMixDataset
        train_data_x = inner_dataset.data
        train_data_y = inner_dataset.targets
        
    train_loader_no_shuffle = create_dataset(train_data_x, train_data_y,
                                             config=preset["target_data"],
                                             shuffle=False, is_training=False)
    
    # ============================================================
    # CREATE MODELS
    # ============================================================
    netF, netB, netC = create_models(
        var_x_shape=var_x_shape,
        num_users=num_users,
        num_classes=num_classes,
        bottleneck_dim=preset["training"]["bottleneck"]
    )
    
    # Load source model weights
    netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(source_dir, "source_B.pt")))
    netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt")))
    
    # ============================================================
    # DUAL BRANCH SETUP (if enabled)
    # ============================================================
    dual_branch = preset["training"].get("dual_branch", False)
    netF_teacher, netB_teacher, netC_teacher = None, None, None
    
    if dual_branch:
        print("\n" + "="*60)
        print("DUAL BRANCH ADAPTATION MODE")
        print("="*60)
        
        # ========================================
        # BRANCH 1: SHOT-IM ADAPTATION (Discriminative)
        # ========================================
        print("\n" + "="*50)
        print("BRANCH 1: SHOT-IM Adaptation (Discriminative Focus)")
        print("="*50)
        
        branch1_F_path = osp.join(output_dir, "branch1_adapted_F.pt")
        
        if not os.path.exists(branch1_F_path):
            print("Training Branch 1 from scratch...")
            
            # Branch 1 config: SHOT-IM only
            branch1_config = {
                "ent_par": preset["training"].get("branch1_ent_par", 0.1),
                "gent": preset["training"].get("branch1_gent", True),
                "max_epoch": preset["training"].get("branch1_epochs", 30)
            }
            
            # Run Branch 1 adaptation
            branch1_models = adapt_branch1(
                source_dir, output_dir, train_loader, valid_loader, out_file,
                branch1_config, num_users, num_classes, var_x_shape
            )
            
            print("✓ Branch 1 adaptation complete")
        else:
            print("✓ Branch 1 already trained, loading from disk...")
        
        # ========================================
        # Load Branch 1 as TEACHER (frozen)
        # ========================================
        netF_teacher, netB_teacher, netC_teacher = create_models(
            var_x_shape=var_x_shape,
            num_users=num_users,
            num_classes=num_classes,
            bottleneck_dim=preset["training"]["bottleneck"]
        )
        
        netF_teacher.load_state_dict(torch.load(osp.join(output_dir, "branch1_adapted_F.pt")))
        netB_teacher.load_state_dict(torch.load(osp.join(output_dir, "branch1_adapted_B.pt")))
        netC_teacher.load_state_dict(torch.load(osp.join(output_dir, "branch1_adapted_C.pt")))
        
        # Freeze teacher
        for param in netF_teacher.parameters():
            param.requires_grad = False
        for param in netB_teacher.parameters():
            param.requires_grad = False
        for param in netC_teacher.parameters():
            param.requires_grad = False
        
        netF_teacher.eval()
        netB_teacher.eval()
        netC_teacher.eval()
        
        print("✓ Branch 1 teacher models loaded and frozen")
        
        # ========================================
        # BRANCH 2: Reinitialize from source (student)
        # ========================================
        print("\n" + "="*50)
        print("BRANCH 2: SSL + Distillation (Occupancy Focus)")
        print("="*50)
        
        # Reload fresh source weights for Branch 2 student
        netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt")))
        netB.load_state_dict(torch.load(osp.join(source_dir, "source_B.pt")))
        netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt")))
        
        print("✓ Branch 2 (student) reinitialized from source")
        
        # Distillation hyperparameters
        feature_distill_weight = preset["training"].get("feature_distill_weight", 1.0)
        logit_distill_weight = preset["training"].get("logit_distill_weight", 0.5)
        temperature = preset["training"].get("distill_temperature", 4.0)
        
        print(f"  Feature distillation weight: {feature_distill_weight}")
        print(f"  Logit distillation weight: {logit_distill_weight}")
        print(f"  Temperature: {temperature}")
    
    # ============================================================
    # SSL ROTATION PRE-TRAINING (if enabled)
    # ============================================================
    ssl_metrics = None
    netR = None
    if preset["training"]["ssl"] > 0:
        print("\n" + "="*50)
        print("SSL ROTATION PRE-TRAINING")
        print("="*50)
        netR = network.feat_classifier(class_num=2, bottleneck_dim=2*preset["training"]["bottleneck"]).cuda()
        netR_dict, ssl_metrics = train_target_rot(output_dir, out_file, train_loader, netF, netB, num_users, num_classes)
        get_model_device_safe(netR).load_state_dict(netR_dict)
        print(f"SSL Rotation Accuracy: {ssl_metrics['best_rotation_accuracy']:.2f}% (Epoch {ssl_metrics['best_epoch']})\n")
    
    # ============================================================
    # WRAP IN DATAPARALLEL
    # ============================================================
    if torch.cuda.device_count() > 1:
        netF, netB, netC = nn.DataParallel(netF), nn.DataParallel(netB), nn.DataParallel(netC)
        if netR is not None:
            netR = nn.DataParallel(netR)
        
        # Wrap teacher models if dual_branch
        if dual_branch:
            netF_teacher = nn.DataParallel(netF_teacher)
            netB_teacher = nn.DataParallel(netB_teacher)
            netC_teacher = nn.DataParallel(netC_teacher)
    
    # ============================================================
    # SETUP TRAINABLE PARAMETERS
    # ============================================================
    # Check if we should unfreeze classifier
    unfreeze_classifier = preset["training"].get("unfreeze_classifier", False)
    
    # Freeze or unfreeze classifier
    if unfreeze_classifier:
        for v in get_model_device_safe(netC).parameters():
            v.requires_grad = True
    else:
        for v in get_model_device_safe(netC).parameters():
            v.requires_grad = False
    
    # Build parameter groups
    param_group = []
    for v in get_model_device_safe(netF).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
    for v in get_model_device_safe(netB).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
    
    # Add netC if unfrozen
    if unfreeze_classifier:
        for v in get_model_device_safe(netC).parameters():
            param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
    
    # Add netR to optimizer if SSL enabled
    if netR is not None:
        for v in get_model_device_safe(netR).parameters():
            param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
        netR.train()
    
    # ============================================================
    # DOMAIN DISCRIMINATOR FOR ADVERSARIAL ALIGNMENT (if enabled)
    # ============================================================
    netD = None
    if use_uda and preset["training"].get("use_adversarial", False):
        netD = network.feat_classifier(
            class_num=2,  # Binary: source vs target
            bottleneck_dim=preset["training"]["bottleneck"]
        ).cuda()
        if torch.cuda.device_count() > 1:
            netD = nn.DataParallel(netD)
        
        # Add discriminator to optimizer
        for v in get_model_device_safe(netD).parameters():
            param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
    
    # Create optimizer
    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["target_lr"], weight_decay=1e-4))
    
    # ============================================================
    # TRAINING SETUP
    # ============================================================
    max_epochs = preset["training"]["max_epoch"]
    best_acc = 0
    best_f1_micro = 0
    best_f1_macro = 0
    best_hamming = float('inf')
    best_epoch = 0
    best_metrics = {}
    best_netF = None
    best_netB = None
    best_netC = None
    
    # Store adaptation history
    adaptation_history = []
    
    # ============================================================
    # PSEUDO-LABELING SETUP (if enabled)
    # ============================================================
    mem_pseudo_labels = None
    target_centroids = None
    cls_par = preset["training"].get("cls_par", 0.0)
    
    if cls_par > 0:
        print("\n" + "="*50)
        print("BUILDING INITIAL CENTROIDS & PSEUDO-LABELS")
        print("="*50)
        
        # Build centroids from target data
        print("Building per-class centroids...")
        target_centroids = build_target_centroids(
            train_loader_no_shuffle, netF, netB, netC, num_users, num_classes, conf_thresh=0.5
        )
        print(f"Centroids built for {num_classes} classes")
        
        # Generate pseudo-labels using centroids
        print("Generating pseudo-labels from centroids...")
        mem_pseudo_labels, mem_confidences = obtain_pseudo_labels_with_centroids(
            train_loader_no_shuffle, netF, netB, netC, target_centroids, num_users, num_classes, out_file
        )
        print(f"Pseudo-labeling loss weight: {cls_par}")
    
    # ============================================================
    # ESTIMATE SOURCE CLASS DISTRIBUTION (once, before adaptation)
    # ============================================================
    print("\n" + "="*60)
    print("ESTIMATING SOURCE CLASS DISTRIBUTION ON TARGET DATA")
    print("="*60)
    
    netF.eval()
    netB.eval()
    netC.eval()
    
    source_marginal_list = []
    with torch.no_grad():
        for inputs_target, _, _ in train_loader:
            inputs_target = inputs_target.cuda()
            
            # Simple forward pass - don't need features here
            outputs_target = forward_with_reshape(
                netF, netB, netC, inputs_target, num_users, num_classes
            )
            B = outputs_target.shape[0]
            
            # Flatten to [B*6, num_classes+1]
            outputs_flat = outputs_target.view(B * num_users, num_classes + 1)
            
            # Softmax to get probabilities
            softmax_out = F.softmax(outputs_flat, dim=1)  # [B*6, num_classes+1]
            pred = softmax_out.argmax(dim=1)
            no_person_argmax = (pred == num_classes).float().mean().item() * 100
            no_person_marg = softmax_out.mean(dim=0)[num_classes].item() * 100
            
            print(f"no_person(argmax) = {no_person_argmax:.2f}%")
            print(f"no_person(marginal)= {no_person_marg:.2f}%")
            
            # Accumulate batch marginal
            source_marginal_list.append(softmax_out.mean(dim=0))  # [num_classes+1]
    
    # Average across all batches
    source_marginal = torch.stack(source_marginal_list).mean(dim=0)  # [num_classes+1]
    
    print("\nEstimated source class distribution:")
    for k in range(num_classes):
        print(f"  Activity {k}: {source_marginal[k].item()*100:.2f}%")
    print(f"  No person: {source_marginal[num_classes].item()*100:.2f}%")
    print("="*60 + "\n")
    
    # ============================================================
    # MAIN TRAINING LOOP
    # ============================================================
    print("\nStarting target adaptation...")
    # Log actual weights used (preset is read at process start; matches batch job logs)
    _ssl = preset["training"]["ssl"]
    _ent = preset["training"].get("ent_par", 1.0)
    print(f"ADAPTATION WEIGHTS (from preset): ssl={_ssl}, ent_par={_ent}")
    
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train()
        netB.train()
        netC.train() if unfreeze_classifier else netC.eval()
        if netD is not None:
            netD.train()
        
        # ========================================================
        # UPDATE CENTROIDS & PSEUDO-LABELS (periodically)
        # ========================================================
        if cls_par > 0 and epoch > 0 and epoch % 10 == 0:
            print(f"\n" + "="*50)
            print(f"UPDATING CENTROIDS & PSEUDO-LABELS (Epoch {epoch+1})")
            print("="*50)
            
            # Set models to eval mode for pseudo-label generation
            netF.eval()
            netB.eval()
            
            # Rebuild centroids
            print("Rebuilding per-class centroids...")
            target_centroids = build_target_centroids(
                train_loader_no_shuffle, netF, netB, netC, num_users, num_classes, conf_thresh=0.5
            )
            
            # Generate new pseudo-labels
            print("Regenerating pseudo-labels from centroids...")
            mem_pseudo_labels, mem_confidences = obtain_pseudo_labels_with_centroids(
                train_loader_no_shuffle, netF, netB, netC, target_centroids, num_users, num_classes, out_file
            )
            
            # Restore training mode
            netF.train()
            netB.train()
            netC.train() if unfreeze_classifier else netC.eval()
        
        epoch_loss = 0
        num_batches = 0
        batch_idx = 0
        
        # Create source iterator for UDA
        if use_uda:
            source_iter = iter(source_train_loader)
        
        # ========================================================
        # BATCH TRAINING LOOP
        # ========================================================
        for inputs_target, _, target_indices in train_loader: 
            inputs_target = inputs_target.cuda()
            target_indices = target_indices.cuda() # Indices of the current batch samples
            # ====================================================
            # GET SOURCE BATCH (if UDA enabled)
            # ====================================================
            if use_uda:
                try:
                    inputs_source, labels_source = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_train_loader)
                    inputs_source, labels_source = next(source_iter)
                
                inputs_source = inputs_source.cuda()
                labels_source = labels_source.cuda()
            
            # ====================================================
            # FORWARD PASS - STUDENT (Branch 2 or standard)
            # ====================================================
            # Determine if we need features
            need_features = (
                use_uda or 
                preset["training"].get("feature_coverage", False) or
                dual_branch  # Always need features for distillation
            )
            
            if need_features:
                features_student, outputs_student = get_features_and_outputs(
                    netF, netB, netC, inputs_target, num_users, num_classes
                )
                features_batch = features_student  # Keep old variable name for UDA compatibility
            else:
                outputs_student = forward_with_reshape(
                    netF, netB, netC, inputs_target, num_users, num_classes
                )
                features_student = None
                features_batch = None
            
            # Alias for backward compatibility
            outputs_target = outputs_student
            B = outputs_target.shape[0]
            
            # ====================================================
            # FORWARD PASS - SOURCE (for UDA)
            # ====================================================
            source_loss = torch.tensor(0.0).cuda()
            alignment_loss = torch.tensor(0.0).cuda()
            
            if use_uda:
                features_source, outputs_source = get_features_and_outputs(
                    netF, netB, netC, inputs_source, num_users, num_classes
                )
                B_src = outputs_source.shape[0]
                
                # ------------------------------------------------
                # SUPERVISED LOSS ON SOURCE
                # ------------------------------------------------
                if preset["training"].get("use_source_supervision", True):
                    # Hungarian matching for source
                    matched_pred_indices, matched_gt_indices = hungarian_matching_batch(
                        outputs_source, labels_source, num_classes
                    )
                    
                    # Compute CE loss
                    criterion = CrossEntropyLabelSmooth(
                        num_classes=num_classes + 1,
                        epsilon=preset["training"].get("smooth", 0.1),
                        use_gpu=True
                    )
                    
                    batch_source_loss = 0
                    for b in range(B_src):
                        pred_slots = outputs_source[b]
                        gt_slots = labels_source[b]
                        pred_indices = matched_pred_indices[b]
                        gt_indices = matched_gt_indices[b]
                        
                        for i in range(6):
                            pred_slot_idx = pred_indices[i]
                            gt_slot_idx = gt_indices[i]
                            pred_logits = pred_slots[pred_slot_idx]
                            gt_class = gt_slots[gt_slot_idx]
                            
                            batch_source_loss += criterion(
                                pred_logits.unsqueeze(0),
                                gt_class.unsqueeze(0)
                            )
                    
                    source_loss = batch_source_loss / (B_src * 6)
                    source_loss *= preset["training"].get("source_weight", 1.0)
                
                # ------------------------------------------------
                # MMD ALIGNMENT
                # ------------------------------------------------
                if preset["training"].get("use_mmd", False):
                    def mmd_rbf(x, y, sigma=None, eps=1e-8):
                        # Normalize for stability
                        x = F.normalize(x, dim=1)
                        y = F.normalize(y, dim=1)
                        
                        # Median heuristic sigma
                        with torch.no_grad():
                            if sigma is None:
                                z = torch.cat([x, y], dim=0)
                                if z.size(0) > 256:
                                    idx = torch.randperm(z.size(0), device=z.device)[:256]
                                    z = z[idx]
                                d2 = torch.cdist(z, z).pow(2)
                                sigma2 = torch.median(d2[d2 > 0])
                                sigma = torch.sqrt(sigma2 + eps).item()
                        
                        gamma = 1.0 / (2 * sigma * sigma + eps)
                        
                        xx = torch.cdist(x, x).pow(2)
                        yy = torch.cdist(y, y).pow(2)
                        xy = torch.cdist(x, y).pow(2)
                        
                        K_xx = torch.exp(-gamma * xx)
                        K_yy = torch.exp(-gamma * yy)
                        K_xy = torch.exp(-gamma * xy)
                        
                        return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean(), sigma
                    
                    mmd_w = preset["training"].get("mmd_weight", 0.1)
                    mmd_loss, sigma = mmd_rbf(features_source.detach(), features_batch, sigma=None)
                    alignment_loss += mmd_w * mmd_loss
                    
                    if batch_idx == 0:
                        print(f"[Epoch {epoch+1}] MMD sigma={sigma:.4f} raw={mmd_loss.item():.6f} | weighted={(mmd_w*mmd_loss).item():.6f}")
                
                # ------------------------------------------------
                # ADVERSARIAL ALIGNMENT
                # ------------------------------------------------
                if netD is not None:
                    # Domain labels: 0=source, 1=target
                    domain_label_src = torch.zeros(B_src).long().cuda()
                    domain_label_tgt = torch.ones(B).long().cuda()
                    
                    # Discriminator predictions
                    domain_pred_src = netD(features_source)
                    domain_pred_tgt = netD(features_batch)
                    
                    # Discriminator loss (train to classify domains)
                    disc_loss = nn.CrossEntropyLoss()(domain_pred_src, domain_label_src) + \
                                nn.CrossEntropyLoss()(domain_pred_tgt, domain_label_tgt)
                    
                    # Feature extractor loss (train to confuse discriminator)
                    confusion_loss = nn.CrossEntropyLoss()(domain_pred_tgt, domain_label_src)
                    
                    alignment_loss += preset["training"].get("adv_weight", 0.1) * confusion_loss
                
                # ------------------------------------------------
                # CORAL ALIGNMENT
                # ------------------------------------------------
                if preset["training"].get("use_coral", False):
                    def coral_loss(source, target, eps=1e-8):
                        d = source.size(1)
                        
                        # Center
                        source = source - source.mean(dim=0, keepdim=True)
                        target = target - target.mean(dim=0, keepdim=True)
                        
                        # Covariance
                        ns = source.size(0)
                        nt = target.size(0)
                        
                        cs = (source.t() @ source) / (ns - 1 + eps)
                        ct = (target.t() @ target) / (nt - 1 + eps)
                        
                        # Frobenius distance (scaled)
                        return ((cs - ct) ** 2).sum() / (4.0 * d * d)
                    
                    # Detach source so CORAL doesn't fight source supervision
                    src = F.normalize(features_source.detach(), dim=1)
                    tgt = F.normalize(features_batch, dim=1)
                    
                    coral = coral_loss(src, tgt)
                    coral_w = preset["training"].get("coral_weight", 0.1)
                    alignment_loss += coral_w * coral
                    
                    if batch_idx == 0:
                        print(f"[Epoch {epoch+1}] CORAL raw={coral.item():.6f} | weighted={(coral_w*coral).item():.6f}")
            
            # ====================================================
            # TASK LOSSES (Entropy + SSL + Occupancy)
            # ====================================================
            outputs_flat = outputs_student.view(B * num_users, num_classes + 1)
            softmax_out = F.softmax(outputs_flat, dim=1)
            
            # Choose entropy variant based on dual_branch or config
            if dual_branch:
                # Branch 2: Always use occupancy-focused entropy
                entropy_loss = occupancy_conditioned_entropy(outputs_student, num_classes)
                occ_count_loss = occupancy_count_consistency(outputs_student, num_classes)
                entropy_loss = entropy_loss + preset["training"].get("occ_count_weight", 0.3) * occ_count_loss
            elif preset["training"].get("use_occupancy_conditioned", False):
                # Standard occupancy-conditioned
                entropy_loss = occupancy_conditioned_entropy(outputs_student, num_classes)
                occ_count_loss = occupancy_count_consistency(outputs_student, num_classes)
                entropy_loss = entropy_loss + preset["training"].get("occ_count_weight", 0.3) * occ_count_loss
            else:
                # Standard entropy minimization
                entropy_per_slot = -softmax_out * torch.log(softmax_out + 1e-5)
                entropy_per_slot = entropy_per_slot.sum(dim=1)
                entropy_loss = entropy_per_slot.mean()
            
            # ====================================================
            # FEATURE SPACE COVERAGE REGULARIZATION
            # ====================================================
            if preset["training"].get("feature_coverage", False):
                # Expand features to match slot structure
                features_expanded = features_batch.unsqueeze(1).expand(B, num_users, -1)
                features_flat = features_expanded.reshape(B * num_users, -1)
                
                # Compute feature coverage loss
                coverage_loss = feature_coverage_regularization(features_flat)
                
                # Add to entropy loss
                entropy_loss += preset["training"].get("coverage_weight", 0.1) * coverage_loss
            
            # ====================================================
            # SLOT DIVERSITY REGULARIZATION
            # ====================================================
            if preset["training"].get("slot_diversity", False):
                slot_div_loss = slot_diversity_regularization(outputs_target, num_users, num_classes)
                entropy_loss += preset["training"].get("slot_div_weight", 0.3) * slot_div_loss
            
            # ====================================================
            # GENT: DIVERSITY REGULARIZATION
            # ====================================================
            if preset["training"].get("gent", False):
                use_class_balanced = preset["training"].get("use_class_balanced_gent", False)
                
                if use_class_balanced:
                    # Class-balanced GENT
                    current_marginal = softmax_out.mean(dim=0)
                    kl_loss = F.kl_div(
                        current_marginal.log(),
                        source_marginal,
                        reduction='batchmean'
                    )
                    entropy_loss += kl_loss
                elif preset["training"].get("use_slot_diversity_gent", False):
                    # NEW: Slot-aware diversity for transformer queries
                    slot_diversity = slot_occupancy_diversity_loss(outputs_target, num_classes)
                    entropy_loss -= slot_diversity  # Negative because we want HIGH diversity
                                        
                elif preset["training"].get("use_occupancy_weighted_gent", True):
                    # Occupancy-weighted GENT
                    softmax_classes = softmax_out[:, :num_classes]
                    occupied_prob = softmax_classes.sum(dim=1, keepdim=True)
                    weighted_softmax_classes = softmax_classes * occupied_prob
                    msoftmax = weighted_softmax_classes.mean(dim=0)
                    msoftmax = msoftmax / (msoftmax.sum() + 1e-8)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                    entropy_loss -= gentropy_loss
                    
                elif preset["training"].get("use_hierarchical_gent", False):
                    # Hierarchical GENT
                    activity_probs = softmax_out[:, :num_classes]
                    occupancy_probs = activity_probs.sum(dim=1)
                    empty_probs = softmax_out[:, num_classes]
                    
                    occupancy_marginal = torch.stack([
                        occupancy_probs.mean(),
                        empty_probs.mean()
                    ])
                    occupancy_entropy = -torch.sum(
                        occupancy_marginal * torch.log(occupancy_marginal + 1e-5)
                    )
                    
                    activity_weights = occupancy_probs.unsqueeze(1)
                    weighted_activities = activity_probs * activity_weights
                    activity_marginal = weighted_activities.mean(dim=0)
                    activity_marginal = activity_marginal / (activity_marginal.sum() + 1e-8)
                    activity_entropy = -torch.sum(
                        activity_marginal * torch.log(activity_marginal + 1e-5)
                    )
                    
                    hierarchical_gentropy = occupancy_entropy + activity_entropy
                    entropy_loss -= hierarchical_gentropy
                else:
                    # Standard GENT
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                    entropy_loss -= gentropy_loss
            
            # ====================================================
            # DISTILLATION LOSSES (Dual Branch only)
            # ====================================================
            distill_loss = torch.tensor(0.0, device=inputs_target.device)
            
            if dual_branch and features_student is not None:
                with torch.no_grad():
                    features_teacher, outputs_teacher = get_features_and_outputs(
                        netF_teacher, netB_teacher, netC_teacher,
                        inputs_target, num_users, num_classes
                    )
            
                # ============================================================
                # 1) FEATURE DISTILLATION (sample-level)
                # ============================================================
                # (Optional but usually keep small weight; can also normalize)
                fs = F.normalize(features_student, dim=1)
                ft = F.normalize(features_teacher, dim=1)
                feature_distill_loss = F.mse_loss(fs, ft)
            
                # ============================================================
                # 2) CONDITIONAL ACTIVITY KD (slot-matched, occupancy-weighted)
                #    - distill ONLY activities (0..K-1)
                #    - DO NOT distill "no_person" directly
                # ============================================================
                T = preset["training"].get("distill_temperature", 4.0)
            
                # probs with temperature, shape [B, M, K+1]
                p_s = F.softmax(outputs_student / T, dim=-1)
                p_t = F.softmax(outputs_teacher / T, dim=-1)
            
                # occupancy prob from teacher, [B, M]
                occ_t = 1.0 - p_t[..., num_classes]  # 1 - p(no_person)
            
                logit_distill_accum = torch.tensor(0.0, device=inputs_target.device)
                weight_accum = torch.tensor(0.0, device=inputs_target.device)
            
                eps = 1e-8
            
                for b in range(B):
                    # --- Conditional activity distributions q(y|occ) over K classes ---
                    ps_act = p_s[b, :, :num_classes]  # [M, K]
                    pt_act = p_t[b, :, :num_classes]  # [M, K]
            
                    qs = ps_act / (ps_act.sum(dim=-1, keepdim=True) + eps)  # [M, K]
                    qt = pt_act / (pt_act.sum(dim=-1, keepdim=True) + eps)  # [M, K]
            
                    # --- Hungarian match teacher slots -> student slots ---
                    # Similarity on conditional activity distributions
                    # cost = - cosine-like dot product (since they sum to 1, dot works well)
                    cost = -(qs @ qt.t())  # [M, M]
                    _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            
                    idx = torch.tensor(col_ind, device=inputs_target.device, dtype=torch.long)
            
                    qt_m = qt[idx]                 # teacher matched, [M, K]
                    occ_w = occ_t[b, idx].detach() # teacher occupancy weight for matched slots, [M]
            
                    # --- KL( teacher || student ) on activities only ---
                    # Use logits directly for numerical stability:
                    # student log prob over activities (conditional)
                    stud_logits_act = outputs_student[b, :, :num_classes] / T
                    log_qs = F.log_softmax(stud_logits_act, dim=-1)  # [M, K]
            
                    # teacher probs (conditional) already in qt_m
                    per_slot_kl = F.kl_div(log_qs, qt_m, reduction="none").sum(dim=-1)  # [M]
            
                    # Weight by teacher occupancy so empty slots don't dominate KD
                    logit_distill_accum += (occ_w * per_slot_kl).sum()
                    weight_accum += occ_w.sum()
            
                if weight_accum.item() > 1e-6:
                    logit_distill_loss = (logit_distill_accum / (weight_accum + eps)) * (T ** 2)
                else:
                    logit_distill_loss = torch.tensor(0.0, device=inputs_target.device)
            
                # ============================================================
                # COMBINE
                # ============================================================
                w_feat = preset["training"].get("feature_distill_weight", 0.1)  # recommend small
                w_logit = preset["training"].get("logit_distill_weight", 0.05)  # recommend MUCH smaller than 0.5
            
                distill_loss = w_feat * feature_distill_loss + w_logit * logit_distill_loss
            
                if batch_idx % 20 == 0:
                    print(f"  [Distill] feat={feature_distill_loss.item():.4f} | "
                          f"actKD={logit_distill_loss.item():.4f} | "
                          f"w_occ={(weight_accum.item() / (B * num_users)):.3f} | "
                          f"total={distill_loss.item():.4f}")

            # ====================================================
            # INFORMATION MAXIMIZATION LOSS
            # ====================================================
            im_loss = entropy_loss * preset["training"]["ent_par"]
            
            # ====================================================
            # PSEUDO-LABELING LOSS
            # ====================================================
            pseudo_loss = torch.tensor(0.0).cuda()
            if cls_par > 0 and mem_pseudo_labels is not None:
                current_batch_pseudo_labels = mem_pseudo_labels[target_indices.cpu().numpy()]
                
                pseudo_labels_batch = torch.from_numpy(current_batch_pseudo_labels).cuda()
                
                # Hungarian matching
                matched_pred_indices, matched_gt_indices = hungarian_matching_batch(
                    outputs_target, pseudo_labels_batch, num_classes
                )
                
                # Compute CE loss on matched pairs
                criterion = nn.CrossEntropyLoss()
                batch_pseudo_loss = 0
                for b in range(B):
                    pred_slots = outputs_target[b]
                    pseudo_slots = pseudo_labels_batch[b]
                    pred_indices = matched_pred_indices[b]
                    gt_indices = matched_gt_indices[b]
                    
                    for i in range(6):
                        pred_slot_idx = pred_indices[i]
                        gt_slot_idx = gt_indices[i]
                        pred_logits = pred_slots[pred_slot_idx]
                        pseudo_class = pseudo_slots[gt_slot_idx]
                        
                        batch_pseudo_loss += criterion(
                            pred_logits.unsqueeze(0),
                            pseudo_class.unsqueeze(0)
                        )
                
                pseudo_loss = cls_par * (batch_pseudo_loss / (B * 6))
            
            # ====================================================
            # TEMPORAL CONSISTENCY LOSS
            # ====================================================
            temporal_loss = torch.tensor(0.0).cuda()
            if preset["training"].get("temporal_consistency", False):
                temporal_loss = compute_temporal_consistency_loss(
                    inputs_target, netF, netB, netC, num_users, num_classes
                )
                temporal_loss = preset["training"].get("temporal_weight", 0.5) * temporal_loss
            
            # ====================================================
            # TOTAL LOSS
            # ====================================================
            total_loss = im_loss + pseudo_loss + temporal_loss + source_loss + alignment_loss + distill_loss
            
            # ====================================================
            # SSL ROTATION LOSS
            # ====================================================
            if preset["training"]["ssl"] > 0 and netR is not None:
                r_labels = torch.randint(0, 2, (inputs_target.shape[0],), dtype=torch.long, device=inputs_target.device)
                r_inputs = rotation.rotate_batch_with_labels(inputs_target, r_labels)
                
                f_outputs = netB(netF(inputs_target)).detach()
                f_r_outputs = netB(netF(r_inputs))
                
                # ========================================
                # HANDLE DETR MODE: Pool spatial features
                # ========================================
                use_transformer = preset["training"].get("use_transformer", False)
                if use_transformer:
                    # DETR mode: pool [B, seq_len, D] → [B, D]
                    f_outputs = f_outputs.mean(dim=1)
                    f_r_outputs = f_r_outputs.mean(dim=1)
                
                r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))
                
                rotation_loss = preset["training"]["ssl"] * nn.CrossEntropyLoss()(r_outputs, r_labels)
                total_loss = total_loss + rotation_loss
            
            # ====================================================
            # BACKPROPAGATION
            # ====================================================
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            batch_idx += 1
        
        # ========================================================
        # EPOCH LOGGING
        # ========================================================
        avg_loss = epoch_loss / num_batches
        
        if epoch % 1 == 0:
            print(f"im={im_loss.item():.3f} "
                  f"src={source_loss.item():.3f} "
                  f"align={alignment_loss.item():.3f} "
                  f"distill={distill_loss.item():.3f} "
                  f"rot={(rotation_loss.item() if 'rotation_loss' in locals() else 0):.3f}")
        
        if epoch % 5 == 0:
            print(f"IM Loss: {im_loss.item():.4f}, Consistency Loss: {temporal_loss.item():.4f}")
            pred_classes = softmax_out.argmax(dim=1)
            no_person_count = (pred_classes == num_classes).sum().item()
            total_slots = pred_classes.shape[0]
            print(f"Epoch {epoch}: Predicting 'no person' for {no_person_count}/{total_slots} slots ({100*no_person_count/total_slots:.1f}%)")
        
        # ========================================================
        # VALIDATION
        # ========================================================
        netF.eval()
        netB.eval()
        valid_metrics = cal_acc(valid_loader, netF, netB, netC, num_users, num_classes)
        
        # Store epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'valid_slot_wise_accuracy': valid_metrics['slot_wise_accuracy'],
            'valid_exact_match_accuracy': valid_metrics['exact_match_accuracy'],
            'valid_f1_micro': valid_metrics['f1_micro'],
            'valid_f1_macro': valid_metrics['f1_macro'],
            'valid_activity_macro_f1': valid_metrics['activity_macro_f1'],
            'valid_occupancy_mae': valid_metrics['occupancy_mae'],
            'valid_occupancy_exact_match': valid_metrics['occupancy_exact_match'],
            'valid_hamming_loss': valid_metrics['hamming_loss'],
            'valid_per_activity_f1': valid_metrics['per_activity_f1'],
        }
        adaptation_history.append(epoch_results)
        
        # ========================================================
        # SAVE BEST MODEL
        # ========================================================
        if valid_metrics['slot_wise_accuracy'] > best_acc:
            best_acc = valid_metrics['slot_wise_accuracy']
            best_f1_micro = valid_metrics['f1_micro']
            best_f1_macro = valid_metrics['f1_macro']
            best_hamming = valid_metrics['hamming_loss']
            best_epoch = epoch + 1
            
            best_netF = copy.deepcopy(get_model_device_safe(netF).state_dict())
            best_netB = copy.deepcopy(get_model_device_safe(netB).state_dict())
            if unfreeze_classifier:
                best_netC = copy.deepcopy(get_model_device_safe(netC).state_dict())
            
            best_metrics = {
                "valid_slot_wise_accuracy": float(valid_metrics['slot_wise_accuracy']),
                "valid_exact_match_accuracy": float(valid_metrics['exact_match_accuracy']),
                "valid_f1_micro": float(valid_metrics['f1_micro']),
                "valid_f1_macro": float(valid_metrics['f1_macro']),
                "valid_activity_macro_f1": float(valid_metrics['activity_macro_f1']),
                "valid_occupancy_mae": float(valid_metrics['occupancy_mae']),
                "valid_occupancy_exact_match": float(valid_metrics['occupancy_exact_match']),
                "valid_hamming_loss": float(valid_metrics['hamming_loss']),
                "valid_per_activity_f1": valid_metrics['per_activity_f1'],
                "epoch": best_epoch
            }
            
            # Save validation report
            savename = f'ent_{preset["training"]["ent_par"]}'
            if dual_branch:
                savename += '_dual_branch'
            
            report_filename = osp.join(output_dir, f"target_adapted_best_validation_report_{savename}.txt")
            with open(report_filename, 'w') as f:
                f.write(f"Best Target Adapted Model Performance on Validation Set (Epoch {epoch+1})\n")
                f.write("="*60 + "\n\n")
                f.write(f"Slot-wise Accuracy: {valid_metrics['slot_wise_accuracy']:.2f}%\n")
                f.write(f"Exact Match Accuracy: {valid_metrics['exact_match_accuracy']:.2f}%\n")
                f.write(f"F1-Micro (all classes): {valid_metrics['f1_micro']:.2f}%\n")
                f.write(f"F1-Macro (all classes): {valid_metrics['f1_macro']:.2f}%\n")
                f.write(f"Activity Macro-F1 (9 activities): {valid_metrics['activity_macro_f1']:.2f}%\n")
                f.write(f"Occupancy MAE: {valid_metrics['occupancy_mae']:.4f}\n")
                f.write(f"Occupancy Exact Match: {valid_metrics['occupancy_exact_match']:.2f}%\n")
                f.write(f"Hamming Loss: {valid_metrics['hamming_loss']:.2f}%\n\n")
                f.write("Per-Activity F1 Scores:\n")
                f.write("-"*40 + "\n")
                for act_key, f1_val in valid_metrics['per_activity_f1'].items():
                    f.write(f"  {act_key}: {f1_val:.2f}%\n")
                f.write("\n" + "="*60 + "\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(valid_metrics['classification_report'])
        
        log_str = f'Target Adaptation - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Valid Slot-Acc: {valid_metrics["slot_wise_accuracy"]:.2f}%, Activity-F1: {valid_metrics["activity_macro_f1"]:.2f}%'
        out_file.write(log_str + '\n')
        out_file.flush()
        print(log_str)
    
    # ============================================================
    # TEST SET EVALUATION
    # ============================================================
    print("\nEvaluating best adapted model on TEST set...")
    
    # Load best model states
    base_netF = get_model_device_safe(netF) if hasattr(netF, 'module') else netF
    base_netB = get_model_device_safe(netB) if hasattr(netB, 'module') else netB
    base_netC = get_model_device_safe(netC) if hasattr(netC, 'module') else netC
    
    base_netF.load_state_dict(best_netF)
    base_netB.load_state_dict(best_netB)
    if unfreeze_classifier:
        base_netC.load_state_dict(best_netC)
    
    # Wrap in DataParallel if needed
    if torch.cuda.device_count() > 1:
        netF = nn.DataParallel(base_netF)
        netB = nn.DataParallel(base_netB)
        netC = nn.DataParallel(base_netC)
    else:
        netF = base_netF
        netB = base_netB
        netC = base_netC
    
    netF.eval()
    netB.eval()
    netC.eval()
    
    test_metrics = cal_acc(test_loader, netF, netB, netC, num_users, num_classes)
    
    # Save test results
    savename = f'ent_{preset["training"]["ent_par"]}'
    if dual_branch:
        savename += '_dual_branch'
    
    test_report_filename = osp.join(output_dir, f"target_adapted_best_test_report_{savename}.txt")
    with open(test_report_filename, 'w') as f:
        f.write(f"Best Target Adapted Model Performance on TEST Set\n")
        f.write("="*60 + "\n\n")
        f.write(f"Slot-wise Accuracy: {test_metrics['slot_wise_accuracy']:.2f}%\n")
        f.write(f"Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.2f}%\n")
        f.write(f"F1-Micro (all classes): {test_metrics['f1_micro']:.2f}%\n")
        f.write(f"F1-Macro (all classes): {test_metrics['f1_macro']:.2f}%\n")
        f.write(f"Activity Macro-F1 (9 activities): {test_metrics['activity_macro_f1']:.2f}%\n")
        f.write(f"Occupancy MAE: {test_metrics['occupancy_mae']:.4f}\n")
        f.write(f"Occupancy Exact Match: {test_metrics['occupancy_exact_match']:.2f}%\n")
        f.write(f"Hamming Loss: {test_metrics['hamming_loss']:.2f}%\n\n")
        f.write("Per-Activity F1 Scores:\n")
        f.write("-"*40 + "\n")
        for act_key, f1_val in test_metrics['per_activity_f1'].items():
            f.write(f"  {act_key}: {f1_val:.2f}%\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(test_metrics['classification_report'])
    
    # Save test metrics as JSON
    test_metrics_file = osp.join(output_dir, f"target_adapted_best_test_metrics_{savename}.json")
    test_metrics_for_json = {k: v for k, v in test_metrics.items() if k != 'classification_report'}
    with open(test_metrics_file, 'w') as f:
        json.dump(test_metrics_for_json, f, indent=2)
    
    # Save best adapted models
    torch.save(best_netF, osp.join(output_dir, f"adapted_F_{savename}.pt"))
    torch.save(best_netB, osp.join(output_dir, f"adapted_B_{savename}.pt"))
    
    log_str = f'\nFinal Adapted Model - Valid Slot-Acc: {best_acc:.2f}%, Test Slot-Acc: {test_metrics["slot_wise_accuracy"]:.2f}%'
    out_file.write(log_str + '\n')
    out_file.flush()
    print(log_str)
    
    # Save adaptation history
    history_file = osp.join(output_dir, "target_adaptation_history.json")
    with open(history_file, 'w') as f:
        json.dump(adaptation_history, f, indent=2)
    
    # ============================================================
    # RETURN RESULTS
    # ============================================================
    result = {
        # Best validation metrics
        'best_valid_slot_wise_accuracy': best_acc,
        'best_valid_exact_match_accuracy': best_metrics.get('valid_exact_match_accuracy', 0.0),
        'best_valid_f1_micro': best_f1_micro,
        'best_valid_f1_macro': best_f1_macro,
        'best_valid_activity_macro_f1': best_metrics.get('valid_activity_macro_f1', 0.0),
        'best_valid_occupancy_mae': best_metrics.get('valid_occupancy_mae', 0.0),
        'best_valid_occupancy_exact_match': best_metrics.get('valid_occupancy_exact_match', 0.0),
        'best_valid_hamming_loss': best_hamming,
        'best_epoch': best_epoch,
        # Test metrics
        'test_slot_wise_accuracy': test_metrics['slot_wise_accuracy'],
        'test_exact_match_accuracy': test_metrics['exact_match_accuracy'],
        'test_f1_micro': test_metrics['f1_micro'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_activity_macro_f1': test_metrics['activity_macro_f1'],
        'test_occupancy_mae': test_metrics['occupancy_mae'],
        'test_occupancy_exact_match': test_metrics['occupancy_exact_match'],
        'test_hamming_loss': test_metrics['hamming_loss'],
        'test_per_activity_f1': test_metrics['per_activity_f1'],
        # Training history
        'adaptation_history': adaptation_history
    }
    
    # Add SSL metrics if they exist
    if ssl_metrics is not None:
        result['ssl_metrics'] = ssl_metrics
    
    return result


def adapt_branch1(source_dir, output_dir, train_loader, valid_loader, out_file,
                  config, num_users, num_classes, var_x_shape):
    """
    Branch 1: Pure SHOT-IM adaptation (discriminative focus)
    No SSL, no occupancy conditioning - just entropy minimization + GENT
    """
    print("\nInitializing Branch 1 models...")
    
    # Create models
    netF, netB, netC = create_models(
        var_x_shape=var_x_shape,
        num_users=num_users,
        num_classes=num_classes,
        bottleneck_dim=preset["training"]["bottleneck"]
    )
    
    # Load source weights
    netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(source_dir, "source_B.pt")))
    netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt")))
    
    # Freeze classifier
    for param in netC.parameters():
        param.requires_grad = False
    
    # Wrap in DataParallel if needed
    if torch.cuda.device_count() > 1:
        netF = nn.DataParallel(netF)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
    
    # Setup optimizer (only F and B)
    param_group = []
    base_netF = get_model_device_safe(netF)
    base_netB = get_model_device_safe(netB)
    
    for v in base_netF.parameters():
        param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
    for v in base_netB.parameters():
        param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
    

    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["lr"], weight_decay=1e-4))
    
    best_acc = 0
    best_netF, best_netB = None, None
    max_epochs = config["max_epoch"]
    
    print(f"Starting Branch 1 adaptation for {max_epochs} epochs...")
    
    # Training loop
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        
        netF.train()
        netB.train()
        netC.eval()
        
        epoch_loss = 0
        num_batches = 0
        
        for inputs_target, _, _ in train_loader:
            inputs_target = inputs_target.cuda()
            
            outputs = forward_with_reshape(netF, netB, netC, inputs_target, num_users, num_classes)
            B = outputs.shape[0]
            
            # SHOT-IM: Entropy minimization
            outputs_flat = outputs.view(B * num_users, num_classes + 1)
            softmax_out = F.softmax(outputs_flat, dim=1)
            
            entropy_per_slot = -softmax_out * torch.log(softmax_out + 1e-5)
            entropy_loss = entropy_per_slot.sum(dim=1).mean()
            
            # GENT: Diversity (if enabled)
            if config.get("gent", False):
                msoftmax = softmax_out.mean(dim=0)
                gentropy = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss -= gentropy
            
            loss = config["ent_par"] * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Validation
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            netF.eval()
            netB.eval()
            valid_metrics = cal_acc(valid_loader, netF, netB, netC, num_users, num_classes)
            
            log_str = (f"Branch 1 - Epoch {epoch+1}/{max_epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Valid Slot-Acc={valid_metrics['slot_wise_accuracy']:.2f}%, "
                      f"Activity-F1={valid_metrics['activity_macro_f1']:.2f}%")
            print(log_str)
            out_file.write(log_str + '\n')
            out_file.flush()
            
            if valid_metrics['slot_wise_accuracy'] > best_acc:
                best_acc = valid_metrics['slot_wise_accuracy']
                best_netF = copy.deepcopy(get_model_device_safe(netF).state_dict())
                best_netB = copy.deepcopy(get_model_device_safe(netB).state_dict())
    
    # Load best and save
    get_model_device_safe(netF).load_state_dict(best_netF)
    get_model_device_safe(netB).load_state_dict(best_netB)
    
    torch.save(best_netF, osp.join(output_dir, "branch1_adapted_F.pt"))
    torch.save(best_netB, osp.join(output_dir, "branch1_adapted_B.pt"))
    torch.save(get_model_device_safe(netC).state_dict(), osp.join(output_dir, "branch1_adapted_C.pt"))
    
    print(f"✓ Branch 1 best validation accuracy: {best_acc:.2f}%")
    
    return netF, netB, netC

def get_domain_name(config_key):
    config = preset[config_key]
    env = "_".join(config["environment"])
    wifi = "_".join(config["wifi_band"]) + "GHz"
    users = f"users_{min(config['num_users'])}-{max(config['num_users'])}"
    
    # Add data type
    data_type = config.get("data_type", "amp")
    
    # Add normalization info
    normalize = config.get("normalize", None)
    if normalize:
        norm_str = f"norm_{normalize}"
    else:
        norm_str = "no_norm"
    
    return f"{env}_{wifi}_{users}_{data_type}_{norm_str}"


def get_adaptation_config_name():
    """Generate a unique name for the current adaptation configuration"""
    config_parts = []
    arch_name = get_architecture_name()
    config_parts.append(arch_name)
    
    # Data configuration (from source_data)
    source_config = preset["source_data"]
    data_type = source_config.get("data_type", "amp")
    normalize = source_config.get("normalize", None)
    
    # Shortened data type names
    data_type_map = {
        "amp": "amp",
        "phase_raw": "ph_raw",
        "phase_sanitized": "ph_san",
        "ratio_amp": "r_amp",
        "ratio_phase_raw": "r_ph_raw",
        "ratio_phase_sanitized": "r_ph_san"
    }
    data_str = data_type_map.get(data_type, data_type)
    config_parts.append(data_str)
    
    # Add normalization
    if normalize:
        config_parts.append(f"n_{normalize}")
    else:
        config_parts.append("n_none")
    
    # SSL configuration
    ssl = preset["training"]["ssl"]
    if ssl > 0:
        config_parts.append(f"ssl{ssl}")
    else:
        config_parts.append("nossl")
        
    # Diversity/GENT configuration
    if preset["training"].get("gent", False):
        config_parts.append("gent")
    else:
        config_parts.append("nogent")
        
    # Pseudo-labeling configuration
    cls_par = preset["training"].get("cls_par", 0.0)
    if cls_par > 0:
        config_parts.append(f"cls{cls_par}")
    else:
        config_parts.append("nocls")
        
    # Entropy parameter
    ent_par = preset["training"].get("ent_par", 1.0)
    config_parts.append(f"ent{ent_par}")
    
    return "_".join(config_parts)
def get_architecture_name():
    """Generate architecture identifier for saving models"""
    if preset["training"].get("use_transformer", False):
        num_layers = preset["training"].get("num_decoder_layers", 4)
        nhead = preset["training"].get("nhead", 4)
        return f"transformer_L{num_layers}_H{nhead}"
    else:
        return "linear_classifier"
        
def print_config():
    s = "==========================================\nEXPERIMENT CONFIGURATION\n==========================================\n"
    for key, content in preset.items():
        if key != "encoding": s += f"{key}: {content}\n"
    s += "==========================================\n"
    return s

def run_single_experiment(run_idx, seed, base_output_dir, source_dir):
    """Executes a single run of the experiment with a given seed - runs source baseline, SHOT"""
    # Set seed for this run
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)

    # Create a unique directory for this run
    output_dir = osp.join(base_output_dir, f"run_{run_idx}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting Run {run_idx+1}/{preset['repeat']} with Seed {seed} | Output: {output_dir}")

    # Initialize run results dictionary
    run_results = {
        'run_index': run_idx,
        'seed': seed,
        'experiment_config': {k: v for k, v in preset.items() if k != 'encoding'}
    }

    # --- Source Training ---
    source_results = None
    source_run_dir = osp.join(source_dir, f"run_{run_idx}")
    os.makedirs(source_run_dir, exist_ok=True)

    if not osp.exists(osp.join(source_run_dir, 'source_F.pt')):
        print("\n" + "="*50 + "\nPHASE 1: SOURCE DOMAIN TRAINING\n" + "="*50)
        with open(osp.join(source_run_dir, 'log_src.txt'), 'w') as out_file:
            out_file.write(print_config() + '\n')
            source_results = train_source(source_run_dir, out_file, random_state=seed)
    else:
        print("Source model already exists, loading existing results...")
        # Try to load existing source results if available
        try:
            existing_results_file = osp.join(source_run_dir, "run_complete_results.json")
            if osp.exists(existing_results_file):
                with open(existing_results_file, 'r') as f:
                    existing_results = json.load(f)
                    source_results = existing_results.get('source_domain', {})
        except:
            pass
    
    # --- Target Baseline Testing (Source Model on Target) ---
    print("\n" + "="*50 + "\nPHASE 2: TARGET DOMAIN BASELINE (Source Model)\n" + "="*50)
    with open(osp.join(output_dir, 'log_baseline.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        target_baseline_results = test_target_baseline(source_run_dir, output_dir, out_file, random_state=seed)

    # ============================================================
    # ============================================================
    # PHASE 3: TARGET ADAPTATION - SHOT
    # ============================================================
    # ============================================================
    print("\n" + "="*50 + "\nPHASE 3: TARGET DOMAIN ADAPTATION - SHOT\n" + "="*50)


    with open(osp.join(output_dir, 'log_tar_shot.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        out_file.write("="*50 + "\n")
        out_file.write("METHOD: SHOT (Entropy Minimization)\n")
        out_file.write("="*50 + "\n\n")

        target_shot_results = train_target_shot(
            source_run_dir, output_dir, out_file,
            random_state=seed,
        )
    
    # ============================================================
    # Compile all results
    run_results.update({
        'source_domain': source_results,
        'target_domain_baseline': target_baseline_results,
        'target_domain_adapted': target_shot_results
    })
    
    # Save comprehensive results for this run
    save_run_results(output_dir, run_results)
    
    # Return ALL metrics for aggregation - source baseline and SHOT
    return {
        'source_baseline': {
            'slot_wise_accuracy': target_baseline_results['slot_wise_accuracy'],
            'exact_match_accuracy': target_baseline_results['exact_match_accuracy'],
            'f1_micro': target_baseline_results['f1_micro'],
            'f1_macro': target_baseline_results['f1_macro'],
            'activity_macro_f1': target_baseline_results['activity_macro_f1'],
            'occupancy_mae': target_baseline_results['occupancy_mae'],
            'occupancy_exact_match': target_baseline_results['occupancy_exact_match'],
            'hamming_loss': target_baseline_results['hamming_loss'],
            'per_activity_f1': target_baseline_results['per_activity_f1'],
        },
        'shot': {
            'best_valid_slot_wise_accuracy': target_shot_results['best_valid_slot_wise_accuracy'],
            'best_valid_exact_match_accuracy': target_shot_results['best_valid_exact_match_accuracy'],
            'best_valid_activity_macro_f1': target_shot_results['best_valid_activity_macro_f1'],
            'test_slot_wise_accuracy': target_shot_results['test_slot_wise_accuracy'],
            'test_exact_match_accuracy': target_shot_results['test_exact_match_accuracy'],
            'test_f1_micro': target_shot_results['test_f1_micro'],
            'test_f1_macro': target_shot_results['test_f1_macro'],
            'test_activity_macro_f1': target_shot_results['test_activity_macro_f1'],
            'test_occupancy_mae': target_shot_results['test_occupancy_mae'],
            'test_occupancy_exact_match': target_shot_results['test_occupancy_exact_match'],
            'test_hamming_loss': target_shot_results['test_hamming_loss'],
            'test_per_activity_f1': target_shot_results['test_per_activity_f1'],
        },
        'ssl_metrics': target_shot_results.get('ssl_metrics', None)
    }

def main():
    """Main function to run the experiment multiple times and aggregate results."""
    if preset["training"]["gpu_id"] != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = preset["training"]["gpu_id"]
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No GPUs available. This code requires CUDA.")
    
    initial_seed = preset["training"]["seed"]
    
    # Define the base output directory
    experiment_name = f"{preset['source_task']}_to_{preset['target_task']}"
    source_domain = get_domain_name("source_data")
    target_domain = get_domain_name("target_data")
    
    # Get script directory to determine paths
    script_dir = osp.dirname(osp.abspath(__file__))
    wimans_dir = osp.dirname(script_dir)
    shotplus_dir = osp.join(wimans_dir, "SHOTPlus")
    
    # ============================================================
    # NEW: Get architecture name for source models
    # ============================================================
    arch_name = get_architecture_name()
    
    adaptation_config = get_adaptation_config_name()
    save_dir = preset["path"]["save_dir"]
    if not osp.isabs(save_dir):
        save_dir = osp.join(script_dir, save_dir)
    
    # Source checkpoints: same tree as preset save_dir (override via WIMANS_FIXED_SOURCE_DIR).
    fixed_src = os.environ.get("WIMANS_FIXED_SOURCE_DIR")
    if fixed_src:
        source_base_dir = fixed_src
    else:
        source_base_dir = osp.join(
            save_dir,
            experiment_name,
            f"{source_domain}_to_{target_domain}",
            "source",
            f"seed_{initial_seed}",
        )
    
    base_output_dir = osp.join(
        save_dir,
        experiment_name,
        f"{source_domain}_to_{target_domain}",
        "adaptation",
        adaptation_config,
        f"seed_{initial_seed}"
    )
    
    debug_data_availability()
    os.makedirs(source_base_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Architecture: {arch_name}")  # ← NEW: Show architecture
    print(f"Experiment: {experiment_name}")
    print(f"Source → Target: {source_domain} → {target_domain}")
    print(f"Adaptation Config: {adaptation_config}")
    print(f"Number of Runs: {preset['repeat']}")
    print(f"Initial Seed: {initial_seed}")
    print(f"Source Model Directory: {source_base_dir}")
    print(f"Adaptation Results Directory: {osp.abspath(base_output_dir)}")
    print(f"{'='*80}\n")
    
    # Lists to store results from all runs - SEPARATE for each method
    all_source_baseline_metrics = []
    all_shot_metrics = []
    all_run_summaries = []

    for i in range(preset["repeat"]):
        current_seed = initial_seed + i
        final_metrics = run_single_experiment(i, current_seed, base_output_dir, source_base_dir)
        
        if final_metrics:
            all_source_baseline_metrics.append(final_metrics['source_baseline'])
            all_shot_metrics.append(final_metrics['shot'])
            
            # Load the complete run results for summary
            run_dir = osp.join(base_output_dir, f"run_{i}")
            run_results_file = osp.join(run_dir, "run_complete_results.json")
            if osp.exists(run_results_file):
                with open(run_results_file, 'r') as f:
                    run_data = json.load(f)
                    all_run_summaries.append(run_data)
    
    # --- Aggregate and Save Final Results ---
    if not all_source_baseline_metrics or not all_shot_metrics:
        print("\nNo results to aggregate. Exiting.")
        return
    
    # Helper function to calculate statistics
    def calculate_stats(metrics_list, metric_name):
        values = [m[metric_name] for m in metrics_list]
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values
        }

    # Helper function to aggregate per-activity F1 scores
    def aggregate_per_activity_f1(metrics_list):
        """
        Aggregates per-activity F1 scores across multiple runs.
        Handles both 'per_activity_f1' and 'test_per_activity_f1' keys.
        """
        all_activities = {}
        
        for m in metrics_list:
            # Try both possible keys
            activity_f1s = m.get('test_per_activity_f1') or m.get('per_activity_f1', {})
            
            for act_key, f1_val in activity_f1s.items():
                if act_key not in all_activities:
                    all_activities[act_key] = []
                all_activities[act_key].append(f1_val)
        
        # Calculate stats for each activity
        activity_stats = {}
        for act_key, values in all_activities.items():
            activity_stats[act_key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
        
        return activity_stats

    # Aggregated statistics for SOURCE BASELINE (no adaptation)
    source_baseline_stats = {
        'slot_wise_accuracy': calculate_stats(all_source_baseline_metrics, 'slot_wise_accuracy'),
        'exact_match_accuracy': calculate_stats(all_source_baseline_metrics, 'exact_match_accuracy'),
        'f1_micro': calculate_stats(all_source_baseline_metrics, 'f1_micro'),
        'f1_macro': calculate_stats(all_source_baseline_metrics, 'f1_macro'),
        'activity_macro_f1': calculate_stats(all_source_baseline_metrics, 'activity_macro_f1'),
        'occupancy_mae': calculate_stats(all_source_baseline_metrics, 'occupancy_mae'),
        'occupancy_exact_match': calculate_stats(all_source_baseline_metrics, 'occupancy_exact_match'),
        'hamming_loss': calculate_stats(all_source_baseline_metrics, 'hamming_loss'),
        'per_activity_f1': aggregate_per_activity_f1(all_source_baseline_metrics),
    }

    # Aggregated statistics for SHOT (with adaptation)
    shot_stats = {
        'test_slot_wise_accuracy': calculate_stats(all_shot_metrics, 'test_slot_wise_accuracy'),
        'test_exact_match_accuracy': calculate_stats(all_shot_metrics, 'test_exact_match_accuracy'),
        'test_f1_micro': calculate_stats(all_shot_metrics, 'test_f1_micro'),
        'test_f1_macro': calculate_stats(all_shot_metrics, 'test_f1_macro'),
        'test_activity_macro_f1': calculate_stats(all_shot_metrics, 'test_activity_macro_f1'),
        'test_occupancy_mae': calculate_stats(all_shot_metrics, 'test_occupancy_mae'),
        'test_occupancy_exact_match': calculate_stats(all_shot_metrics, 'test_occupancy_exact_match'),
        'test_hamming_loss': calculate_stats(all_shot_metrics, 'test_hamming_loss'),
        'test_per_activity_f1': aggregate_per_activity_f1(all_shot_metrics),
    }

    # Aggregate SSL metrics if they exist
    ssl_stats = None
    ssl_metrics_list = [m.get('ssl_metrics') for m in all_shot_metrics if m.get('ssl_metrics') is not None]
    if ssl_metrics_list:
        ssl_rotation_accs = [m['best_rotation_accuracy'] for m in ssl_metrics_list if 'best_rotation_accuracy' in m]
        if ssl_rotation_accs:
            ssl_stats = {
                'best_rotation_accuracy': {
                    'mean': float(np.mean(ssl_rotation_accs)),
                    'std': float(np.std(ssl_rotation_accs)),
                    'values': ssl_rotation_accs
                }
            }

    aggregated_stats = {
        'num_runs': len(all_source_baseline_metrics),
        'initial_seed': initial_seed,
        'source_baseline_method': source_baseline_stats,
        'shot_method': shot_stats,
    }

    if ssl_stats is not None:
        aggregated_stats['ssl_stats'] = ssl_stats

    # If we have source domain training results, aggregate those
    if all_run_summaries:
        # Source domain training results with NEW metric names
        source_train_slot_accs = []
        source_train_exact_match = []
        source_train_activity_f1 = []
        source_train_f1_micros = []
        source_train_f1_macros = []

        for run_summary in all_run_summaries:
            if 'source_domain' in run_summary and run_summary['source_domain']:
                sd = run_summary['source_domain']
                # Handle both old and new metric names for backward compatibility
                source_train_slot_accs.append(sd.get('test_slot_wise_accuracy', sd.get('test_accuracy', 0)))
                source_train_exact_match.append(sd.get('test_exact_match_accuracy', 0))
                source_train_activity_f1.append(sd.get('test_activity_macro_f1', 0))
                source_train_f1_micros.append(sd.get('test_f1_micro', 0))
                source_train_f1_macros.append(sd.get('test_f1_macro', 0))

        if source_train_slot_accs:
            aggregated_stats['source_domain_training'] = {
                'test_slot_wise_accuracy': {
                    'mean': float(np.mean(source_train_slot_accs)),
                    'std': float(np.std(source_train_slot_accs)),
                    'values': source_train_slot_accs
                },
                'test_exact_match_accuracy': {
                    'mean': float(np.mean(source_train_exact_match)),
                    'std': float(np.std(source_train_exact_match)),
                    'values': source_train_exact_match
                },
                'test_activity_macro_f1': {
                    'mean': float(np.mean(source_train_activity_f1)),
                    'std': float(np.std(source_train_activity_f1)),
                    'values': source_train_activity_f1
                },
                'test_f1_micro': {
                    'mean': float(np.mean(source_train_f1_micros)),
                    'std': float(np.std(source_train_f1_micros)),
                    'values': source_train_f1_micros
                },
                'test_f1_macro': {
                    'mean': float(np.mean(source_train_f1_macros)),
                    'std': float(np.std(source_train_f1_macros)),
                    'values': source_train_f1_macros
                }
            }
    
    # Create comprehensive summary string
    summary_str = f"\n{'='*80}\n"
    summary_str += f"FINAL AGGREGATED RESULTS OVER {preset['repeat']} RUNS\n"
    summary_str += f"{'='*80}\n"
    summary_str += f"Experiment: {experiment_name}\n"
    summary_str += f"Source Domain: {source_domain}\n"
    summary_str += f"Target Domain: {target_domain}\n"
    summary_str += f"Initial Seed: {initial_seed}\n\n"
    
    # Source domain training performance
    if 'source_domain_training' in aggregated_stats:
        summary_str += "SOURCE DOMAIN TRAINING PERFORMANCE (on source test set):\n"
        summary_str += "-" * 80 + "\n"
        src_train_slot_acc = aggregated_stats['source_domain_training']['test_slot_wise_accuracy']
        src_train_exact = aggregated_stats['source_domain_training']['test_exact_match_accuracy']
        src_train_activity_f1 = aggregated_stats['source_domain_training']['test_activity_macro_f1']
        summary_str += f"Slot-wise Accuracy:     {src_train_slot_acc['mean']:.2f}% ± {src_train_slot_acc['std']:.2f}\n"
        summary_str += f"Exact Match Accuracy:   {src_train_exact['mean']:.2f}% ± {src_train_exact['std']:.2f}\n"
        summary_str += f"Activity Macro-F1:      {src_train_activity_f1['mean']:.2f}% ± {src_train_activity_f1['std']:.2f}\n\n"

    # ============================================================
    # METHOD 1: SOURCE BASELINE (No Adaptation)
    # ============================================================
    summary_str += "="*80 + "\n"
    summary_str += "METHOD 1: SOURCE BASELINE (No Adaptation)\n"
    summary_str += "="*80 + "\n"
    summary_str += "Direct transfer: Source model tested on target domain without any adaptation\n\n"

    source_baseline_slot_acc = source_baseline_stats['slot_wise_accuracy']
    source_baseline_exact = source_baseline_stats['exact_match_accuracy']
    source_baseline_activity_f1 = source_baseline_stats['activity_macro_f1']
    source_baseline_occ_mae = source_baseline_stats['occupancy_mae']
    source_baseline_occ_exact = source_baseline_stats['occupancy_exact_match']

    summary_str += "Performance on Target Test Set:\n"
    summary_str += "-" * 80 + "\n"
    summary_str += f"Slot-wise Accuracy:      {source_baseline_slot_acc['mean']:.2f}% ± {source_baseline_slot_acc['std']:.2f} (Range: {source_baseline_slot_acc['min']:.2f}-{source_baseline_slot_acc['max']:.2f})\n"
    summary_str += f"Exact Match Accuracy:    {source_baseline_exact['mean']:.2f}% ± {source_baseline_exact['std']:.2f} (Range: {source_baseline_exact['min']:.2f}-{source_baseline_exact['max']:.2f})\n"
    summary_str += f"Activity Macro-F1:       {source_baseline_activity_f1['mean']:.2f}% ± {source_baseline_activity_f1['std']:.2f} (Range: {source_baseline_activity_f1['min']:.2f}-{source_baseline_activity_f1['max']:.2f})\n"
    summary_str += f"Occupancy MAE:           {source_baseline_occ_mae['mean']:.4f} ± {source_baseline_occ_mae['std']:.4f}\n"
    summary_str += f"Occupancy Exact Match:   {source_baseline_occ_exact['mean']:.2f}% ± {source_baseline_occ_exact['std']:.2f}\n\n"

    # ============================================================
    # METHOD 2: SHOT (Entropy Minimization)
    # ============================================================
    summary_str += "="*80 + "\n"
    summary_str += "METHOD 2: SHOT (Entropy Minimization Adaptation)\n"
    summary_str += "="*80 + "\n"
    summary_str += "Adaptation via entropy minimization\n\n"

    # SSL Metrics (if available)
    if 'ssl_stats' in aggregated_stats:
        ssl_rot_acc = aggregated_stats['ssl_stats']['best_rotation_accuracy']
        summary_str += f"SSL Rotation Pre-training: {ssl_rot_acc['mean']:.2f}% ± {ssl_rot_acc['std']:.2f}\n\n"

    shot_slot_acc = shot_stats['test_slot_wise_accuracy']
    shot_exact = shot_stats['test_exact_match_accuracy']
    shot_activity_f1 = shot_stats['test_activity_macro_f1']
    shot_occ_mae = shot_stats['test_occupancy_mae']
    shot_occ_exact = shot_stats['test_occupancy_exact_match']
    shot_f1_micro = shot_stats['test_f1_micro']
    shot_f1_macro = shot_stats['test_f1_macro']
    shot_hamming = shot_stats['test_hamming_loss']

    summary_str += "Performance on Target Test Set:\n"
    summary_str += "-" * 80 + "\n"
    summary_str += f"Slot-wise Accuracy:      {shot_slot_acc['mean']:.2f}% ± {shot_slot_acc['std']:.2f} (Range: {shot_slot_acc['min']:.2f}-{shot_slot_acc['max']:.2f})\n"
    summary_str += f"Exact Match Accuracy:    {shot_exact['mean']:.2f}% ± {shot_exact['std']:.2f} (Range: {shot_exact['min']:.2f}-{shot_exact['max']:.2f})\n"
    summary_str += f"Activity Macro-F1 (9):   {shot_activity_f1['mean']:.2f}% ± {shot_activity_f1['std']:.2f} (Range: {shot_activity_f1['min']:.2f}-{shot_activity_f1['max']:.2f})\n"
    summary_str += f"F1-Micro (all classes):  {shot_f1_micro['mean']:.2f}% ± {shot_f1_micro['std']:.2f} (Range: {shot_f1_micro['min']:.2f}-{shot_f1_micro['max']:.2f})\n"
    summary_str += f"F1-Macro (all classes):  {shot_f1_macro['mean']:.2f}% ± {shot_f1_macro['std']:.2f} (Range: {shot_f1_macro['min']:.2f}-{shot_f1_macro['max']:.2f})\n"
    summary_str += f"Occupancy MAE:           {shot_occ_mae['mean']:.4f} ± {shot_occ_mae['std']:.4f}\n"
    summary_str += f"Occupancy Exact Match:   {shot_occ_exact['mean']:.2f}% ± {shot_occ_exact['std']:.2f}\n"
    summary_str += f"Hamming Loss:            {shot_hamming['mean']:.2f}% ± {shot_hamming['std']:.2f} (Range: {shot_hamming['min']:.2f}-{shot_hamming['max']:.2f})\n\n"

    summary_str += "\nImprovement over Source Baseline:\n"
    summary_str += "-" * 80 + "\n"
    slot_acc_improvement = shot_slot_acc['mean'] - source_baseline_slot_acc['mean']
    exact_improvement = shot_exact['mean'] - source_baseline_exact['mean']
    activity_f1_improvement = shot_activity_f1['mean'] - source_baseline_activity_f1['mean']
    occ_mae_improvement = source_baseline_occ_mae['mean'] - shot_occ_mae['mean']  # Lower is better
    occ_exact_improvement = shot_occ_exact['mean'] - source_baseline_occ_exact['mean']
    summary_str += f"Slot-wise Accuracy:      {slot_acc_improvement:+.2f}%\n"
    summary_str += f"Exact Match Accuracy:    {exact_improvement:+.2f}%\n"
    summary_str += f"Activity Macro-F1:       {activity_f1_improvement:+.2f}%\n"
    summary_str += f"Occupancy MAE:           {occ_mae_improvement:+.4f} (lower is better)\n"
    summary_str += f"Occupancy Exact Match:   {occ_exact_improvement:+.2f}%\n\n"

    # Save all results files
    # 1. Aggregated statistics (JSON)
    aggregated_results_file = osp.join(base_output_dir, "aggregated_results.json")
    aggregated_stats['timestamp'] = datetime.now().isoformat()
    aggregated_stats['experiment_config'] = {k: v for k, v in preset.items() if k != 'encoding'}
    
    with open(aggregated_results_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    
    # 2. Human-readable summary
    summary_file_path = osp.join(base_output_dir, "final_aggregated_results.txt")
    with open(summary_file_path, 'w') as f:
        f.write(summary_str)
    
    # 3. All runs summary (JSON)
    all_runs_file = osp.join(base_output_dir, "all_runs_complete_results.json")
    with open(all_runs_file, 'w') as f:
        json.dump({
            'experiment_info': {
                'experiment_name': experiment_name,
                'source_domain': source_domain,
                'target_domain': target_domain,
                'num_runs': len(all_run_summaries),
                'timestamp': datetime.now().isoformat()
            },
            'runs': all_run_summaries,
            'aggregated_statistics': aggregated_stats
        }, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - Aggregated results: {aggregated_results_file}")
    print(f"  - Summary: {summary_file_path}")
    print(f"  - Complete results: {all_runs_file}")

if __name__ == "__main__":
    main()