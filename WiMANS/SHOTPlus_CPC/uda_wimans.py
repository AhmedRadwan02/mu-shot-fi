"""WiMANS SHOT with CPC: target adaptation, CPC pre-training, and full metrics logging."""
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
import gc

# Project modules
import network
from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y
import torch.nn.functional as F
import pandas as pd
from augmentations import AugMixDataset
import rotation

# CPC modules
if preset["training"].get("old_cpc",True):  
    from cpc_networks import CPCModel, init_weights as cpc_init_weights
    from cpc_utils import pretrain_cpc, evaluate_cpc, compute_cpc_loss
# CPC modules (per-slot version)
else:
    from cpc_networks2 import CPCModelPerSlot as CPCModel, init_weights as cpc_init_weights
    from cpc_utils2 import pretrain_cpc, evaluate_cpc, compute_cpc_loss

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

def convert_onehot_to_indices(data_y, num_classes):
    """Convert one-hot encoded labels [N, 6, num_classes] to class indices [N, 6]."""
    N, num_users = data_y.shape[0], data_y.shape[1]
    indices = np.zeros((N, num_users), dtype=np.int64)
    for i in range(N):
        for j in range(num_users):
            user_label = data_y[i, j]
            if user_label.sum() == 0:
                indices[i, j] = num_classes
            else:
                indices[i, j] = np.argmax(user_label)
    return indices

def hungarian_matching_batch(pred_logits, gt_indices, num_classes):
    """Perform Hungarian matching between predicted slots and ground truth slots."""
    B = pred_logits.shape[0]
    matched_pred_indices = []
    matched_gt_indices = []
    pred_probs = F.softmax(pred_logits, dim=-1)
    
    for b in range(B):
        cost_matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                gt_class = gt_indices[b, j].item()
                cost_matrix[i, j] = -torch.log(pred_probs[b, i, gt_class] + 1e-8).item()
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_pred_indices.append(row_ind)
        matched_gt_indices.append(col_ind)
    
    matched_pred_indices = np.array(matched_pred_indices)
    matched_gt_indices = np.array(matched_gt_indices)
    return matched_pred_indices, matched_gt_indices

def debug_data_availability():
    """Debug what data is available for different user counts"""
    print("Debugging data availability...")
    data_pd_y_all = pd.read_csv(preset["path"]["data_y"], dtype=str)
    
    print(f"Total samples in dataset: {len(data_pd_y_all)}")
    print("\nSamples by number of users:")
    print(data_pd_y_all["number_of_users"].value_counts().sort_index())
    print("\nSamples by environment:")
    print(data_pd_y_all["environment"].value_counts())
    print("\nSamples by WiFi band:")
    print(data_pd_y_all["wifi_band"].value_counts())
    
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

def create_dataset(data_x, data_y, config=None, batch_size=None, shuffle=True, is_training=True):
    """Create PyTorch dataset and dataloader with optional AugMix"""
    if batch_size is None:
        batch_size = preset["training"]["batch_size"]

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        # Each GPU gets batch_size // gpu_count samples
        # So if batch_size=128 and 2 GPUs -> 64 per GPU
        batch_size = batch_size // gpu_count
        print(f"DataParallel detected: Using batch_size={batch_size} per GPU ({batch_size * gpu_count} total)")
    
    dataset_size = len(data_x)
    if dataset_size < batch_size:
        batch_size = max(1, dataset_size // 2)
        print(f"Warning: Dataset size ({dataset_size}) smaller than batch size. Adjusted to {batch_size}")
    
    use_augmix = config.get("use_augmix", False) if config is not None else False
    use_augmix = use_augmix and is_training
    
    if use_augmix:
        print(f"Using AugMix (training mode) with ops: {config['augmix_ops']}")
        dataset = AugMixDataset(data_x, data_y, config)
    else:
        dataset = TensorDataset(torch.FloatTensor(data_x), torch.LongTensor(data_y))
    
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
    data_x = load_data_x(preset["path"]["data_x"], var_label_list)
    
    task = preset["source_task"] if config_key == "source_data" else preset["target_task"]
    data_y = encode_data_y(data_pd_y, task)
    
    from sklearn.model_selection import train_test_split
    data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(
        data_x, data_y, test_size=0.2, shuffle=True, random_state=random_state
    )
    
    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(
        data_test_x, data_test_y, test_size=0.5, shuffle=True, random_state=random_state
    )
    
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    
    var_x_shape = data_train_x[0].shape
    num_users = 6
    task_encoding = preset["encoding"][task]
    num_task_classes = len(list(task_encoding.values())[0])
    
    data_train_y = convert_onehot_to_indices(data_train_y, num_task_classes)
    data_valid_y = convert_onehot_to_indices(data_valid_y, num_task_classes)
    data_test_y = convert_onehot_to_indices(data_test_y, num_task_classes)
    
    classifier_output_size = 6 * (num_task_classes + 1)
    
    train_loader = create_dataset(data_train_x, data_train_y, config=config, shuffle=True, is_training=True)
    valid_loader = create_dataset(data_valid_x, data_valid_y, config=config, shuffle=False, is_training=False)
    test_loader = create_dataset(data_test_x, data_test_y, config=config, shuffle=False, is_training=False)
    
    return train_loader, valid_loader, test_loader, classifier_output_size, var_x_shape, num_users, num_task_classes

def cal_acc(loader, netF, netB, netC, num_users, num_classes):
    """
    Calculate comprehensive metrics with slot-based predictions and Hungarian matching.
    Returns dictionary with all metrics including slot_wise_accuracy, exact_match_accuracy, 
    per_activity_f1, activity_macro_f1, occupancy metrics, etc.
    """
    netF.eval()
    netB.eval()
    netC.eval()
    
    all_preds = []
    all_labels = []
    all_sample_preds = []
    all_sample_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
            
            outputs = netC(netB(netF(inputs)))
            B = outputs.shape[0]
            outputs = outputs.view(B, num_users, num_classes + 1)
            
            matched_pred_indices, matched_gt_indices = hungarian_matching_batch(
                outputs, targets, num_classes
            )
            
            for b in range(B):
                pred_slots = outputs[b]
                gt_slots = targets[b]
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
    all_sample_preds = np.array(all_sample_preds)
    all_sample_labels = np.array(all_sample_labels)
    
    # Main metrics
    slot_wise_accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    hamming = (1 - slot_wise_accuracy / 100) * 100
    
    report = classification_report(all_labels, all_preds, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    
    # Per-activity F1 scores
    activity_mask = all_labels < num_classes
    activity_preds = all_preds[activity_mask]
    activity_labels = all_labels[activity_mask]
    
    per_activity_f1 = {}
    if len(activity_labels) > 0:
        for class_idx in range(num_classes):
            binary_labels = (activity_labels == class_idx).astype(int)
            binary_preds = (activity_preds == class_idx).astype(int)
            if binary_labels.sum() > 0:
                f1 = f1_score(binary_labels, binary_preds, zero_division=0) * 100
            else:
                f1 = 0.0
            per_activity_f1[f'activity_{class_idx}'] = f1
        activity_macro_f1 = np.mean(list(per_activity_f1.values()))
    else:
        per_activity_f1 = {f'activity_{i}': 0.0 for i in range(num_classes)}
        activity_macro_f1 = 0.0
    
    # Occupancy metrics
    true_occupancy = (all_sample_labels < num_classes).sum(axis=1)
    pred_occupancy = (all_sample_preds < num_classes).sum(axis=1)
    occupancy_mae = np.mean(np.abs(true_occupancy - pred_occupancy))
    occupancy_exact_match = (true_occupancy == pred_occupancy).mean() * 100
    
    # Exact match accuracy
    exact_matches = (all_sample_preds == all_sample_labels).all(axis=1)
    exact_match_accuracy = exact_matches.mean() * 100
    
    metrics = {
        'slot_wise_accuracy': float(slot_wise_accuracy),
        'exact_match_accuracy': float(exact_match_accuracy),
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'per_activity_f1': {k: float(v) for k, v in per_activity_f1.items()},
        'activity_macro_f1': float(activity_macro_f1),
        'occupancy_mae': float(occupancy_mae),
        'occupancy_exact_match': float(occupancy_exact_match),
        'hamming_loss': float(hamming),
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
    run_results['timestamp'] = datetime.now().isoformat()
    run_results['run_directory'] = output_dir
    
    with open(results_file, 'w') as f:
        json.dump(run_results, f, indent=2)
    
    def format_metric(value, default='N/A'):
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return f"{value:.2f}%"
        return str(value)
    
    summary_file = osp.join(output_dir, "run_results_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPLETE RUN RESULTS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Run Directory: {output_dir}\n")
        f.write(f"Timestamp: {run_results['timestamp']}\n")
        f.write(f"Random Seed: {run_results['seed']}\n\n")
        
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
        
        if 'target_domain_baseline' in run_results:
            f.write("TARGET DOMAIN BASELINE (Source Model on Target):\n")
            f.write("-" * 30 + "\n")
            target_baseline = run_results['target_domain_baseline']
            f.write(f"Slot-wise Accuracy: {format_metric(target_baseline.get('slot_wise_accuracy'))}\n")
            f.write(f"Exact Match Accuracy: {format_metric(target_baseline.get('exact_match_accuracy'))}\n")
            f.write(f"Activity Macro-F1: {format_metric(target_baseline.get('activity_macro_f1'))}\n")
            f.write(f"F1-Micro: {format_metric(target_baseline.get('f1_micro'))}\n")
            f.write(f"F1-Macro: {format_metric(target_baseline.get('f1_macro'))}\n\n")
        
        if 'target_domain_adapted' in run_results:
            f.write("TARGET DOMAIN RESULTS (SHOT+CPC Adaptation):\n")
            f.write("-" * 30 + "\n")
            target_adapted = run_results['target_domain_adapted']
            f.write(f"Best Valid Slot-wise Accuracy: {format_metric(target_adapted.get('best_valid_slot_wise_accuracy'))}\n")
            f.write(f"Test Slot-wise Accuracy: {format_metric(target_adapted.get('test_slot_wise_accuracy'))}\n")
            f.write(f"Test Exact Match Accuracy: {format_metric(target_adapted.get('test_exact_match_accuracy'))}\n")
            f.write(f"Test Activity Macro-F1: {format_metric(target_adapted.get('test_activity_macro_f1'))}\n")
            f.write(f"Test F1-Micro: {format_metric(target_adapted.get('test_f1_micro'))}\n")
            f.write(f"Test F1-Macro: {format_metric(target_adapted.get('test_f1_macro'))}\n")
            f.write(f"Best Epoch: {target_adapted.get('best_epoch', 'N/A')}\n\n")
            
            if 'cpc_metrics' in target_adapted and target_adapted['cpc_metrics']:
                cpc = target_adapted['cpc_metrics']
                f.write(f"CPC Pre-training Accuracy: {format_metric(cpc.get('best_cpc_accuracy'))}\n\n")
        
        if 'target_domain_baseline' in run_results and 'target_domain_adapted' in run_results:
            f.write("PERFORMANCE IMPROVEMENT (SHOT+CPC vs Source Baseline):\n")
            f.write("-" * 30 + "\n")
            baseline_acc = run_results['target_domain_baseline'].get('slot_wise_accuracy', 0)
            adapted_acc = run_results['target_domain_adapted'].get('test_slot_wise_accuracy', 0)
            if isinstance(baseline_acc, (int, float)) and isinstance(adapted_acc, (int, float)):
                acc_improvement = adapted_acc - baseline_acc
                f.write(f"Slot-wise Accuracy Improvement: {acc_improvement:+.2f}%\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"Run results saved to: {results_file}")
    print(f"Run summary saved to: {summary_file}")



def train_source(output_dir, out_file, random_state):
    """Train source domain model"""
    print("Loading source data...")
    train_loader, valid_loader, test_loader, classifier_output_size, var_x_shape, num_users, num_classes = dset_loaders("source_data", random_state=random_state)
    
    netF = network.CNN2DBase(var_x_shape).cuda()
    netB = network.feat_bottleneck(feature_dim=netF.in_features, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    netC = network.feat_classifier(class_num=classifier_output_size, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    
    if torch.cuda.device_count() > 1:
        netF, netB, netC = nn.DataParallel(netF), nn.DataParallel(netB), nn.DataParallel(netC)
    
    param_group = []
    for v in get_model_device_safe(netF).parameters(): param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netB).parameters(): param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netC).parameters(): param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["lr"], weight_decay=1e-4))
    
    max_epochs = preset["training"]["max_epoch"]
    best_acc = 0
    best_train_acc = 0
    best_valid_acc = 0
    
    smooth_epsilon = preset["training"].get("smooth", 0.1)
    criterion = CrossEntropyLabelSmooth(
        num_classes=num_classes + 1,
        epsilon=smooth_epsilon,
        use_gpu=torch.cuda.is_available()
    )
    
    training_history = []
    
    print("Starting source training...")
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.train()
        
        epoch_loss = 0
        num_batches = 0
        for inputs_source, labels_source in train_loader:
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            outputs_source = netC(netB(netF(inputs_source)))
            B = outputs_source.shape[0]
            outputs_source = outputs_source.view(B, num_users, num_classes + 1)
            
            matched_pred_indices, matched_gt_indices = hungarian_matching_batch(
                outputs_source, labels_source, num_classes
            )
            
            total_loss = 0
            for b in range(B):
                pred_slots = outputs_source[b]
                gt_slots = labels_source[b]
                pred_indices = matched_pred_indices[b]
                gt_indices = matched_gt_indices[b]
                
                for i in range(6):
                    pred_slot_idx = pred_indices[i]
                    gt_slot_idx = gt_indices[i]
                    pred_logits = pred_slots[pred_slot_idx]
                    gt_class = gt_slots[gt_slot_idx]
                    total_loss += criterion(pred_logits.unsqueeze(0), gt_class.unsqueeze(0))
            
            classifier_loss = total_loss / (B * 6)
            
            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
            
            epoch_loss += classifier_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        netF.eval(); netB.eval(); netC.eval()
        train_metrics = cal_acc(train_loader, netF, netB, netC, num_users, num_classes)
        valid_metrics = cal_acc(valid_loader, netF, netB, netC, num_users, num_classes)
        
        epoch_results = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'train_slot_wise_accuracy': train_metrics['slot_wise_accuracy'],
            'train_exact_match_accuracy': train_metrics['exact_match_accuracy'],
            'train_f1_micro': train_metrics['f1_micro'],
            'train_f1_macro': train_metrics['f1_macro'],
            'train_activity_macro_f1': train_metrics['activity_macro_f1'],
            'train_occupancy_mae': train_metrics['occupancy_mae'],
            'train_occupancy_exact_match': train_metrics['occupancy_exact_match'],
            'train_hamming_loss': train_metrics['hamming_loss'],
            'train_per_activity_f1': train_metrics['per_activity_f1'],
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
    
    print("\nEvaluating best model on TEST set...")
    base_netF = get_model_device_safe(netF) if hasattr(netF, 'module') else netF
    base_netB = get_model_device_safe(netB) if hasattr(netB, 'module') else netB
    base_netC = get_model_device_safe(netC) if hasattr(netC, 'module') else netC
    
    base_netF.load_state_dict(best_netF)
    base_netB.load_state_dict(best_netB)
    base_netC.load_state_dict(best_netC)
    
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
    
    test_metrics_file = osp.join(output_dir, "source_best_test_metrics.json")
    test_metrics_for_json = {k: v for k, v in test_metrics.items() if k != 'classification_report'}
    with open(test_metrics_file, 'w') as f:
        json.dump(test_metrics_for_json, f, indent=2)
    
    log_str = f'\nFinal Source Model - Valid Slot-Acc: {best_valid_acc:.2f}%, Test Slot-Acc: {test_metrics["slot_wise_accuracy"]:.2f}%'
    out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
    
    torch.save(base_netF.state_dict(), osp.join(output_dir, "source_F.pt"))
    torch.save(base_netB.state_dict(), osp.join(output_dir, "source_B.pt"))
    torch.save(base_netC.state_dict(), osp.join(output_dir, "source_C.pt"))
    
    history_file = osp.join(output_dir, "source_training_history.json")
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Source training completed. Best test slot-wise accuracy: {test_metrics['slot_wise_accuracy']:.2f}%")
    
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
        'per_activity_f1': test_metrics['per_activity_f1'],
        'training_history': training_history
    }

def test_target_baseline(source_dir, output_dir, out_file, random_state):
    """Test on target domain using the trained source model"""
    print("Loading target data for baseline testing...")
    _, _, test_loader, classifier_output_size, var_x_shape, num_users, num_classes = dset_loaders("target_data", random_state=random_state)
    
    netF = network.CNN2DBase(var_x_shape).cuda()
    netB = network.feat_bottleneck(feature_dim=netF.in_features, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    netC = network.feat_classifier(class_num=classifier_output_size, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    
    netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(source_dir, "source_B.pt")))
    netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt")))
    
    if torch.cuda.device_count() > 1:
        netF, netB, netC = nn.DataParallel(netF), nn.DataParallel(netB), nn.DataParallel(netC)
    
    netF.eval(); netB.eval(); netC.eval()
    
    test_metrics = cal_acc(test_loader, netF, netB, netC, num_users, num_classes)
    
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
    
    baseline_metrics_file = osp.join(output_dir, "target_baseline_metrics.json")
    baseline_metrics_for_json = {k: v for k, v in test_metrics.items() if k != 'classification_report'}
    with open(baseline_metrics_file, 'w') as f:
        json.dump(baseline_metrics_for_json, f, indent=2)
    
    log_str = f'Target Test (Source Model) - Slot-Acc: {test_metrics["slot_wise_accuracy"]:.2f}%, Exact-Match: {test_metrics["exact_match_accuracy"]:.2f}%, Activity-F1: {test_metrics["activity_macro_f1"]:.2f}%'
    out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
    
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

    Returns:
        best_netR: Best rotation classifier state dict
        ssl_metrics: Dictionary containing SSL training history and final metrics
    """

    # Initialize rotation classifier (2 classes: 0° and 180°)
    bottleneck_dim = preset["training"]["bottleneck"]
    netR = network.feat_classifier(class_num=2, bottleneck_dim=2*bottleneck_dim).cuda()

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

        for inputs_target, _ in train_loader:
            inputs_target = inputs_target.cuda()

            # Generate random rotations (binary: 0 or 180 degrees)
            r_labels = torch.randint(0, 2, (inputs_target.shape[0],), dtype=torch.long, device=inputs_target.device)
            r_inputs = rotation.rotate_batch_with_labels(inputs_target, r_labels)

            # Forward pass
            f_outputs = netB(netF(inputs_target))
            f_r_outputs = netB(netF(r_inputs))
            r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))

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
    """Calculate rotation prediction accuracy"""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.cuda()

            # Generate random rotations
            r_labels = torch.randint(0, 2, (inputs.shape[0],), dtype=torch.long, device=inputs.device)
            r_inputs = rotation.rotate_batch_with_labels(inputs, r_labels)

            # Forward pass
            f_outputs = netB(netF(inputs))
            f_r_outputs = netB(netF(r_inputs))
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
        for inputs, _ in loader:
            inputs = inputs.cuda()

            # Forward pass
            outputs = netC(netB(netF(inputs)))  # [B, 6*(num_classes+1)]
            B = outputs.shape[0]

            # Reshape to [B, 6, num_classes+1]
            outputs = outputs.view(B, num_users, num_classes + 1)

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


def build_target_centroids(loader, netF, netB, netC, num_users, num_classes, conf_thresh=0.7):
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
        for inputs, _ in loader:
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
        for inputs, _ in loader:
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


def train_target_shot(source_dir, output_dir, out_file, random_state):
    """
    Domain adaptation training on target with SHOT + CPC integration.

    MERGED VERSION:
    - Comprehensive metrics tracking from SHOTPlus
    - CPC self-supervised pre-training from SHOTPlus_CPC
    - SSL rotation pre-training
    - All new metrics: slot-wise accuracy, exact match, per-activity F1, occupancy, etc.
    """
    print("Loading target data for adaptation...")
    train_loader, valid_loader, test_loader, classifier_output_size, var_x_shape, num_users, num_classes = dset_loaders("target_data", random_state=random_state)

    # Create non-shuffled loader for pseudo-labeling
    train_loader_no_shuffle, _, _, _, _, _, _ = dset_loaders("target_data", random_state=random_state)
    # Override with non-shuffled version
    train_data_x = train_loader.dataset.tensors[0].cpu().numpy()
    train_data_y = train_loader.dataset.tensors[1].cpu().numpy()
    train_loader_no_shuffle = create_dataset(train_data_x, train_data_y,
                                              config=preset["target_data"],
                                              shuffle=False, is_training=False)

    netF = network.CNN2DBase(var_x_shape).cuda()
    netB = network.feat_bottleneck(feature_dim=netF.in_features, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    netC = network.feat_classifier(class_num=classifier_output_size, bottleneck_dim=preset["training"]["bottleneck"]).cuda()

    netF.load_state_dict(torch.load(osp.join(source_dir, "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(source_dir, "source_B.pt")))
    netC.load_state_dict(torch.load(osp.join(source_dir, "source_C.pt")))

    # ===== SSL ROTATION PRE-TRAINING =====
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

        # Clean up after rotation training
        torch.cuda.empty_cache()
        gc.collect()
    # ===== CPC SELF-SUPERVISED PRE-TRAINING =====
    cpc_model = None
    cpc_metrics = None
    if preset["training"].get("old_cpc",True):  
        if preset["training"].get("cpc_weight", 0) > 0 and preset["training"].get("cpc_pretrain", False):
            print("\n" + "="*50)
            print("CPC SELF-SUPERVISED PRE-TRAINING")
            print("="*50)
            # ===== MEMORY CLEANUP BEFORE CPC =====
            print("Cleaning up GPU memory before CPC training...")
            # Move unused models to CPU temporarily to free GPU memory
            if netR is not None:
                netR = netR.cpu()
            netC = netC.cpu()
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()
            # Print memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
            # ===== DETERMINE INPUT_FREQ FROM ACTUAL DATA =====
            # ===== DETERMINE INPUT_FREQ FROM ACTUAL DATA =====
            print("Determining CPC input dimensions from data...")
            # Get a sample batch to inspect shape
            sample_batch = next(iter(train_loader))
            sample_input = sample_batch[0]  # Shape: (B, Time, Features)
            
            if len(sample_input.shape) != 3:
                raise ValueError(f"Expected 3D input (B, Time, Features), got shape {sample_input.shape}")
            
            B, T, feat_dim = sample_input.shape  # Use feat_dim instead of F
            input_freq = feat_dim  # Features dimension is what CPC expects
            
            print(f"Input shape: (B={B}, Time={T}, Features={feat_dim})")
            print(f"CPC input_freq = {input_freq}")
            
            # Validate input_freq is reasonable
            if input_freq <= 0 or input_freq > 10000:
                raise ValueError(f"Invalid input_freq={input_freq}. Check your data shape.")
            
            del sample_batch, sample_input  # Clean up
            torch.cuda.empty_cache()
            # ================================================
            
            print("Initializing CPC model for self-supervised learning...")
            cpc_model = CPCModel(
                input_freq=input_freq,  # Dynamically determined from Features dimension
                embedding_dim=preset["training"].get("cpc_embedding_dim", 256),
                hidden_dim=preset["training"].get("cpc_hidden_dim", 512),
                projection_dim=preset["training"].get("cpc_projection_dim", 256),
                num_gru_layers=preset["training"].get("cpc_num_gru_layers", 2),
                temperature=preset["training"].get("cpc_temperature", 0.07),
                window_size=preset["training"].get("cpc_window_size", 30),
                prediction_steps=preset["training"].get("cpc_prediction_steps", 9),
                use_masking=preset["training"].get("cpc_use_masking", False),
                mask_prob=preset["training"].get("cpc_mask_prob", 0.5),
                mask_ratio=preset["training"].get("cpc_mask_ratio", 0.15)
            ).cuda()
            
            # ===== WRAP IN DATAPARALLEL BEFORE PRETRAINING =====
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs for CPC pre-training")
                cpc_model = nn.DataParallel(cpc_model)
            # ====================================================
            
            # Pre-train CPC on target domain (unsupervised)
            print("Pre-training CPC on target domain...")
            best_cpc_state, cpc_metrics = pretrain_cpc(
                cpc_model=cpc_model,
                data_loader=train_loader,
                device=torch.device('cuda'),
                num_epochs=preset["training"].get("cpc_pretrain_epochs", 70),
                lr=preset["training"].get("cpc_pretrain_lr", 1e-3),
                out_file=out_file
            )
            
            # UNWRAP from DataParallel and load best weights
            base_cpc_model = get_model_device_safe(cpc_model)
            base_cpc_model.load_state_dict(best_cpc_state)
            cpc_model = base_cpc_model  # Use unwrapped version
            print(f"CPC Pre-training completed! Best accuracy: {cpc_metrics['best_cpc_accuracy']:.4f}\n")
            
            # ===== MEMORY CLEANUP AFTER CPC =====
            print("Cleaning up GPU memory after CPC training...")
            # Move models back to GPU
            if netR is not None:
                netR = netR.cuda()
            netC = netC.cuda()
            cpc_model = cpc_model.cuda()
            # Clear cache again
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # ===== CPC SELF-SUPERVISED PRE-TRAINING =====
    else:  
        cpc_model = None
        cpc_metrics = None
        if preset["training"].get("cpc_weight", 0) > 0 and preset["training"].get("cpc_pretrain", False):
            print("\n" + "="*50)
            print("CPC SELF-SUPERVISED PRE-TRAINING (Per-Slot)")
            print("="*50)
            # ===== MEMORY CLEANUP BEFORE CPC =====
            print("Cleaning up GPU memory before CPC training...")
            if netR is not None:
                netR = netR.cpu()
            netC = netC.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
            # ===== DETERMINE INPUT DIMENSIONS FROM ACTUAL DATA =====
            print("Determining CPC input dimensions from data...")
            sample_batch = next(iter(train_loader))
            sample_input = sample_batch[0]  # Shape: (B, Time, Features) or (B, Time, Features, Channels)
            
            # Get num_slots from preset or infer from data
            num_slots = preset["training"].get("cpc_num_slots", None)
            if len(sample_input.shape) == 4:
                # (B, T, F, C) - C could be slots
                B, T, F_dim, C = sample_input.shape  # Changed F to F_dim
                if num_slots is None:
                    num_slots = C  # Treat channels as slots
                input_freq = F_dim  # Features per slot
                print(f"Input shape: (B={B}, T={T}, F={F_dim}, C={C})")
                print(f"Detected {num_slots} slots, {input_freq} features per slot")
            elif len(sample_input.shape) == 3:
                # (B, T, F) - need to split F into slots
                B, T, feat_dim = sample_input.shape
                if num_slots is None:
                    raise ValueError(
                        "Input is (B, T, F) but cpc_num_slots not specified in preset. "
                        "Please set preset['training']['cpc_num_slots']"
                    )
                input_freq = feat_dim // num_slots  # Features per slot
                print(f"Input shape: (B={B}, T={T}, Features={feat_dim})")
                print(f"Configured {num_slots} slots, {input_freq} features per slot")
                
                if feat_dim % num_slots != 0:
                    raise ValueError(
                        f"feat_dim={feat_dim} not divisible by num_slots={num_slots}"
                    )
            else:
                raise ValueError(f"Expected 3D or 4D input, got shape {sample_input.shape}")
            
            # Validate dimensions
            if input_freq <= 0 or input_freq > 10000:
                raise ValueError(f"Invalid input_freq={input_freq}. Check your data shape.")
            
            del sample_batch, sample_input
            torch.cuda.empty_cache()
            # ================================================
            
            print("Initializing Per-Slot CPC model for self-supervised learning...")
            cpc_model = CPCModel(
                input_freq=input_freq,  # Features PER SLOT
                num_slots=num_slots,    # Number of slots
                embedding_dim=preset["training"].get("cpc_embedding_dim", 256),
                hidden_dim=preset["training"].get("cpc_hidden_dim", 512),
                projection_dim=preset["training"].get("cpc_projection_dim", 256),
                num_gru_layers=preset["training"].get("cpc_num_gru_layers", 2),
                temperature=preset["training"].get("cpc_temperature", 0.07),
                window_size=preset["training"].get("cpc_window_size", 30),
                prediction_steps=preset["training"].get("cpc_prediction_steps", 9),
                use_masking=preset["training"].get("cpc_use_masking", False),
                mask_prob=preset["training"].get("cpc_mask_prob", 0.5),
                mask_ratio=preset["training"].get("cpc_mask_ratio", 0.15),
                # New per-slot parameters
                negative_mode=preset["training"].get("cpc_negative_mode", "cross_batch"),
                return_per_slot_loss=preset["training"].get("cpc_return_per_slot_loss", False)
            ).cuda()
            
            # ===== WRAP IN DATAPARALLEL BEFORE PRETRAINING =====
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs for CPC pre-training")
                cpc_model = nn.DataParallel(cpc_model)
            # ====================================================
            
            # Pre-train CPC on target domain (unsupervised)
            print(f"Pre-training Per-Slot CPC on target domain ({num_slots} slots)...")
            best_cpc_state, cpc_metrics = pretrain_cpc(
                cpc_model=cpc_model,
                data_loader=train_loader,
                device=torch.device('cuda'),
                num_epochs=preset["training"].get("cpc_pretrain_epochs", 70),
                lr=preset["training"].get("cpc_pretrain_lr", 1e-3),
                out_file=out_file
            )
            
            # UNWRAP from DataParallel and load best weights
            base_cpc_model = get_model_device_safe(cpc_model)
            base_cpc_model.load_state_dict(best_cpc_state)
            cpc_model = base_cpc_model
            print(f"CPC Pre-training completed! Best accuracy: {cpc_metrics['best_cpc_accuracy']:.4f}\n")
            
            # ===== MEMORY CLEANUP AFTER CPC =====
            print("Cleaning up GPU memory after CPC training...")
            if netR is not None:
                netR = netR.cuda()
            netC = netC.cuda()
            cpc_model = cpc_model.cuda()
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        # =====================================

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        # =====================================

        if torch.cuda.device_count() > 1:
            netF, netB, netC = nn.DataParallel(netF), nn.DataParallel(netB), nn.DataParallel(netC)
            if netR is not None:
                netR = nn.DataParallel(netR)
            if cpc_model is not None:
                # Only wrap if not already wrapped
                if not isinstance(cpc_model, nn.DataParallel):
                    cpc_model = nn.DataParallel(cpc_model)

    for v in get_model_device_safe(netC).parameters(): v.requires_grad = False

    param_group = []
    for v in get_model_device_safe(netF).parameters(): param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
    for v in get_model_device_safe(netB).parameters(): param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]

    # Add netR to optimizer if SSL enabled
    if netR is not None:
        for v in get_model_device_safe(netR).parameters():
            param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
        netR.train()

    # Add CPC to optimizer if enabled
    if cpc_model is not None:
        for v in get_model_device_safe(cpc_model).parameters():
            param_group += [{'params': v, 'lr': preset["training"]["target_lr"]}]
        cpc_model.train()

    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["target_lr"], weight_decay=1e-4))

    max_epochs = preset["training"]["max_epoch"]
    best_acc = 0
    best_f1_micro = 0
    best_f1_macro = 0
    best_hamming = float('inf')
    best_epoch = 0
    best_metrics = {}
    best_test_results = {}
    best_netF = None
    best_netB = None


    # Store adaptation history
    adaptation_history = []

    # ===== PSEUDO-LABELING SETUP (Centroid-based) =====
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
            train_loader_no_shuffle, netF, netB, netC, num_users, num_classes, conf_thresh=0.7
        )
        print(f"Centroids built for {num_classes} classes")

        # Generate pseudo-labels using centroids
        print("Generating pseudo-labels from centroids...")
        mem_pseudo_labels, mem_confidences = obtain_pseudo_labels_with_centroids(
            train_loader_no_shuffle, netF, netB, netC, target_centroids, num_users, num_classes, out_file
        )
        print(f"Pseudo-labeling loss weight: {cls_par}")
    # =================================

    print("\nStarting target adaptation...")
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.eval()

        # Update centroids and pseudo-labels periodically
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
                train_loader_no_shuffle, netF, netB, netC, num_users, num_classes, conf_thresh=0.7
            )

            # Generate new pseudo-labels
            print("Regenerating pseudo-labels from centroids...")
            mem_pseudo_labels, mem_confidences = obtain_pseudo_labels_with_centroids(
                train_loader_no_shuffle, netF, netB, netC, target_centroids, num_users, num_classes, out_file
            )

            # Restore training mode
            netF.train()
            netB.train()

        # Adaptation training loop
        epoch_loss = 0
        num_batches = 0
        batch_idx = 0
        for inputs_target, _ in train_loader:
            inputs_target = inputs_target.cuda()
            outputs_target = netC(netB(netF(inputs_target)))  # [B, 6*(num_classes+1)]
            B = outputs_target.shape[0]

            # Reshape to [B, 6, num_classes+1]
            outputs_target = outputs_target.view(B, num_users, num_classes + 1)

            # ============================================================
            # SHOT ENTROPY MINIMIZATION (multiclass version)
            # ============================================================
            outputs_flat = outputs_target.view(B * num_users, num_classes + 1)  # [B*6, num_classes+1]

            # Apply softmax
            softmax_out = F.softmax(outputs_flat, dim=1)  # [B*6, num_classes+1]

            # ============================================================
            # CONDITIONAL ENTROPY: Minimize for confident predictions
            # ============================================================
            entropy_per_slot = -softmax_out * torch.log(softmax_out + 1e-5)
            entropy_per_slot = entropy_per_slot.sum(dim=1)  # [B*6]

            # Mean entropy across all slots
            entropy_loss = entropy_per_slot.mean()

            # ============================================================
            # WEIGHTED GENT: Diversity regularization for occupied slots
            # ============================================================
            if preset["training"].get("gent", False):
                if preset["training"].get("use_occupancy_weighted_gent", True):
                    # Extract predictions for actual activity/location classes (exclude no_person)
                    softmax_classes = softmax_out[:, :num_classes]  # [B*6, num_classes]
    
                    # Compute probability that each slot is occupied (not "no_person")
                    # occupied_prob: [B*6, 1]
                    # High value = model confident slot is occupied
                    # Low value = model thinks slot is empty
                    occupied_prob = softmax_classes.sum(dim=1, keepdim=True)  # [B*6, 1]
    
                    # Weight each slot's class probabilities by its occupancy confidence
                    # This gives more importance to slots that are likely occupied
                    # and reduces impact of slots that are likely empty
                    weighted_softmax_classes = softmax_classes * occupied_prob  # [B*6, num_classes]
    
                    # Average across all slots in the batch
                    # msoftmax: [num_classes] - marginal distribution over activity/location classes
                    msoftmax = weighted_softmax_classes.mean(dim=0)  # [num_classes]
    
                    # Normalize to make it a proper probability distribution
                    msoftmax = msoftmax / (msoftmax.sum() + 1e-8)
    
                    # Compute entropy of this marginal distribution
                    # High entropy = predictions are diverse across different activities/locations
                    # Low entropy = model collapsed to always predicting same activity
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    
                    # SUBTRACT to MAXIMIZE diversity (prevent collapse)
                    # This encourages the model to predict different activities/locations
                    # for different occupied slots, rather than predicting the same
                    # dominant activity for all occupied slots
                    entropy_loss -= gentropy_loss
                else:
                    # ========================================================
                    # STANDARD UNWEIGHTED GENT (Original SHOT/SHOT++)
                    # ========================================================
                    # Compute marginal distribution across ALL classes (including "no_person")
                    msoftmax = softmax_out.mean(dim=0)  # [num_classes+1]
                    
                    # Compute entropy of marginal distribution
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                    
                    # SUBTRACT to MAXIMIZE diversity (prevent collapse)
                    entropy_loss -= gentropy_loss
            

            # ============================================================
            # TOTAL INFORMATION MAXIMIZATION LOSS
            # ============================================================
            im_loss = entropy_loss * preset["training"]["ent_par"]
            print(f"Entropy loss: {im_loss.item()}")
            # ============================================================
            # PSEUDO-LABELING LOSS (if enabled)
            # ============================================================
            pseudo_loss = torch.tensor(0.0).cuda()

            if cls_par > 0 and mem_pseudo_labels is not None:
                # Get pseudo-labels for this batch
                batch_start = batch_idx * train_loader.batch_size
                batch_end = batch_start + B
                pseudo_labels_batch = torch.from_numpy(
                    mem_pseudo_labels[batch_start:batch_end]
                ).cuda()  # [B, 6]

                # Compute pseudo-labeling loss with Hungarian matching
                # outputs_target: [B, 6, num_classes+1]
                # pseudo_labels_batch: [B, 6]

                # Hungarian matching
                matched_pred_indices, matched_gt_indices = hungarian_matching_batch(
                    outputs_target, pseudo_labels_batch, num_classes
                )

                # Compute CE loss on matched pairs
                criterion = nn.CrossEntropyLoss()
                batch_pseudo_loss = 0

                for b in range(B):
                    pred_slots = outputs_target[b]         # [6, num_classes+1]
                    pseudo_slots = pseudo_labels_batch[b]  # [6]

                    # Get matched pairs
                    pred_indices = matched_pred_indices[b]
                    gt_indices = matched_gt_indices[b]

                    for i in range(6):
                        pred_slot_idx = pred_indices[i]
                        gt_slot_idx = gt_indices[i]

                        pred_logits = pred_slots[pred_slot_idx]   # [num_classes+1]
                        pseudo_class = pseudo_slots[gt_slot_idx]   # scalar

                        # CE loss for this matched pair
                        batch_pseudo_loss += criterion(
                            pred_logits.unsqueeze(0),
                            pseudo_class.unsqueeze(0)
                        )

                # Average over batch and 6 slots
                pseudo_loss = cls_par * (batch_pseudo_loss / (B * 6))

            # ============================================================
            # SSL ROTATION LOSS (if enabled)
            # ============================================================
            total_loss = im_loss + pseudo_loss

            if preset["training"]["ssl"] > 0 and netR is not None:
                # Generate rotations
                r_labels = torch.randint(0, 2, (inputs_target.shape[0],), dtype=torch.long, device=inputs_target.device)
                r_inputs = rotation.rotate_batch_with_labels(inputs_target, r_labels)

                # Forward with detached original features
                f_outputs = netB(netF(inputs_target)).detach()
                f_r_outputs = netB(netF(r_inputs))
                r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))

                rotation_loss = preset["training"]["ssl"] * nn.CrossEntropyLoss()(r_outputs, r_labels)
                total_loss = total_loss + rotation_loss
                print(f"SSL loss: {rotation_loss.item()}")

            # ============================================================
            # CPC SELF-SUPERVISED LOSS (if enabled)
            # ============================================================
            if cpc_model is not None and preset["training"].get("cpc_weight", 0) > 0:
                # Compute CPC loss
                cpc_loss, cpc_acc = compute_cpc_loss(
                    cpc_model=cpc_model,
                    inputs=inputs_target
                )

                # Weighted CPC loss
                weighted_cpc_loss = preset["training"]["cpc_weight"] * cpc_loss
                total_loss = total_loss + weighted_cpc_loss
                print(f"CPC loss: {weighted_cpc_loss.item()}")

            # ============================================================
            # BACKPROPAGATION
            # ============================================================
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            num_batches += 1
            batch_idx += 1

        avg_loss = epoch_loss / num_batches

        # Evaluation with COMPREHENSIVE METRICS
        netF.eval(); netB.eval()
        valid_metrics = cal_acc(valid_loader, netF, netB, netC, num_users, num_classes)

        # Store epoch results with ALL metrics
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

        if valid_metrics['slot_wise_accuracy'] > best_acc:
            best_acc = valid_metrics['slot_wise_accuracy']
            best_f1_micro = valid_metrics['f1_micro']
            best_f1_macro = valid_metrics['f1_macro']
            best_hamming = valid_metrics['hamming_loss']
            best_epoch = epoch + 1


            best_netF = copy.deepcopy(get_model_device_safe(netF).state_dict())
            best_netB = copy.deepcopy(get_model_device_safe(netB).state_dict())

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

            # Save best adapted model's detailed VALIDATION report
            savename = f'ent_{preset["training"]["ent_par"]}'
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
        out_file.write(log_str + '\n'); out_file.flush(); print(log_str)

    print("\nEvaluating best adapted model on TEST set...")

    # Load best model states into base models
    base_netF = get_model_device_safe(netF) if hasattr(netF, 'module') else netF
    base_netB = get_model_device_safe(netB) if hasattr(netB, 'module') else netB
    base_netC = get_model_device_safe(netC) if hasattr(netC, 'module') else netC

    base_netF.load_state_dict(best_netF)
    base_netB.load_state_dict(best_netB)

    # Wrap in DataParallel if needed
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
    savename = f'ent_{preset["training"]["ent_par"]}'
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
    out_file.write(log_str + '\n'); out_file.flush(); print(log_str)


    # Save adaptation history
    history_file = osp.join(output_dir, "target_adaptation_history.json")
    with open(history_file, 'w') as f:
        json.dump(adaptation_history, f, indent=2)

    # Prepare comprehensive return dictionary with ALL metrics + CPC
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
        'per_activity_f1': test_metrics['per_activity_f1'],
        # Training history
        'adaptation_history': adaptation_history
    }

    # Add SSL metrics if they exist
    if ssl_metrics is not None:
        result['ssl_metrics'] = ssl_metrics

    # Add CPC metrics if they exist
    if cpc_metrics is not None:
        result['cpc_metrics'] = cpc_metrics

    return result

def get_domain_name(config_key):
    config = preset[config_key]
    env = "_".join(config["environment"])
    wifi = "_".join(config["wifi_band"]) + "GHz"
    users = f"users_{min(config['num_users'])}-{max(config['num_users'])}"
    return f"{env}_{wifi}_{users}"

def get_adaptation_config_name():
    """Generate a unique name for the current adaptation configuration"""
    config_parts = []

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

def print_config():
    s = "==========================================\nEXPERIMENT CONFIGURATION\n==========================================\n"
    for key, content in preset.items():
        if key != "encoding": s += f"{key}: {content}\n"
    s += "==========================================\n"
    return s

def run_single_experiment(run_idx, seed, base_output_dir, source_dir):
    """Executes a single run of the experiment with a given seed - runs source baseline, SHOT+CPC"""
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
    
    # --- Target SHOT+CPC Adaptation ---
    print("\n" + "="*50 + "\nPHASE 3: TARGET DOMAIN ADAPTATION - SHOT+CPC\n" + "="*50)
    with open(osp.join(output_dir, 'log_tar_shot.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        out_file.write("="*50 + "\n")
        out_file.write("METHOD: SHOT+CPC (Entropy Minimization + CPC)\n")
        out_file.write("="*50 + "\n\n")
        target_shot_results = train_target_shot(
            source_run_dir, output_dir, out_file,
            random_state=seed,
        )
    
    # Compile all results
    run_results.update({
        'source_domain': source_results,
        'target_domain_baseline': target_baseline_results,
        'target_domain_adapted': target_shot_results
    })
    
    # Save comprehensive results for this run
    save_run_results(output_dir, run_results)
    
    # Return ALL comprehensive metrics for aggregation - source baseline and SHOT+CPC
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
            'per_activity_f1': target_shot_results['per_activity_f1'],
        },
        'ssl_metrics': target_shot_results.get('ssl_metrics', None),
        'cpc_metrics': target_shot_results.get('cpc_metrics', None)
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
    script_dir = osp.dirname(osp.abspath(__file__))  # WiMANS/SHOTPlus_CPC
    wimans_dir = osp.dirname(script_dir)  # WiMANS
    shotplus_dir = osp.join(wimans_dir, "SHOTPlus")  # WiMANS/SHOTPlus
    
    # Load source models from WiMANS/SHOTPlus (NOT from SHOTPlus_CPC)
    source_base_dir = osp.join(
        shotplus_dir,
        "SHOTPlus_WiMANS_Results",  # SHOTPlus uses this save_dir
        experiment_name,
        f"{source_domain}_to_{target_domain}",
        "source",
        f"seed_{initial_seed}"
    )
    
    # Create adaptation-specific directory
    adaptation_config = get_adaptation_config_name()
    base_output_dir = osp.join(
        preset["path"]["save_dir"],
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
    print(f"Experiment: {experiment_name}")
    print(f"Source → Target: {source_domain} → {target_domain}")
    print(f"Adaptation Config: {adaptation_config}")
    print(f"Running Configurations:")
    print(f"  1. Source Baseline (Source model on target, no adaptation)")
    print(f"  2. SHOT+CPC (Entropy minimization + CPC self-supervision)")
    print(f"Number of Runs: {preset['repeat']}")
    print(f"Initial Seed: {initial_seed}")
    print(f"Source Model Directory: {source_base_dir}")
    print(f"Adaptation Results Directory: {base_output_dir}")
    print(f"{'='*80}\n")
    
    # Lists to store results from all runs
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
        all_activities = {}
        for m in metrics_list:
            for act_key, f1_val in m['per_activity_f1'].items():
                if act_key not in all_activities:
                    all_activities[act_key] = []
                all_activities[act_key].append(f1_val)
        
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
    
    # Aggregated statistics for SHOT+CPC (with adaptation)
    shot_stats = {
        'test_slot_wise_accuracy': calculate_stats(all_shot_metrics, 'test_slot_wise_accuracy'),
        'test_exact_match_accuracy': calculate_stats(all_shot_metrics, 'test_exact_match_accuracy'),
        'test_f1_micro': calculate_stats(all_shot_metrics, 'test_f1_micro'),
        'test_f1_macro': calculate_stats(all_shot_metrics, 'test_f1_macro'),
        'test_activity_macro_f1': calculate_stats(all_shot_metrics, 'test_activity_macro_f1'),
        'test_occupancy_mae': calculate_stats(all_shot_metrics, 'test_occupancy_mae'),
        'test_occupancy_exact_match': calculate_stats(all_shot_metrics, 'test_occupancy_exact_match'),
        'test_hamming_loss': calculate_stats(all_shot_metrics, 'test_hamming_loss'),
        'per_activity_f1': aggregate_per_activity_f1(all_shot_metrics),
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
    
    # Aggregate CPC metrics if they exist
    cpc_stats = None
    cpc_metrics_list = [m.get('cpc_metrics') for m in all_shot_metrics if m.get('cpc_metrics') is not None]
    if cpc_metrics_list:
        cpc_accuracies = [m['best_cpc_accuracy'] for m in cpc_metrics_list if 'best_cpc_accuracy' in m]
        if cpc_accuracies:
            cpc_stats = {
                'best_cpc_accuracy': {
                    'mean': float(np.mean(cpc_accuracies)),
                    'std': float(np.std(cpc_accuracies)),
                    'values': cpc_accuracies
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
    if cpc_stats is not None:
        aggregated_stats['cpc_stats'] = cpc_stats
    
    # If we have source domain training results, aggregate those
    if all_run_summaries:
        source_train_slot_accs = []
        source_train_exact_match = []
        source_train_activity_f1 = []
        source_train_f1_micros = []
        source_train_f1_macros = []
        
        for run_summary in all_run_summaries:
            if 'source_domain' in run_summary and run_summary['source_domain']:
                sd = run_summary['source_domain']
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
    # METHOD 2: SHOT+CPC (Entropy Minimization + CPC)
    # ============================================================
    summary_str += "="*80 + "\n"
    summary_str += "METHOD 2: SHOT+CPC (Entropy Minimization + CPC Self-Supervision)\n"
    summary_str += "="*80 + "\n"
    summary_str += "Adaptation via entropy minimization with CPC pre-training\n\n"
    
    # SSL Metrics (if available)
    if 'ssl_stats' in aggregated_stats:
        ssl_rot_acc = aggregated_stats['ssl_stats']['best_rotation_accuracy']
        summary_str += f"SSL Rotation Pre-training: {ssl_rot_acc['mean']:.2f}% ± {ssl_rot_acc['std']:.2f}\n"
    
    # CPC Metrics (if available)
    if 'cpc_stats' in aggregated_stats:
        cpc_acc = aggregated_stats['cpc_stats']['best_cpc_accuracy']
        summary_str += f"CPC Pre-training Accuracy: {cpc_acc['mean']:.4f} ± {cpc_acc['std']:.4f}\n"
    
    if 'ssl_stats' in aggregated_stats or 'cpc_stats' in aggregated_stats:
        summary_str += "\n"
    
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
    
    summary_str += "Improvement over Source Baseline:\n"
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
    aggregated_results_file = osp.join(base_output_dir, "aggregated_results.json")
    aggregated_stats['timestamp'] = datetime.now().isoformat()
    aggregated_stats['experiment_config'] = {k: v for k, v in preset.items() if k != 'encoding'}
    
    with open(aggregated_results_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    
    summary_file_path = osp.join(base_output_dir, "final_aggregated_results.txt")
    with open(summary_file_path, 'w') as f:
        f.write(summary_str)
    
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