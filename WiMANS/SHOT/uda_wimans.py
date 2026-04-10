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
import pickle
import json
from datetime import datetime
from sklearn.metrics import hamming_loss, f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Project modules
import network
import loss
from preset import preset
from load_data import load_data_x, load_data_y, encode_data_y

import pandas as pd  # Add this import if not already there

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
    
def create_dataset(data_x, data_y, batch_size=None, shuffle=True):
    """Create PyTorch dataset and dataloader"""
    if batch_size is None:
        batch_size = preset["training"]["batch_size"]
    
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        batch_size = batch_size * gpu_count
    
    # Adjust batch size if dataset is smaller
    dataset_size = len(data_x)
    if dataset_size < batch_size:
        batch_size = max(1, dataset_size // 2)  # Use half the dataset size, minimum 1
        print(f"Warning: Dataset size ({dataset_size}) smaller than batch size. Adjusted to {batch_size}")
    
    dataset = TensorDataset(torch.FloatTensor(data_x), torch.FloatTensor(data_y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                           drop_last=False,  # Change to False to keep smaller batches
                           num_workers=4, pin_memory=True)
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
    
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    
    var_x_shape = data_train_x[0].shape
    var_y_shape = data_train_y[0].reshape(-1).shape
    
    # CHANGE TO MATCH THEIR APPROACH ( In paper ) :
    classifier_output_size = var_y_shape[-1]  # Get size directly from actual data
    
    num_users = 6  # Always 6 in WiMANS dataset
    task_encoding = preset["encoding"][task]
    num_task_classes = len(list(task_encoding.values())[0])
    
    train_loader = create_dataset(data_train_x, data_train_y, shuffle=True)
    test_loader = create_dataset(data_test_x, data_test_y, shuffle=False)
    
    return train_loader, test_loader, classifier_output_size, var_x_shape, var_y_shape, num_users, num_task_classes

def cal_acc(loader, netF, netB, netC, num_users, num_classes):
    """Calculate accuracy on given loader and generate a classification report."""
    netF.eval()
    netB.eval()
    netC.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = netC(netB(netF(inputs)))
            preds = (torch.sigmoid(outputs) > 0.5).float()
            preds_np = preds.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            preds_reshaped = preds_np.reshape(-1, num_classes)
            targets_reshaped = targets_np.reshape(-1, num_classes)
            all_preds.append(preds_reshaped)
            all_labels.append(targets_reshaped)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    accuracy = accuracy_score(all_labels.astype(int), all_preds.astype(int)) * 100
    f1_micro = f1_score(all_labels.astype(int), all_preds.astype(int), average='micro', zero_division=0) * 100
    f1_macro = f1_score(all_labels.astype(int), all_preds.astype(int), average='macro', zero_division=0) * 100
    hamming = hamming_loss(all_labels.astype(int), all_preds.astype(int)) * 100
    report = classification_report(all_labels.astype(int), all_preds.astype(int), zero_division=0)
    
    return accuracy, f1_micro, f1_macro, hamming, report

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
    
    # Also save a human-readable summary
    summary_file = osp.join(output_dir, "run_results_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPLETE RUN RESULTS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Run Directory: {output_dir}\n")
        f.write(f"Timestamp: {run_results['timestamp']}\n")
        f.write(f"Random Seed: {run_results['seed']}\n\n")
        
        f.write("SOURCE DOMAIN RESULTS:\n")
        f.write("-" * 30 + "\n")
        source_results = run_results['source_domain']
        f.write(f"Best Training Accuracy: {source_results['best_train_accuracy']:.2f}%\n")
        f.write(f"Test Accuracy: {source_results['test_accuracy']:.2f}%\n")
        f.write(f"Test F1-Micro: {source_results['test_f1_micro']:.2f}%\n")
        f.write(f"Test F1-Macro: {source_results['test_f1_macro']:.2f}%\n")
        f.write(f"Test Hamming Loss: {source_results['test_hamming_loss']:.2f}%\n\n")
        
        f.write("TARGET DOMAIN RESULTS (Source Model):\n")
        f.write("-" * 30 + "\n")
        target_baseline = run_results['target_domain_baseline']
        f.write(f"Accuracy: {target_baseline['accuracy']:.2f}%\n")
        f.write(f"F1-Micro: {target_baseline['f1_micro']:.2f}%\n")
        f.write(f"F1-Macro: {target_baseline['f1_macro']:.2f}%\n")
        f.write(f"Hamming Loss: {target_baseline['hamming_loss']:.2f}%\n\n")
        
        f.write("TARGET DOMAIN RESULTS (After Adaptation):\n")
        f.write("-" * 30 + "\n")
        target_adapted = run_results['target_domain_adapted']
        f.write(f"Best Accuracy: {target_adapted['best_accuracy']:.2f}%\n")
        f.write(f"Best F1-Micro: {target_adapted['best_f1_micro']:.2f}%\n")
        f.write(f"Best F1-Macro: {target_adapted['best_f1_macro']:.2f}%\n")
        f.write(f"Best Hamming Loss: {target_adapted['best_hamming_loss']:.2f}%\n")
        f.write(f"Best Epoch: {target_adapted['best_epoch']}\n\n")
        
        f.write("PERFORMANCE IMPROVEMENTS:\n")
        f.write("-" * 30 + "\n")
        acc_improvement = target_adapted['best_accuracy'] - target_baseline['accuracy']
        f1_micro_improvement = target_adapted['best_f1_micro'] - target_baseline['f1_micro']
        f1_macro_improvement = target_adapted['best_f1_macro'] - target_baseline['f1_macro']
        f.write(f"Accuracy Improvement: {acc_improvement:+.2f}%\n")
        f.write(f"F1-Micro Improvement: {f1_micro_improvement:+.2f}%\n")
        f.write(f"F1-Macro Improvement: {f1_macro_improvement:+.2f}%\n")
        f.write("="*60 + "\n")
    
    print(f"Run results saved to: {results_file}")
    print(f"Run summary saved to: {summary_file}")

def train_source(output_dir, out_file, random_state):
    """Train source domain model"""
    print("Loading source data...")
    train_loader, test_loader, classifier_output_size, var_x_shape, var_y_shape, num_users, num_classes = dset_loaders("source_data", random_state=random_state)
    
    # Model setup
    netF = network.CNN2DBase(var_x_shape).cuda()
    netB = network.feat_bottleneck(feature_dim=netF.in_features, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    netC = network.feat_classifier(class_num=classifier_output_size, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6] * var_y_shape[-1]).to(device))
    
    # Store training history
    training_history = []
    
    print("Starting source training...")
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.train()
        
        # Training loop
        epoch_loss = 0
        num_batches = 0
        for inputs_source, labels_source in train_loader:
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            outputs_source = netC(netB(netF(inputs_source)))
            labels_flattened = labels_source.reshape(labels_source.shape[0], -1).float()
            classifier_loss = criterion(outputs_source, labels_flattened)
            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
            epoch_loss += classifier_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation
        netF.eval(); netB.eval(); netC.eval()
        
        # Training accuracy
        acc_s_tr, f1_micro_tr, f1_macro_tr, hamming_tr, report_tr = cal_acc(train_loader, netF, netB, netC, num_users, num_classes)
        
        # Test accuracy
        acc_s_te, f1_micro_te, f1_macro_te, hamming_te, report_te = cal_acc(test_loader, netF, netB, netC, num_users, num_classes)
        
        # Store epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'train_accuracy': acc_s_tr,
            'train_f1_micro': f1_micro_tr,
            'train_f1_macro': f1_macro_tr,
            'train_hamming_loss': hamming_tr,
            'test_accuracy': acc_s_te,
            'test_f1_micro': f1_micro_te,
            'test_f1_macro': f1_macro_te,
            'test_hamming_loss': hamming_te
        }
        training_history.append(epoch_results)
        
        log_str = f'Source Training - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Train Acc: {acc_s_tr:.2f}%; Test Acc: {acc_s_te:.2f}%'
        out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
        
        if acc_s_te >= best_acc:
            best_acc = acc_s_te
            best_train_acc = acc_s_tr
            best_netF = copy.deepcopy(get_model_device_safe(netF).state_dict())
            best_netB = copy.deepcopy(get_model_device_safe(netB).state_dict())
            best_netC = copy.deepcopy(get_model_device_safe(netC).state_dict())
            
            # Save best model's detailed report
            best_report_file = osp.join(output_dir, "source_best_classification_report.txt")
            with open(best_report_file, 'w') as f:
                f.write(f"Best Source Model Performance (Epoch {epoch+1})\n")
                f.write(f"Test Accuracy: {acc_s_te:.2f}%\n")
                f.write(f"Test F1-Micro: {f1_micro_te:.2f}%\n")
                f.write(f"Test F1-Macro: {f1_macro_te:.2f}%\n")
                f.write(f"Test Hamming Loss: {hamming_te:.2f}%\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(report_te)

    # Save models
    torch.save(best_netF, osp.join(output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(output_dir, "source_C.pt"))
    
    # Save training history
    history_file = osp.join(output_dir, "source_training_history.json")
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Source training completed. Best test accuracy: {best_acc:.2f}%")
    
    # Return source domain results
    return {
        'best_train_accuracy': best_train_acc,
        'test_accuracy': best_acc,
        'test_f1_micro': f1_micro_te,
        'test_f1_macro': f1_macro_te,
        'test_hamming_loss': hamming_te,
        'training_history': training_history
    }

def test_target_baseline(output_dir, out_file, random_state):
    """Test on target domain using the trained source model"""
    print("Loading target data for baseline testing...")
    _, test_loader, classifier_output_size, var_x_shape, var_y_shape, num_users, num_classes = dset_loaders("target_data", random_state=random_state)
    
    netF = network.CNN2DBase(var_x_shape).cuda()
    netB = network.feat_bottleneck(feature_dim=netF.in_features, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    netC = network.feat_classifier(class_num=classifier_output_size, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    
    netF.load_state_dict(torch.load(osp.join(output_dir, "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(output_dir, "source_B.pt")))
    netC.load_state_dict(torch.load(osp.join(output_dir, "source_C.pt")))
    
    if torch.cuda.device_count() > 1:
        netF, netB, netC = nn.DataParallel(netF), nn.DataParallel(netB), nn.DataParallel(netC)
        
    netF.eval(); netB.eval(); netC.eval()
    
    acc, f1_micro, f1_macro, hamming, report = cal_acc(test_loader, netF, netB, netC, num_users, num_classes)
    
    # Save baseline results
    baseline_report_file = osp.join(output_dir, "target_baseline_classification_report.txt")
    with open(baseline_report_file, 'w') as f:
        f.write("Target Domain Baseline Performance (Source Model)\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"F1-Micro: {f1_micro:.2f}%\n")
        f.write(f"F1-Macro: {f1_macro:.2f}%\n")
        f.write(f"Hamming Loss: {hamming:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    
    log_str = f'Target Test (Source Model) - Accuracy: {acc:.2f}%, F1-micro: {f1_micro:.2f}%, F1-macro: {f1_macro:.2f}%'
    out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
    
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'hamming_loss': hamming,
        'classification_report': report
    }

def train_target(output_dir, out_file, random_state):
    """Domain adaptation training on target"""
    print("Loading target data for adaptation...")
    train_loader, test_loader, classifier_output_size, var_x_shape, var_y_shape, num_users, num_classes = dset_loaders("target_data", random_state=random_state)
    
    netF = network.CNN2DBase(var_x_shape).cuda()
    netB = network.feat_bottleneck(feature_dim=netF.in_features, bottleneck_dim=preset["training"]["bottleneck"]).cuda()
    netC = network.feat_classifier(class_num=classifier_output_size, bottleneck_dim=preset["training"]["bottleneck"]).cuda()

    netF.load_state_dict(torch.load(osp.join(output_dir, "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(output_dir, "source_B.pt")))
    netC.load_state_dict(torch.load(osp.join(output_dir, "source_C.pt")))

    if torch.cuda.device_count() > 1:
        netF, netB, netC = nn.DataParallel(netF), nn.DataParallel(netB), nn.DataParallel(netC)

    for v in get_model_device_safe(netC).parameters(): v.requires_grad = False
        
    param_group = []
    for v in get_model_device_safe(netF).parameters(): param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netB).parameters(): param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["lr"], weight_decay=1e-4))
    
    max_epochs = preset["training"]["max_epoch"]
    best_acc = 0
    best_f1_micro = 0
    best_f1_macro = 0
    best_hamming = float('inf')
    best_epoch = 0
    best_metrics = {}
    
    # Store adaptation history
    adaptation_history = []
    
    print("Starting target adaptation...")
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.eval()
        
        # Adaptation training loop
        epoch_loss = 0
        num_batches = 0
        for inputs_target, _ in train_loader:
            inputs_target = inputs_target.cuda()
            outputs_target = netC(netB(netF(inputs_target)))
            
            sigmoid_out = torch.sigmoid(outputs_target)
            entropy_loss = -torch.mean(sigmoid_out * torch.log(sigmoid_out + 1e-8) + (1 - sigmoid_out) * torch.log(1 - sigmoid_out + 1e-8))
            im_loss = entropy_loss * preset["training"]["ent_par"]
            
            optimizer.zero_grad()
            im_loss.backward()
            optimizer.step()
            epoch_loss += im_loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        
        # Evaluation
        netF.eval(); netB.eval()
        acc, f1_micro, f1_macro, hamming, report = cal_acc(test_loader, netF, netB, netC, num_users, num_classes)
        
        # Store epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': acc,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'hamming_loss': hamming
        }
        adaptation_history.append(epoch_results)
        
        if acc > best_acc:
            best_acc = acc
            best_f1_micro = f1_micro
            best_f1_macro = f1_macro
            best_hamming = hamming
            best_epoch = epoch + 1
            best_metrics = {
                "accuracy": float(acc), 
                "f1_micro": float(f1_micro), 
                "f1_macro": float(f1_macro),
                "hamming_loss": float(hamming),
                "epoch": best_epoch
            }
            
            # Save best adapted model's detailed report
            savename = f'ent_{preset["training"]["ent_par"]}'
            report_filename = osp.join(output_dir, f"target_adapted_best_classification_report_{savename}.txt")
            with open(report_filename, 'w') as f:
                f.write(f"Best Target Adapted Model Performance (Epoch {epoch+1})\n")
                f.write(f"Accuracy: {acc:.2f}%\n")
                f.write(f"F1-Micro: {f1_micro:.2f}%\n")
                f.write(f"F1-Macro: {f1_macro:.2f}%\n")
                f.write(f"Hamming Loss: {hamming:.2f}%\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(report)

        log_str = f'Target Adaptation - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Accuracy: {acc:.2f}%, F1-micro: {f1_micro:.2f}%'
        out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
    
    # Save adaptation history
    history_file = osp.join(output_dir, "target_adaptation_history.json")
    with open(history_file, 'w') as f:
        json.dump(adaptation_history, f, indent=2)
    
    return {
        'best_accuracy': best_acc,
        'best_f1_micro': best_f1_micro,
        'best_f1_macro': best_f1_macro,
        'best_hamming_loss': best_hamming,
        'best_epoch': best_epoch,
        'adaptation_history': adaptation_history
    }

def get_domain_name(config_key):
    config = preset[config_key]
    env = "_".join(config["environment"])
    wifi = "_".join(config["wifi_band"]) + "GHz"
    users = f"users_{min(config['num_users'])}-{max(config['num_users'])}"
    return f"{env}_{wifi}_{users}"

def print_config():
    s = "==========================================\nEXPERIMENT CONFIGURATION\n==========================================\n"
    for key, content in preset.items():
        if key != "encoding": s += f"{key}: {content}\n"
    s += "==========================================\n"
    return s

def run_single_experiment(run_idx, seed, base_output_dir):
    """Executes a single run of the experiment with a given seed."""
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
    if not osp.exists(osp.join(output_dir, 'source_F.pt')):
        print("\n" + "="*50 + "\nPHASE 1: SOURCE DOMAIN TRAINING\n" + "="*50)
        with open(osp.join(output_dir, 'log_src.txt'), 'w') as out_file:
            out_file.write(print_config() + '\n')
            source_results = train_source(output_dir, out_file, random_state=seed)
    else:
        print("Source model already exists, loading existing results...")
        # Try to load existing source results if available
        try:
            existing_results_file = osp.join(output_dir, "run_complete_results.json")
            if osp.exists(existing_results_file):
                with open(existing_results_file, 'r') as f:
                    existing_results = json.load(f)
                    source_results = existing_results.get('source_domain', {})
        except:
            pass
    
    # --- Target Baseline Testing ---
    print("\n" + "="*50 + "\nPHASE 2: TARGET DOMAIN BASELINE (Source Model)\n" + "="*50)
    with open(osp.join(output_dir, 'log_baseline.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        target_baseline_results = test_target_baseline(output_dir, out_file, random_state=seed)

    # --- Target Adaptation ---
    print("\n" + "="*50 + "\nPHASE 3: TARGET DOMAIN ADAPTATION\n" + "="*50)
    savename = f'ent_{preset["training"]["ent_par"]}'
    with open(osp.join(output_dir, f'log_tar_{savename}.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        target_adapted_results = train_target(output_dir, out_file, random_state=seed)

    # Compile all results
    run_results.update({
        'source_domain': source_results,
        'target_domain_baseline': target_baseline_results,
        'target_domain_adapted': target_adapted_results
    })
    
    # Save comprehensive results for this run
    save_run_results(output_dir, run_results)
    
    # Return metrics needed for aggregation
    return {
        'accuracy': target_adapted_results['best_accuracy'],
        'f1_micro': target_adapted_results['best_f1_micro'],
        'f1_macro': target_adapted_results['best_f1_macro'],
        'hamming_loss': target_adapted_results['best_hamming_loss']
    }

def main():
    """Main function to run the experiment multiple times and aggregate results."""
    if preset["training"]["gpu_id"] != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = preset["training"]["gpu_id"]
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No GPUs available. This code requires CUDA.")

    initial_seed = preset["training"]["seed"]
    
    # Define the base output directory, without the run number
    experiment_name = f"{preset['source_task']}_to_{preset['target_task']}"
    source_domain = get_domain_name("source_data")
    target_domain = get_domain_name("target_data")
    base_output_dir = osp.join(
        preset["path"]["save_dir"], 
        experiment_name,
        f"{source_domain}_to_{target_domain}",
        f"seed_{initial_seed}"
    )
    debug_data_availability()
    os.makedirs(base_output_dir, exist_ok=True)

    # Lists to store results from all runs
    all_final_metrics = []
    all_run_summaries = []

    for i in range(preset["repeat"]):
        current_seed = initial_seed + i
        final_metrics = run_single_experiment(i, current_seed, base_output_dir)
        if final_metrics:
            all_final_metrics.append(final_metrics)
            
            # Load the complete run results for summary
            run_dir = osp.join(base_output_dir, f"run_{i}")
            run_results_file = osp.join(run_dir, "run_complete_results.json")
            if osp.exists(run_results_file):
                with open(run_results_file, 'r') as f:
                    run_data = json.load(f)
                    all_run_summaries.append(run_data)

    # --- Aggregate and Save Final Results ---
    if not all_final_metrics:
        print("\nNo results to aggregate. Exiting.")
        return

    # Calculate statistics for target adaptation results
    accs = [m['accuracy'] for m in all_final_metrics]
    f1_micros = [m['f1_micro'] for m in all_final_metrics]
    f1_macros = [m['f1_macro'] for m in all_final_metrics]
    hamming_losses = [m['hamming_loss'] for m in all_final_metrics]

    # Aggregated statistics
    aggregated_stats = {
        'num_runs': len(all_final_metrics),
        'initial_seed': initial_seed,
        'target_adaptation': {
            'accuracy': {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'min': float(np.min(accs)),
                'max': float(np.max(accs)),
                'values': accs
            },
            'f1_micro': {
                'mean': float(np.mean(f1_micros)),
                'std': float(np.std(f1_micros)),
                'min': float(np.min(f1_micros)),
                'max': float(np.max(f1_micros)),
                'values': f1_micros
            },
            'f1_macro': {
                'mean': float(np.mean(f1_macros)),
                'std': float(np.std(f1_macros)),
                'min': float(np.min(f1_macros)),
                'max': float(np.max(f1_macros)),
                'values': f1_macros
            },
            'hamming_loss': {
                'mean': float(np.mean(hamming_losses)),
                'std': float(np.std(hamming_losses)),
                'min': float(np.min(hamming_losses)),
                'max': float(np.max(hamming_losses)),
                'values': hamming_losses
            }
        }
    }
    
    # If we have source and baseline results, aggregate those too
    if all_run_summaries:
        # Source domain results
        source_accs = []
        source_f1_micros = []
        source_f1_macros = []
        
        # Target baseline results
        baseline_accs = []
        baseline_f1_micros = []
        baseline_f1_macros = []
        
        for run_summary in all_run_summaries:
            if 'source_domain' in run_summary and run_summary['source_domain']:
                source_accs.append(run_summary['source_domain']['test_accuracy'])
                source_f1_micros.append(run_summary['source_domain']['test_f1_micro'])
                source_f1_macros.append(run_summary['source_domain']['test_f1_macro'])
            
            if 'target_domain_baseline' in run_summary:
                baseline_accs.append(run_summary['target_domain_baseline']['accuracy'])
                baseline_f1_micros.append(run_summary['target_domain_baseline']['f1_micro'])
                baseline_f1_macros.append(run_summary['target_domain_baseline']['f1_macro'])
        
        if source_accs:
            aggregated_stats['source_domain'] = {
                'accuracy': {
                    'mean': float(np.mean(source_accs)),
                    'std': float(np.std(source_accs)),
                    'values': source_accs
                },
                'f1_micro': {
                    'mean': float(np.mean(source_f1_micros)),
                    'std': float(np.std(source_f1_micros)),
                    'values': source_f1_micros
                },
                'f1_macro': {
                    'mean': float(np.mean(source_f1_macros)),
                    'std': float(np.std(source_f1_macros)),
                    'values': source_f1_macros
                }
            }
        
        if baseline_accs:
            aggregated_stats['target_baseline'] = {
                'accuracy': {
                    'mean': float(np.mean(baseline_accs)),
                    'std': float(np.std(baseline_accs)),
                    'values': baseline_accs
                },
                'f1_micro': {
                    'mean': float(np.mean(baseline_f1_micros)),
                    'std': float(np.std(baseline_f1_micros)),
                    'values': baseline_f1_micros
                },
                'f1_macro': {
                    'mean': float(np.mean(baseline_f1_macros)),
                    'std': float(np.std(baseline_f1_macros)),
                    'values': baseline_f1_macros
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
    
    if 'source_domain' in aggregated_stats:
        summary_str += "SOURCE DOMAIN PERFORMANCE:\n"
        summary_str += "-" * 40 + "\n"
        src_acc = aggregated_stats['source_domain']['accuracy']
        src_f1_micro = aggregated_stats['source_domain']['f1_micro']
        src_f1_macro = aggregated_stats['source_domain']['f1_macro']
        summary_str += f"Accuracy:      {src_acc['mean']:.2f}% ± {src_acc['std']:.2f}\n"
        summary_str += f"F1-Micro:      {src_f1_micro['mean']:.2f}% ± {src_f1_micro['std']:.2f}\n"
        summary_str += f"F1-Macro:      {src_f1_macro['mean']:.2f}% ± {src_f1_macro['std']:.2f}\n\n"
    
    if 'target_baseline' in aggregated_stats:
        summary_str += "TARGET DOMAIN BASELINE (Source Model):\n"
        summary_str += "-" * 40 + "\n"
        base_acc = aggregated_stats['target_baseline']['accuracy']
        base_f1_micro = aggregated_stats['target_baseline']['f1_micro']
        base_f1_macro = aggregated_stats['target_baseline']['f1_macro']
        summary_str += f"Accuracy:      {base_acc['mean']:.2f}% ± {base_acc['std']:.2f}\n"
        summary_str += f"F1-Micro:      {base_f1_micro['mean']:.2f}% ± {base_f1_micro['std']:.2f}\n"
        summary_str += f"F1-Macro:      {base_f1_macro['mean']:.2f}% ± {base_f1_macro['std']:.2f}\n\n"
    
    summary_str += "TARGET DOMAIN ADAPTATION (Final Results):\n"
    summary_str += "-" * 40 + "\n"
    adapt_acc = aggregated_stats['target_adaptation']['accuracy']
    adapt_f1_micro = aggregated_stats['target_adaptation']['f1_micro']
    adapt_f1_macro = aggregated_stats['target_adaptation']['f1_macro']
    adapt_hamming = aggregated_stats['target_adaptation']['hamming_loss']
    summary_str += f"Accuracy:      {adapt_acc['mean']:.2f}% ± {adapt_acc['std']:.2f} (Range: {adapt_acc['min']:.2f}-{adapt_acc['max']:.2f})\n"
    summary_str += f"F1-Micro:      {adapt_f1_micro['mean']:.2f}% ± {adapt_f1_micro['std']:.2f} (Range: {adapt_f1_micro['min']:.2f}-{adapt_f1_micro['max']:.2f})\n"
    summary_str += f"F1-Macro:      {adapt_f1_macro['mean']:.2f}% ± {adapt_f1_macro['std']:.2f} (Range: {adapt_f1_macro['min']:.2f}-{adapt_f1_macro['max']:.2f})\n"
    summary_str += f"Hamming Loss:  {adapt_hamming['mean']:.2f}% ± {adapt_hamming['std']:.2f} (Range: {adapt_hamming['min']:.2f}-{adapt_hamming['max']:.2f})\n\n"
    
    if 'target_baseline' in aggregated_stats:
        summary_str += "IMPROVEMENT FROM BASELINE:\n"
        summary_str += "-" * 40 + "\n"
        acc_improvement = adapt_acc['mean'] - base_acc['mean']
        f1_micro_improvement = adapt_f1_micro['mean'] - base_f1_micro['mean']
        f1_macro_improvement = adapt_f1_macro['mean'] - base_f1_macro['mean']
        summary_str += f"Accuracy:      {acc_improvement:+.2f}%\n"
        summary_str += f"F1-Micro:      {f1_micro_improvement:+.2f}%\n"
        summary_str += f"F1-Macro:      {f1_macro_improvement:+.2f}%\n\n"
    
    summary_str += f"{'='*80}\n"
    
    print(summary_str)

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
    
    print(f"Results saved to:")
    print(f"  - Aggregated results: {aggregated_results_file}")
    print(f"  - Summary: {summary_file_path}")
    print(f"  - Complete results: {all_runs_file}")
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()