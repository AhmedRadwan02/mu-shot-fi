"""
SHOT (Source Hypothesis Transfer) Training Script
Organized structure for domain adaptation experiments
"""
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F

# Custom modules
import network
import rotation
import temporal_shift
from preset import preset
from load_data import load_widar_data

# For pseudo-labeling (clustering)
from scipy.spatial.distance import cdist
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_device_and_parallel():
    """Setup device and determine if we should use DataParallel"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_parallel = torch.cuda.device_count() > 1
    
    if use_parallel:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        print(f"Using device: {device}")
    
    return device, use_parallel


def get_model_device_safe(model):
    """Get the underlying model whether it's wrapped in DataParallel or not"""
    return model.module if hasattr(model, 'module') else model


def save_model_weights(model, filepath, use_parallel=False):
    """Save model weights properly handling DataParallel"""
    if use_parallel:
        torch.save(model.module.state_dict(), filepath)
    else:
        torch.save(model.state_dict(), filepath)


def load_model_weights(model, filepath, device, strict=True):
    """Load model weights properly handling device mapping"""
    state_dict = torch.load(filepath, map_location=device)
    
    if strict:
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded weights from {filepath} (strict mode)")
    else:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {filepath} (lenient mode)")
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")
    
    return model

def lr_scheduler(optimizer, epoch, max_epochs, gamma=10, power=1.0):
    """Learning rate scheduler for epoch-based training (source domain)"""
    decay = (1 + gamma * epoch / max_epochs) ** (-power)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
    
    return decay

def lr_scheduler_iter(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """Learning rate scheduler for iteration-based training (target adaptation)"""
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    
    return optimizer

def get_domain_name(config_key):
    """Generate domain name for WiDAR configuration"""
    config = preset[config_key]
    rooms = "_".join(config["rooms"])
    users = f"users_{len(config['users'])}"
    
    if "csi_components" in config:
        components = config["csi_components"]
        if len(components) == 1 and components[0] == "amplitude":
            phase_info = "amp_only"
        elif len(components) == 1 and components[0] == "phase":
            phase_info = "phase_only"
        elif len(components) == 2:
            phase_info = "amp_phase"
        else:
            phase_info = f"{'_'.join(components)}"
    else:
        phase_info = "default"
    
    return f"{rooms}_{users}_{phase_info}"


def print_config():
    """Print experiment configuration"""
    s = "="*60 + "\nEXPERIMENT CONFIGURATION\n" + "="*60 + "\n"
    for key, content in preset.items():
        if key != "encoding": 
            s += f"{key}: {content}\n"
    s += "="*60 + "\n"
    return s


def create_experiment_identifier(preset):
    """Create a comprehensive experiment identifier from config parameters"""
    dataset_type = preset.get("dataset_type", "unknown")
    source_task = preset.get("source_task", "src")
    target_task = preset.get("target_task", "tgt")
    
    epochs = preset.get("training", {}).get("max_epoch", "default")
    lr = preset.get("training", {}).get("lr", "default")
    batch_size = preset.get("training", {}).get("batch_size", "default")
    
    model_name = preset.get("model", {}).get("name", "shot")
    
    experiment_id = f"{dataset_type}_{source_task}to{target_task}_{model_name}"
    experiment_id += f"_ep{epochs}_lr{lr}_bs{batch_size}"
    
    return experiment_id


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def cal_acc(loader, netF, netB, netC, encoder, device):
    """Calculate accuracy on given loader and generate a classification report"""
    netF.eval()
    netB.eval()
    netC.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = netC(netB(netF(inputs)))
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    class_names = encoder.classes_
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    return accuracy, f1_micro, f1_macro, 0.0, report



# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_source(output_dir, out_file, random_state, device, use_parallel):
    """Train source domain model"""
    print("Loading source data...")
    
    data_info = load_widar_data("source_data", random_state)
    
    train_loader = data_info['train_loader']
    val_loader = data_info['val_loader']
    test_loader = data_info['test_loader']
    num_classes = data_info['num_classes']
    var_x_shape = data_info['sample_shape']
    encoder = data_info['label_encoder']
    input_channels = data_info['input_channels']
    
    print(f"Network setup:")
    print(f"  Sample shape: {var_x_shape}") 
    print(f"  Input channels: {input_channels}")
    print(f"  Num classes: {num_classes}")
    
    # Model setup
    netF = network.CNN2DBase(var_x_shape, input_channels=input_channels).to(device)
    netB = network.feat_bottleneck(feature_dim=netF.in_features, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    netC = network.feat_classifier(class_num=num_classes, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    
    if use_parallel:
        netF = nn.DataParallel(netF)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
    
    # Setup optimizer
    param_group = []
    for v in get_model_device_safe(netF).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netB).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netC).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    
    optimizer = optim.SGD(param_group, momentum=0.9)
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    
    max_epochs = preset["training"]["max_epoch"]
    best_val_acc = 0
    best_train_acc = 0
    best_f1_micro = 0
    best_f1_macro = 0
    
    # Use label smoothing as in original SHOT paper
    criterion = CrossEntropyLabelSmooth(num_classes=num_classes, 
                                   epsilon=preset["training"].get("smooth", 0.1),
                                   use_gpu=torch.cuda.is_available())
    training_history = []
    
    print("Starting source training...")
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.train()
        
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}')
        for inputs_source, labels_source in pbar:
            inputs_source = inputs_source.to(device).float()
            labels_source = labels_source.to(device).long()
            inputs_source = torch.nan_to_num(inputs_source, nan=0.0, posinf=0.0, neginf=0.0)
            
            features = netF(inputs_source)
            bottleneck_features = netB(features)
            outputs_source = netC(bottleneck_features)
            
            classifier_loss = criterion(outputs_source, labels_source)
            
            optimizer.zero_grad()
            classifier_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(get_model_device_safe(netF).parameters()) +
                list(get_model_device_safe(netB).parameters()) +
                list(get_model_device_safe(netC).parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            epoch_loss += classifier_loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{classifier_loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        
        netF.eval(); netB.eval(); netC.eval()
        
        acc_s_tr, f1_micro_tr, f1_macro_tr, hamming_tr, report_tr = cal_acc(train_loader, netF, netB, netC, encoder, device)
        acc_s_val, f1_micro_val, f1_macro_val, hamming_val, report_val = cal_acc(val_loader, netF, netB, netC, encoder, device)
        
        epoch_results = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'train_accuracy': acc_s_tr,
            'train_f1_micro': f1_micro_tr,
            'train_f1_macro': f1_macro_tr,
            'val_accuracy': acc_s_val,
            'val_f1_micro': f1_micro_val,
            'val_f1_macro': f1_macro_val
        }
        training_history.append(epoch_results)
        
        log_str = f'Source Training - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Train Acc: {acc_s_tr:.2f}%; Val Acc: {acc_s_val:.2f}%'
        out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
        
        if acc_s_val >= best_val_acc:
            best_val_acc = acc_s_val
            best_train_acc = acc_s_tr
            best_f1_micro = f1_micro_val
            best_f1_macro = f1_macro_val
            
            save_model_weights(netF, osp.join(output_dir, "source_F.pt"), use_parallel)
            save_model_weights(netB, osp.join(output_dir, "source_B.pt"), use_parallel)
            save_model_weights(netC, osp.join(output_dir, "source_C.pt"), use_parallel)
            
            best_report_file = osp.join(output_dir, "source_best_classification_report.txt")
            with open(best_report_file, 'w') as f:
                f.write(f"Best Source Model Performance (Epoch {epoch+1})\n")
                f.write(f"Validation Accuracy: {acc_s_val:.2f}%\n")
                f.write(f"Validation F1-Micro: {f1_micro_val:.2f}%\n")
                f.write(f"Validation F1-Macro: {f1_macro_val:.2f}%\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(report_val)
    
    print("\nEvaluating best model on source test set...")
    netF.eval(); netB.eval(); netC.eval()
    acc_s_test, f1_micro_test, f1_macro_test, hamming_test, report_test = cal_acc(test_loader, netF, netB, netC, encoder, device)
    
    log_str = f'Source Test Set - Accuracy: {acc_s_test:.2f}%; F1-Micro: {f1_micro_test:.2f}%; F1-Macro: {f1_macro_test:.2f}%'
    out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
    
    test_report_file = osp.join(output_dir, "source_test_classification_report.txt")
    with open(test_report_file, 'w') as f:
        f.write("Source Model Test Set Performance\n")
        f.write(f"Test Accuracy: {acc_s_test:.2f}%\n")
        f.write(f"Test F1-Micro: {f1_micro_test:.2f}%\n")
        f.write(f"Test F1-Macro: {f1_macro_test:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report_test)
    
    history_file = osp.join(output_dir, "source_training_history.json")
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Source training completed. Best val accuracy: {best_val_acc:.2f}%, Test accuracy: {acc_s_test:.2f}%")
    
    source_results_dict = {
        'best_train_accuracy': best_train_acc,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': acc_s_test,
        'test_f1_micro': f1_micro_test,
        'test_f1_macro': f1_macro_test,
        'test_hamming_loss': 0.0,
        'training_history': training_history
    }
    
    source_results_file = osp.join(output_dir, 'source_training_results.json')
    with open(source_results_file, 'w') as f:
        json.dump(source_results_dict, f, indent=2)
    print(f"Source results saved to: {source_results_file}")
    
    return source_results_dict


def test_target_baseline(output_dir, out_file, random_state, device, use_parallel):
    """Test on target domain using the trained source model"""
    print("Loading target data for baseline testing...")
    
    data_info = load_widar_data("target_data", random_state)
    
    test_loader = data_info['test_loader']
    num_classes = data_info['num_classes']
    var_x_shape = data_info['sample_shape']
    encoder = data_info['label_encoder']
    input_channels = data_info['input_channels']
    
    netF = network.CNN2DBase(var_x_shape, input_channels=input_channels).to(device)
    netB = network.feat_bottleneck(feature_dim=netF.in_features, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    netC = network.feat_classifier(class_num=num_classes, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    
    load_model_weights(netF, osp.join(output_dir, "source_F.pt"), device)
    load_model_weights(netB, osp.join(output_dir, "source_B.pt"), device)
    load_model_weights(netC, osp.join(output_dir, "source_C.pt"), device)
    
    if use_parallel:
        netF = nn.DataParallel(netF)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
    
    netF.eval(); netB.eval(); netC.eval()
    
    acc, f1_micro, f1_macro, hamming, report = cal_acc(test_loader, netF, netB, netC, encoder, device)
    
    baseline_report_file = osp.join(output_dir, "target_baseline_classification_report.txt")
    with open(baseline_report_file, 'w') as f:
        f.write("Target Domain Baseline Performance (Source Model)\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"F1-Micro: {f1_micro:.2f}%\n")
        f.write(f"F1-Macro: {f1_macro:.2f}%\n\n")
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

def unpack_batch(batch, batch_counter=0):
    """
    Safely unpack batch whether it returns 2 or 3 values.
    Returns: (inputs, labels, indices)
    """
    if len(batch) == 3:
        inputs, labels, indices = batch
        return inputs, labels, indices
    elif len(batch) == 2:
        inputs, labels = batch
        # Generate pseudo-indices
        indices = torch.arange(batch_counter, batch_counter + len(inputs))
        return inputs, labels, indices
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
        
def train_target_rot(output_dir, out_file, dset_loaders, netF, netB, device, use_parallel):
    """Pre-train rotation classifier on target domain"""
    
    # Initialize rotation classifier
    bottleneck_dim = preset["training"]["bottleneck"]
    netR = network.feat_classifier(type='linear', class_num=2, 
                                   bottleneck_dim=2*bottleneck_dim).to(device)
    
    # Load pre-trained source model weights
    load_model_weights(netF, osp.join(output_dir, 'source_F.pt'), device)
    load_model_weights(netB, osp.join(output_dir, 'source_B.pt'), device)
    
    # Freeze feature extractor and bottleneck
    netF.eval()
    for k, v in get_model_device_safe(netF).named_parameters():
        v.requires_grad = False
    netB.eval()
    for k, v in get_model_device_safe(netB).named_parameters():
        v.requires_grad = False
    
    if use_parallel:
        netR = nn.DataParallel(netR)
    
    # Setup optimizer - only train netR
    param_group = []
    for k, v in get_model_device_safe(netR).named_parameters():
        param_group += [{'params': v, 'lr': preset["training"]["adaptation_lr"]}]
    
    netR.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    max_iter = preset["training"]["ssl_max_epoch"] * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0
    rot_acc = 0
    best_netR = None
    
    print("Pre-training rotation classifier...")
    iter_test = None
    batch_counter = 0
    
    # Initialize progress bar
    pbar = tqdm(total=max_iter, desc='SSL Rotation Pre-training', 
                unit='iter', dynamic_ncols=True)
    
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            batch = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            batch = next(iter_test)
            batch_counter = 0
        
        # Safely unpack batch
        inputs_test, _, tar_idx = unpack_batch(batch, batch_counter)
        batch_counter += len(inputs_test)
        
        if inputs_test.size(0) == 1:
            continue
        
        inputs_test = inputs_test.to(device)
        inputs_test = torch.nan_to_num(inputs_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        iter_num += 1
        lr_scheduler_iter(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        # Generate random rotations (torch, on the same device)
        r_labels_target = torch.randint(0, 2, (inputs_test.shape[0],), dtype=torch.long, device=inputs_test.device)
        r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)

        # Forward pass
        f_outputs = netB(netF(inputs_test))
        f_r_outputs = netB(netF(r_inputs_target))
        r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))
        
        rotation_loss = nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
        rotation_loss.backward()
        optimizer.step()
        
        # Update progress bar
        current_epoch = (iter_num * preset["training"]["ssl_max_epoch"]) // max_iter
        pbar.update(1)
        pbar.set_postfix({
            'Epoch': f'{current_epoch}/{preset["training"]["ssl_max_epoch"]}',
            'Loss': f'{rotation_loss.item():.4f}',
            'Best Acc': f'{rot_acc:.2f}%'
        })
        
        # Evaluation
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netR.eval()
            acc_rot = cal_acc_rot(dset_loaders['target'], netF, netB, netR, device)
            log_str = 'SSL Rotation Pre-training - Iter:{}/{}; Rotation Accuracy = {:.2f}%'.format(
                iter_num, max_iter, acc_rot
            )
            out_file.write(log_str + '\n')
            out_file.flush()
            tqdm.write(log_str)
            netR.train()
            
            if rot_acc < acc_rot:
                rot_acc = acc_rot
                best_netR = get_model_device_safe(netR).state_dict()
                
                # Update progress bar with new best
                pbar.set_postfix({
                    'Epoch': f'{current_epoch}/{preset["training"]["ssl_max_epoch"]}',
                    'Loss': f'{rotation_loss.item():.4f}',
                    'Best Acc': f'{rot_acc:.2f}%'
                })
    
    pbar.close()
    
    log_str = 'Best Rotation Accuracy = {:.2f}%'.format(rot_acc)
    out_file.write(log_str + '\n')
    out_file.flush()
    print(log_str)
    
    return best_netR, rot_acc



def cal_acc_rot(loader, netF, netB, netR, device):
    """Calculate rotation prediction accuracy"""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            inputs = inputs.to(device)
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)

            # Generate random rotations
            r_labels = np.random.randint(0, 2, len(inputs))
            r_inputs = rotation.rotate_batch_with_labels(inputs, r_labels)
            r_labels = torch.from_numpy(r_labels)
            r_inputs = r_inputs.to(device)

            # Forward pass
            f_outputs = netB(netF(inputs))
            f_r_outputs = netB(netF(r_inputs))
            r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))

            if start_test:
                all_output = r_outputs.float().cpu()
                all_label = r_labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, r_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, r_labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return accuracy * 100


def train_target_tsd(output_dir, out_file, dset_loaders, netF, netB, device, use_parallel):
    """Pre-train temporal-shift discriminator on target domain"""

    # Get shift configuration from preset
    shift_values = preset["training"]["tsd_shifts"]
    num_shifts = len(shift_values)

    # Initialize TSD discriminator (multi-class classifier)
    bottleneck_dim = preset["training"]["bottleneck"]
    netD = network.feat_classifier(type='linear', class_num=num_shifts,
                                   bottleneck_dim=2*bottleneck_dim).to(device)

    # Load pre-trained source model weights
    load_model_weights(netF, osp.join(output_dir, 'source_F.pt'), device)
    load_model_weights(netB, osp.join(output_dir, 'source_B.pt'), device)

    # Freeze feature extractor and bottleneck
    netF.eval()
    for k, v in get_model_device_safe(netF).named_parameters():
        v.requires_grad = False
    netB.eval()
    for k, v in get_model_device_safe(netB).named_parameters():
        v.requires_grad = False

    if use_parallel:
        netD = nn.DataParallel(netD)

    # Setup optimizer - only train netD
    param_group = []
    for k, v in get_model_device_safe(netD).named_parameters():
        param_group += [{'params': v, 'lr': preset["training"]["adaptation_lr"]}]

    netD.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = preset["training"]["tsd_max_epoch"] * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0
    tsd_acc = 0
    best_netD = None

    print(f"Pre-training TSD discriminator with {num_shifts} shift classes: {shift_values}")
    iter_test = None
    batch_counter = 0

    # Initialize progress bar
    pbar = tqdm(total=max_iter, desc='TSD Pre-training',
                unit='iter', dynamic_ncols=True)

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            batch = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            batch = next(iter_test)
            batch_counter = 0

        # Safely unpack batch
        inputs_test, _, tar_idx = unpack_batch(batch, batch_counter)
        batch_counter += len(inputs_test)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.to(device)
        inputs_test = torch.nan_to_num(inputs_test, nan=0.0, posinf=0.0, neginf=0.0)

        iter_num += 1
        lr_scheduler_iter(optimizer, iter_num=iter_num, max_iter=max_iter)

        # Generate random temporal shifts
        shift_labels = torch.randint(0, num_shifts, (inputs_test.shape[0],),
                                    dtype=torch.long, device=inputs_test.device)
        shifted_inputs = temporal_shift.shift_batch_with_labels(inputs_test, shift_labels, shift_values)

        # Forward pass
        f_outputs = netB(netF(inputs_test))
        f_shifted_outputs = netB(netF(shifted_inputs))
        shift_outputs = netD(torch.cat((f_outputs, f_shifted_outputs), 1))

        tsd_loss = nn.CrossEntropyLoss()(shift_outputs, shift_labels)
        tsd_loss.backward()
        optimizer.step()

        # Update progress bar
        current_epoch = (iter_num * preset["training"]["tsd_max_epoch"]) // max_iter
        pbar.update(1)
        pbar.set_postfix({
            'Epoch': f'{current_epoch}/{preset["training"]["tsd_max_epoch"]}',
            'Loss': f'{tsd_loss.item():.4f}',
            'Best Acc': f'{tsd_acc:.2f}%'
        })

        # Evaluation
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netD.eval()
            acc_tsd = cal_acc_tsd(dset_loaders['target'], netF, netB, netD, device, shift_values)
            log_str = 'TSD Pre-training - Iter:{}/{}; TSD Accuracy = {:.2f}%'.format(
                iter_num, max_iter, acc_tsd
            )
            out_file.write(log_str + '\n')
            out_file.flush()
            tqdm.write(log_str)
            netD.train()

            if tsd_acc < acc_tsd:
                tsd_acc = acc_tsd
                best_netD = get_model_device_safe(netD).state_dict()

                # Update progress bar with new best
                pbar.set_postfix({
                    'Epoch': f'{current_epoch}/{preset["training"]["tsd_max_epoch"]}',
                    'Loss': f'{tsd_loss.item():.4f}',
                    'Best Acc': f'{tsd_acc:.2f}%'
                })

    pbar.close()

    log_str = 'Best TSD Accuracy = {:.2f}%'.format(tsd_acc)
    out_file.write(log_str + '\n')
    out_file.flush()
    print(log_str)

    return best_netD, tsd_acc


def cal_acc_tsd(loader, netF, netB, netD, device, shift_values):
    """Calculate temporal-shift discrimination accuracy"""
    num_shifts = len(shift_values)
    start_test = True

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            inputs = inputs.to(device)
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)

            # Generate random temporal shifts
            shift_labels = np.random.randint(0, num_shifts, len(inputs))
            shifted_inputs = temporal_shift.shift_batch_with_labels(inputs, shift_labels, shift_values)
            shift_labels = torch.from_numpy(shift_labels)
            shifted_inputs = shifted_inputs.to(device)

            # Forward pass
            f_outputs = netB(netF(inputs))
            f_shifted_outputs = netB(netF(shifted_inputs))
            shift_outputs = netD(torch.cat((f_outputs, f_shifted_outputs), 1))

            if start_test:
                all_output = shift_outputs.float().cpu()
                all_label = shift_labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, shift_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, shift_labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return accuracy * 100

def train_target_shot(output_dir, out_file, random_state, device, use_parallel):
    """Domain adaptation training on target using SHOT"""
    print("Loading target data for adaptation...")
    
    data_info = load_widar_data("target_data", random_state)
    
    dset_loaders = {}
    dset_loaders["target"] = data_info['train_loader']  # Shuffled for training
    dset_loaders["target_te"] = data_info['train_loader_no_shuffle']  # Non-shuffled for pseudo-label generation
    dset_loaders["val"] = data_info['val_loader']  # For model selection during training
    dset_loaders["test"] = data_info['test_loader']  # For final evaluation only
    
    num_classes = data_info['num_classes']
    var_x_shape = data_info['sample_shape']
    encoder = data_info['label_encoder']
    input_channels = data_info['input_channels']
    
    # Initialize networks
    netF = network.CNN2DBase(var_x_shape, input_channels=input_channels).to(device)
    netB = network.feat_bottleneck(feature_dim=netF.in_features,
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    netC = network.feat_classifier(class_num=num_classes,
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)

    # Initialize SSL and TSD results
    acc_rot = None
    acc_tsd = None

    # SSL rotation network (if enabled)
    if preset["training"]["ssl"] > 0:
        netR = network.feat_classifier(type='linear', class_num=2,
                                      bottleneck_dim=2*preset["training"]["bottleneck"]).to(device)
        print("Pre-training rotation classifier...")
        netR_dict, acc_rot = train_target_rot(output_dir, out_file, dset_loaders,
                                              netF, netB, device, use_parallel)
        netR.load_state_dict(netR_dict)

    # TSD network (if enabled)
    if preset["training"]["tsd"] > 0:
        shift_values = preset["training"]["tsd_shifts"]
        num_shifts = len(shift_values)
        netD = network.feat_classifier(type='linear', class_num=num_shifts,
                                      bottleneck_dim=2*preset["training"]["bottleneck"]).to(device)
        print("Pre-training TSD discriminator...")
        netD_dict, acc_tsd = train_target_tsd(output_dir, out_file, dset_loaders,
                                              netF, netB, device, use_parallel)
        netD.load_state_dict(netD_dict)
    
    # Load pre-trained source model
    print("Loading pre-trained source model weights...")
    load_model_weights(netF, osp.join(output_dir, "source_F.pt"), device)
    load_model_weights(netB, osp.join(output_dir, "source_B.pt"), device)
    load_model_weights(netC, osp.join(output_dir, "source_C.pt"), device)
    for param in netF.parameters():
        param.requires_grad = True
    for param in netB.parameters():
        param.requires_grad = True

    if use_parallel:
        netF = nn.DataParallel(netF)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
        if preset["training"]["ssl"] > 0:
            netR = nn.DataParallel(netR)
        if preset["training"]["tsd"] > 0:
            netD = nn.DataParallel(netD)
    

    netC.eval()
    for k, v in get_model_device_safe(netC).named_parameters():
        v.requires_grad = False
    
    # Setup optimizer - only for netF, netB, and optionally netR and netD
    param_group = []
    for k, v in get_model_device_safe(netF).named_parameters():
        param_group += [{'params': v, 'lr': preset["training"]["adaptation_lr"]}]
    for k, v in get_model_device_safe(netB).named_parameters():
        param_group += [{'params': v, 'lr': preset["training"]["adaptation_lr"]}]
    if preset["training"]["ssl"] > 0:
        for k, v in get_model_device_safe(netR).named_parameters():
            param_group += [{'params': v, 'lr': preset["training"]["adaptation_lr"]}]
        netR.train()
    if preset["training"]["tsd"] > 0:
        for k, v in get_model_device_safe(netD).named_parameters():
            param_group += [{'params': v, 'lr': preset["training"]["adaptation_lr"]}]
        netD.train()
    
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    max_iter = preset["training"]["adaptation_max_epoch"] * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])  # Evaluate every epoch
    iter_num = 0
    
    # Tracking variables for best model
    best_val_acc = 0
    best_val_f1_micro = 0
    best_val_f1_macro = 0
    best_epoch = 0
    adaptation_history = []
    
    # Initialize pseudo-label memory
    mem_label = None
    
    # Initialize batch counter and iterator
    batch_counter = 0
    iter_test = None
    
    print("Starting target adaptation...")
    
    # Set netF and netB to train mode
    netF.train()
    netB.train()
    
    # Initialize progress bar for entire adaptation
    pbar = tqdm(total=max_iter, desc='Target Adaptation', 
                unit='iter', dynamic_ncols=True)
        
    while iter_num < max_iter:
        # Generate pseudo-labels at START of each epoch
        if iter_num % interval_iter == 0 and preset["training"]["cls_par"] > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['target_te'], netF, netB, netC, 
                                    encoder, device, out_file)
            mem_label = torch.from_numpy(mem_label).to(device)
            netF.train()
            netB.train()
            batch_counter = 0
        
        # Get batch
        try:
            batch = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            batch = next(iter_test)
            batch_counter = 0
        
        # Unpack batch
        inputs_test, _, tar_idx = unpack_batch(batch, batch_counter)
        
        if inputs_test.size(0) == 1:
            continue
        
        batch_counter += len(inputs_test)
        
        iter_num += 1
        lr_scheduler_iter(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        inputs_test = inputs_test.to(device)
        inputs_test = torch.nan_to_num(inputs_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # === FORWARD PASS ===
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        
        # === PSEUDO-LABELING LOSS ===
        if preset["training"]["cls_par"] > 0:
            tar_idx = tar_idx.long().to(device)
            pred = mem_label[tar_idx]
            classifier_loss = preset["training"]["cls_par"] * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0, device=device)
        
        # === INFORMATION MAXIMIZATION LOSS ===
        if preset["training"]["ent"]:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            
            if preset["training"]["gent"]:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss -= gentropy_loss
            
            im_loss = entropy_loss * preset["training"]["ent_par"]
            classifier_loss += im_loss
        
        # === BACKWARD PASS (MAIN TASK) ===
        optimizer.zero_grad()
        # ============ DEBUGGING ============
        if preset["training"]["ent"]:
            print(f"im_loss: {im_loss}")
            classifier_loss.backward()
        
        # === SSL ROTATION LOSS ===
        if preset["training"]["ssl"] > 0:
            # Generate rotations (2 classes for non-square, 4 for square)
            num_rotations = 2  # or 4 if you pad to square
            r_labels_target = np.random.randint(0, num_rotations, len(inputs_test))
            r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target).to(device)
            r_inputs_target = r_inputs_target.to(device)

            # Forward pass with detached original features
            f_outputs = netB(netF(inputs_test))
            f_outputs = f_outputs.detach()
            f_r_outputs = netB(netF(r_inputs_target))
            r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

            rotation_loss = preset["training"]["ssl"] * nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
            rotation_loss.backward()

        # === TSD LOSS ===
        if preset["training"]["tsd"] > 0:
            # Get shift configuration
            shift_values = preset["training"]["tsd_shifts"]
            num_shifts = len(shift_values)

            # Generate random temporal shifts
            shift_labels_target = torch.randint(0, num_shifts, (len(inputs_test),),
                                               dtype=torch.long, device=device)
            shifted_inputs_target = temporal_shift.shift_batch_with_labels(inputs_test, shift_labels_target, shift_values)

            # Forward pass with detached original features
            f_outputs = netB(netF(inputs_test))
            f_outputs = f_outputs.detach()
            f_shifted_outputs = netB(netF(shifted_inputs_target))
            shift_outputs_target = netD(torch.cat((f_outputs, f_shifted_outputs), 1))

            tsd_loss = preset["training"]["tsd"] * nn.CrossEntropyLoss()(shift_outputs_target, shift_labels_target)
            tsd_loss.backward()

        # === OPTIMIZER STEP ===
        optimizer.step()

        
        # Update progress bar with current metrics
        current_epoch = iter_num // interval_iter
        pbar.update(1)
        pbar.set_postfix({
            'Epoch': f'{current_epoch}/{preset["training"]["adaptation_max_epoch"]}',
            'Best Val Acc': f'{best_val_acc:.2f}%'
        })
        
        # Evaluation at end of each epoch
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            epoch = iter_num // interval_iter
            netF.eval()
            netB.eval()
            
            # Evaluate on VALIDATION set for model selection
            val_acc, val_f1_micro, val_f1_macro, val_hamming, val_report = cal_acc(
                dset_loaders['val'], netF, netB, netC, encoder, device
            )
            
            log_str = 'Epoch {}/{} (Iter:{}/{}); Val Acc = {:.2f}%, Val F1-Micro = {:.2f}%, Val F1-Macro = {:.2f}%'.format(
                epoch, preset["training"]["adaptation_max_epoch"], iter_num, max_iter, 
                val_acc, val_f1_micro, val_f1_macro
            )
            out_file.write(log_str + '\n')
            out_file.flush()
            
            # Update progress bar with validation results
            pbar.set_postfix({
                'Epoch': f'{epoch}/{preset["training"]["adaptation_max_epoch"]}',
                'Val Acc': f'{val_acc:.2f}%',
                'Val F1-Micro': f'{val_f1_micro:.2f}%',
                'Best': f'{best_val_acc:.2f}%'
            })
            tqdm.write(log_str)  # Print without disrupting progress bar
            
            # Track history
            adaptation_history.append({
                'epoch': epoch,
                'iteration': iter_num,
                'val_accuracy': val_acc,
                'val_f1_micro': val_f1_micro,
                'val_f1_macro': val_f1_macro
            })
            
            # Save best model based on validation accuracy
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_f1_micro = val_f1_micro
                best_val_f1_macro = val_f1_macro
                best_epoch = epoch
                
                log_str = '  --> New best validation accuracy! Saving model...'
                out_file.write(log_str + '\n')
                out_file.flush()
                tqdm.write(log_str)
                
                # Save best models
                if preset["training"]["save_models"]:
                    savename = f'par_{preset["training"]["cls_par"]}'
                    if preset["training"]["ssl"] > 0:
                        savename += f'_ssl_{preset["training"]["ssl"]}'
                    if preset["training"]["tsd"] > 0:
                        savename += f'_tsd_{preset["training"]["tsd"]}'
                    save_model_weights(netF, osp.join(output_dir, f"target_F_{savename}_best.pt"), use_parallel)
                    save_model_weights(netB, osp.join(output_dir, f"target_B_{savename}_best.pt"), use_parallel)
                    save_model_weights(netC, osp.join(output_dir, f"target_C_{savename}_best.pt"), use_parallel)
                    if preset["training"]["ssl"] > 0:
                        save_model_weights(netR, osp.join(output_dir, f"target_R_{savename}_best.pt"), use_parallel)
                    if preset["training"]["tsd"] > 0:
                        save_model_weights(netD, osp.join(output_dir, f"target_D_{savename}_best.pt"), use_parallel)
                
                # Save best validation report
                best_report_file = osp.join(output_dir, "target_best_validation_report.txt")
                with open(best_report_file, 'w') as f:
                    f.write(f"Best Target Model Validation Performance (Epoch {epoch})\n")
                    f.write(f"Validation Accuracy: {val_acc:.2f}%\n")
                    f.write(f"Validation F1-Micro: {val_f1_micro:.2f}%\n")
                    f.write(f"Validation F1-Macro: {val_f1_macro:.2f}%\n\n")
                    f.write("Validation Set Classification Report:\n")
                    f.write(val_report)
            
            # Return to training mode
            netF.train()
            netB.train()
    
    # Close progress bar
    pbar.close()
    
    # ============================================================================
    # FINAL TEST EVALUATION - Only done once at the very end
    # ============================================================================
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION (using best validation checkpoint)")
    print("="*60)
    
    # Load best model
    if preset["training"]["save_models"]:
        savename = f'par_{preset["training"]["cls_par"]}'
        if preset["training"]["ssl"] > 0:
            savename += f'_ssl_{preset["training"]["ssl"]}'
        if preset["training"]["tsd"] > 0:
            savename += f'_tsd_{preset["training"]["tsd"]}'

        print("Loading best model for final test evaluation...")
        load_model_weights(netF, osp.join(output_dir, f"target_F_{savename}_best.pt"), device)
        load_model_weights(netB, osp.join(output_dir, f"target_B_{savename}_best.pt"), device)
        load_model_weights(netC, osp.join(output_dir, f"target_C_{savename}_best.pt"), device)
    
    netF.eval()
    netB.eval()
    netC.eval()
    
    # Final test evaluation
    test_acc, test_f1_micro, test_f1_macro, test_hamming, test_report = cal_acc(
        dset_loaders['test'], netF, netB, netC, encoder, device
    )
    
    # Save final test report
    test_report_file = osp.join(output_dir, "target_final_test_report.txt")
    with open(test_report_file, 'w') as f:
        f.write(f"Final Test Set Performance (Best Model from Epoch {best_epoch})\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Test F1-Micro: {test_f1_micro:.2f}%\n")
        f.write(f"Test F1-Macro: {test_f1_macro:.2f}%\n\n")
        f.write("Test Set Classification Report:\n")
        f.write(test_report)
    
    # Save adaptation history
    history_file = osp.join(output_dir, "target_adaptation_history.json")
    with open(history_file, 'w') as f:
        json.dump(adaptation_history, f, indent=2)
    
    log_str = '\n' + '='*60 + '\n'
    log_str += 'ADAPTATION COMPLETE\n'
    log_str += '='*60 + '\n'

    # SSL and TSD Pre-training Results
    if acc_rot is not None or acc_tsd is not None:
        log_str += '\nSELF-SUPERVISED PRE-TRAINING RESULTS:\n'
        log_str += '-'*40 + '\n'
        if acc_rot is not None:
            log_str += f'SSL Rotation Accuracy: {acc_rot:.2f}%\n'
        if acc_tsd is not None:
            log_str += f'TSD Shift Accuracy: {acc_tsd:.2f}%\n'
        log_str += '\n'

    log_str += f'Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})\n'
    log_str += f'Best Validation F1-Micro: {best_val_f1_micro:.2f}%\n'
    log_str += f'Best Validation F1-Macro: {best_val_f1_macro:.2f}%\n'
    log_str += f'\nFINAL TEST SET RESULTS:\n'
    log_str += f'Test Accuracy: {test_acc:.2f}%\n'
    log_str += f'Test F1-Micro: {test_f1_micro:.2f}%\n'
    log_str += f'Test F1-Macro: {test_f1_macro:.2f}%\n'
    log_str += '='*60 + '\n'
    out_file.write(log_str)
    out_file.flush()
    print(log_str)
    
    return {
        'best_val_accuracy': best_val_acc,
        'best_val_f1_micro': best_val_f1_micro,
        'best_val_f1_macro': best_val_f1_macro,
        'test_accuracy': test_acc,
        'test_f1_micro': test_f1_micro,
        'test_f1_macro': test_f1_macro,
        'test_hamming_loss': 0.0,
        'classification_report': test_report,
        'best_epoch': best_epoch,
        'adaptation_history': adaptation_history,
        'ssl_rotation_accuracy': acc_rot,
        'tsd_shift_accuracy': acc_tsd
    }
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
            targets: ground truth labels, shape (batch_size,)
        Returns:
            loss value
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: 
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss

def obtain_label(loader, netF, netB, netC, encoder, device, out_file):
    """Generate pseudo-labels using k-nearest centroid clustering"""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
            
            feas = netB(netF(inputs))
            outputs = netC(feas)
            
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    # Normalize features
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()
    
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    
    # Refinement round
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    
    log_str = 'Pseudo-label Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    out_file.write(log_str + '\n')
    out_file.flush()
    print(log_str)
    
    return pred_label.astype('int')
def op_copy(optimizer):
    """Copy optimizer learning rate to lr0"""
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def Entropy(input_):
    """Calculate entropy"""
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def save_run_results(output_dir, run_results):
    """Save comprehensive results for a single run"""
    results_file = osp.join(output_dir, "run_complete_results.json")
    
    run_results['timestamp'] = datetime.now().isoformat()
    run_results['run_directory'] = output_dir
    
    with open(results_file, 'w') as f:
        json.dump(run_results, f, indent=2)
    
    # Human-readable summary
    summary_file = osp.join(output_dir, "run_results_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPLETE RUN RESULTS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Run Directory: {output_dir}\n")
        f.write(f"Timestamp: {run_results['timestamp']}\n")
        f.write(f"Random Seed: {run_results['seed']}\n\n")
        
        # SOURCE DOMAIN RESULTS
        f.write("SOURCE DOMAIN RESULTS:\n")
        f.write("-" * 30 + "\n")
        source_results = run_results.get('source_domain')
        
        if source_results is not None:
            f.write(f"Best Training Accuracy: {source_results.get('best_train_accuracy', 0.0):.2f}%\n")
            f.write(f"Test Accuracy: {source_results.get('test_accuracy', 0.0):.2f}%\n")
            f.write(f"Test F1-Micro: {source_results.get('test_f1_micro', 0.0):.2f}%\n")
            f.write(f"Test F1-Macro: {source_results.get('test_f1_macro', 0.0):.2f}%\n")
            if 'note' in source_results:
                f.write(f"Note: {source_results['note']}\n")
        else:
            f.write("Source results not available (model loaded from checkpoint)\n")
        f.write("\n")
        
        # TARGET DOMAIN BASELINE
        f.write("TARGET DOMAIN RESULTS (Source Model):\n")
        f.write("-" * 30 + "\n")
        target_baseline = run_results.get('target_domain_baseline')
        
        if target_baseline is not None:
            f.write(f"Accuracy: {target_baseline.get('accuracy', 0.0):.2f}%\n")
            f.write(f"F1-Micro: {target_baseline.get('f1_micro', 0.0):.2f}%\n")
            f.write(f"F1-Macro: {target_baseline.get('f1_macro', 0.0):.2f}%\n")
        else:
            f.write("Baseline results not available\n")
        f.write("\n")
        
        # TARGET DOMAIN ADAPTED
        f.write("TARGET DOMAIN RESULTS (After SHOT Adaptation):\n")
        f.write("-" * 30 + "\n")
        target_adapted = run_results.get('target_domain_adapted')

        if target_adapted is not None:
            # SSL/TSD Pre-training Results
            ssl_acc = target_adapted.get('ssl_rotation_accuracy')
            tsd_acc = target_adapted.get('tsd_shift_accuracy')
            if ssl_acc is not None or tsd_acc is not None:
                f.write("Self-Supervised Pre-training:\n")
                if ssl_acc is not None:
                    f.write(f"  SSL Rotation Accuracy: {ssl_acc:.2f}%\n")
                if tsd_acc is not None:
                    f.write(f"  TSD Shift Accuracy: {tsd_acc:.2f}%\n")
                f.write("\n")

            f.write(f"Best Val Accuracy: {target_adapted.get('best_val_accuracy', 0.0):.2f}%\n")
            f.write(f"Best Val F1-Micro: {target_adapted.get('best_val_f1_micro', 0.0):.2f}%\n")
            f.write(f"Best Val F1-Macro: {target_adapted.get('best_val_f1_macro', 0.0):.2f}%\n")
            f.write(f"Test Accuracy: {target_adapted.get('test_accuracy', 0.0):.2f}%\n")
            f.write(f"Test F1-Micro: {target_adapted.get('test_f1_micro', 0.0):.2f}%\n")
            f.write(f"Test F1-Macro: {target_adapted.get('test_f1_macro', 0.0):.2f}%\n")
            f.write(f"Best Epoch: {target_adapted.get('best_epoch', 0)}\n")
        else:
            f.write("Adapted results not available\n")
        f.write("\n")
        
        # PERFORMANCE IMPROVEMENTS (only if both baseline and adapted are available)
        if target_baseline is not None and target_adapted is not None:
            f.write("PERFORMANCE IMPROVEMENTS:\n")
            f.write("-" * 30 + "\n")
            acc_improvement = target_adapted.get('test_accuracy', 0.0) - target_baseline.get('accuracy', 0.0)
            f1_micro_improvement = target_adapted.get('test_f1_micro', 0.0) - target_baseline.get('f1_micro', 0.0)
            f1_macro_improvement = target_adapted.get('test_f1_macro', 0.0) - target_baseline.get('f1_macro', 0.0)
            f.write(f"Test Accuracy Improvement: {acc_improvement:+.2f}%\n")
            f.write(f"Test F1-Micro Improvement: {f1_micro_improvement:+.2f}%\n")
            f.write(f"Test F1-Macro Improvement: {f1_macro_improvement:+.2f}%\n")
        
        f.write("="*60 + "\n")
    
    print(f"Run results saved to: {results_file}")
    print(f"Run summary saved to: {summary_file}")


def run_single_experiment(run_idx, seed, base_output_dir):
    """Execute a single run of the experiment with a given seed"""
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device, use_parallel = setup_device_and_parallel()
    
    run_output_dir = osp.join(base_output_dir, f"run_{run_idx}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"RUN {run_idx+1}/{preset['repeat']} | Seed: {seed}")
    print(f"Output: {run_output_dir}")
    print(f"{'='*60}")
    
    run_results = {
        'run_index': run_idx,
        'seed': seed,
        'device': str(device),
        'use_parallel': use_parallel,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'experiment_config': {k: v for k, v in preset.items() if k != 'encoding'}
    }
    
    # PHASE 1: Source Training
    source_results = None
    if not osp.exists(osp.join(run_output_dir, 'source_F.pt')):
        print("\n" + "="*60)
        print("PHASE 1: SOURCE DOMAIN TRAINING")
        print("="*60)
        with open(osp.join(run_output_dir, 'log_src.txt'), 'w') as out_file:
            out_file.write(print_config() + '\n')
            source_results = train_source(run_output_dir, out_file, random_state=seed, 
                                        device=device, use_parallel=use_parallel)
    else:
        print("\nSource model already exists, skipping training...")
        print("Loading existing source results if available...")
        
        # Try to load existing source results from JSON
        source_results_file = osp.join(run_output_dir, 'source_training_results.json')
        if osp.exists(source_results_file):
            try:
                with open(source_results_file, 'r') as f:
                    source_results = json.load(f)
                print(f"Loaded existing source results: Test Acc = {source_results.get('test_accuracy', 'N/A'):.2f}%")
            except Exception as e:
                print(f"Could not load source results: {e}")
                source_results = None
        
        # If still None, create placeholder results
        if source_results is None:
            print("No existing source results found. Creating placeholder results...")
            source_results = {
                'best_train_accuracy': 0.0,
                'best_val_accuracy': 0.0,
                'test_accuracy': 0.0,
                'test_f1_micro': 0.0,
                'test_f1_macro': 0.0,
                'test_hamming_loss': 0.0,
                'training_history': [],
                'note': 'Source model loaded from existing checkpoint - metrics not available'
            }
    
    # PHASE 2: Target Baseline Testing
    print("\n" + "="*60)
    print("PHASE 2: TARGET DOMAIN BASELINE")
    print("="*60)
    with open(osp.join(run_output_dir, 'log_baseline.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        target_baseline_results = test_target_baseline(run_output_dir, out_file, random_state=seed, 
                                                     device=device, use_parallel=use_parallel)
    
    # PHASE 3: Target Adaptation (SHOT)
    print("\n" + "="*60)
    print("PHASE 3: TARGET DOMAIN ADAPTATION (SHOT)")
    print("="*60)
    savename = f'par_{preset["training"]["cls_par"]}'
    if preset["training"]["ssl"] > 0:
        savename += f'_ssl_{preset["training"]["ssl"]}'
    if preset["training"]["tsd"] > 0:
        savename += f'_tsd_{preset["training"]["tsd"]}'
    
    with open(osp.join(run_output_dir, f'log_tar_{savename}.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        target_adapted_results = train_target_shot(run_output_dir, out_file, random_state=seed, 
                                                  device=device, use_parallel=use_parallel)
    
    # Save results
    run_results['source_domain'] = source_results
    run_results['target_domain_baseline'] = target_baseline_results
    run_results['target_domain_adapted'] = target_adapted_results
    
    save_run_results(run_output_dir, run_results)
    
    return {
        'accuracy': target_adapted_results['test_accuracy'],
        'f1_micro': target_adapted_results['test_f1_micro'],
        'f1_macro': target_adapted_results['test_f1_macro'],
        'hamming_loss': target_adapted_results.get('test_hamming_loss', 0.0)
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main function to run the experiment multiple times and aggregate results"""
    print("\n" + "="*80)
    print("SHOT DOMAIN ADAPTATION EXPERIMENT")
    print("="*80)
    
    # Setup GPU
    if preset["training"]["gpu_id"] != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = preset["training"]["gpu_id"]
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU.")
    else:
        print(f"CUDA available. GPU count: {torch.cuda.device_count()}")
    
    initial_seed = preset["training"]["seed"]
    
    scenario_name = preset.get("scenario_name", "shot_experiment")
    experiment_id = create_experiment_identifier(preset)
    
    experiment_name = f"{preset['source_task']}_to_{preset['target_task']}"
    source_domain = get_domain_name("source_data")
    target_domain = get_domain_name("target_data")
    
    base_output_dir = osp.join(
        preset["path"]["save_dir"],
        f"{scenario_name}_{preset['dataset_type']}",
        experiment_name,
        f"{source_domain}_to_{target_domain}",
        f"seed_{initial_seed}_epochs_{preset.get('training', {}).get('max_epoch', 'default')}"
    )

    os.makedirs(base_output_dir, exist_ok=True)
    
    all_final_metrics = []
    all_run_summaries = []
    
    print(f"\nRunning {preset['repeat']} experiment(s) with initial seed {initial_seed}")
    print(f"Method: SHOT (Source Hypothesis Transfer)")
    print(f"Output directory: {base_output_dir}\n")
    
    # Run multiple experiments
    for i in range(preset["repeat"]):
        current_seed = initial_seed + i
        try:
            final_metrics = run_single_experiment(i, current_seed, base_output_dir)
            if final_metrics:
                all_final_metrics.append(final_metrics)
                
                run_dir = osp.join(base_output_dir, f"run_{i}")
                run_results_file = osp.join(run_dir, "run_complete_results.json")
                if osp.exists(run_results_file):
                    with open(run_results_file, 'r') as f:
                        run_data = json.load(f)
                        all_run_summaries.append(run_data)
                        
        except Exception as e:
            print(f"Error in run {i}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    if not all_final_metrics:
        print("\nNo results to aggregate. Exiting.")
        return
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    accuracies = [m['accuracy'] for m in all_final_metrics]
    f1_micros = [m['f1_micro'] for m in all_final_metrics]
    f1_macros = [m['f1_macro'] for m in all_final_metrics]
    hamming_losses = [m['hamming_loss'] for m in all_final_metrics]
    
    aggregated_stats = {
        'experiment_info': {
            'scenario_name': scenario_name,
            'experiment_id': experiment_id,
            'dataset_type': preset.get("dataset_type"),
            'source_task': preset['source_task'],
            'target_task': preset['target_task'],
            'source_domain': source_domain,
            'target_domain': target_domain,
            'method': 'SHOT'
        },
        'num_runs': len(all_final_metrics),
        'initial_seed': initial_seed,
        'results': {
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'values': accuracies
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
    
    # Create summary
    summary_str = f"\n{'='*80}\n"
    summary_str += f"FINAL RESULTS - SHOT METHOD (Test Set Performance)\n"
    summary_str += f"AGGREGATED OVER {preset['repeat']} RUNS\n"
    summary_str += f"{'='*80}\n"
    summary_str += f"Scenario: {scenario_name}\n"
    summary_str += f"Experiment ID: {experiment_id}\n"
    summary_str += f"Dataset: {preset.get('dataset_type')}\n"
    summary_str += f"Tasks: {preset['source_task']} → {preset['target_task']}\n"
    summary_str += f"Domains: {source_domain} → {target_domain}\n"
    summary_str += f"Initial Seed: {initial_seed}\n\n"
    
    results = aggregated_stats['results']
    summary_str += f"SHOT RESULTS:\n"
    summary_str += f"{'-'*40}\n"
    summary_str += f"Accuracy:     {results['accuracy']['mean']:.2f}% ± {results['accuracy']['std']:.2f}%\n"
    summary_str += f"F1-Micro:     {results['f1_micro']['mean']:.2f}% ± {results['f1_micro']['std']:.2f}%\n"
    summary_str += f"F1-Macro:     {results['f1_macro']['mean']:.2f}% ± {results['f1_macro']['std']:.2f}%\n"
    summary_str += f"Hamming Loss: {results['hamming_loss']['mean']:.2f}% ± {results['hamming_loss']['std']:.2f}%\n"
    summary_str += f"Range:        [{results['accuracy']['min']:.2f}%, {results['accuracy']['max']:.2f}%]\n"
    summary_str += f"{'='*80}\n"
    
    print(summary_str)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    aggregated_results_file = osp.join(base_output_dir, f"aggregated_results_shot_{timestamp}.json")
    aggregated_stats['timestamp'] = datetime.now().isoformat()
    aggregated_stats['experiment_config'] = {k: v for k, v in preset.items() if k != 'encoding'}
    
    with open(aggregated_results_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    
    summary_file_path = osp.join(base_output_dir, f"final_summary_shot_{timestamp}.txt")
    with open(summary_file_path, 'w') as f:
        f.write(summary_str)
    
    all_runs_file = osp.join(base_output_dir, f"all_runs_complete_{timestamp}.json")
    with open(all_runs_file, 'w') as f:
        json.dump({
            'experiment_info': aggregated_stats['experiment_info'],
            'runs': all_run_summaries,
            'aggregated_statistics': aggregated_stats
        }, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - Aggregated results: {aggregated_results_file}")
    print(f"  - Summary: {summary_file_path}")
    print(f"  - Complete results: {all_runs_file}")
    print(f"  - Output directory: {base_output_dir}")
    print("\nSHOT experiment completed successfully!")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()