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
import warnings
warnings.filterwarnings('ignore')

# Project modules
import network
import loss
from preset import preset
from load_data import load_widar_data, apply_windowing


def debug_data_availability():
    """Debug what data is available for different configurations"""
    print("Debugging WiDAR data availability...")
    
    from load_data import WidarDatasetReader
    
    reader = WidarDatasetReader(preset["path"]["base_path"])
    
    print(f"Base path: {preset['path']['base_path']}")
    
    # Test source data
    print(f"\nTesting source data:")
    print(f"  Rooms: {preset['source_data']['rooms']}")
    print(f"  Users: {preset['source_data']['users']}")
    print(f"  Gestures: {preset['source_data']['gestures']}")
    
    try:
        X_src, y_src, metas_src, encoder_src = reader.load_dataset(
            rooms=preset["source_data"]["rooms"],
            users=preset["source_data"]["users"],
            receivers=preset["source_data"]["receivers"],
            torso_locations=preset["source_data"]["torso_locations"],
            gestures=preset["source_data"]["gestures"],
            exclude_digits=preset["source_data"]["exclude_digits"],
            min_time_steps=preset["source_data"]["min_time_steps"],
            max_samples_per_class=preset["source_data"]["max_samples_per_class"],
            target_len=preset["source_data"]["target_len"],
            include_phase=preset["source_data"]["include_phase"]
        )
        print(f"  Source samples found: {len(X_src)}")
        print(f"  Source shape: {X_src.shape}")
        print(f"  Source classes: {list(encoder_src.classes_)}")
    except Exception as e:
        print(f"  Error loading source data: {e}")
    
    # Test target data
    print(f"\nTesting target data:")
    print(f"  Rooms: {preset['target_data']['rooms']}")
    print(f"  Users: {preset['target_data']['users']}")
    print(f"  Gestures: {preset['target_data']['gestures']}")
    
    try:
        X_tgt, y_tgt, metas_tgt, encoder_tgt = reader.load_dataset(
            rooms=preset["target_data"]["rooms"],
            users=preset["target_data"]["users"],
            receivers=preset["target_data"]["receivers"],
            torso_locations=preset["target_data"]["torso_locations"],
            gestures=preset["target_data"]["gestures"],
            exclude_digits=preset["target_data"]["exclude_digits"],
            min_time_steps=preset["target_data"]["min_time_steps"],
            max_samples_per_class=preset["target_data"]["max_samples_per_class"],
            target_len=preset["target_data"]["target_len"],
            include_phase=preset["target_data"]["include_phase"]
        )
        print(f"  Target samples found: {len(X_tgt)}")
        print(f"  Target shape: {X_tgt.shape}")
        print(f"  Target classes: {list(encoder_tgt.classes_)}")
    except Exception as e:
        print(f"  Error loading target data: {e}")


def cal_acc(loader, netF, netB, netC, encoder, device):
    """Calculate accuracy on given loader and generate a classification report."""
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
    
    # Convert numeric labels back to class names for report
    class_names = encoder.classes_
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    return accuracy, f1_micro, f1_macro, 0.0, report  # hamming_loss=0 for multiclass


def lr_scheduler(optimizer, epoch, max_epochs, gamma=10, power=0.75):
    decay = (1 + gamma * epoch / max_epochs) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def get_model_device_safe(model):
    """Get the underlying model whether it's wrapped in DataParallel or not"""
    return model.module if hasattr(model, 'module') else model


def setup_device_and_parallel():
    """Setup device and determine if we should use DataParallel"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_parallel = torch.cuda.device_count() > 1
    
    if use_parallel:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        print(f"Using device: {device}")
    
    return device, use_parallel


def save_model_weights(model, filepath, use_parallel=False):
    """Save model weights properly handling DataParallel"""
    if use_parallel:
        torch.save(model.module.state_dict(), filepath)
    else:
        torch.save(model.state_dict(), filepath)


def load_model_weights(model, filepath, device):
    """Load model weights properly handling device mapping"""
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    return model


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
        
        f.write("SOURCE DOMAIN RESULTS:\n")
        f.write("-" * 30 + "\n")
        source_results = run_results['source_domain']
        f.write(f"Best Training Accuracy: {source_results['best_train_accuracy']:.2f}%\n")
        f.write(f"Test Accuracy: {source_results['test_accuracy']:.2f}%\n")
        f.write(f"Test F1-Micro: {source_results['test_f1_micro']:.2f}%\n")
        f.write(f"Test F1-Macro: {source_results['test_f1_macro']:.2f}%\n\n")
        
        f.write("TARGET DOMAIN RESULTS (Source Model):\n")
        f.write("-" * 30 + "\n")
        target_baseline = run_results['target_domain_baseline']
        f.write(f"Accuracy: {target_baseline['accuracy']:.2f}%\n")
        f.write(f"F1-Micro: {target_baseline['f1_micro']:.2f}%\n")
        f.write(f"F1-Macro: {target_baseline['f1_macro']:.2f}%\n\n")
        
        f.write("TARGET DOMAIN RESULTS (After Adaptation):\n")
        f.write("-" * 30 + "\n")
        target_adapted = run_results['target_domain_adapted']
        f.write(f"Best Accuracy: {target_adapted['best_accuracy']:.2f}%\n")
        f.write(f"Best F1-Micro: {target_adapted['best_f1_micro']:.2f}%\n")
        f.write(f"Best F1-Macro: {target_adapted['best_f1_macro']:.2f}%\n")
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


def train_source(output_dir, out_file, random_state, device, use_parallel):
    """Train source domain model with proper device handling"""
    print("Loading source data...")
    
    # Get the full data info to access input_channels directly
    data_info = load_widar_data("source_data", random_state)
    
    # Extract loaders and info
    train_loader = data_info['train_loader']
    test_loader = data_info['test_loader']
    num_classes = data_info['num_classes']
    var_x_shape = data_info['sample_shape']
    encoder = data_info['label_encoder']
    has_phase = data_info['has_phase']
    input_channels = data_info['input_channels']
    
    print(f"Network setup:")
    print(f"  Sample shape: {var_x_shape}")
    print(f"  Input channels: {input_channels}")
    print(f"  Num classes: {num_classes}")
    
    # Model setup with correct input channels
    netF = network.CNN2DBase(var_x_shape, input_channels=input_channels).to(device)
    netB = network.feat_bottleneck(feature_dim=netF.in_features, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    netC = network.feat_classifier(class_num=num_classes, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    
    # Apply DataParallel if multiple GPUs available
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
    
    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["lr"], weight_decay=1e-4))
    
    max_epochs = preset["training"]["max_epoch"]
    best_acc = 0
    best_train_acc = 0
    best_f1_micro = 0
    best_f1_macro = 0
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    
    training_history = []
    
    print("Starting source training...")
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.train()
        
        epoch_loss = 0
        num_batches = 0
        
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}')
        for inputs_source, labels_source in pbar:
            inputs_source = inputs_source.to(device).float()
            labels_source = labels_source.to(device).long()
            inputs_source = torch.nan_to_num(inputs_source, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Forward pass
            features = netF(inputs_source)
            bottleneck_features = netB(features)
            outputs_source = netC(bottleneck_features)
            
            classifier_loss = criterion(outputs_source, labels_source)
            
            optimizer.zero_grad()
            classifier_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(get_model_device_safe(netF).parameters()) +
                list(get_model_device_safe(netB).parameters()) +
                list(get_model_device_safe(netC).parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            epoch_loss += classifier_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{classifier_loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation
        netF.eval(); netB.eval(); netC.eval()
        
        # Training accuracy
        acc_s_tr, f1_micro_tr, f1_macro_tr, hamming_tr, report_tr = cal_acc(train_loader, netF, netB, netC, encoder, device)
        
        # Test accuracy
        acc_s_te, f1_micro_te, f1_macro_te, hamming_te, report_te = cal_acc(test_loader, netF, netB, netC, encoder, device)
        
        # Store epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'train_accuracy': acc_s_tr,
            'train_f1_micro': f1_micro_tr,
            'train_f1_macro': f1_macro_tr,
            'test_accuracy': acc_s_te,
            'test_f1_micro': f1_micro_te,
            'test_f1_macro': f1_macro_te
        }
        training_history.append(epoch_results)
        
        log_str = f'Source Training - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Train Acc: {acc_s_tr:.2f}%; Test Acc: {acc_s_te:.2f}%'
        out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
        
        # Save best model
        if acc_s_te >= best_acc:
            best_acc = acc_s_te
            best_train_acc = acc_s_tr
            best_f1_micro = f1_micro_te
            best_f1_macro = f1_macro_te
            
            # Save best model weights properly
            save_model_weights(netF, osp.join(output_dir, "source_F.pt"), use_parallel)
            save_model_weights(netB, osp.join(output_dir, "source_B.pt"), use_parallel)
            save_model_weights(netC, osp.join(output_dir, "source_C.pt"), use_parallel)
            
            # Save best model's detailed report
            best_report_file = osp.join(output_dir, "source_best_classification_report.txt")
            with open(best_report_file, 'w') as f:
                f.write(f"Best Source Model Performance (Epoch {epoch+1})\n")
                f.write(f"Test Accuracy: {acc_s_te:.2f}%\n")
                f.write(f"Test F1-Micro: {f1_micro_te:.2f}%\n")
                f.write(f"Test F1-Macro: {f1_macro_te:.2f}%\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(report_te)
    
    # Save training history
    history_file = osp.join(output_dir, "source_training_history.json")
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Source training completed. Best test accuracy: {best_acc:.2f}%")
    
    return {
        'best_train_accuracy': best_train_acc,
        'test_accuracy': best_acc,
        'test_f1_micro': best_f1_micro,
        'test_f1_macro': best_f1_macro,
        'test_hamming_loss': 0.0,
        'training_history': training_history
    }


def test_target_baseline(output_dir, out_file, random_state, device, use_parallel):
    """Test on target domain using the trained source model"""
    print("Loading target data for baseline testing...")
    
    # Use new data loading function
    data_info = load_widar_data("target_data", random_state)
    
    test_loader = data_info['test_loader']
    num_classes = data_info['num_classes']
    var_x_shape = data_info['sample_shape']
    encoder = data_info['label_encoder']
    input_channels = data_info['input_channels']
    
    # Initialize models
    netF = network.CNN2DBase(var_x_shape, input_channels=input_channels).to(device)
    netB = network.feat_bottleneck(feature_dim=netF.in_features, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    netC = network.feat_classifier(class_num=num_classes, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    
    # Load weights
    load_model_weights(netF, osp.join(output_dir, "source_F.pt"), device)
    load_model_weights(netB, osp.join(output_dir, "source_B.pt"), device)
    load_model_weights(netC, osp.join(output_dir, "source_C.pt"), device)
    
    # Apply DataParallel if needed
    if use_parallel:
        netF = nn.DataParallel(netF)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
    
    netF.eval(); netB.eval(); netC.eval()
    
    acc, f1_micro, f1_macro, hamming, report = cal_acc(test_loader, netF, netB, netC, encoder, device)
    
    # Save baseline results
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



def train_target(output_dir, out_file, random_state, device, use_parallel):
    #Domain adaptation training on target
    print("Loading target data for adaptation...")
    
    # Use new data loading function
    data_info = load_widar_data("target_data", random_state)
    
    train_loader = data_info['train_loader']
    test_loader = data_info['test_loader']
    num_classes = data_info['num_classes']
    var_x_shape = data_info['sample_shape']
    encoder = data_info['label_encoder']
    input_channels = data_info['input_channels']
    
    # Initialize models
    netF = network.CNN2DBase(var_x_shape, input_channels=input_channels).to(device)
    netB = network.feat_bottleneck(feature_dim=netF.in_features, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    netC = network.feat_classifier(class_num=num_classes, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    
    # Load pre-trained weights
    print("Loading pre-trained source model weights...")
    load_model_weights(netF, osp.join(output_dir, "source_F.pt"), device)
    load_model_weights(netB, osp.join(output_dir, "source_B.pt"), device)
    load_model_weights(netC, osp.join(output_dir, "source_C.pt"), device)
    
    # Apply DataParallel if needed
    if use_parallel:
        netF = nn.DataParallel(netF)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
    
    # IMPORTANT: Evaluate model immediately after loading weights to verify correct loading
    print("Evaluating loaded pre-trained model on target data (before adaptation)...")
    netF.eval(); netB.eval(); netC.eval()
    initial_acc, initial_f1_micro, initial_f1_macro, initial_hamming, initial_report = cal_acc(
        test_loader, netF, netB, netC, encoder, device
    )
    
    print(f"Initial Model Performance (Pre-trained on Source, Tested on Target):")
    print(f"  Accuracy: {initial_acc:.2f}%")
    print(f"  F1-Micro: {initial_f1_micro:.2f}%")
    print(f"  F1-Macro: {initial_f1_macro:.2f}%")
    print(f"  Hamming Loss: {initial_hamming:.2f}%")
    
    # Log initial performance
    log_str = f'Initial Pre-trained Model - Accuracy: {initial_acc:.2f}%, F1-micro: {initial_f1_micro:.2f}%, F1-macro: {initial_f1_macro:.2f}%, Hamming: {initial_hamming:.2f}%'
    out_file.write(log_str + '\n'); out_file.flush()
    
    # Save initial evaluation report
    savename = f'ent_{preset["training"]["ent_par"]}'
    initial_report_filename = osp.join(output_dir, f"target_initial_classification_report_{savename}.txt")
    with open(initial_report_filename, 'w') as f:
        f.write("Initial Pre-trained Model Performance on Target Data (Before Adaptation)\n")
        f.write(f"Accuracy: {initial_acc:.2f}%\n")
        f.write(f"F1-Micro: {initial_f1_micro:.2f}%\n")
        f.write(f"F1-Macro: {initial_f1_macro:.2f}%\n")
        f.write(f"Hamming Loss: {initial_hamming:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(initial_report)
    
    # Freeze classifier for domain adaptation
    for v in get_model_device_safe(netC).parameters(): 
        v.requires_grad = False

    # Setup optimizer for adaptation
    param_group = []
    for v in get_model_device_safe(netF).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netB).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    optimizer = optim.SGD(param_group)  # No momentum, no weight decay
    optimizer = op_copy(optimizer)
        
    max_epochs = preset["training"]["max_epoch"]
    best_acc = initial_acc  # Initialize with initial performance
    best_f1_micro = initial_f1_micro
    best_f1_macro = initial_f1_macro
    best_hamming = initial_hamming
    best_epoch = 0
    
    adaptation_history = []
    
    # Store initial results in history
    initial_results = {
        'epoch': 0,
        'loss': 0.0,
        'accuracy': initial_acc,
        'f1_micro': initial_f1_micro,
        'f1_macro': initial_f1_macro,
        'hamming_loss': initial_hamming,
        'note': 'Initial pre-trained model performance'
    }
    adaptation_history.append(initial_results)
    
    print("Starting target adaptation...")
    for epoch in range(max_epochs):
       # lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.eval()
        
        epoch_loss = 0
        num_batches = 0
        
        # Adaptation loop with progress bar
        pbar = tqdm(train_loader, desc=f'Adaptation Epoch {epoch+1}/{max_epochs}')
        for inputs_target, _ in pbar:
            inputs_target = inputs_target.to(device)
            inputs_target = torch.nan_to_num(inputs_target, nan=0.0, posinf=0.0, neginf=0.0)
            features_target = netB(netF(inputs_target))
            outputs_target = netC(features_target)
    
            # Entropy loss for domain adaptation
            # Entropy loss for domain adaptation
            #softmax_out = torch.softmax(outputs_target, dim=1)
            #entropy_loss = torch.mean(torch.sum(-softmax_out * torch.log(softmax_out + 1e-5), dim=1))

            #im_loss = entropy_loss * preset["training"]["ent_par"]


            # Entropy loss for domain adaptation
            softmax_out = torch.softmax(outputs_target, dim=1)
            entropy_loss = torch.mean(torch.sum(-softmax_out * torch.log(softmax_out + 1e-5), dim=1))
            
            # GENT regularization - encourage batch diversity
            #msoftmax = softmax_out.mean(dim=0)
            #entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            
            im_loss = entropy_loss * preset["training"]["ent_par"]
            optimizer.zero_grad()
            im_loss.backward()
            optimizer.step()
            epoch_loss += im_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Entropy': f'{entropy_loss.item():.4f}',
                'Total': f'{im_loss.item():.4f}'
            })
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation
        netF.eval(); netB.eval()
        acc, f1_micro, f1_macro, hamming, report = cal_acc(test_loader, netF, netB, netC, encoder, device)
        
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
        
        # Save best model (only if better than initial)
        if acc > best_acc:
            best_acc = acc
            best_f1_micro = f1_micro
            best_f1_macro = f1_macro
            best_hamming = hamming
            best_epoch = epoch + 1
            
            # Save best adapted model weights
            if preset["training"]["save_models"]:
                savename = f'ent_{preset["training"]["ent_par"]}'
                save_model_weights(netF, osp.join(output_dir, f"adapted_F_{savename}.pt"), use_parallel)
                save_model_weights(netB, osp.join(output_dir, f"adapted_B_{savename}.pt"), use_parallel)
            
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
        
        # Compare with initial performance to detect issues
        acc_change = acc - initial_acc
        change_indicator = "↑" if acc_change > 0 else "↓" if acc_change < 0 else "="
        
        log_str = f'Target Adaptation - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Accuracy: {acc:.2f}% ({change_indicator}{abs(acc_change):.2f}%), F1-micro: {f1_micro:.2f}%'
        out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
        
        # Warning if performance drops significantly from initial
        if acc < initial_acc - 5.0:  # More than 5% drop
            print(f"WARNING: Performance dropped by {initial_acc - acc:.2f}% from initial model!")
    
    # Final summary
    print(f"\nAdaptation Summary:")
    print(f"  Initial Performance: {initial_acc:.2f}% accuracy")
    print(f"  Best Performance: {best_acc:.2f}% accuracy (Epoch {best_epoch})")
    print(f"  Performance Change: {best_acc - initial_acc:+.2f}%")
    
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

def train_target_wisfdagr(output_dir, out_file, random_state, device, use_parallel):
    """Wi-SFDAGR: WiFi-Based Source-Free Domain Adaptation for Gesture Recognition"""
    print("Loading target data for adaptation...")
    
    # Use new data loading function
    data_info = load_widar_data("target_data", random_state)
    
    train_loader = data_info['train_loader']
    test_loader = data_info['test_loader']
    num_classes = data_info['num_classes']
    var_x_shape = data_info['sample_shape']
    encoder = data_info['label_encoder']
    input_channels = data_info['input_channels']
    
    # Initialize models
    netF = network.CNN2DBase(var_x_shape, input_channels=input_channels).to(device)
    netB = network.feat_bottleneck(feature_dim=netF.in_features, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    netC = network.feat_classifier(class_num=num_classes, 
                                   bottleneck_dim=preset["training"]["bottleneck"]).to(device)
    
    # Load pre-trained weights
    print("Loading pre-trained source model weights...")
    load_model_weights(netF, osp.join(output_dir, "source_F.pt"), device)
    load_model_weights(netB, osp.join(output_dir, "source_B.pt"), device)
    load_model_weights(netC, osp.join(output_dir, "source_C.pt"), device)
    
    # Apply DataParallel if needed
    if use_parallel:
        netF = nn.DataParallel(netF)
        netB = nn.DataParallel(netB)
        netC = nn.DataParallel(netC)
    
    # IMPORTANT: Evaluate model immediately after loading weights to verify correct loading
    print("Evaluating loaded pre-trained model on target data (before adaptation)...")
    netF.eval(); netB.eval(); netC.eval()
    initial_acc, initial_f1_micro, initial_f1_macro, initial_hamming, initial_report = cal_acc(
        test_loader, netF, netB, netC, encoder, device
    )
    
    print(f"Initial Model Performance (Pre-trained on Source, Tested on Target):")
    print(f"  Accuracy: {initial_acc:.2f}%")
    print(f"  F1-Micro: {initial_f1_micro:.2f}%")
    print(f"  F1-Macro: {initial_f1_macro:.2f}%")
    print(f"  Hamming Loss: {initial_hamming:.2f}%")
    
    # Log initial performance
    log_str = f'Initial Pre-trained Model - Accuracy: {initial_acc:.2f}%, F1-micro: {initial_f1_micro:.2f}%, F1-macro: {initial_f1_macro:.2f}%, Hamming: {initial_hamming:.2f}%'
    out_file.write(log_str + '\n'); out_file.flush()
    
    # Save initial evaluation report
    savename = f'wi_sfdagr'
    initial_report_filename = osp.join(output_dir, f"target_initial_classification_report_{savename}.txt")
    with open(initial_report_filename, 'w') as f:
        f.write("Initial Pre-trained Model Performance on Target Data (Before Adaptation)\n")
        f.write(f"Accuracy: {initial_acc:.2f}%\n")
        f.write(f"F1-Micro: {initial_f1_micro:.2f}%\n")
        f.write(f"F1-Macro: {initial_f1_macro:.2f}%\n")
        f.write(f"Hamming Loss: {initial_hamming:.2f}%\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(initial_report)
    
    # Freeze classifier for domain adaptation (as in Wi-SFDAGR)
    for v in get_model_device_safe(netC).parameters(): 
        v.requires_grad = False

    # Setup optimizer for adaptation
    param_group = []
    for v in get_model_device_safe(netF).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["lr"]}]
    for v in get_model_device_safe(netB).parameters(): 
        param_group += [{'params': v, 'lr': preset["training"]["lr"]}]

    optimizer = op_copy(optim.Adam(param_group, lr=preset["training"]["lr"], weight_decay=1e-4))
    
    max_epochs = preset["training"]["max_epoch"]
    best_acc = initial_acc  # Initialize with initial performance
    best_f1_micro = initial_f1_micro
    best_f1_macro = initial_f1_macro
    best_hamming = initial_hamming
    best_epoch = 0
    
    adaptation_history = []
    
    # Store initial results in history
    initial_results = {
        'epoch': 0,
        'loss': 0.0,
        'accuracy': initial_acc,
        'f1_micro': initial_f1_micro,
        'f1_macro': initial_f1_macro,
        'hamming_loss': initial_hamming,
        'note': 'Initial pre-trained model performance'
    }
    adaptation_history.append(initial_results)
    
    # Wi-SFDAGR parameters
    K_neighbors = 10  # Number of nearest neighbors
    lambda_weight = 1.0  # Weight balance between attraction and dispersion
    
    # Memory banks for features and predictions (Wi-SFDAGR approach)
    feature_memory = []
    prediction_memory = []
    
    print("Starting Wi-SFDAGR target adaptation...")
    for epoch in range(max_epochs):
        lr_scheduler(optimizer, epoch, max_epochs)
        netF.train(); netB.train(); netC.eval()
        
        epoch_loss = 0
        num_batches = 0
        
        # Clear memory banks at start of each epoch
        feature_memory = []
        prediction_memory = []
        
        # First pass: collect all features and predictions for this epoch
        print("Collecting features and predictions...")
        with torch.no_grad():
            netF.eval(); netB.eval(); netC.eval()
            for inputs_target, _ in train_loader:
                inputs_target = inputs_target.to(device)
                inputs_target = torch.nan_to_num(inputs_target, nan=0.0, posinf=0.0, neginf=0.0)
                features_target = netB(netF(inputs_target))
                outputs_target = netC(features_target)
                predictions = torch.softmax(outputs_target, dim=1)
                
                feature_memory.append(features_target.cpu())
                prediction_memory.append(predictions.cpu())
            
            # Concatenate all features and predictions
            all_features = torch.cat(feature_memory, dim=0).to(device)
            all_predictions = torch.cat(prediction_memory, dim=0).to(device)
        
        # Second pass: adaptation with attraction-dispersion network
        netF.train(); netB.train(); netC.eval()
        
        # Adaptation loop with progress bar
        pbar = tqdm(train_loader, desc=f'Wi-SFDAGR Epoch {epoch+1}/{max_epochs}')
        batch_idx = 0
        
        for inputs_target, _ in pbar:
            inputs_target = inputs_target.to(device)
            inputs_target = torch.nan_to_num(inputs_target, nan=0.0, posinf=0.0, neginf=0.0)
            batch_size = inputs_target.size(0)
            
            features_target = netB(netF(inputs_target))
            outputs_target = netC(features_target)
            predictions = torch.softmax(outputs_target, dim=1)
            
            total_loss = 0.0
            
            # Wi-SFDAGR: Attraction-Dispersion Loss for each sample in batch
            for i in range(batch_size):
                current_idx = batch_idx * train_loader.batch_size + i
                if current_idx >= all_features.size(0):
                    continue
                    
                current_feature = features_target[i].unsqueeze(0)  # [1, feature_dim]
                current_pred = predictions[i]  # [num_classes]
                
                # Compute cosine similarity with all features in memory
                with torch.no_grad():
                    # Normalize features for cosine similarity
                    current_feature_norm = F.normalize(current_feature, p=2, dim=1)
                    all_features_norm = F.normalize(all_features, p=2, dim=1)
                    similarities = torch.mm(current_feature_norm, all_features_norm.t()).squeeze(0)
                    
                    # Get K nearest neighbors (excluding self)
                    similarities[current_idx] = -1  # Exclude self
                    _, neighbor_indices = torch.topk(similarities, K_neighbors)
                    
                    # Get non-neighbor indices (farthest samples)
                    _, non_neighbor_indices = torch.topk(similarities, K_neighbors, largest=False)
                
                # Attraction loss: encourage similar predictions for nearest neighbors
                attraction_loss = 0.0
                for neighbor_idx in neighbor_indices:
                    neighbor_pred = all_predictions[neighbor_idx]
                    
                    # Compute uncertainty weight (Wi-SFDAGR uncertainty estimation)
                    avg_pred = (current_pred + neighbor_pred) / 2.0
                    entropy = -torch.sum(avg_pred * torch.log(avg_pred + 1e-8))
                    uncertainty_weight = torch.exp(-entropy / torch.log(torch.tensor(num_classes, dtype=torch.float32)))
                    
                    # Weighted attraction (encourage similar predictions)
                    attraction_loss += uncertainty_weight * torch.sum(-current_pred * torch.log(neighbor_pred + 1e-8))
                
                attraction_loss = attraction_loss / K_neighbors
                
                # Dispersion loss: encourage different predictions for distant samples
                dispersion_loss = 0.0
                for non_neighbor_idx in non_neighbor_indices:
                    non_neighbor_pred = all_predictions[non_neighbor_idx]
                    # Encourage different predictions (maximize entropy between predictions)
                    dispersion_loss += torch.sum(current_pred * torch.log(non_neighbor_pred + 1e-8))
                
                dispersion_loss = dispersion_loss / K_neighbors
                
                # Wi-SFDAGR combined loss (Equation 17 from paper)
                sample_loss = attraction_loss + lambda_weight * dispersion_loss
                total_loss += sample_loss
            
            # Average loss over batch
            if batch_size > 0:
                im_loss = total_loss / batch_size
            else:
                im_loss = torch.tensor(0.0, requires_grad=True, device=device)
            
            optimizer.zero_grad()
            im_loss.backward()
            optimizer.step()
            
            epoch_loss += im_loss.item()
            num_batches += 1
            batch_idx += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Wi-SFDAGR Loss': f'{im_loss.item():.4f}',
            })
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Evaluation
        netF.eval(); netB.eval()
        acc, f1_micro, f1_macro, hamming, report = cal_acc(test_loader, netF, netB, netC, encoder, device)
        
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
        
        # Save best model (only if better than initial)
        if acc > best_acc:
            best_acc = acc
            best_f1_micro = f1_micro
            best_f1_macro = f1_macro
            best_hamming = hamming
            best_epoch = epoch + 1
            
            # Save best adapted model weights
            if preset["training"]["save_models"]:
                save_model_weights(netF, osp.join(output_dir, f"adapted_F_wi_sfdagr.pt"), use_parallel)
                save_model_weights(netB, osp.join(output_dir, f"adapted_B_wi_sfdagr.pt"), use_parallel)
            
            # Save best adapted model's detailed report
            report_filename = osp.join(output_dir, f"target_adapted_best_classification_report_wi_sfdagr.txt")
            with open(report_filename, 'w') as f:
                f.write(f"Best Wi-SFDAGR Adapted Model Performance (Epoch {epoch+1})\n")
                f.write(f"Accuracy: {acc:.2f}%\n")
                f.write(f"F1-Micro: {f1_micro:.2f}%\n")
                f.write(f"F1-Macro: {f1_macro:.2f}%\n")
                f.write(f"Hamming Loss: {hamming:.2f}%\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(report)
        
        # Compare with initial performance to detect issues
        acc_change = acc - initial_acc
        change_indicator = "↑" if acc_change > 0 else "↓" if acc_change < 0 else "="
        
        log_str = f'Wi-SFDAGR Adaptation - Epoch {epoch+1}/{max_epochs}; Loss: {avg_loss:.4f}; Accuracy: {acc:.2f}% ({change_indicator}{abs(acc_change):.2f}%), F1-micro: {f1_micro:.2f}%'
        out_file.write(log_str + '\n'); out_file.flush(); print(log_str)
        
        # Warning if performance drops significantly from initial
        if acc < initial_acc - 5.0:  # More than 5% drop
            print(f"WARNING: Performance dropped by {initial_acc - acc:.2f}% from initial model!")
    
    # Final summary
    print(f"\nWi-SFDAGR Adaptation Summary:")
    print(f"  Initial Performance: {initial_acc:.2f}% accuracy")
    print(f"  Best Performance: {best_acc:.2f}% accuracy (Epoch {best_epoch})")
    print(f"  Performance Change: {best_acc - initial_acc:+.2f}%")
    
    # Save adaptation history
    history_file = osp.join(output_dir, "wi_sfdagr_adaptation_history.json")
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
    """Generate domain name for WiDAR configuration"""
    config = preset[config_key]
    rooms = "_".join(config["rooms"])
    users = f"users_{len(config['users'])}"
    phase_info = "amp_phase" if config["include_phase"] else "amp_only"
    return f"{rooms}_{users}_{phase_info}"


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
    
    # Setup device and parallel processing
    device, use_parallel = setup_device_and_parallel()
    
    # Create a unique directory for this run
    output_dir = osp.join(base_output_dir, f"run_{run_idx}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nStarting Run {run_idx+1}/{preset['repeat']} with Seed {seed} | Output: {output_dir}")
    
    # Initialize run results dictionary
    run_results = {
        'run_index': run_idx,
        'seed': seed,
        'device': str(device),
        'use_parallel': use_parallel,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'experiment_config': {k: v for k, v in preset.items() if k != 'encoding'}
    }
    
    # --- Source Training ---
    source_results = None
    if not osp.exists(osp.join(output_dir, 'source_F.pt')):
        print("\n" + "="*50 + "\nPHASE 1: SOURCE DOMAIN TRAINING\n" + "="*50)
        with open(osp.join(output_dir, 'log_src.txt'), 'w') as out_file:
            out_file.write(print_config() + '\n')
            source_results = train_source(output_dir, out_file, random_state=seed, 
                                        device=device, use_parallel=use_parallel)
    else:
        print("Source model already exists, loading existing results...")
        # Try to load existing source results if available
        try:
            existing_results_file = osp.join(output_dir, "run_complete_results.json")
            if osp.exists(existing_results_file):
                with open(existing_results_file, 'r') as f:
                    existing_results = json.load(f)
                    source_results = existing_results.get('source_domain', {})
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    # --- Target Baseline Testing ---
    print("\n" + "="*50 + "\nPHASE 2: TARGET DOMAIN BASELINE (Source Model)\n" + "="*50)
    with open(osp.join(output_dir, 'log_baseline.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        target_baseline_results = test_target_baseline(output_dir, out_file, random_state=seed, 
                                                     device=device, use_parallel=use_parallel)
    
    # --- Target Adaptation ---
    print("\n" + "="*50 + "\nPHASE 3: TARGET DOMAIN ADAPTATION\n" + "="*50)
    savename = f'ent_{preset["training"]["ent_par"]}'
    with open(osp.join(output_dir, f'log_tar_{savename}.txt'), 'w') as out_file:
        out_file.write(print_config() + '\n')
        target_adapted_results = train_target(output_dir, out_file, random_state=seed, 
                                            device=device, use_parallel=use_parallel)
    
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


def create_experiment_identifier(preset):
    """Create a comprehensive experiment identifier from config parameters."""
    
    # Core experiment info
    dataset_type = preset.get("dataset_type", "unknown")
    source_task = preset.get("source_task", "src")
    target_task = preset.get("target_task", "tgt")
    
    # Training parameters
    epochs = preset.get("training", {}).get("epochs", "default")
    lr = preset.get("training", {}).get("learning_rate", preset.get("training", {}).get("lr", "default"))
    batch_size = preset.get("training", {}).get("batch_size", "default")
    
    # Model architecture
    model_name = preset.get("model", {}).get("name", "model")
    
    # Adaptation method (if any)
    adaptation_method = preset.get("adaptation", {}).get("method", "none")
    
    # Create experiment identifier
    experiment_id = f"{dataset_type}_{source_task}to{target_task}_{model_name}"
    
    if adaptation_method != "none":
        experiment_id += f"_{adaptation_method}"
    
    experiment_id += f"_ep{epochs}_lr{lr}_bs{batch_size}"
    
    return experiment_id

def main():
    """Main function to run the experiment multiple times and aggregate results."""
    # Set up GPU environment
    if preset["training"]["gpu_id"] != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = preset["training"]["gpu_id"]
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU.")
    else:
        print(f"CUDA available. GPU count: {torch.cuda.device_count()}")
    
    initial_seed = preset["training"]["seed"]
    
    # Option 1: Add a scenario variable for custom naming
    scenario_name = preset.get("scenario_name", "default_scenario")
    
    # Option 2: Create comprehensive experiment identifier
    experiment_id = create_experiment_identifier(preset)
    
    # Define the base output directory with more descriptive naming
    experiment_name = f"{preset['source_task']}_to_{preset['target_task']}"
    source_domain = get_domain_name("source_data")
    target_domain = get_domain_name("target_data")
    
    # Choose one of these approaches:
    
    # Approach 1: Use scenario name + key params
    base_output_dir = osp.join(
        preset["path"]["save_dir"],
        f"{scenario_name}_{preset['dataset_type']}",
        experiment_name,
        f"{source_domain}_to_{target_domain}",
        f"seed_{initial_seed}_epochs_{preset.get('training', {}).get('epochs', 'default')}"
    )
    
    # Approach 2: Use comprehensive experiment ID
    # base_output_dir = osp.join(
    #     preset["path"]["save_dir"],
    #     experiment_id,
    #     f"{source_domain}_to_{target_domain}",
    #     f"seed_{initial_seed}"
    # )
    
    # Approach 3: Include timestamp for uniqueness
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_output_dir = osp.join(
    #     preset["path"]["save_dir"],
    #     f"{scenario_name}_{preset['dataset_type']}_{timestamp}",
    #     experiment_name,
    #     f"{source_domain}_to_{target_domain}",
    #     f"seed_{initial_seed}"
    # )
    
    # Debug data availability
    debug_data_availability()
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Lists to store results from all runs
    all_final_metrics = []
    all_run_summaries = []
    
    print(f"\nRunning {preset['repeat']} experiment(s) with initial seed {initial_seed}")
    print(f"Output directory: {base_output_dir}")
    
    for i in range(preset["repeat"]):
        current_seed = initial_seed + i
        try:
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
                        
        except Exception as e:
            print(f"Error in run {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
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
        'experiment_info': {
            'scenario_name': scenario_name,
            'experiment_id': experiment_id,
            'dataset_type': preset.get("dataset_type"),
            'source_task': preset['source_task'],
            'target_task': preset['target_task'],
            'source_domain': source_domain,
            'target_domain': target_domain
        },
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
    
    # [Rest of the aggregation code remains the same...]
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
    summary_str += f"Scenario: {scenario_name}\n"
    summary_str += f"Experiment ID: {experiment_id}\n"
    summary_str += f"Dataset: {preset.get('dataset_type')}\n"
    summary_str += f"Tasks: {preset['source_task']} → {preset['target_task']}\n"
    summary_str += f"Domains: {source_domain} → {target_domain}\n"
    summary_str += f"Initial Seed: {initial_seed}\n\n"
    
    # [Rest of summary generation remains the same...]
    
    # Save results with more descriptive filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Aggregated statistics (JSON) - include key params in filename
    aggregated_results_file = osp.join(
        base_output_dir, 
        f"aggregated_results_{scenario_name}_{timestamp}.json"
    )
    aggregated_stats['timestamp'] = datetime.now().isoformat()
    aggregated_stats['experiment_config'] = {k: v for k, v in preset.items() if k != 'encoding'}
    
    with open(aggregated_results_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    
    # 2. Human-readable summary
    summary_file_path = osp.join(
        base_output_dir, 
        f"final_summary_{scenario_name}_{timestamp}.txt"
    )
    with open(summary_file_path, 'w') as f:
        f.write(summary_str)
    
    # 3. All runs summary (JSON)
    all_runs_file = osp.join(
        base_output_dir, 
        f"all_runs_{scenario_name}_{timestamp}.json"
    )
    with open(all_runs_file, 'w') as f:
        json.dump({
            'experiment_info': {
                'scenario_name': scenario_name,
                'experiment_id': experiment_id,
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
    print(f"  - Output directory: {base_output_dir}")
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()