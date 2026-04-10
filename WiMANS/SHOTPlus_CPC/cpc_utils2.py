# cpc_utils2.py — training utilities for per-slot CPC (pre-train, evaluate, loss).
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def pretrain_cpc(cpc_model, data_loader, device, num_epochs=50, lr=1e-3, out_file=None):
    """
    Pre-train CPC model on target domain (unsupervised) with comprehensive tracking.
    Supports both per-sample and per-slot CPC models.
    
    Args:
        cpc_model: CPCModel or CPCModelPerSlot instance (can be wrapped in DataParallel)
        data_loader: DataLoader for target domain
        device: torch.device
        num_epochs: Number of pre-training epochs
        lr: Learning rate
        out_file: File handle for logging
    
    Returns:
        best_state_dict: Best model weights (unwrapped if DataParallel)
        cpc_metrics: Dictionary containing CPC training history and final metrics
    """
    is_parallel = isinstance(cpc_model, nn.DataParallel)
    
    cpc_model.to(device)
    cpc_model.train()
    
    model_params = cpc_model.module.parameters() if is_parallel else cpc_model.parameters()
    optimizer = optim.Adam(model_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_accuracy = 0
    best_epoch = 0
    best_state_dict = None
    cpc_history = []
    
    # Check if model returns per-slot loss
    actual_model = cpc_model.module if is_parallel else cpc_model
    return_per_slot = getattr(actual_model, 'return_per_slot_loss', False)
    
    if is_parallel:
        num_gpus = torch.cuda.device_count()
        print(f"Pre-training CPC on {num_gpus} GPU(s)")
        print(f"Batch size per GPU: {data_loader.batch_size}, Effective batch size: {data_loader.batch_size * num_gpus}")
    
    print(f"Training for {num_epochs} epochs...")
    if return_per_slot:
        print("Using per-slot CPC loss")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        pbar = tqdm(data_loader, desc=f'CPC Pre-train Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            inputs = batch[0].to(device).float()
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
            
            if inputs.size(0) <= 1:
                continue
            
            # Forward pass
            loss, accuracy = cpc_model(inputs)
            
            # Handle different loss shapes
            if is_parallel:
                loss = loss.mean()
                accuracy = accuracy.mean() if isinstance(accuracy, torch.Tensor) else accuracy
            
            # Per-slot loss returns (B, num_slots) - average over slots
            if loss.dim() > 0:
                loss = loss.mean()
            
            if isinstance(accuracy, torch.Tensor) and accuracy.dim() > 0:
                accuracy = accuracy.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            if is_parallel:
                torch.nn.utils.clip_grad_norm_(cpc_model.module.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(cpc_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            acc_val = accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy
            
            epoch_loss += loss_val
            epoch_accuracy += acc_val
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss_val:.4f}',
                'Acc': f'{acc_val:.4f}'
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'cpc_loss': float(avg_loss),
            'cpc_accuracy': float(avg_accuracy)
        }
        cpc_history.append(epoch_metrics)
        
        log_str = f'CPC Pre-train Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}'
        print(log_str)
        if out_file:
            out_file.write(log_str + '\n')
            out_file.flush()
        
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_epoch = epoch + 1
            
            model_to_save = cpc_model.module if is_parallel else cpc_model
            best_state_dict = {k: v.cpu().clone() for k, v in model_to_save.state_dict().items()}
            
            print(f'  → New best CPC accuracy: {best_accuracy:.4f}')
    
    print(f'CPC pre-training completed. Best accuracy: {best_accuracy:.4f} (Epoch {best_epoch})')
    
    cpc_metrics = {
        'best_cpc_accuracy': float(best_accuracy),
        'best_epoch': best_epoch,
        'training_history': cpc_history
    }
    
    return best_state_dict, cpc_metrics


def evaluate_cpc(cpc_model, data_loader, device):
    """
    Evaluate CPC model on a dataset.
    Supports both per-sample and per-slot CPC models.
    
    Args:
        cpc_model: CPCModel or CPCModelPerSlot instance (can be wrapped in DataParallel)
        data_loader: DataLoader
        device: torch.device
    
    Returns:
        avg_loss: Average InfoNCE loss
        avg_accuracy: Average contrastive accuracy
    """
    is_parallel = isinstance(cpc_model, nn.DataParallel)
    cpc_model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device).float()
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
            
            if inputs.size(0) <= 1:
                continue
            
            loss, accuracy = cpc_model(inputs)
            
            # Handle different loss shapes
            if is_parallel:
                loss = loss.mean()
                accuracy = accuracy.mean() if isinstance(accuracy, torch.Tensor) else accuracy
            
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            
            if isinstance(accuracy, torch.Tensor) and accuracy.dim() > 0:
                accuracy = accuracy.mean()
            
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            acc_val = accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy
            
            total_loss += loss_val
            total_accuracy += acc_val
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_accuracy


def compute_cpc_loss(cpc_model, inputs, return_per_slot=False):
    """
    Compute CPC loss for a batch during adaptation.
    Supports both per-sample and per-slot CPC models.
    
    Args:
        cpc_model: CPCModel or CPCModelPerSlot instance (can be wrapped in DataParallel)
        inputs: (B, T, F) or (B, num_slots, T, F_per_slot) or (B, T, F, C)
        return_per_slot: If True and model supports it, return per-slot losses
    
    Returns:
        loss: InfoNCE loss (scalar or (B, num_slots) if return_per_slot)
        accuracy: Contrastive accuracy (scalar)
    """
    is_parallel = isinstance(cpc_model, nn.DataParallel)
    
    # Temporarily set return_per_slot_loss if needed
    actual_model = cpc_model.module if is_parallel else cpc_model
    original_return_per_slot = getattr(actual_model, 'return_per_slot_loss', False)
    
    if return_per_slot and hasattr(actual_model, 'return_per_slot_loss'):
        actual_model.return_per_slot_loss = True
    
    loss, accuracy = cpc_model(inputs)
    
    # Restore original setting
    if hasattr(actual_model, 'return_per_slot_loss'):
        actual_model.return_per_slot_loss = original_return_per_slot
    
    # Handle DataParallel
    if is_parallel:
        if loss.dim() > 1:
            # Per-slot: (num_gpus, B, num_slots) -> mean over GPUs
            loss = loss.mean(dim=0)
        else:
            loss = loss.mean()
        accuracy = accuracy.mean() if isinstance(accuracy, torch.Tensor) else accuracy
    
    # Average per-slot loss to scalar if not returning per-slot
    if not return_per_slot and isinstance(loss, torch.Tensor) and loss.dim() > 0:
        loss = loss.mean()
    
    return loss, accuracy


def compute_cpc_loss_per_slot(cpc_model, inputs):
    """
    Compute per-slot CPC loss for use with Hungarian matching during adaptation.
    
    Args:
        cpc_model: CPCModelPerSlot instance
        inputs: (B, num_slots, T, F_per_slot) or auto-reshaped
    
    Returns:
        slot_losses: (num_slots,) - loss per slot
        accuracy: Scalar mean accuracy
    """
    return compute_cpc_loss(cpc_model, inputs, return_per_slot=True)


def extract_slot_features(cpc_model, inputs, flatten=True):
    """
    Extract per-slot features using trained CPC encoder.
    Useful for downstream tasks with Hungarian matching.
    
    Args:
        cpc_model: CPCModelPerSlot instance
        inputs: Input data
        flatten: If True, return (B, num_slots * hidden_dim), else (B, num_slots, hidden_dim)
    
    Returns:
        features: Per-slot representations
    """
    is_parallel = isinstance(cpc_model, nn.DataParallel)
    actual_model = cpc_model.module if is_parallel else cpc_model
    
    actual_model.eval()
    with torch.no_grad():
        if flatten:
            features = actual_model.encode_slots_flat(inputs)
        else:
            features = actual_model.encode_sequence(inputs)
    
    return features