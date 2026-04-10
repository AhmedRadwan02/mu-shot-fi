# cpc_utils.py - Utility functions for CPC training

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def pretrain_cpc(cpc_model, data_loader, device, num_epochs=50, lr=1e-3, out_file=None):
    """
    Pre-train CPC model on target domain (unsupervised).
    Uses windowing with random sampling for context/prediction.

    Args:
        cpc_model: CPCModel instance
        data_loader: DataLoader for target domain
        device: torch.device
        num_epochs: Number of pre-training epochs
        lr: Learning rate
        out_file: File handle for logging

    Returns:
        best_state_dict: Best model weights
        best_accuracy: Best contrastive accuracy
    """
    cpc_model.to(device)
    cpc_model.train()

    # Optimizer
    optimizer = optim.Adam(cpc_model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_accuracy = 0
    best_state_dict = None

    print(f"Pre-training CPC for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0

        pbar = tqdm(data_loader, desc=f'CPC Pre-train Epoch {epoch+1}/{num_epochs}')

        for batch in pbar:
            inputs = batch[0].to(device).float()
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)

            # Skip if batch too small
            if inputs.size(0) <= 1:
                continue

            # Forward pass
            loss, accuracy = cpc_model(inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cpc_model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy.item():.4f}'
            })

        # Update learning rate
        scheduler.step()

        # Average metrics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0

        log_str = f'CPC Pre-train Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}'
        print(log_str)
        if out_file:
            out_file.write(log_str + '\n')
            out_file.flush()

        # Save best model
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_state_dict = cpc_model.state_dict().copy()
            print(f'  → New best CPC accuracy: {best_accuracy:.4f}')

    print(f'CPC pre-training completed. Best accuracy: {best_accuracy:.4f}')

    return best_state_dict, best_accuracy


def evaluate_cpc(cpc_model, data_loader, device):
    """
    Evaluate CPC model on a dataset.
    Uses windowing with random sampling for context/prediction.

    Args:
        cpc_model: CPCModel instance
        data_loader: DataLoader
        device: torch.device

    Returns:
        avg_loss: Average InfoNCE loss
        avg_accuracy: Average contrastive accuracy
    """
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

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0

    return avg_loss, avg_accuracy


def compute_cpc_loss(cpc_model, inputs):
    """
    Compute CPC loss for a batch during adaptation.
    Uses windowing with random sampling for context/prediction.

    Args:
        cpc_model: CPCModel instance
        inputs: (B, T, F, C)

    Returns:
        loss: InfoNCE loss
        accuracy: Contrastive accuracy
    """
    loss, accuracy = cpc_model(inputs)
    return loss, accuracy
