# cpc_networks2.py — CPC with per-slot prediction for multi-user CSI (contrastive pre-training).
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CPCRandomMasking:
    """
    Random masking augmentation for CSI data before CPC encoding.
    Supports both per-sample and per-slot masking.
    """
    def __init__(self, mask_prob=0.5, mask_ratio=0.15, n_blocks=3,
                 min_ar=0.3, max_ar=3.0, mask_value=0.0):
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.n_blocks = n_blocks
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.mask_value = mask_value
    
    def __call__(self, x):
        """
        Apply random masking to input tensor.
        Args:
            x: Input tensor of shape (B, T, F) or (B, num_slots, T, F_per_slot)
        Returns:
            Masked tensor of same shape
        """
        if np.random.rand() > self.mask_prob:
            return x
        
        x_masked = x.clone()
        
        if x.dim() == 3:
            # (B, T, F) - original behavior
            B, T, F = x.shape
            total_elements = T * F
            area_per_block = max(1, int(round((self.mask_ratio * total_elements) / self.n_blocks)))
            
            for _ in range(self.n_blocks):
                ar = float(np.random.uniform(self.min_ar, self.max_ar))
                h = int(max(1, round(np.sqrt(area_per_block / ar))))
                w = int(max(1, round(h * ar)))
                h, w = min(h, T), min(w, F)
                t0 = 0 if T == h else np.random.randint(0, T - h + 1)
                f0 = 0 if F == w else np.random.randint(0, F - w + 1)
                x_masked[:, t0:t0+h, f0:f0+w] = self.mask_value
                
        elif x.dim() == 4:
            # (B, num_slots, T, F_per_slot) - per-slot masking
            B, num_slots, T, F = x.shape
            total_elements = T * F
            area_per_block = max(1, int(round((self.mask_ratio * total_elements) / self.n_blocks)))
            
            # Apply masking independently per slot (different masks)
            for s in range(num_slots):
                for _ in range(self.n_blocks):
                    ar = float(np.random.uniform(self.min_ar, self.max_ar))
                    h = int(max(1, round(np.sqrt(area_per_block / ar))))
                    w = int(max(1, round(h * ar)))
                    h, w = min(h, T), min(w, F)
                    t0 = 0 if T == h else np.random.randint(0, T - h + 1)
                    f0 = 0 if F == w else np.random.randint(0, F - w + 1)
                    x_masked[:, s, t0:t0+h, f0:f0+w] = self.mask_value
        
        return x_masked


class CPCEncoderPerSlot(nn.Module):
    """
    2D CNN encoder for per-slot windowed CSI encoding.
    Shared weights across all slots - processes each slot independently.
    
    Input: (B, num_slots, T, F_per_slot) or (B, T, F) with reshaping
    Output: (B, num_slots, num_windows, embedding_dim)
    """
    def __init__(self, input_freq, embedding_dim=256, window_size=30):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_freq = input_freq  # F_per_slot (features per slot)
        self.window_size = window_size
        
        # 2D CNN to process windowed temporal-frequency structure
        # Input shape per window: (B*num_slots, 1, F_per_slot, window_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=1, padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.projection = nn.Linear(512, embedding_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, num_slots, T, F_per_slot)
        Returns:
            embeddings: (B, num_slots, num_windows, embedding_dim)
        """
        B, num_slots, T, F_dim = x.shape
        
        if F_dim != self.input_freq:
            raise ValueError(f"Expected feature dimension {self.input_freq}, got {F_dim}")
        
        num_windows = T // self.window_size
        T_truncated = num_windows * self.window_size
        x = x[:, :, :T_truncated, :].contiguous()  # (B, num_slots, T_truncated, F_per_slot)
        
        # Reshape for batch processing: (B * num_slots, T_truncated, F_per_slot)
        x = x.reshape(B * num_slots, T_truncated, F_dim)
        
        # Reshape into windows: (B * num_slots, num_windows, window_size, F_per_slot)
        x = x.reshape(B * num_slots, num_windows, self.window_size, F_dim)
        
        # Process each window
        embeddings = []
        for w in range(num_windows):
            window = x[:, w, :, :]  # (B * num_slots, window_size, F_per_slot)
            window = window.permute(0, 2, 1)  # (B * num_slots, F_per_slot, window_size)
            window = window.unsqueeze(1)  # (B * num_slots, 1, F_per_slot, window_size)
            
            feat = self.encoder(window)  # (B * num_slots, 512, 1, 1)
            feat = feat.squeeze(-1).squeeze(-1)  # (B * num_slots, 512)
            emb = self.projection(feat)  # (B * num_slots, embedding_dim)
            embeddings.append(emb)
        
        # Stack: (B * num_slots, num_windows, embedding_dim)
        embeddings = torch.stack(embeddings, dim=1)
        
        # Reshape back: (B, num_slots, num_windows, embedding_dim)
        embeddings = embeddings.reshape(B, num_slots, num_windows, self.embedding_dim)
        
        return embeddings


class CPCContextNetworkPerSlot(nn.Module):
    """
    Autoregressive context network - shared across slots.
    Processes each slot's sequence independently with shared GRU weights.
    
    Input: (B, num_slots, T_context, embedding_dim)
    Output: (B, num_slots, hidden_dim)
    """
    def __init__(self, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (B, num_slots, T_context, embedding_dim)
        Returns:
            c_t: (B, num_slots, hidden_dim)
        """
        B, num_slots, T_context, emb_dim = embeddings.shape
        
        # Reshape: (B * num_slots, T_context, embedding_dim)
        embeddings_flat = embeddings.reshape(B * num_slots, T_context, emb_dim)
        
        # Process through shared GRU
        output, h_n = self.gru(embeddings_flat)
        
        # Get final context: (B * num_slots, hidden_dim)
        c_t = h_n[-1]
        
        # Reshape back: (B, num_slots, hidden_dim)
        c_t = c_t.reshape(B, num_slots, self.hidden_dim)
        
        return c_t


class CPCModelPerSlot(nn.Module):
    """
    CPC model with per-slot prediction for multi-user/activity recognition.
    
    Key differences from original CPCModel:
    - Input is (B, num_slots, T, F_per_slot) or auto-reshaped from (B, T, F)
    - Computes separate CPC loss for each slot
    - Encoder and context network weights are SHARED across slots
    - Returns per-slot losses that can be used with Hungarian matching
    
    Still fully SSL - no labels used!
    """
    def __init__(self,
                 input_freq,          # F_per_slot: features per slot
                 num_slots=None,      # If None, inferred from input or must reshape externally
                 embedding_dim=256,
                 hidden_dim=512,
                 projection_dim=256,
                 num_gru_layers=2,
                 temperature=0.07,
                 window_size=30,
                 prediction_steps=9,
                 use_masking=False,
                 mask_prob=0.5,
                 mask_ratio=0.15,
                 negative_mode='cross_batch',  # 'cross_batch', 'cross_slot', 'both'
                 return_per_slot_loss=False):  # If True, return (B, num_slots) losses
        super().__init__()
        
        self.input_freq = input_freq
        self.num_slots = num_slots
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.use_masking = use_masking
        self.negative_mode = negative_mode
        self.return_per_slot_loss = return_per_slot_loss
        
        # Shared encoder across slots
        self.encoder = CPCEncoderPerSlot(
            input_freq=input_freq,
            embedding_dim=embedding_dim,
            window_size=window_size
        )
        
        # Shared context network across slots
        self.context_net = CPCContextNetworkPerSlot(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gru_layers
        )
        
        # Shared projection head
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Shared prediction heads W_k
        self.W_k = nn.ModuleList([
            nn.Linear(hidden_dim, projection_dim, bias=False)
            for _ in range(prediction_steps)
        ])
        
        # Optional masking
        if use_masking:
            self.masking = CPCRandomMasking(mask_prob=mask_prob, mask_ratio=mask_ratio)
        else:
            self.masking = None
    
    def reshape_input(self, inputs):
        """
        Reshape input to (B, num_slots, T, F_per_slot) if needed.
        
        Handles:
        - (B, num_slots, T, F_per_slot): Already correct, pass through
        - (B, T, F): Reshape assuming F = num_slots * F_per_slot
        - (B, T, F, C): Treat C as num_slots, F as F_per_slot
        """
        if inputs.dim() == 4:
            # Could be (B, num_slots, T, F_per_slot) or (B, T, F, C)
            if inputs.shape[3] == self.input_freq:
                # (B, num_slots, T, F_per_slot) - already correct
                return inputs.contiguous()
            elif inputs.shape[2] == self.input_freq:
                # (B, T, F_per_slot, num_slots) - need to permute
                return inputs.permute(0, 3, 1, 2).contiguous()  # -> (B, num_slots, T, F_per_slot)
            else:
                # (B, T, F, C) - treat last dim as slots
                # Reshape to (B, C, T, F)
                return inputs.permute(0, 3, 1, 2).contiguous()
        
        elif inputs.dim() == 3:
            # (B, T, F) - need to split F into slots
            B, T, F = inputs.shape
            
            if self.num_slots is None:
                raise ValueError(
                    "Input is (B, T, F) but num_slots not specified. "
                    "Either provide num_slots in __init__ or input (B, num_slots, T, F_per_slot)"
                )
            
            F_per_slot = F // self.num_slots
            if F_per_slot != self.input_freq:
                raise ValueError(
                    f"F={F} / num_slots={self.num_slots} = {F_per_slot}, "
                    f"but input_freq={self.input_freq}"
                )
            
            # Reshape: (B, T, num_slots, F_per_slot) -> (B, num_slots, T, F_per_slot)
            inputs = inputs.reshape(B, T, self.num_slots, F_per_slot)
            inputs = inputs.permute(0, 2, 1, 3).contiguous()
            return inputs
        
        else:
            raise ValueError(f"Expected 3D or 4D input, got {inputs.dim()}D")
    
    def forward(self, inputs):
        """
        Per-slot CPC forward pass.
        
        Args:
            inputs: (B, num_slots, T, F_per_slot) or (B, T, F) or (B, T, F, C)
        
        Returns:
            loss: Scalar (mean across slots) or (B, num_slots) if return_per_slot_loss
            accuracy: Scalar (mean across slots and steps)
        """
        # Reshape to (B, num_slots, T, F_per_slot)
        inputs = self.reshape_input(inputs)
        B, num_slots, T, F_dim = inputs.shape
        
        # Apply masking if enabled
        if self.masking is not None and self.training:
            inputs = self.masking(inputs)
        
        # 1. Encode all slots and windows (shared encoder)
        # Output: (B, num_slots, num_windows, embedding_dim)
        window_embeddings = self.encoder(inputs)
        num_windows = window_embeddings.shape[2]
        
        # 2. Sample context endpoint (same for all slots for consistency)
        if self.training:
            max_t = num_windows - self.prediction_steps - 1
            t_samples = torch.randint(0, max(1, max_t), size=(1,)).item() if max_t >= 1 else 0
        else:
            t_samples = max(0, int(num_windows * 0.75) - self.prediction_steps - 1)
        
        # 3. Extract future windows as targets for each slot
        # Shape: (K, B, num_slots, projection_dim)
        encode_samples = []
        for i in range(1, self.prediction_steps + 1):
            future_idx = t_samples + i
            if future_idx >= num_windows:
                break
            # (B, num_slots, embedding_dim)
            future_window = window_embeddings[:, :, future_idx, :]
            # Flatten for projection: (B * num_slots, embedding_dim)
            future_flat = future_window.reshape(B * num_slots, -1)
            z_future = self.projection(future_flat)  # (B * num_slots, projection_dim)
            z_future = z_future.reshape(B, num_slots, -1)  # (B, num_slots, projection_dim)
            encode_samples.append(z_future)
        
        encode_samples = torch.stack(encode_samples, dim=0)  # (K, B, num_slots, projection_dim)
        actual_K = encode_samples.shape[0]
        
        # 4. Get context for each slot (shared GRU)
        # (B, num_slots, t_samples+1, embedding_dim)
        context_embeddings = window_embeddings[:, :, :t_samples+1, :].contiguous()
        c_t = self.context_net(context_embeddings)  # (B, num_slots, hidden_dim)
        
        # 5. Predict future windows for each slot
        # Shape: (K, B, num_slots, projection_dim)
        predictions = []
        for i in range(actual_K):
            # (B * num_slots, hidden_dim)
            c_t_flat = c_t.reshape(B * num_slots, -1)
            pred = self.W_k[i](c_t_flat)  # (B * num_slots, projection_dim)
            pred = pred.reshape(B, num_slots, -1)  # (B, num_slots, projection_dim)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (K, B, num_slots, projection_dim)
        
        # 6. Compute per-slot InfoNCE loss
        slot_losses = []
        slot_accuracies = []
        
        for s in range(num_slots):
            slot_loss = 0
            slot_acc = 0
            
            for k in range(actual_K):
                pred_k_s = predictions[k, :, s, :]  # (B, projection_dim)
                target_k_s = encode_samples[k, :, s, :]  # (B, projection_dim)
                
                if self.negative_mode == 'cross_batch':
                    # Standard: negatives from other samples in batch
                    scores = torch.matmul(pred_k_s, target_k_s.T) / self.temperature  # (B, B)
                    labels = torch.arange(B, device=inputs.device)
                    
                elif self.negative_mode == 'cross_slot':
                    # Negatives from other slots in same sample
                    # (B, num_slots, projection_dim)
                    all_targets = encode_samples[k]
                    # For each sample, compute similarity to all slots
                    # pred_k_s: (B, projection_dim), all_targets: (B, num_slots, projection_dim)
                    scores = torch.einsum('bd,bnd->bn', pred_k_s, all_targets) / self.temperature  # (B, num_slots)
                    labels = torch.full((B,), s, device=inputs.device)
                    
                elif self.negative_mode == 'both':
                    # Combine cross-batch and cross-slot negatives
                    # Cross-batch: (B, B)
                    cross_batch = torch.matmul(pred_k_s, target_k_s.T) / self.temperature
                    
                    # Cross-slot for same sample: (B, num_slots-1)
                    other_slots = [encode_samples[k, :, other_s, :] for other_s in range(num_slots) if other_s != s]
                    if other_slots:
                        other_slots = torch.stack(other_slots, dim=1)  # (B, num_slots-1, projection_dim)
                        cross_slot = torch.einsum('bd,bnd->bn', pred_k_s, other_slots) / self.temperature  # (B, num_slots-1)
                        
                        # Combine: positive is diagonal of cross_batch, negatives are off-diagonal + cross_slot
                        # For simplicity, use cross_batch loss + auxiliary cross_slot loss
                        labels = torch.arange(B, device=inputs.device)
                        loss_batch = F.cross_entropy(cross_batch, labels)
                        
                        # Cross-slot: predict that slot s is different from other slots
                        # Use margin or just add as regularization
                        # Here we just use cross-batch as primary
                        scores = cross_batch
                    else:
                        scores = cross_batch
                        labels = torch.arange(B, device=inputs.device)
                else:
                    raise ValueError(f"Unknown negative_mode: {self.negative_mode}")
                
                loss_k = F.cross_entropy(scores, labels)
                slot_loss += loss_k
                
                with torch.no_grad():
                    preds = scores.argmax(dim=1)
                    acc_k = (preds == labels).float().mean()
                    slot_acc += acc_k
            
            slot_losses.append(slot_loss / actual_K)
            slot_accuracies.append(slot_acc / actual_K)
        
        # Stack: (num_slots,)
        slot_losses = torch.stack(slot_losses)
        slot_accuracies = torch.stack(slot_accuracies)
        
        if self.return_per_slot_loss:
            # Return per-slot losses for potential use with Hungarian matching
            # Expand to (B, num_slots) by repeating
            loss = slot_losses.unsqueeze(0).expand(B, -1)
            accuracy = slot_accuracies.mean()
        else:
            # Return scalar mean
            loss = slot_losses.mean()
            accuracy = slot_accuracies.mean()
        
        return loss, accuracy
    
    def encode_sequence(self, inputs):
        """
        Encode a sequence and return per-slot context representations.
        
        Args:
            inputs: (B, num_slots, T, F_per_slot) or (B, T, F)
        
        Returns:
            c_t: (B, num_slots, hidden_dim) - per-slot context representations
        """
        inputs = self.reshape_input(inputs)
        window_embeddings = self.encoder(inputs)
        c_t = self.context_net(window_embeddings)
        return c_t
    
    def encode_slots_flat(self, inputs):
        """
        Encode and return flattened representation for downstream tasks.
        
        Args:
            inputs: (B, num_slots, T, F_per_slot) or (B, T, F)
        
        Returns:
            features: (B, num_slots * hidden_dim) - flattened slot representations
        """
        c_t = self.encode_sequence(inputs)  # (B, num_slots, hidden_dim)
        return c_t.reshape(c_t.shape[0], -1)


# Backward compatible alias
CPCModel = CPCModelPerSlot


def init_weights(m):
    """Initialize network weights"""
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)