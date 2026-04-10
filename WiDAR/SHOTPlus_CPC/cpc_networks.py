# cpc_networks.py - Contrastive Predictive Coding Networks for WiFi CSI

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CPCRandomMasking:
    """
    Random masking augmentation for CSI data before CPC encoding.

    Masks the phase channel with rectangular patches to encourage
    robustness in the learned representations.
    """
    def __init__(self, mask_prob=0.5, mask_ratio=0.15, n_blocks=3,
                 min_ar=0.3, max_ar=3.0, mask_value=0.0, mask_phase_only=True):
        """
        Args:
            mask_prob: Probability of applying masking (0 to 1)
            mask_ratio: Ratio of data to mask (0 to 1)
            n_blocks: Number of rectangular blocks to mask
            min_ar: Minimum aspect ratio of mask blocks
            max_ar: Maximum aspect ratio of mask blocks
            mask_value: Value to use for masked regions
            mask_phase_only: If True and C>1, only mask phase channel (assumes phase is last channel)
        """
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.n_blocks = n_blocks
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.mask_value = mask_value
        self.mask_phase_only = mask_phase_only

    def __call__(self, x):
        """
        Apply random masking to input tensor.

        Args:
            x: Input tensor of shape (B, T, F, C) where:
               B = batch size
               T = time steps
               F = frequency features (180)
               C = channels (1 or 2, where C=2 means [amplitude, phase])

        Returns:
            Masked tensor of same shape
        """
        # Only apply with probability mask_prob
        if np.random.rand() > self.mask_prob:
            return x

        B, T, F, C = x.shape
        x_masked = x.clone()

        # Determine which channel to mask
        if self.mask_phase_only and C > 1:
            # Mask only phase channel (last channel)
            channel_idx = C - 1
            total_elements = T * F
        else:
            # Mask all channels
            channel_idx = None
            total_elements = T * F

        # Calculate area to mask per block
        area_per_block = max(1, int(round((self.mask_ratio * total_elements) / self.n_blocks)))

        # Apply n_blocks masks
        for _ in range(self.n_blocks):
            # Random aspect ratio
            ar = float(np.random.uniform(self.min_ar, self.max_ar))

            # Calculate block dimensions
            h = int(max(1, round(np.sqrt(area_per_block / ar))))  # time dimension
            w = int(max(1, round(h * ar)))  # frequency dimension

            # Clip to valid range
            h = min(h, T)
            w = min(w, F)

            # Random position
            t0 = 0 if T == h else np.random.randint(0, T - h + 1)
            f0 = 0 if F == w else np.random.randint(0, F - w + 1)

            # Apply mask
            if channel_idx is not None:
                # Mask only specific channel
                x_masked[:, t0:t0+h, f0:f0+w, channel_idx] = self.mask_value
            else:
                # Mask all channels
                x_masked[:, t0:t0+h, f0:f0+w, :] = self.mask_value

        return x_masked


class CPCEncoder(nn.Module):
    """
    2D CNN encoder for windowed CSI encoding.

    Uses windowing approach: splits T timesteps into windows of size W.
    Processes each window with for loop (like GitHub CPC processes timesteps).
    Each window format: (B, C=1, F=180, T=window_size) - same as source format.
    Input: (B, T, F=180, C)
    Output: (B, num_windows, embedding_dim)
    """
    def __init__(self, input_freq=180, embedding_dim=256, num_tx=6, num_subcarrier=30, window_size=10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_freq = input_freq
        self.window_size = window_size

        # 2D CNN to process windowed temporal-frequency structure
        # Input shape per window: (B, C=1, F=180, T=window_size)
        # This matches source format: (B, 1, 180, T) where T is time dimension
        self.encoder = nn.Sequential(
            # Layer 1: Frequency-temporal convolution
            # (180, 10) -> (180, 10) with stride 1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 3), stride=1, padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 2: Reduce dimensions
            # (180, 10) -> (90, 5)
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: Further processing
            # (90, 5) -> (90, 5)
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 4: Final compression
            # (90, 5) -> (88, 3)
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Global average pooling: (88, 3) -> (1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Final projection to embedding_dim
        self.projection = nn.Linear(512, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, T, F=180, C) for windowed sequence encoding

        Returns:
            embeddings: (B, num_windows, embedding_dim)
        """
        B, T, F_dim, C = x.shape

        # Calculate number of windows
        num_windows = T // self.window_size

        # Truncate to fit window size exactly
        T_truncated = num_windows * self.window_size
        x = x[:, :T_truncated, :, :]  # (B, T_truncated, F, C)

        # Reshape into windows: (B, num_windows, window_size, F, C)
        x = x.view(B, num_windows, self.window_size, F_dim, C)

        # Average over channels if C > 1, keep dimension for consistency
        # This handles both C=1 (phase only) and C=2 (amplitude + phase)
        if C > 1:
            x = x.mean(dim=-1, keepdim=True)  # (B, num_windows, window_size, F, 1)
        # x shape: (B, num_windows, window_size, F, 1) for both cases

        # Process each window with a for loop (like GitHub CPC)
        embeddings = []
        for w in range(num_windows):
            window = x[:, w, :, :, :]  # (B, window_size, F, 1)

            # Transpose to match source format: (B, F, T)
            # window: (B, window_size, F, 1) -> (B, F, window_size, 1)
            window = window.permute(0, 2, 1, 3)  # (B, F=180, T=window_size, C=1)

            # Permute to Conv2D format: (B, C, F, T)
            window = window.permute(0, 3, 1, 2)  # (B, 1, 180, 10)

            # Encode this window
            feat = self.encoder(window)  # (B, 512, 1, 1)
            feat = feat.squeeze(-1).squeeze(-1)  # (B, 512)
            emb = self.projection(feat)  # (B, embedding_dim)

            embeddings.append(emb)

        # Stack all window embeddings: (B, num_windows, embedding_dim)
        embeddings = torch.stack(embeddings, dim=1)

        return embeddings


class CPCContextNetwork(nn.Module):
    """
    Autoregressive context network using GRU.

    Summarizes historical context from encoded timesteps.
    Input: (B, T_context, embedding_dim)
    Output: c_t (B, hidden_dim) - context representation
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
            embeddings: (B, T_context, embedding_dim)

        Returns:
            c_t: (B, hidden_dim) - final context representation
        """
        # GRU processes sequence autoregressively
        output, h_n = self.gru(embeddings)
        # output: (B, T, hidden_dim) - all hidden states
        # h_n: (num_layers, B, hidden_dim) - final hidden state per layer

        # Use final hidden state from last layer as context
        c_t = h_n[-1]  # (B, hidden_dim)

        return c_t


class CPCModel(nn.Module):
    """
    Complete CPC model with windowing and multi-step prediction.
    Uses window-based encoding and predicts K future windows.
    """
    def __init__(self,
                 input_freq=180,
                 embedding_dim=256,
                 hidden_dim=512,
                 projection_dim=256,
                 num_gru_layers=2,
                 temperature=0.07,
                 num_tx=6,
                 num_subcarrier=30,
                 window_size=10,
                 prediction_steps=9,
                 use_masking=False,
                 mask_prob=0.5,
                 mask_ratio=0.15):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.use_masking = use_masking

        # Encoder: CSI -> window embeddings
        self.encoder = CPCEncoder(
            input_freq=input_freq,
            embedding_dim=embedding_dim,
            num_tx=num_tx,
            num_subcarrier=num_subcarrier,
            window_size=window_size
        )

        # Context network: embeddings -> c_t
        self.context_net = CPCContextNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gru_layers
        )

        # Projection head for target windows
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Multiple prediction heads W_k for each of K future windows
        self.W_k = nn.ModuleList([
            nn.Linear(hidden_dim, projection_dim, bias=False)
            for _ in range(prediction_steps)
        ])

        # Optional random masking augmentation
        if use_masking:
            self.masking = CPCRandomMasking(
                mask_prob=mask_prob,
                mask_ratio=mask_ratio,
                mask_phase_only=True
            )
        else:
            self.masking = None

    def forward(self, inputs):
        """
        Complete CPC forward pass with random window sampling and multi-step prediction.
        Follows the GitHub CPC pattern with windowing instead of timesteps.

        Args:
            inputs: (B, T=1200, F=180, C) where C=1 or 2

        Returns:
            loss: InfoNCE contrastive loss (averaged over K prediction steps)
            accuracy: Fraction of correct positive identifications (averaged over K steps)
        """
        B, T, F_dim, C = inputs.shape

        # Apply random masking augmentation if enabled (during training)
        if self.masking is not None and self.training:
            inputs = self.masking(inputs)

        # 1. Encode all windows (using for loop like GitHub CPC encodes timesteps)
        # This is 'f' in GitHub code: f = f.view(batch_size, sequence_length, feature_dim)
        window_embeddings = self.encoder(inputs)  # (B, num_windows, embedding_dim)
        num_windows = window_embeddings.shape[1]

        # 2. Randomly sample context endpoint (like GitHub CPC's t_samples)
        # GitHub: t_samples = torch.randint(sequence_length - timestep, size=(1,))
        if self.training:
            max_t = num_windows - self.prediction_steps - 1
            if max_t < 1:
                t_samples = 0
            else:
                t_samples = torch.randint(0, max_t, size=(1,)).item()
        else:
            t_samples = int(num_windows * 0.75) - self.prediction_steps - 1
            t_samples = max(0, t_samples)

        # 3. Extract future K windows as targets (like GitHub's encode_samples)
        # GitHub: encode_samples[i-1] = f[:, t_samples+i, :]
        encode_samples = []
        for i in range(1, self.prediction_steps + 1):
            future_idx = t_samples + i
            if future_idx >= num_windows:
                break
            future_window = window_embeddings[:, future_idx, :]  # (B, embedding_dim)
            z_future = self.projection(future_window)  # (B, projection_dim)
            encode_samples.append(z_future)

        # Stack: (K, B, projection_dim) - matches GitHub pattern
        encode_samples = torch.stack(encode_samples, dim=0)
        actual_K = encode_samples.shape[0]

        # 4. Get context representation using GRU (like GitHub's autoregressive_model)
        # GitHub: forward_seq = f[:, :t_samples+1, :]
        #         output, _ = autoregressive_model(forward_seq)
        #         c_t = output[:, t_samples, :]
        context_embeddings = window_embeddings[:, :t_samples+1, :]  # (B, t_samples+1, embedding_dim)
        c_t = self.context_net(context_embeddings)  # (B, hidden_dim)

        # 5. Predict future windows using W_k heads (like GitHub's pred)
        # GitHub: for i in range(timestep):
        #             pred[i] = Wk[i](c_t)
        predictions = []
        for i in range(actual_K):
            pred = self.W_k[i](c_t)  # (B, projection_dim)
            predictions.append(pred)

        # Stack: (K, B, projection_dim)
        predictions = torch.stack(predictions, dim=0)

        # 6. Compute InfoNCE loss for each prediction step
        total_loss = 0
        total_accuracy = 0
        labels = torch.arange(B, device=inputs.device)

        for i in range(actual_K):
            pred_i = predictions[i]  # (B, projection_dim)
            target_i = encode_samples[i]  # (B, projection_dim)

            # Compute similarity scores
            scores = torch.matmul(pred_i, target_i.T) / self.temperature  # (B, B)

            # InfoNCE loss
            loss_i = F.cross_entropy(scores, labels)
            total_loss += loss_i

            # Accuracy
            with torch.no_grad():
                preds = scores.argmax(dim=1)
                acc_i = (preds == labels).float().mean()
                total_accuracy += acc_i

        # Average loss and accuracy over K steps
        loss = total_loss / actual_K
        accuracy = total_accuracy / actual_K

        return loss, accuracy

    def encode_sequence(self, inputs):
        """
        Encode a sequence and return context representation.
        Useful for feature extraction after pre-training.

        Args:
            inputs: (B, T, F, C)

        Returns:
            c_t: (B, hidden_dim) - context representation
        """
        # Encode all windows
        window_embeddings = self.encoder(inputs)  # (B, num_windows, embedding_dim)

        # Get context from all windows
        c_t = self.context_net(window_embeddings)  # (B, hidden_dim)

        return c_t


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
