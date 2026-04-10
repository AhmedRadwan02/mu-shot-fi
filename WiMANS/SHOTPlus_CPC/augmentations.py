"""
[file]          augmentations.py
[description]   Generic CSI data augmentations and AugMix implementation
                Works with both amplitude and phase CSI data
"""
import numpy as np
import torch

def signal_scale(csi_data, severity):
    """Stronger scaling for CSI data"""
    # Increase from 0.3 to 1.0 (±100% at severity=10)
    scale_factor = 1.0 + (severity / 10.0) * np.random.uniform(-1.0, 1.0)
    return csi_data * scale_factor


def additive_noise(csi_data, severity):
    """Stronger noise for CSI data"""
    # Increase from 0.1 to 0.5 (50% of std at severity=10)
    noise_std = (severity / 10.0) * 0.5 * np.std(csi_data)
    noise = np.random.normal(0, noise_std, csi_data.shape)
    return csi_data + noise


def feature_dropout(csi_data, severity):
    """Stronger dropout for CSI data"""
    # Increase from 15% to 40% max dropout
    dropout_prob = (severity / 10.0) * 0.4
    mask = np.random.rand(csi_data.shape[1]) > dropout_prob
    masked_data = csi_data.copy()
    masked_data[:, ~mask] = 0
    return masked_data


def temporal_shift(csi_data, severity):
    """Stronger temporal shift for CSI data"""
    # Increase from 5% to 15% max shift
    max_shift = int((severity / 10.0) * csi_data.shape[0] * 0.15)
    shift_amount = np.random.randint(-max_shift, max_shift + 1)
    
    if shift_amount > 0:
        shifted = np.concatenate([np.zeros((shift_amount, csi_data.shape[1])), 
                                  csi_data[:-shift_amount]], axis=0)
    elif shift_amount < 0:
        shifted = np.concatenate([csi_data[-shift_amount:], 
                                  np.zeros((-shift_amount, csi_data.shape[1]))], axis=0)
    else:
        shifted = csi_data.copy()
    
    return shifted


def multipath_fading(csi_data, severity):
    """Stronger multipath fading"""
    num_paths = np.random.randint(1, min(6, severity + 1))  # More paths
    faded = csi_data.copy()
    
    for _ in range(num_paths):
        delay = np.random.randint(1, 20)  # Longer delays
        attenuation = np.random.uniform(0.2, 0.8)  # Stronger attenuation
        
        delayed_signal = np.zeros_like(csi_data)
        if delay < csi_data.shape[0]:
            delayed_signal[delay:] = csi_data[:-delay] * attenuation
            faded += delayed_signal
    
    return faded


def path_loss(csi_data, severity):
    """Stronger path loss"""
    path_loss_exp = np.random.uniform(2.0, 4.0)
    # Increase distance variation from 0.5 to 2.0
    distance_factor = 1.0 + (severity / 10.0) * np.random.uniform(-2.0, 2.0)
    distance_factor = max(0.1, distance_factor)  # Avoid negative
    
    attenuation = distance_factor ** (-path_loss_exp / 2)
    return csi_data * attenuation


def frequency_selective_fading(csi_data, severity):
    """Stronger frequency-selective fading"""
    num_features = csi_data.shape[1]
    num_notches = np.random.randint(2, max(3, severity // 2 + 2))  # More notches
    freq_response = np.ones(num_features)
    
    for _ in range(num_notches):
        center = np.random.randint(0, num_features)
        width = np.random.randint(5, min(30, num_features // 3))  # Wider notches
        attenuation = np.random.uniform(0.1, 0.7)  # Stronger attenuation
        
        for i in range(num_features):
            distance = abs(i - center)
            if distance < width:
                freq_response[i] *= (attenuation + (1 - attenuation) * (distance / width))
    
    faded = csi_data * freq_response[np.newaxis, :]
    return faded


def environmental_interference(csi_data, severity):
    """Stronger environmental interference"""
    # Increase narrowband from 0.2 to 0.5
    narrowband_power = (severity / 10.0) * 0.5
    narrowband_freq = np.random.randint(1, 8)
    t = np.arange(csi_data.shape[0])
    interference = narrowband_power * np.sin(2 * np.pi * narrowband_freq * t / csi_data.shape[0])
    interference = interference[:, np.newaxis] * np.ones((1, csi_data.shape[1]))
    
    # Increase broadband from 0.05 to 0.15
    broadband = np.random.randn(*csi_data.shape) * np.std(csi_data) * 0.15
    
    return csi_data + interference + broadband


def doppler_shift(csi_data, severity):
    """
    Simulate Doppler effects from user movement
    Different rooms = different movement patterns/velocities
    
    Args:
        csi_data: numpy array of shape (time_steps, features)
        severity: int (1-10), controls Doppler frequency
    
    Returns:
        Augmented CSI data with Doppler effects
    """
    max_doppler = (severity / 10.0) * 5  # Hz
    doppler_freq = np.random.uniform(-max_doppler, max_doppler)
    
    # Create phase ramp
    t = np.arange(csi_data.shape[0])
    phase_shift = 2 * np.pi * doppler_freq * t / csi_data.shape[0]
    
    # Apply as amplitude modulation (simplified)
    modulation = 1 + 0.1 * np.sin(phase_shift)
    
    return csi_data * modulation[:, np.newaxis]


# ============================================================
# AUGMENTATION OPERATION REGISTRY
# Generic names that work for both amplitude and phase
# ============================================================

AUGMENTATION_OPS = {
    # Basic augmentations
    "signal_scale": signal_scale,
    "additive_noise": additive_noise,
    "feature_dropout": feature_dropout,
    "temporal_shift": temporal_shift,
    
    # Physics-based augmentations for environmental domain shift
    "multipath_fading": multipath_fading,
    "path_loss": path_loss,
    "frequency_selective_fading": frequency_selective_fading,
    "environmental_interference": environmental_interference,
    "doppler_shift": doppler_shift,
}


# ============================================================
# AUGMIX IMPLEMENTATION
# ============================================================

def augment_and_mix(csi_data, severity, width, depth, alpha, augmentation_ops):
    """
    Apply AugMix to a single CSI sample
    
    Args:
        csi_data: numpy array of shape (time_steps, features)
        severity: Augmentation severity (1-10)
        width: Number of augmentation chains
        depth: Depth of each chain (-1 for random 1-3)
        alpha: Dirichlet/Beta distribution parameter
        augmentation_ops: List of augmentation operation names
    
    Returns:
        Mixed augmented CSI data
    """
    # Convert op names to functions
    ops = [AUGMENTATION_OPS[op_name] for op_name in augmentation_ops]
    
    # Sample mixing weights
    ws = np.random.dirichlet([alpha] * width)
    m = np.random.beta(alpha, alpha)
    
    mix = np.zeros_like(csi_data)
    
    # Create augmentation chains
    for i in range(width):
        # Determine chain depth
        chain_depth = depth if depth > 0 else np.random.randint(1, 4)
        
        # Start with original data
        aug_data = csi_data.copy()
        
        # Apply augmentation chain
        for _ in range(chain_depth):
            # Randomly select operation
            op = np.random.choice(ops)
            # Apply with random severity
            severity_i = np.random.randint(1, severity + 1)
            aug_data = op(aug_data, severity_i)
        
        # Add to mixture
        mix += ws[i] * aug_data
    
    # Final mixing with original
    mixed = (1 - m) * csi_data + m * mix
    
    return mixed


def apply_augmix_batch(batch_data, config):
    """
    Apply AugMix to a batch of CSI data
    
    Args:
        batch_data: numpy array of shape (batch_size, time_steps, features)
        config: Dictionary with AugMix parameters
                {
                    "use_augmix": bool,
                    "augmix_severity": int,
                    "augmix_width": int,
                    "augmix_depth": int,
                    "augmix_alpha": float,
                    "augmix_ops": list of str
                }
    
    Returns:
        Augmented batch data (same shape)
    """
    if not config.get("use_augmix", False):
        return batch_data
    
    batch_size = batch_data.shape[0]
    augmented_batch = np.zeros_like(batch_data)
    
    for i in range(batch_size):
        augmented_batch[i] = augment_and_mix(
            batch_data[i],
            severity=config["augmix_severity"],
            width=config["augmix_width"],
            depth=config["augmix_depth"],
            alpha=config["augmix_alpha"],
            augmentation_ops=config["augmix_ops"]
        )
    
    return augmented_batch


# ============================================================
# PYTORCH DATALOADER INTEGRATION
# ============================================================

class AugMixDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies AugMix on-the-fly during training
    Returns BOTH clean and augmented samples for JSD loss
    Works with both amplitude and phase CSI data
    """
    def __init__(self, data_x, data_y, config):
        """
        Args:
            data_x: numpy array of shape (num_samples, time_steps, features)
            data_y: numpy array of labels
            config: AugMix configuration dict
        """
        self.data_x = data_x
        self.data_y = data_y
        self.config = config
        self.use_augmix = config.get("use_augmix", False)
    
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        x_clean = self.data_x[idx]
        y = self.data_y[idx]

        x_aug = augment_and_mix(
            x_clean,
            severity=self.config["augmix_severity"],
            width=self.config["augmix_width"],
            depth=self.config["augmix_depth"],
            alpha=self.config["augmix_alpha"],
            augmentation_ops=self.config["augmix_ops"]
        )
        # Return: augmented, label
        # NOTE: y is now indices [6] as int64, not one-hot
        return (torch.FloatTensor(x_aug),
                torch.LongTensor(y))


# ============================================================
# TESTING FUNCTIONS
# ============================================================

def test_augmentations():
    """Test individual augmentation operations"""
    print("Testing CSI Augmentations...")
    print("(Works for both amplitude and phase data)\n")
    
    # Create dummy CSI data: (100 time steps, 90 features)
    csi_data = np.random.randn(100, 90)
    
    print(f"Original shape: {csi_data.shape}")
    print(f"Original range: [{csi_data.min():.3f}, {csi_data.max():.3f}]")
    print(f"Original mean: {csi_data.mean():.3f}")
    print(f"Original std: {csi_data.std():.3f}\n")
    
    print("="*60)
    print("BASIC AUGMENTATIONS")
    print("="*60)
    
    # Test basic augmentations
    basic_ops = ["signal_scale", "additive_noise", "feature_dropout", "temporal_shift"]
    for op_name in basic_ops:
        op_func = AUGMENTATION_OPS[op_name]
        aug_data = op_func(csi_data, severity=5)
        print(f"\n{op_name}:")
        print(f"  Shape: {aug_data.shape}")
        print(f"  Range: [{aug_data.min():.3f}, {aug_data.max():.3f}]")
        print(f"  Mean: {aug_data.mean():.3f} (diff: {abs(aug_data.mean() - csi_data.mean()):.3f})")
        print(f"  Std: {aug_data.std():.3f}")
    
    print("\n" + "="*60)
    print("PHYSICS-BASED AUGMENTATIONS")
    print("="*60)
    
    # Test physics-based augmentations
    physics_ops = ["multipath_fading", "path_loss", "frequency_selective_fading", 
                   "environmental_interference", "doppler_shift"]
    for op_name in physics_ops:
        op_func = AUGMENTATION_OPS[op_name]
        aug_data = op_func(csi_data, severity=5)
        print(f"\n{op_name}:")
        print(f"  Shape: {aug_data.shape}")
        print(f"  Range: [{aug_data.min():.3f}, {aug_data.max():.3f}]")
        print(f"  Mean: {aug_data.mean():.3f} (diff: {abs(aug_data.mean() - csi_data.mean()):.3f})")
        print(f"  Std: {aug_data.std():.3f}")


def test_augmix():
    """Test AugMix implementation"""
    print("="*60)
    print("Testing AugMix Implementation...")
    print("="*60 + "\n")
    
    # Create dummy data
    csi_data = np.random.randn(100, 90)
    
    config = {
        "use_augmix": True,
        "augmix_severity": 3,
        "augmix_width": 3,
        "augmix_depth": -1,
        "augmix_alpha": 1.0,
        "augmix_ops": ["signal_scale", "additive_noise", "feature_dropout", "temporal_shift"]
    }
    
    print(f"AugMix Config:")
    print(f"  Severity: {config['augmix_severity']}")
    print(f"  Width: {config['augmix_width']}")
    print(f"  Depth: {config['augmix_depth']} (random 1-3)")
    print(f"  Alpha: {config['augmix_alpha']}")
    print(f"  Operations: {config['augmix_ops']}\n")
    
    mixed = augment_and_mix(
        csi_data,
        severity=config["augmix_severity"],
        width=config["augmix_width"],
        depth=config["augmix_depth"],
        alpha=config["augmix_alpha"],
        augmentation_ops=config["augmix_ops"]
    )
    
    print(f"Original shape: {csi_data.shape}")
    print(f"Mixed shape: {mixed.shape}")
    print(f"Original range: [{csi_data.min():.3f}, {csi_data.max():.3f}]")
    print(f"Mixed range: [{mixed.min():.3f}, {mixed.max():.3f}]")
    print(f"Original mean: {csi_data.mean():.3f}")
    print(f"Mixed mean: {mixed.mean():.3f}")
    print(f"Mean difference: {abs(mixed.mean() - csi_data.mean()):.3f}")
    print(f"Correlation: {np.corrcoef(csi_data.flatten(), mixed.flatten())[0, 1]:.3f}")


def test_dataset_integration():
    """Test AugMixDataset class"""
    print("\n" + "="*60)
    print("Testing Dataset Integration...")
    print("="*60 + "\n")
    
    # Create dummy dataset
    num_samples = 10
    data_x = np.random.randn(num_samples, 100, 90)
    data_y = np.random.randint(0, 2, (num_samples, 6))
    
    config = {
        "use_augmix": True,
        "augmix_severity": 3,
        "augmix_width": 3,
        "augmix_depth": -1,
        "augmix_alpha": 1.0,
        "augmix_ops": ["signal_scale", "additive_noise", "feature_dropout", "temporal_shift"]
    }
    
    dataset = AugMixDataset(data_x, data_y, config)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Testing __getitem__...")
    
    x, y = dataset[0]
    print(f"  Sample shape: {x.shape}")
    print(f"  Label shape: {y.shape}")
    print(f"  Sample type: {type(x)}")
    print(f"  Label type: {type(y)}")
    print("\nDataset integration successful!")


if __name__ == "__main__":
    test_augmentations()
    test_augmix()
    test_dataset_integration()