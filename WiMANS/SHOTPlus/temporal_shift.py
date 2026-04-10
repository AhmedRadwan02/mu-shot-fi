# temporal_shift.py - Temporal-Shift Discrimination (TSD) for CSI data
import torch
import numpy as np
from typing import Union, List


def temporal_shift_single(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Apply temporal shift to a single sample using torch.roll.

    Args:
        x: Input tensor of shape (T, F, C) or (T, F) - single sample
        shift: Integer shift value (negative = shift left, positive = shift right)
               shift=0 means no shift

    Returns:
        Shifted tensor with same shape as input
    """
    if shift == 0:
        return x

    # Roll along time dimension (dimension 0)
    # Positive shift: data moves forward in time (right shift)
    # Negative shift: data moves backward in time (left shift)
    shifted = torch.roll(x, shifts=shift, dims=0)

    return shifted


def shift_single_with_label(x: torch.Tensor, label: int, shift_values: List[int]) -> torch.Tensor:
    """
    Apply temporal shift based on label index.

    Args:
        x: Input tensor of shape (T, F, C) or (T, F) - single sample
        label: Integer label index into shift_values list
        shift_values: List of shift values (e.g., [-2, -1, 0, 1, 2])

    Returns:
        Shifted tensor with same shape as input
    """
    shift = shift_values[label]
    return temporal_shift_single(x, shift)


@torch.no_grad()
def shift_batch_with_labels(batch: torch.Tensor,
                           labels: Union[torch.Tensor, np.ndarray],
                           shift_values: List[int]) -> torch.Tensor:
    """
    Apply temporal shifts to a batch based on labels.

    Args:
        batch: (B, T, F, C) or (B, T, F) batch OR (T, F, C) or (T, F) single sample
        labels: (B,) integer labels indexing into shift_values
                Can be numpy array or torch tensor
        shift_values: List of shift values (e.g., [-2, -1, 0, 1, 2])

    Returns:
        Shifted batch with same shape as input
    """
    # Detect if input is single sample or batch
    # Single: (T, F) [2D] or (T, F, C) [3D with C<=2]
    # Batch: (B, T, F) [3D with F>2] or (B, T, F, C) [4D]
    is_single = (batch.dim() == 2) or (batch.dim() == 3 and batch.shape[-1] <= 2)
    if is_single:
        batch = batch.unsqueeze(0)
        labels = labels.unsqueeze(0) if torch.is_tensor(labels) and labels.dim() == 0 else labels

    # Convert labels to torch tensor if it's numpy
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    labels = labels.to(batch.device).long()

    # Shift each sample based on its label
    shifted_samples = []
    for sample, label in zip(batch, labels):
        shifted = shift_single_with_label(sample, int(label), shift_values)
        shifted_samples.append(shifted)

    # Stack along batch dimension
    out = torch.stack(shifted_samples, dim=0)

    if is_single:
        out = out.squeeze(0)

    return out


def generate_random_shift_labels(batch_size: int,
                                 num_shifts: int,
                                 device: torch.device = None) -> torch.Tensor:
    """
    Generate random shift labels for a batch.

    Args:
        batch_size: Number of samples in batch
        num_shifts: Number of different shift values
        device: Device to place tensor on

    Returns:
        Random integer labels in range [0, num_shifts)
    """
    labels = torch.randint(0, num_shifts, (batch_size,), dtype=torch.long)
    if device is not None:
        labels = labels.to(device)
    return labels


# Example shift configurations
SHIFT_CONFIGS = {
    "5_class": [-2, -1, 0, 1, 2],      # 5 classes: large shifts
    "3_class": [-1, 0, 1],              # 3 classes: small shifts
    "7_class": [-3, -2, -1, 0, 1, 2, 3], # 7 classes: very large shifts
}


def get_shift_config(config_name: str = "5_class") -> List[int]:
    """
    Get predefined shift configuration.

    Args:
        config_name: Name of configuration ("5_class", "3_class", or "7_class")

    Returns:
        List of shift values
    """
    if config_name not in SHIFT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(SHIFT_CONFIGS.keys())}")
    return SHIFT_CONFIGS[config_name]


# Test function
def test_temporal_shift():
    """Test temporal shift transformations."""
    print("="*60)
    print("TESTING TEMPORAL SHIFT TRANSFORMATIONS")
    print("="*60)

    # Create test data
    batch_size = 4
    T, F, C = 1200, 180, 1

    # Test with batch
    x = torch.randn(batch_size, T, F, C)
    shift_values = [-2, -1, 0, 1, 2]

    print(f"\nInput shape: {x.shape}")
    print(f"Shift values: {shift_values}")

    # Generate random labels
    labels = generate_random_shift_labels(batch_size, len(shift_values))
    print(f"Random labels: {labels}")

    # Apply shifts
    shifted = shift_batch_with_labels(x, labels, shift_values)
    print(f"Output shape: {shifted.shape}")

    # Verify shift for first sample
    if labels[0] != 2:  # If not 0 shift
        shift_val = shift_values[labels[0]]
        print(f"\nVerifying shift for sample 0 (label={labels[0]}, shift={shift_val}):")

        # Check if values match after shift
        if shift_val > 0:
            # Positive shift: x[t] should match shifted[t+shift]
            match = torch.allclose(x[0, :-shift_val], shifted[0, shift_val:], rtol=1e-5)
        elif shift_val < 0:
            # Negative shift: x[t] should match shifted[t+shift]
            match = torch.allclose(x[0, -shift_val:], shifted[0, :shift_val], rtol=1e-5)
        else:
            match = torch.allclose(x[0], shifted[0], rtol=1e-5)

        print(f"  Shift verification: {'PASS' if match else 'FAIL'}")

    # Test with single sample
    print("\nTesting single sample:")
    x_single = torch.randn(T, F, C)
    label_single = 2  # No shift
    shifted_single = shift_batch_with_labels(x_single, torch.tensor(label_single), shift_values)
    print(f"  Input shape: {x_single.shape}")
    print(f"  Output shape: {shifted_single.shape}")
    print(f"  Label: {label_single} (shift={shift_values[label_single]})")

    # Verify no-shift case
    if label_single == 2:  # 0 shift
        match = torch.allclose(x_single, shifted_single, rtol=1e-5)
        print(f"  No-shift verification: {'PASS' if match else 'FAIL'}")

    print("\n" + "="*60)
    print("TEMPORAL SHIFT TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_temporal_shift()
