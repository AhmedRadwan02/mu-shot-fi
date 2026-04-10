# rotation.py - simplified for non-square CSI data (only 0° and 180°)
import torch
import numpy as np
from torchvision import datasets, transforms

def _rot_tf(x: torch.Tensor, k: int):
    """
    Rotate in the (T, F) plane by k*90 degrees, channels last.
    x: (..., T, F, C)
    """
    return torch.rot90(x, k=k, dims=(-3, -2))

def tensor_rot_180(x):
    return _rot_tf(x, k=2)

def rotate_single_with_label(img: torch.Tensor, label: int):
    """
    img: (T, F, C) tensor
    label: 0=no rotation, 1=180° rotation
    Returns: rotated (T, F, C) - dimensions preserved
    """
    if label == 1:
        return tensor_rot_180(img)
    else:
        return img

@torch.no_grad()
def rotate_batch_with_labels(batch: torch.Tensor, labels: torch.Tensor):
    """
    batch: (B, T, F, C) channels-last OR (T, F, C) single sample.
    labels: (B,) ints in {0, 1} (0=no rotation, 1=180°)
    Returns: rotated batch with same shape as input.
    
    Note: Limited to 0° and 180° rotations to avoid dimension mismatch
    with non-square CSI data (T=1200, F=180).
    """
    is_single = (batch.dim() == 3)
    if is_single:
        batch = batch.unsqueeze(0)
        labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
    
    labels = labels.to(batch.device).long()
    
    # Rotate each sample
    imgs = []
    for img, lab in zip(batch, labels):
        rotated = rotate_single_with_label(img, int(lab))
        imgs.append(rotated)
    
    # Stack along batch dimension - all same shape now
    out = torch.stack(imgs, dim=0)
    
    if is_single:
        out = out.squeeze(0)
    
    return out

@torch.no_grad()
def rotate_batch_with_labels(batch: torch.Tensor, labels: torch.Tensor):
    """
    batch: (B, T, F, C) channels-last OR (T, F, C) single sample.
    labels: (B,) ints in {0, 1} (0=no rotation, 1=180°) - can be numpy or torch
    Returns: rotated batch with same shape as input.
    
    Note: Limited to 0° and 180° rotations to avoid dimension mismatch
    with non-square CSI data (T=1200, F=180).
    """
    is_single = (batch.dim() == 3)
    if is_single:
        batch = batch.unsqueeze(0)
        labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
    
    # Convert labels to torch tensor if it's numpy
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    labels = labels.to(batch.device).long()
    
    # Rotate each sample
    imgs = []
    for img, lab in zip(batch, labels):
        rotated = rotate_single_with_label(img, int(lab))
        imgs.append(rotated)
    
    # Stack along batch dimension - all same shape now
    out = torch.stack(imgs, dim=0)
    
    if is_single:
        out = out.squeeze(0)
    
    return out

class RotateImageFolder(datasets.ImageFolder):
    def __init__(self, traindir, train_transform, original=True, rotation=True, rotation_transform=None):
        super(RotateImageFolder, self).__init__(traindir, train_transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform        
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img_input = self.loader(path)
        if self.transform is not None:
            img = self.transform(img_input)
        else:
            img = img_input
        
        results = []
        if self.original:
            results.append(img)
            results.append(target)
        
        if self.rotation:
            if self.rotation_transform is not None:
                img = self.rotation_transform(img_input)
            target_ssh = np.random.randint(0, 2, 1)[0]  # Binary: 0 or 1
            img_ssh = rotate_single_with_label(img, target_ssh)
            results.append(img_ssh)
            results.append(target_ssh)
        
        return results
    
    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation