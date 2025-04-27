import torch
import numpy as np
import random
from hashlib import sha256
from pathlib import Path

# Ensure reproducibility by setting seeds globally
def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def generate_unique_seed(*args):
    # Create a string representation of the combined parameters
    combined_str = ','.join(map(str, args))
    # Use SHA256 to hash the combined string
    hash_digest = sha256(combined_str.encode()).hexdigest()
    # Convert the hash to an integer and ensure it fits in the range of a typical seed
    return int(hash_digest, 16) % (2**31 - 1)

def override_symlink(source: Path, link:str=None):
    if link.exists():
        link.unlink()
    link.symlink_to(source)

def gpu_check():
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA devices:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

def l2_distance(tensor):
    return torch.sqrt((tensor ** 2).sum(axis=1))

def manhattan_distance(tensor):
    return torch.abs(tensor).sum(axis=1)

def balance_dataset(dataset, num_bins=100, distance_fn=l2_distance, labels_are_classes=False, target_mode='samples_over_bins'):
    x = dataset.data['x']
    y = dataset.data['y']

    distances = distance_fn(y)
    bins = torch.linspace(distances.min(), distances.max(), num_bins + 1)
    if labels_are_classes:
        bin_indices = y.ravel().to(int) 
    else:
        bin_indices = torch.bucketize(distances, bins) - 1

    if target_mode == 'samples_over_bins':
        target_per_bin = max(len(y) // num_bins, 1)
    elif target_mode == 'minimum_bin':
        bincount = torch.bincount(y.ravel().to(int))
        target_per_bin = bincount.min() 
    
    # Sample from each bin
    balanced_indices = []
    for i in range(num_bins):
        bin_samples = torch.arange(len(y))[bin_indices == i]
        if len(bin_samples) > 0:
            if len(bin_samples) <= target_per_bin:
                balanced_indices.append(bin_samples)
            else:
                sampled = bin_samples[torch.randperm(len(bin_samples))[:target_per_bin]]
                balanced_indices.append(sampled)
    
    balanced_indices = torch.cat(balanced_indices) if balanced_indices else torch.arange(len(y))
    
    # Report statistics
    if labels_are_classes:
        print("Class distribution:")
        unique_labels = torch.unique(y)
        for label in unique_labels:
            before_count = (y == label).sum().item()
            after_count = (y[balanced_indices] == label).sum().item()
            print(f" Class {int(label.item())}: {before_count} → {after_count} samples")
    else:
        print("Value distribution (quartiles):")
        quartiles = [0, 25, 50, 75, 100]
        before_quartiles = [torch.quantile(y, q/100).item() for q in quartiles]
        after_quartiles = [torch.quantile(y[balanced_indices], q/100).item() for q in quartiles]
        for i, q in enumerate(quartiles):
            print(f" {q}%: {before_quartiles[i]:.4f} → {after_quartiles[i]:.4f}")
    
    # Update dataset with balanced indices
    dataset.data['x'] = x[balanced_indices]
    dataset.data['y'] = y[balanced_indices]
    
    # Report overall reduction
    n_before, n_after = len(x), len(balanced_indices)
    print(f'Balanced dataset from {n_before} samples to {n_after} ({(n_before - n_after) / n_before * 100:.2f}% reduction)')

    return dataset


if __name__ == '__main__':
    gpu_check()
