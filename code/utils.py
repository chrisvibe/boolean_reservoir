import torch
import numpy as np
import random
from collections.abc import Iterable
from pathlib import Path

# Ensure reproducibility by setting seeds globally
def set_seed(seed=42):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def make_folders(out_path: Path | str, folders: Iterable[str]):
    if isinstance(out_path, str):
        out_path = Path(out_path)
    for f in folders:
        p = out_path / f 
        if not p.exists():
            p.mkdir(parents=True)

def gpu_check():
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA devices:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

def balance_dataset(dataset, num_bins=100):
    x = dataset.data['x']
    y = dataset.data['y']
    
    # Step 1: Calculate the distance of each point from the origin
    distances = torch.sqrt((y ** 2).sum(dim=1))

    # Step 2: Create distance bins
    bins = torch.linspace(distances.min(), distances.max(), num_bins + 1)

    # Step 3: Assign points to bins
    bin_indices = torch.bucketize(distances, bins) - 1  # subtracting 1 because bucketize returns 1-based index

    # Step 4: Sample points uniformly from each bin
    balanced_indices = []
    target_points_per_bin = len(y) // num_bins  # Number of points you want in each bin

    for i in range(num_bins):
        bin_points_indices = (bin_indices == i).nonzero().squeeze()
        if bin_points_indices.dim() == 0:
            bin_points_indices = bin_points_indices.unsqueeze(0)
        if len(bin_points_indices) > target_points_per_bin:
            sampled_indices = bin_points_indices[torch.randperm(len(bin_points_indices))[:target_points_per_bin]]
        else:
            sampled_indices = bin_points_indices
        balanced_indices.append(sampled_indices)

    # Concatenate all the balanced indices into a single tensor
    balanced_indices = torch.cat(balanced_indices, dim=0)

    # Use the balanced indices to create balanced versions of x and y
    n_before, n_after = len(x), len(balanced_indices)
    dataset.data['x'] = x[balanced_indices]
    dataset.data['y'] = y[balanced_indices]
    print(f'Balanced dataset from {n_before} samples to {n_after} ({(n_before - n_after) / n_before * 100}% reduction)')

def euclidean_distance_accuracy(y_hat, y, radius_threshold, normalize=True):
    distances = torch.sqrt(torch.sum((y_hat - y) ** 2, dim=1))
    correct_predictions = (distances < radius_threshold).sum().item()
    if normalize:
        return correct_predictions / len(y)
    else:
        return correct_predictions

if __name__ == '__main__':
    gpu_check()
