import torch
import numpy as np
import random
from hashlib import sha256
from pathlib import Path
import networkx as nx
import time
import pandas as pd
from os import getpid, replace
from project.boolean_reservoir.code.parameter import Params, load_yaml_config
import yaml
from enum import Enum

# Ensure reproducibility by setting seeds globally
# Wont work across architectures and GPU vs CPU etc
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

def print_pretty_binary_matrix(data, input_nodes=None, reservoir_nodes=None, print_str=True, return_str=False):
    QUADRANT_COLORS = {
        'II': '\033[95m',    # Magenta
        'IR': '\033[93m',    # Yellow
        'RI': '\033[91m',    # Red
        'RR': '\033[92m',    # Green
        'default': '\033[94m',  # Blue
    }

    def infer_quadrant(u, v):
        if u in input_nodes and v in input_nodes:
            return 'II'
        elif u in input_nodes and v in reservoir_nodes:
            return 'IR'
        elif u in reservoir_nodes and v in input_nodes:
            return 'RI'
        elif u in reservoir_nodes and v in reservoir_nodes:
            return 'RR'
        return 'default'

    if isinstance(data, torch.Tensor):
        array = (data.detach().cpu().numpy() != 0).astype(int)
        color_matrix = np.full(array.shape, 'default', dtype=object)

    elif isinstance(data, np.ndarray):
        array = (data != 0).astype(int)
        color_matrix = np.full(array.shape, 'default', dtype=object)

    elif isinstance(data, (nx.Graph, nx.DiGraph)):
        nodes = list(data.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        array = nx.to_numpy_array(data, nodelist=nodes, dtype=bool).astype(int)
        color_matrix = np.full(array.shape, 'default', dtype=object)

        for u, v, attrs in data.edges(data=True):
            i, j = node_index[u], node_index[v]
            if 'quadrant' in attrs:
                q = attrs['quadrant']
            elif input_nodes is not None and reservoir_nodes is not None:
                q = infer_quadrant(u, v)
            else:
                q = 'default'
            color_matrix[i, j] = q
            if not data.is_directed():
                color_matrix[j, i] = q
    else:
        raise TypeError("Input must be a torch.Tensor, np.ndarray, or networkx Graph/DiGraph.")

    lines = []
    for i in range(array.shape[0]):
        row = ''
        for j in range(array.shape[1]):
            val = str(array[i, j])
            color = QUADRANT_COLORS.get(color_matrix[i, j], QUADRANT_COLORS['default'])
            row += f"{color}{val}\033[0m"
        lines.append(row)

    result = '\n'.join(lines)
    if print_str:
        print(result)
    if return_str:
        return result
    
def override_symlink(source: Path, link: Path = None):
    """
    Atomically create/update a symbolic link. Fails silently if unable.
    """
    if link is None:
        link = Path(source.name)
    
    try:
        # Create a temporary symlink with unique name
        temp_link = Path(f"{link}.tmp.{getpid()}.{time.time_ns()}")
        temp_link.symlink_to(source)
        
        # Atomically replace the old symlink
        replace(str(temp_link), str(link))
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_link' in locals() and temp_link.exists():
            try:
                temp_link.unlink()
            except:
                pass
        # Fail silently - symlink is not critical
        pass

def l2_distance(tensor):
    return torch.sqrt((tensor ** 2).sum(axis=1))

def manhattan_distance(tensor):
    return torch.abs(tensor).sum(axis=1)

def balance_dataset(dataset, num_bins=100, distance_fn=l2_distance, labels_are_classes=False, target_mode='samples_over_bins', verbose=False):
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
    if verbose:
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
    if verbose:
        n_before, n_after = len(x), len(balanced_indices)
        print(f'Balanced dataset from {n_before} samples to {n_after} ({(n_before - n_after) / n_before * 100:.2f}% reduction)')

    return dataset

def save_grid_search_results(df: pd.DataFrame, path: Path):
    """Save grid search results to YAML, with Path fields handled by custom representers"""
    df_copy = df.copy()
    df_copy['params'] = df_copy['params'].apply(lambda p: p.model_dump())
    records = df_copy.to_dict(orient='records')
    yaml.dump_all(records, path.open('a'), sort_keys=False, Dumper=yaml.Dumper)


def params_col_to_fields(df, extractions):
    """
    Projects structured parameter objects in `df['params']`
    into a new DataFrame of extracted fields.
    Args:
        df: DataFrame containing a `params` column.
        extractions: List of (prefix, getter, field_set) tuples.
            - prefix: Column prefix or column name if capturing source.
            - getter: Function extracting a sub-model from params.
            - field_set: Set of field names to extract, empty set {} for all fields,
                        or None to capture source object.
    Returns:
        (new_df, factors):
            new_df: DataFrame with extracted fields (and captured sources).
            factors: List of extracted flattened column names.
    """
    rows = []
    factors = []
    for params in df['params']:
        row = {}
        for prefix, get_source, field_set in extractions:
            source = get_source(params)
            if field_set is None:
                row[prefix] = source
                continue
            
            dumped = source.model_dump()
            # If field_set is empty, extract all fields
            fields_to_extract = dumped.keys() if not field_set else field_set
            
            for k in fields_to_extract:
                v = dumped[k]
                col = f"{prefix}_{k}"
                row[col] = str(v) if isinstance(v, Enum) else v
                if col not in factors:
                    factors.append(col)
        rows.append(row)
    return pd.DataFrame(rows), factors

def get_data_path(config_path, filename='log.yaml') -> Path:
    """Derive data path from config's out_path"""
    P = load_yaml_config(config_path)
    return P.L.out_path / filename

def load_params_df_from_yaml(file_path) -> pd.DataFrame:
    """Load multi-document yaml as DataFrame with params column"""
    with open(file_path) as f:
        docs = list(yaml.safe_load_all(f))
    
    params_list = []
    for d in docs:
        params_list.append(Params(**d['params']))
    
    return pd.DataFrame({'params': params_list})


def custom_load_grid_search_data(data_paths=None, config_paths=None, extractions=None, df_filter_mask=None, filename='log.yaml') -> tuple[pd.DataFrame, list[str]]:
    """Core loader - requires explicit data_paths or config_paths + extractions"""
    if data_paths is None and config_paths is None:
        raise ValueError("Must provide data_paths or config_paths")
    
    if data_paths is None:
        if isinstance(config_paths, (str, Path)):
            config_paths = [config_paths]
        data_paths = [get_data_path(p, filename) for p in config_paths]
    
    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]
    
    dfs = []
    factors = []
    for path in data_paths:
        df = load_params_df_from_yaml(path)
        if extractions:
            df, factors = params_col_to_fields(df, extractions)
        if df_filter_mask:
            df = df[df_filter_mask(df)]
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    return df, factors


def load_grid_search_data(data_paths=None, config_paths=None, extractions=None, df_filter_mask=None, filename='log.yaml') -> tuple[pd.DataFrame, list[str]]:
    """Convenience loader with default train_log extraction"""
    if extractions is None:
        extractions = [
            ('P', lambda p: p, None),
            ('T', lambda p: p.L.T, {'accuracy', 'loss'}),
        ]
    
    return custom_load_grid_search_data(data_paths=data_paths, config_paths=config_paths, extractions=extractions, df_filter_mask=df_filter_mask, filename=filename)