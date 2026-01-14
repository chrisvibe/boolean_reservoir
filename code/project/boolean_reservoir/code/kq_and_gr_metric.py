import torch
from torch.utils.data import DataLoader, Dataset
from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from benchmark.temporal.temporal_density_parity_datasets import TemporalDensityDataset
from project.boolean_reservoir.code.parameter import load_yaml_config, Params
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.boolean_reservoir.code.graph import calc_spectral_radius 
import pandas as pd
from project.boolean_reservoir.code.utils.utils import generate_unique_seed, override_symlink, save_grid_search_results
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Tuple

# TODO consider parallelization like in grid search for CPU + GPU

def get_kernel_quality_dataset(p: Params):
    return TemporalDensityDataset(p.D)

def get_generalization_rank_dataset(p: Params):
    dataset = get_kernel_quality_dataset(p)

    # define subsets
    subset_size = p.M.R.n_nodes # square matrix per rank calculation
    m = len(dataset)
    n_subsets = m // subset_size 
    indices = torch.randperm(m)[:n_subsets]

    # override last tao entries with similar inputs
    x = dataset.data['x']
    shape = x.shape
    x = x.view(shape[0], -1)
    tao = p.D.tao
    x[:, -tao:] = x[indices][:, -tao:].repeat_interleave(subset_size, dim=0)
    dataset.data['x'] = x.view(shape)
    return dataset

class DatasetInitKQGR:
    """Dataset initializer that returns both KQ and GR datasets"""
    def __init__(self, get_kq_fn: Callable = get_kernel_quality_dataset, get_gr_fn: Callable = get_generalization_rank_dataset):
        self.get_kq_fn = get_kq_fn
        self.get_gr_fn = get_gr_fn
    
    def __call__(self, P: Params) -> Tuple[Dataset, Dataset]:
        """Generate/load KQ and GR datasets"""
        kq_dataset = self.get_kq_fn(P)
        gr_dataset = self.get_gr_fn(P)
        return kq_dataset, gr_dataset

# def process_batch(model: BooleanReservoir, x: torch.Tensor, metric: str, data: list, config: int, sample: int):
#     # nest history by metric (load from same model and evaluate on KQ and GR)
#     nested_out = model.L.save_path / 'history' / metric
#     new_save_path = nested_out / 'history'
#     model.history = BatchedTensorHistoryWriter(
#         save_path=new_save_path, 
#         buffer_size=model.history.buffer_size
#     )
    
#     # Run model and record states
#     with torch.no_grad():
#         _ = model(x)
#     model.flush_history()
    
#     # Load and calculate rank
#     override_symlink(Path('../../checkpoint'), new_save_path / 'checkpoint')
#     load_dict, history, expanded_meta, meta = model.history.reload_history()
#     df_filter = expanded_meta[expanded_meta['phase'] == 'output_layer']
#     filtered_history = history[df_filter.index].to(torch.float)
#     reservoir_node_history = filtered_history[:, ~model.input_nodes_mask]

#     rank = torch.linalg.matrix_rank(reservoir_node_history)
#     data.append({
#         'config': config,
#         'sample': sample,
#         'metric': metric,
#         'value': rank.item()
#     })

#     # # TODO comment out - debugging
#     # for z in range(10):
#     #    plot_activity_trace(nested_out, save_path=model.L.save_path / 'visualizations', file_name=f"activity_trace_with_phase_{config}_{sample}_{metric}_r_{rank.item()}_t_{z}.png", aggregation_handle=lambda df: df[df['sample_id'] == z])



# def simulate_state_transisions_and_calculate_rank(p: Params, device: torch.device, dataset_kq: Dataset, dataset_gr: Dataset, data: list, config: int):
#     subset_size = p.M.R.n_nodes # square matrix per rank calculation
#     data_loader_kq = DataLoader(dataset_kq.to(device), batch_size=subset_size, shuffle=False, drop_last=True)
#     data_loader_gr = DataLoader(dataset_gr.to(device), batch_size=subset_size, shuffle=False, drop_last=True)

#     for i, ((x_kq, _), (x_gr, _)) in enumerate(zip(data_loader_kq, data_loader_gr)):
#         seed = generate_unique_seed(config, i)
#         p.M.I.seed = seed 
#         p.M.R.seed = seed 
#         p.M.O.seed = seed 
#         model = BooleanReservoir(p).to(device)
#         model.save()
#         model.eval()

#         data.append({
#             'config': config,
#             'sample': i,
#             'metric': 'params',
#             'value': p
#         })

#         spectral_radius = calc_spectral_radius(model.graph)
#         data.append({
#             'config': config,
#             'sample': i,
#             'metric': 'spectral_radius',
#             'value': spectral_radius
#         })

#         process_batch(model, x_kq, 'kq', data, config, i)
#         process_batch(model, x_gr, 'gr', data, config, i)



# def calc_kernel_quality_and_generalization_rank(yaml_path):
#     # get n_node samples for KQ and GR, as well as some graph properties
#     # written with the assumption that n_nodes doesnt change
#     P = load_yaml_config(yaml_path)
#     P = override_samples_in_p(P)
#     param_combinations = generate_param_combinations(P)
#     df = prepare_metrics_data(param_combinations)
#     return df, P

# def override_samples_in_p(p: Params):
#     p.D.samples = p.M.R.n_nodes * p.D.samples # P.D.samples is the number of samples per configuration up until here where it actually becomes the number of samples (overidden)
#     p.D.update_path()
#     return p

# def prepare_metrics_data(param_combinations: list[Params], ignore_gpu=False):
#     data = []
#     device = torch.device("cuda" if torch.cuda.is_available() and not ignore_gpu else "cpu")
#     for i, p in enumerate(tqdm(param_combinations, desc="Processing configurations")):
#         p.D.seed = i
#         # TODO consider overriding tao here if its set to none with a random value
#         dataset_kq, dataset_gr = DatasetInitKQGR(get_gr_fn=get_kernel_quality_dataset, get_kq_fn=get_generalization_rank_dataset)(p)

#         simulate_state_transisions_and_calculate_rank(p, device, dataset_kq, dataset_gr, data, i)

#     df = pd.DataFrame(data)
#     pivot_df = df.pivot_table(index=['config', 'sample'], columns='metric', values='value', aggfunc='first').reset_index()
#     pivot_df['delta'] = pivot_df['kq'] - pivot_df['gr']
#     melted_df = pd.melt(pivot_df, id_vars=['config', 'sample', 'params', 'spectral_radius'], value_vars=['spectral_radius', 'kq', 'gr', 'delta'], var_name='metric', value_name='value')
#     df = melted_df.sort_values(by=['config', 'sample']).reset_index(drop=True)
#     return df

# if __name__ == '__main__':
#     paths = list()
#     paths.append('config/temporal/kq_and_gr/test.yaml')

#     # generate data
#     for path in paths:
#         print(path)
#         df, P = calc_kernel_quality_and_generalization_rank(path)
#         data_file_path = P.L.out_path / 'log.yaml'
#         save_grid_search_results(df, data_file_path)
