import torch
from torch.utils.data import DataLoader, Dataset
from projects.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from benchmarks.temporal.temporal_density_parity_datasets import TemporalDensityDataset
from projects.boolean_reservoir.code.parameters import load_yaml_config, generate_param_combinations, Params
from projects.boolean_reservoir.code.graphs import calc_spectral_radius 
import pandas as pd
from projects.boolean_reservoir.code.utils import generate_unique_seed, override_symlink, print_pretty_binary_matrix, CudaMemoryManager, save_grid_search_results
from pathlib import Path
from tqdm import tqdm

# TODO consider parallelization like in grid search for CPU + GPU

def process_batch(model: BooleanReservoir, x: torch.Tensor, metric: str, data: list, config: int, sample: int):
    # nest history by metric (load from same model and evaluate on KQ and GR)
    nested_out = model.L.save_path / 'history' / metric
    new_save_path = nested_out / 'history'
    model.history = BatchedTensorHistoryWriter(
        save_path=new_save_path, 
        buffer_size=model.history.buffer_size
    )
    
    # Run model and record states
    with torch.no_grad():
        _ = model(x)
    model.flush_history()
    
    # Load and calculate rank
    override_symlink(Path('../../checkpoint'), new_save_path / 'checkpoint')
    load_dict, history, expanded_meta, meta = model.history.reload_history()
    df_filter = expanded_meta[expanded_meta['phase'] == 'output_layer']
    filtered_history = history[df_filter.index].to(torch.float)
    reservoir_node_history = filtered_history[:, ~model.input_nodes_mask]

    rank = torch.linalg.matrix_rank(reservoir_node_history)
    data.append({
        'config': config,
        'sample': sample,
        'metric': metric,
        'value': rank.item()
    })

    # # TODO comment out - debugging
    # for z in range(10):
    #    plot_activity_trace(nested_out, save_path=model.L.save_path / 'visualizations', file_name=f"activity_trace_with_phase_{config}_{sample}_{metric}_r_{rank.item()}_t_{z}.png", aggregation_handle=lambda df: df[df['sample_id'] == z])



def simulate_state_transisions_and_calculate_rank(p: Params, device: torch.device, dataset_kq: Dataset, dataset_gr: Dataset, data: list, config: int):
    subset_size = p.M.R.n_nodes # square matrix per rank calculation
    data_loader_kq = DataLoader(dataset_kq, batch_size=subset_size, shuffle=False, drop_last=True)
    data_loader_gr = DataLoader(dataset_gr, batch_size=subset_size, shuffle=False, drop_last=True)

    for i, ((x_kq, _), (x_gr, _)) in enumerate(zip(data_loader_kq, data_loader_gr)):
        seed = generate_unique_seed(config, i)
        p.M.I.seed = seed 
        p.M.R.seed = seed 
        p.M.O.seed = seed 
        model = BooleanReservoir(p).to(device)
        model.save()
        model.eval()

        data.append({
            'config': config,
            'sample': i,
            'metric': 'params',
            'value': p
        })

        spectral_radius = calc_spectral_radius(model.graph)
        data.append({
            'config': config,
            'sample': i,
            'metric': 'spectral_radius',
            'value': spectral_radius
        })

        process_batch(model, x_kq, 'kq', data, config, i)
        process_batch(model, x_gr, 'gr', data, config, i)

def get_kernel_quality_dataset(p: Params):
    p.D.update_path()
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

def calc_kernel_quality_and_generalization_rank(yaml_path):
    # get n_node samples for KQ and GR, as well as some graph properties
    # written with the assumption that n_nodes doesnt change
    P = load_yaml_config(yaml_path)
    P = override_samples_in_p(P)
    param_combinations = generate_param_combinations(P)
    df = prepare_metrics_data(param_combinations)
    return df, P

def override_samples_in_p(p: Params):
    p.D.samples = p.M.R.n_nodes * p.D.samples # P.D.samples is the number of samples per configuration up until here where it actually becomes the number of samples (overidden)
    p.D.update_path()
    return p

def prepare_metrics_data(param_combinations: list[Params]):
    data = []
    mem = CudaMemoryManager()
    for i, p in enumerate(tqdm(param_combinations, desc="Processing configurations")):
        mem.manage_memory()
        p.D.seed = i
        # TODO consider overriding tao here if its set to none with a random value
        dataset_kq = get_kernel_quality_dataset(p).to(mem.device)
        dataset_gr = get_generalization_rank_dataset(p).to(mem.device)
        simulate_state_transisions_and_calculate_rank(p, mem.device, dataset_kq, dataset_gr, data, i)

    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(index=['config', 'sample'], columns='metric', values='value', aggfunc='first').reset_index()
    pivot_df['delta'] = pivot_df['kq'] - pivot_df['gr']
    melted_df = pd.melt(pivot_df, id_vars=['config', 'sample', 'params', 'spectral_radius'], value_vars=['spectral_radius', 'kq', 'gr', 'delta'], var_name='metric', value_name='value')
    df = melted_df.sort_values(by=['config', 'sample']).reset_index(drop=True)
    return df

if __name__ == '__main__':
    paths = list()
    # paths.append('config/temporal/kq_and_gr/test.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_3/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/fixed_tao/tao_5/heterogeneous_stochastic.yaml')

    paths.append('config/temporal/kq_and_gr/vary_tao/homogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_tao/homogeneous_stochastic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_tao/heterogeneous_deterministic.yaml')
    paths.append('config/temporal/kq_and_gr/vary_tao/heterogeneous_stochastic.yaml')

    # generate data
    for path in paths:
        print(path)
        df, P = calc_kernel_quality_and_generalization_rank(path)
        data_file_path = P.L.out_path / 'log.yaml'
        save_grid_search_results(df, data_file_path)
