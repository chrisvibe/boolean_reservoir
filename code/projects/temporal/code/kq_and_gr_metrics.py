import torch
from torch.utils.data import DataLoader, Dataset
from projects.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from benchmarks.temporal.temporal_replication_study_density_parity_datasets import TemporalDensityDataset
from projects.boolean_reservoir.code.parameters import load_yaml_config, generate_param_combinations, Params
from projects.boolean_reservoir.code.graphs import calc_spectral_radius 
from projects.temporal.code.visualizations import plot_kq_and_gr, group_df_data_by_parameters, plot_kq_and_gr_many_config, plot_optimal_k_vs_k_avg
import pandas as pd
from projects.boolean_reservoir.code.utils import generate_unique_seed, override_symlink
from pathlib import Path
from tqdm import tqdm

def process_batch(model: BooleanReservoir, x: torch.Tensor, metric: str, data: list, config: int, sample: int):
    # nest history by metric (load from same model and evaluate on KQ and GR)
    new_save_path = model.L.save_path / 'history' / metric / 'history'
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
    filter = expanded_meta[expanded_meta['phase'] == 'output_layer']
    filtered_history = history[filter.index].to(torch.float)
    reservoir_node_history = filtered_history[:, ~model.input_nodes_mask]
    
    rank = torch.linalg.matrix_rank(reservoir_node_history)
    data.append({
        'config': config,
        'sample': sample,
        'metric': metric,
        'value': rank.item()
    })


def simulate_state_transisions_and_calculate_rank(p: Params, dataset_kq: Dataset, dataset_gr: Dataset, data: list, config: int):
    subset_size = p.M.R.n_nodes # square matrix per rank calculation
    data_loader_kq = DataLoader(dataset_kq, batch_size=subset_size, shuffle=False, drop_last=True)
    data_loader_gr = DataLoader(dataset_gr, batch_size=subset_size, shuffle=False, drop_last=True)
    
    i = 0
    for (x_kq, _), (x_gr, _) in zip(data_loader_kq, data_loader_gr):
        seed = generate_unique_seed(config, i)
        i += 1
        p.M.I.seed = seed 
        p.M.R.seed = seed 
        p.M.O.seed = seed 
        model = BooleanReservoir(p)
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

def calc_kernel_quality_and_generalization_rank(yaml_path, samples_per_configuration):
    # get n_node samples for KQ and GR, as well as some graph properties
    # written with the assumption that n_nodes doesnt change
    P = load_yaml_config(yaml_path)
    P.D.samples = P.M.R.n_nodes * samples_per_configuration
    P.D.update_path()
    param_combinations = generate_param_combinations(P)
    df = prepare_metrics_data(param_combinations)
    return df, P

def prepare_metrics_data(param_combinations: list[Params]):
    data = []
    for i, p in enumerate(tqdm(param_combinations, desc="Processing configurations")):
        p.D.seed = i
        dataset_kq = get_kernel_quality_dataset(p) 
        dataset_gr = get_generalization_rank_dataset(p)
        simulate_state_transisions_and_calculate_rank(p, dataset_kq, dataset_gr, data, i)

    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(index=['config', 'sample'], columns='metric', values='value', aggfunc='first').reset_index()
    pivot_df['delta'] = pivot_df['kq'] - pivot_df['gr']
    melted_df = pd.melt(pivot_df, id_vars=['config', 'sample', 'params', 'spectral_radius'], value_vars=['spectral_radius', 'kq', 'gr', 'delta'], var_name='metric', value_name='value')
    df = melted_df.sort_values(by=['config', 'sample']).reset_index(drop=True)
    return df

if __name__ == '__main__':
    paths = list()
    paths.append('config/temporal/reservoir/kg_and_gr_homogenous.yaml')
    paths.append('config/temporal/reservoir/kg_and_gr_heterogenous.yaml')
    for path in paths:
        df, P = calc_kernel_quality_and_generalization_rank(path, samples_per_configuration=25)
        df.loc[:, 'k_avg'] = df['params'].apply(lambda p: p.M.R.k_avg)

        grouped_df = group_df_data_by_parameters(df)
        plot_kq_and_gr_many_config(grouped_df, P, 'many_config.png')
        for i, (name, subset) in enumerate(sorted(grouped_df, key=lambda x: x[0])):
            print(i, ':')
            p = subset.iloc[0]['params']
            print(p.M.I)
            print(p.M.R)
            print(p.M.O)
            print(p.D)
            plot_kq_and_gr(subset, P, f'config_{i}_kq_and_gr.png')

        # plot_optimal_k_vs_k_avg(df)

        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'homogenous')]
        # plot_kq_and_gr(subset, P, 'homogenous_kq_and_gr.png')
        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'heterogenous')]
        # plot_kq_and_gr(subset, P, 'heterogenous_kq_and_gr.png')

        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'homogenous') & (df['metric'] == 'kq')]
        # plot_kq_and_gr(subset, P, 'homogenous_kq.png')
        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'heterogenous') & (df['metric'] == 'kq')]
        # plot_kq_and_gr(subset, P, 'heterogenous_kq.png')

        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'homogenous') & (df['metric'] == 'gr')]
        # plot_kq_and_gr(subset, P, 'homogenous_gr.png')
        # subset = df[df['params'].apply(lambda p: p.M.R.mode == 'heterogenous') & (df['metric'] == 'gr')]
        # plot_kq_and_gr(subset, P, 'heterogenous_gr.png')