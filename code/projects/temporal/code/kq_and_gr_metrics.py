import torch
from torch.utils.data import DataLoader, Dataset
from projects.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from benchmarks.temporal.temporal_replication_study_density_parity_datasets import TemporalDensityDataset
from projects.boolean_reservoir.code.parameters import load_yaml_config, generate_param_combinations, Params
from projects.boolean_reservoir.code.graphs import calc_spectral_radius 
from projects.temporal.code.visualizations import plot_kq_and_gr, group_df_data_by_parameters, plot_kq_and_gr_many_config, plot_optimal_k_vs_k_avg
import pandas as pd
from projects.boolean_reservoir.code.utils import generate_unique_seed
from tqdm import tqdm

def process_batch(model: BooleanReservoir, x: torch.Tensor, metric: str, data: list, config: int, sample: int):
    model.history = BatchedTensorHistoryWriter(
        save_dir=model.L.save_dir / 'history' / metric, 
        buffer_size=model.L.history.buffer_size
    )
    
    # Run model and record states
    with torch.no_grad():
        _ = model(x)
    model.flush_history()
    
    # Load and calculate rank
    history, expanded_meta, meta = model.history.reload_history()
    filter = expanded_meta[expanded_meta['phase'] == 'output_layer']
    filtered_history = history[filter.index].to(torch.float)
    
    rank = torch.linalg.matrix_rank(filtered_history)
    data.append({
        'config': config,
        'sample': sample,
        'metric': metric,
        'value': rank.item()
    })


def simulate_state_transisions_and_calculate_rank(p: Params, dataset_kq: Dataset, dataset_gr: Dataset, data: list, config: int):
    subset_size = p.model.reservoir_layer.n_nodes # square matrix per rank calculation
    data_loader_kq = DataLoader(dataset_kq, batch_size=subset_size, shuffle=False, drop_last=True)
    data_loader_gr = DataLoader(dataset_gr, batch_size=subset_size, shuffle=False, drop_last=True)
    
    i = 0
    for (x_kq, _), (x_gr, _) in zip(data_loader_kq, data_loader_gr):
        seed = generate_unique_seed(config, i)
        i += 1
        p.model.input_layer.seed = seed 
        p.model.reservoir_layer.seed = seed 
        p.model.output_layer.seed = seed 
        model = BooleanReservoir(p)
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
    p.dataset.update_path()
    return TemporalDensityDataset(p.dataset)

def get_generalization_rank_dataset(p: Params):
    dataset = get_kernel_quality_dataset(p)

    # define subsets
    subset_size = p.model.reservoir_layer.n_nodes # square matrix per rank calculation
    m = len(dataset)
    n_subsets = m // subset_size 
    indices = torch.randperm(m)[:n_subsets]

    # override last tao entries with similar inputs
    x = dataset.data['x']
    shape = x.shape
    x = x.view(shape[0], -1)
    tao = p.dataset.tao
    x[:, -tao:] = x[indices][:, -tao:].repeat_interleave(subset_size, dim=0)
    dataset.data['x'] = x.view(shape)
    return dataset

def calc_kernel_quality_and_generalization_rank(yaml_path, samples_per_configuration):
    # get n_node samples for KQ and GR, as well as some graph properties
    # written with the assumption that n_nodes doesnt change
    P = load_yaml_config(yaml_path)
    P.dataset.samples = P.model.reservoir_layer.n_nodes * samples_per_configuration
    P.dataset.update_path()
    param_combinations = generate_param_combinations(P)
    df = prepare_metrics_data(param_combinations)
    return df, P

def prepare_metrics_data(param_combinations: list[Params]):
    data = []
    for i, p in enumerate(tqdm(param_combinations, desc="Processing configurations")):
        p.dataset.seed = i
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
    paths.append('config/temporal/reservoir/kg_and_gr_homogenous_w_in_gt_1.yaml')
    paths.append('config/temporal/reservoir/kg_and_gr_heterogenous_w_in_gt_1.yaml')
    for path in paths:
        df, P = calc_kernel_quality_and_generalization_rank(path, samples_per_configuration=25)
        df.loc[:, 'k_avg'] = df['params'].apply(lambda p: p.model.reservoir_layer.k_avg)

        grouped_df = group_df_data_by_parameters(df)
        plot_kq_and_gr_many_config(grouped_df, P, 'many_config.png')
        for i, (name, subset) in enumerate(sorted(grouped_df, key=lambda x: x[0])):
            print(i, ':')
            p = subset.iloc[0]['params']
            print(p.model.input_layer)
            print(p.model.reservoir_layer)
            print(p.model.output_layer)
            print(p.dataset)
            plot_kq_and_gr(subset, P, f'config_{i}_kq_and_gr.png')

        # plot_optimal_k_vs_k_avg(df)


        # subset = df[df['params'].apply(lambda p: p.model.reservoir_layer.mode == 'homogenous')]
        # plot_kq_and_gr(subset, P, 'homogenous_kq_and_gr.png')
        # subset = df[df['params'].apply(lambda p: p.model.reservoir_layer.mode == 'heterogenous')]
        # plot_kq_and_gr(subset, P, 'heterogenous_kq_and_gr.png')

        # subset = df[df['params'].apply(lambda p: p.model.reservoir_layer.mode == 'homogenous') & (df['metric'] == 'kq')]
        # plot_kq_and_gr(subset, P, 'homogenous_kq.png')
        # subset = df[df['params'].apply(lambda p: p.model.reservoir_layer.mode == 'heterogenous') & (df['metric'] == 'kq')]
        # plot_kq_and_gr(subset, P, 'heterogenous_kq.png')

        # subset = df[df['params'].apply(lambda p: p.model.reservoir_layer.mode == 'homogenous') & (df['metric'] == 'gr')]
        # plot_kq_and_gr(subset, P, 'homogenous_gr.png')
        # subset = df[df['params'].apply(lambda p: p.model.reservoir_layer.mode == 'heterogenous') & (df['metric'] == 'gr')]
        # plot_kq_and_gr(subset, P, 'heterogenous_gr.png')