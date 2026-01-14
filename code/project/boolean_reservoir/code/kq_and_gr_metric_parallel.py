import pandas as pd
import torch
from torch.utils.data import DataLoader
from project.boolean_reservoir.code.utils.utils import set_seed, generate_unique_seed, save_grid_search_results, override_symlink
from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.kq_and_gr_metric import DatasetInitKQGR
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations 
from project.boolean_reservoir.code.graph import calc_spectral_radius
from project.parallel_grid_search.code.train_model_parallel import generic_parallel_grid_search
from project.parallel_grid_search.code.parallel_utils import JobInterface
from copy import deepcopy
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


def process_batch(model: BooleanReservoir, x: torch.Tensor, metric: str, data: list, config: int, sample: int):
    """Process one batch through model and compute rank"""
    # Nest history by metric
    nested_out = model.L.save_path / 'history' / metric
    new_save_path = nested_out / 'history'
    model.history = BatchedTensorHistoryWriter(
        save_path=new_save_path, 
        buffer_size=model.history.buffer_size,
        persist_to_disk=model.P.L.save_keys is not None
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


class BooleanReservoirKQGRJob(JobInterface):
    """Job that computes KQ and GR metrics for one config across all samples"""
    
    def __init__(self, i: int, total_configs: int, shared: dict, locks: dict, 
                 P: Params, dataset_init: DatasetInitKQGR):
        # Note: j is not used as we process all samples internally
        super().__init__(i, j=0, total_configs=total_configs, total_samples=1, shared=shared, locks=locks)
        self.P = P
        self.dataset_init = dataset_init
        self.mem_disk_efficient = self.P.L.save_keys is None
    
    def _run(self, device):
        """Run job: process all samples for this config"""
        data = []
        
        # Generate datasets once for this config (expanded to n_nodes * num_samples)
        with self.locks['dataset_lock']:
            kq_dataset, gr_dataset = self.dataset_init(self.P)
            kq_dataset = kq_dataset.to(device)
            gr_dataset = gr_dataset.to(device)
        
        # Create data loaders that batch by n_nodes
        subset_size = self.P.M.R.n_nodes
        kq_loader = DataLoader(kq_dataset, batch_size=subset_size, shuffle=False, drop_last=True)
        gr_loader = DataLoader(gr_dataset, batch_size=subset_size, shuffle=False, drop_last=True)
        
        # Process each batch (each batch = one sample/model seed)
        for j, ((x_kq, _), (x_gr, _)) in enumerate(zip(kq_loader, gr_loader)):
            # Generate unique seed for this model
            seed = generate_unique_seed(self.P.L.grid_search.seed, self.i, j)
            pj = deepcopy(self.P)
            pj.M.I.seed = pj.M.R.seed = pj.M.O.seed = seed
            
            # Create and run model
            model = BooleanReservoir(pj).to(device)
            model.save()
            model.eval()
            
            # Record params and spectral radius
            data.append({
                'config': self.i,
                'sample': j,
                'metric': 'params',
                'value': pj
            })
            
            spectral_radius = calc_spectral_radius(model.graph)
            data.append({
                'config': self.i,
                'sample': j,
                'metric': 'spectral_radius',
                'value': spectral_radius
            })
            
            # Process KQ and GR
            process_batch(model, x_kq, 'kq', data, self.i, j)
            process_batch(model, x_gr, 'gr', data, self.i, j)
        
        # Convert to DataFrame with pivot structure
        df = pd.DataFrame(data)
        pivot_df = df.pivot_table(index=['config', 'sample'], columns='metric', values='value', aggfunc='first').reset_index()
        pivot_df['delta'] = pivot_df['kq'] - pivot_df['gr']
        melted_df = pd.melt(pivot_df, id_vars=['config', 'sample', 'params', 'spectral_radius'], 
                           value_vars=['spectral_radius', 'kq', 'gr', 'delta'], 
                           var_name='metric', value_name='value')
        result_df = melted_df.sort_values(by=['config', 'sample']).reset_index(drop=True)
        
        # Update shared history with all results
        with self.locks['history']:
            for _, row in result_df.iterrows():
                self.shared['history'].append(row.to_dict())
        
        logger.info(f"Config {self.i}: Completed {j+1} samples")
        return {'status': 'completed', 'num_samples': j+1}


def boolean_reservoir_kq_gr_job_factory(P: Params, param_combinations: list, dataset_init: DatasetInitKQGR):
    """Factory for KQ/GR jobs - one job per config"""
    def create_job(i, j, total_configs, total_samples, shared, locks):
        # j is ignored - each job processes all samples internally
        p = param_combinations[i]
        # Override samples to expand for matrix calculation
        p.D.samples = p.M.R.n_nodes * p.D.samples
        p.D.update_path()
        return BooleanReservoirKQGRJob(
            i=i,
            total_configs=total_configs,
            shared=shared,
            locks=locks,
            P=p,
            dataset_init=dataset_init
        )
    return create_job

def boolean_reservoir_kq_gr_grid_search(
    yaml_path: str,
    dataset_init: DatasetInitKQGR = DatasetInitKQGR(),
    param_combinations: list = None,
    gpu_memory_per_job_gb: float = 1,
    cpu_memory_per_job_gb: float = 1,
    cpu_cores_per_job: int = 1,
):
    """Grid search for KQ/GR metrics - one job per config"""
    yaml_path = Path(yaml_path)
    P = load_yaml_config(yaml_path)
    set_seed(P.L.grid_search.seed)
    
    if param_combinations is None:
        param_combinations = generate_param_combinations(P)
    
    # Create job factory
    factory = boolean_reservoir_kq_gr_job_factory(P, param_combinations, dataset_init)
    
    # Define callbacks
    def save_config(output_path):
        save_yaml_config(P, output_path / 'parameters.yaml')
    
    def process_results(history, output_path, done):
        file_path = output_path / 'log.yaml'
        
        if history:
            history_df = pd.DataFrame(history)
            save_grid_search_results(history_df, file_path)
    
    # Run generic grid search with samples_per_config=1 (one job per config)
    generic_parallel_grid_search(
        job_factory=factory,
        total_configs=len(param_combinations),
        samples_per_config=1,  # Each job handles all samples internally
        output_path=P.L.out_path,
        gpu_memory_per_job_gb=gpu_memory_per_job_gb,
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        save_config=save_config,
        process_results=process_results,
    )
    return P
