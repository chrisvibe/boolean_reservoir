import pandas as pd
import torch
from project.boolean_reservoir.code.utils.utils import set_seed, generate_unique_seed, save_grid_search_results, override_symlink
from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from project.boolean_reservoir.code.parameter import Params, LoggingParams, KQGRMetrics, load_yaml_config, save_yaml_config
from project.boolean_reservoir.code.kq_and_gr_metric import DatasetInitKQGR
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.boolean_reservoir.code.graph import calc_spectral_radius
from project.parallel_grid_search.code.train_model_parallel import generic_parallel_grid_search
from project.parallel_grid_search.code.parallel_utils import JobInterface
from copy import deepcopy
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_rank(model: BooleanReservoir, x: torch.Tensor, metric: str) -> int:
    """Run model and compute rank from reservoir states"""
    nested_out = model.L.save_path / 'history' / metric
    new_save_path = nested_out / 'history'
    model.history = BatchedTensorHistoryWriter(
        save_path=new_save_path,
        buffer_size=model.history.buffer_size,
        persist_to_disk=model.P.L.save_keys is not None
    )
    
    with torch.no_grad():
        _ = model(x)
    model.flush_history()
    
    override_symlink(Path('../../checkpoint'), new_save_path / 'checkpoint')
    load_dict, history, expanded_meta, meta = model.history.reload_history()
    df_filter = expanded_meta[expanded_meta['phase'] == 'output_layer']
    filtered_history = history[df_filter.index].to(torch.float)
    reservoir_node_history = filtered_history[:, ~model.input_nodes_mask]
    
    return torch.linalg.matrix_rank(reservoir_node_history).item()


class BooleanReservoirKQGRJob(JobInterface):
    """Job that computes KQ and GR metrics for ONE config + ONE sample"""
    
    def __init__(self, i: int, j: int, total_configs: int, total_samples: int,
                 shared: dict, locks: dict, P: Params, dataset_init: DatasetInitKQGR):
        super().__init__(i, j, total_configs, total_samples, shared, locks)
        self.P = P
        self.dataset_init = dataset_init
    
    def _run(self, device):
        """Run job: compute KQ and GR for single (config, sample) pair"""
        # Generate datasets
        kq_dataset, gr_dataset = self.dataset_init(self.P)
        kq_dataset = kq_dataset.to(device)
        gr_dataset = gr_dataset.to(device)
        
        x_kq = kq_dataset.data['x']
        x_gr = gr_dataset.data['x']
        
        # Create model
        model = BooleanReservoir(self.P).to(device)
        model.save()
        model.eval()
        
        # Compute metrics
        spectral_radius = calc_spectral_radius(model.graph)
        kq_rank = compute_rank(model, x_kq, 'kq')
        gr_rank = compute_rank(model, x_gr, 'gr')
        
        # Store in params
        self.P.L.kqgr = KQGRMetrics(
            config=self.i,
            sample=self.j,
            kq=kq_rank,
            gr=gr_rank,
            delta=kq_rank - gr_rank,
            spectral_radius=spectral_radius
        )
        
        # Append params to history
        with self.locks['history']:
            self.shared['history'].append(self.P)
        
        logger.info(f"Config {self.i}, Sample {self.j}: KQ={kq_rank}, GR={gr_rank}, Δ={kq_rank - gr_rank}")
        return {'status': 'completed'}


def boolean_reservoir_kq_gr_job_factory(P: Params, param_combinations: list, dataset_init: DatasetInitKQGR):
    """Factory - creates one job per (config, sample) pair"""
    def create_job(i, j, total_configs, total_samples, shared, locks):
        p = deepcopy(param_combinations[i])
        
        # Dataset size = n_nodes (single sample)
        p.D.samples = p.M.R.n_nodes
        p.D.update_path()
        
        # Unique seed for this (config, sample)
        seed = generate_unique_seed(P.L.grid_search.seed, i, j)
        p.D.seed = seed
        p.M.I.seed = p.M.R.seed = p.M.O.seed = seed
        
        return BooleanReservoirKQGRJob(
            i=i, j=j,
            total_configs=total_configs,
            total_samples=total_samples,
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
    """Grid search for KQ/GR metrics - one job per (config, sample)"""
    yaml_path = Path(yaml_path)
    P = load_yaml_config(yaml_path)
    set_seed(P.L.grid_search.seed)
    
    if param_combinations is None:
        param_combinations = generate_param_combinations(P)
    
    factory = boolean_reservoir_kq_gr_job_factory(P, param_combinations, dataset_init)
    
    def save_config(output_path):
        save_yaml_config(P, output_path / 'parameters.yaml')
    
    def process_results(history, output_path, done):
        if history:
            df = pd.DataFrame({'params': history})
            save_grid_search_results(df, output_path / 'log.yaml')
    
    generic_parallel_grid_search(
        job_factory=factory,
        total_configs=len(param_combinations),
        samples_per_config=P.L.grid_search.n_samples,
        output_path=P.L.out_path,
        gpu_memory_per_job_gb=gpu_memory_per_job_gb,
        cpu_memory_per_job_gb=cpu_memory_per_job_gb,
        cpu_cores_per_job=cpu_cores_per_job,
        save_config=save_config,
        process_results=process_results,
    )
    return P