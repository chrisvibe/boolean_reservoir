import pandas as pd
from project.boolean_reservoir.code.utils.utils import set_seed, generate_unique_seed, save_grid_search_results, load_grid_search_results
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model import train_and_evaluate
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations 
from project.boolean_reservoir.code.visualization import plot_grid_search
from project.parallel_grid_search.code.train_model_parallel import generic_parallel_grid_search
from project.parallel_grid_search.code.parallel_util import JobInterface
from copy import deepcopy
from typing import Callable
from torch.utils.data import Dataset
from torch import _dynamo, compile

import logging
logger = logging.getLogger(__name__)


class BooleanReservoirJob(JobInterface):
    """Specific implementation for Boolean Reservoir jobs"""
    
    def __init__(self, i: int, j: int, total_configs: int, total_samples: int, shared: dict, locks: dict, 
                 P, dataset_init: Callable[[], Dataset], accuracy):
        super().__init__(i, j, total_configs, total_samples, shared, locks)
        self.P = P
        self.accuracy = accuracy
        self.dataset_init = dataset_init
    
    def _run(self, device):
        """Run job and process results internally"""
        # Train model
        with self.locks['dataset_lock']:
            dataset = self.dataset_init(self.P).to(device)
        model = BooleanReservoir(self.P).to(device)
        if device.type == 'cuda': # slow on cpu atm
            _dynamo.reset()
            model = compile(model)
        best_epoch, trained_model, _ = train_and_evaluate(
            model, dataset, record_stats=False, verbose=False, accuracy=self.accuracy
        )

        if hasattr(trained_model, 'save') and callable(trained_model.save):
            trained_model.save()

        # Process results
        timestamp_utc = trained_model.timestamp_utc
        logger.info(f"{timestamp_utc}: {self.get_log_prefix()}, "
                        f"Loss: {best_epoch['loss']:.4f}, Accuracy: {best_epoch['accuracy']:.4f}")
        
        # Update shared history
        with self.locks['history']:
            self.shared['history'].append({
                'timestamp_utc': timestamp_utc,
                'config': self.i + 1,
                'sample': self.j + 1,
                **best_epoch,
                'params': self.P
            })
            
        # Update best params if needed
        with self.locks['best_params']:
            if self.shared['best_params']['loss'] is None or best_epoch['loss'] < self.shared['best_params']['loss']:
                self.shared['best_params']['loss'] = best_epoch['loss']
                self.shared['best_params']['params'] = self.P

        return {'status': 'success', 'stats': best_epoch, 'timestamp_utc': timestamp_utc}

    
# Factory for Boolean Reservoir jobs
def boolean_reservoir_job_factory(P, param_combinations, dataset_init, accuracy):
    def create_job(i, j, total_configs, total_samples, shared, locks):
        p = param_combinations[i]
        pj = deepcopy(p)
        k = generate_unique_seed(P.L.grid_search.seed, i, j)
        pj.M.I.seed = pj.M.R.seed = pj.M.O.seed = pj.M.T.seed = k
        
        return BooleanReservoirJob(
            i=i, j=j, 
            total_configs=total_configs,
            total_samples=total_samples,
            shared=shared,
            locks=locks,
            P=pj,
            dataset_init=dataset_init,
            accuracy=accuracy
        )
    return create_job

def boolean_reservoir_grid_search(
    yaml_path: str,
    dataset_init,
    accuracy,
    param_combinations: list = None,
    gpu_memory_per_job_gb: float = None,
    cpu_memory_per_job_gb: float = None,
    cpu_cores_per_job: int = 1,
):
    """Boolean Reservoir specific grid search using the generic function"""
    yaml_path = Path(yaml_path)
    P = load_yaml_config(yaml_path)
    set_seed(P.L.grid_search.seed)
    
    if param_combinations is None:
        param_combinations = generate_param_combinations(P)
    
    # Create job factory
    factory = boolean_reservoir_job_factory(P, param_combinations, dataset_init, accuracy)
    
    # Define callbacks
    def save_config(output_path):
        save_yaml_config(P, output_path / 'parameters.yaml')
    
    def process_results(history, best_params, output_path):
        if history:
            history_df = pd.DataFrame(history)
            file_path = output_path / 'log.yaml'
            save_grid_search_results(history_df, file_path)
            plot_grid_search(file_path)
        
        if best_params and 'params' in best_params:
            logger.info(f"Best parameters found: {best_params['params']}")
    
    # Run generic grid search
    history, best_params = generic_parallel_grid_search(
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
    
    return P, best_params.get('params') if best_params else None
