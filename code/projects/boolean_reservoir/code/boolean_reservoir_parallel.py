from projects.boolean_reservoir.code.utils import set_seed, generate_unique_seed
from projects.boolean_reservoir.code.reservoir import BooleanReservoir
from projects.boolean_reservoir.code.train_model import train_and_evaluate
from projects.boolean_reservoir.code.parameters import *
from projects.boolean_reservoir.code.visualizations import plot_grid_search
from projects.parallel_grid_search.code.train_model_parallel import generic_parallel_grid_search
from projects.parallel_grid_search.code.parallel_utils import JobInterface
from copy import deepcopy
import hashlib
import torch

import logging
logger = logging.getLogger(__name__)


class BooleanReservoirJob(JobInterface):
    """Specific implementation for Boolean Reservoir jobs"""
    
    def __init__(self, i: int, j: int, total_configs: int, total_samples: int, shared: dict, locks: dict, 
                 P, dataset_init, accuracy):
        super().__init__(i, j, total_configs, total_samples, shared, locks)
        self.P = P
        self.dataset_init = dataset_init
        self.accuracy = accuracy
    
    def _run(self, device):
        """Run job and process results internally"""
        # Train model
        dataset = self._get_dataset(device)
        model = BooleanReservoir(self.P).to(device)
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

    def _get_dataset(self, device: torch.device): # TODO make this a own class that can be inherited optionally in parallel_utils.py. Problem: self.P, connection to shared and locks...
        """Get dataset from cache or create new one"""
        cache_key = self._get_dataset_key(device)
        logger.debug(f"Loading dataset for key {cache_key} on device {device}")
        
        dataset = self._get_dataset_from_cache(cache_key)
        if dataset is None:
            logger.debug(f"Initializing dataset for key {cache_key} on device {device}")
            dataset = self.dataset_init(P=self.P)
            self._put_dataset_in_cache(cache_key, dataset)
        
        return dataset.to(device)
    
    def _get_dataset_key(self, device: torch.device):
        """Generate cache key for dataset"""
        dataset_params = deepcopy(self.P.D)
        param_str = str(sorted(dataset_params.model_dump()))
        key_str = f"{param_str}:{device}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_dataset_from_cache(self, key):
        with self.locks['dataset_cache']:
            dataset = self.shared['dataset_cache'].get(key, None)
            if dataset is not None:
                self.shared['dataset_scores'][key] = self.shared['dataset_scores'].get(key, 0) + 1
            return dataset

    def _put_dataset_in_cache(self, key, dataset, max_cache_size=10):
        with self.locks['dataset_cache']:
            if key not in self.shared['dataset_cache']:
                if len(self.shared['dataset_cache']) >= max_cache_size:
                    # Evict the dataset with the lowest score
                    if self.shared['dataset_scores']:
                        min_key = min(self.shared['dataset_scores'], key=self.shared['dataset_scores'].get)
                        del self.shared['dataset_cache'][min_key]
                        del self.shared['dataset_scores'][min_key]
                self.shared['dataset_cache'][key] = dataset
                self.shared['dataset_scores'][key] = 1

    
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
            file_path = output_path / 'log.h5'
            history_df.to_hdf(file_path, key='df', mode='w')
            plot_grid_search(file_path)
            logger.info(f"Saved {len(history)} results to {file_path}")
        
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