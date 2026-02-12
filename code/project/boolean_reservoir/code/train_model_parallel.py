import pandas as pd
from project.boolean_reservoir.code.utils.utils import set_seed, generate_unique_seed
from project.boolean_reservoir.code.utils.load_save import save_grid_search_results 
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model import train_and_evaluate
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations 
from project.boolean_reservoir.code.visualization import plot_grid_search
from project.parallel_grid_search.code.train_model_parallel import generic_parallel_grid_search
from project.parallel_grid_search.code.parallel_utils import JobInterface
from project.boolean_reservoir.code.kq_and_gr_metric import compute_rank
from project.boolean_reservoir.code.graph import calc_spectral_radius
from copy import deepcopy
from typing import Callable
from torch.utils.data import Dataset
from torch import compile
import logging
logger = logging.getLogger(__name__)


class BooleanReservoirJob(JobInterface):
    """Grid search job - runs KQGR and/or training based on config"""
    
    def __init__(self, i: int, j: int, total_configs: int, total_samples: int, 
                 shared: dict, locks: dict, P: Params,
                 dataset_init: Optional[Callable[..., Dataset]] = None,
                 accuracy = None):
        super().__init__(i, j, total_configs, total_samples, shared, locks)
        self.P: Params = P
        self.dataset_init = dataset_init
        self.accuracy = accuracy
    
    def _run_kqgr(self, model: BooleanReservoir, device):
        """Compute KQGR metrics on untrained reservoir"""
        # Note that self.P and model.P differ here (accept and avoid re-initializing model)
        # Set parameters for making/using KQ and GR datasets
        self.P.DD.kqgr.samples = self.P.M.R.n_nodes # square history matrix of samples and node states per rank sample
        tau = self.P.DD.kqgr.tau
        self.P.DD.kqgr.tau = 0 # dont set identical bits for GR → KQ
        kq_dataset = self.dataset_init(self.P).to(device)
        self.P.DD.kqgr.tau = tau # set identical bits for GR
        gr_dataset = self.dataset_init(self.P).to(device)

        model.save()
        
        spectral_radius = float(calc_spectral_radius(model.graph))
        kq_rank = compute_rank(model, kq_dataset.data['x'], 'kq')
        gr_rank = compute_rank(model, gr_dataset.data['x'], 'gr')

        model.init_logging() # restore model
        
        self.P.L.kqgr = KQGRMetrics(
            config=self.i,
            sample=self.j,
            kq=kq_rank,
            gr=gr_rank,
            delta=kq_rank - gr_rank,
            spectral_radius=spectral_radius
        )
        
        logger.info(f"Config {self.i}, Sample {self.j}: KQ={kq_rank}, GR={gr_rank}, Δ={kq_rank - gr_rank}")
    
    def _run_training(self, model: BooleanReservoir, device):
        """Train and evaluate model"""
        with self.locks['dataset_lock']:
            dataset = self.dataset_init(self.P).to(device)
        
        model = compile(model)
        best_epoch, trained_model, _ = train_and_evaluate(
            model, dataset, record_stats=False, verbose=False, accuracy=self.accuracy
        )
        
        if hasattr(trained_model, 'save') and callable(trained_model.save):
            trained_model.save()

        self.P.L.train = TrainLog(
            config=self.i,
            sample=self.j,
            accuracy=best_epoch['accuracy'],
            loss=best_epoch['loss'],
            epoch=best_epoch['epoch']
        )
        
        logger.info(f"Config {self.i}, Sample {self.j}: "
                    f"Loss: {self.P.L.train.loss:.4f}, Accuracy: {self.P.L.train.accuracy:.4f}")
    
    def _run(self, device):
        # model.reset_reservoir is called first on each forward pass if reset is enabled (default)
        model = BooleanReservoir(self.P).to(device)

        if self.P.DD.kqgr:
            self._run_kqgr(model, device)

        if self.P.DD.train:
            self._run_training(model, device)
        
        with self.locks['history']:
            self.shared['history'].append(self.P)
        
        return {'status': 'completed'}

    
def apply_seed(p: Params, seed: int):
    """Apply seed to all relevant param sections"""
    p.M.I.seed = p.M.R.seed = p.M.O.seed = seed
    if p.M.T:
        p.M.T.seed = seed

class BooleanReservoirJobFactory: # Note: assumes dataset_init handles KQGR if needed as well as training
    def __init__(self, P: Params, param_combinations: list,
                 dataset_init: Optional[Callable] = None,
                 accuracy = None):
        self.P = P
        self.param_combinations = param_combinations
        self.dataset_init = dataset_init
        self.accuracy = accuracy
        
        if self.P.DD.train and not self.accuracy:
            raise ValueError("Need accuracy function if training.")
    
    def __call__(self, i, j, total_configs, total_samples, shared, locks):
        p = self._override_parameter(self.P, i, j)
        return BooleanReservoirJob(
            i=i, j=j,
            total_configs=total_configs,
            total_samples=total_samples,
            shared=shared,
            locks=locks,
            P=p,
            dataset_init=self.dataset_init,
            accuracy=self.accuracy
        )
    
    def _override_parameter(self, p: Params, i: int, j: int):
        """Override parameters for given config/sample index"""
        p_new = deepcopy(self.param_combinations[i])
        seed = generate_unique_seed(self.P.L.grid_search.seed, i, j)
        apply_seed(p_new, seed)
        return p_new

def boolean_reservoir_grid_search(
    yaml_path: str,
    dataset_init,
    accuracy,
    param_combinations: list = None,
    gpu_memory_per_job_gb: float = 1,
    cpu_memory_per_job_gb: float = 1,
    cpu_cores_per_job: int = 1,
):
    """Boolean Reservoir specific grid search using the generic function"""
    yaml_path = Path(yaml_path)
    P: Params = load_yaml_config(yaml_path)
    set_seed(P.L.grid_search.seed)
    
    if param_combinations is None:
        param_combinations = generate_param_combinations(P)
    
    # Create job factory
    factory = BooleanReservoirJobFactory(P, param_combinations, dataset_init=dataset_init, accuracy=accuracy)
    
    # Define callbacks
    def save_config(output_path: Path):
        save_yaml_config(P, output_path, copy_from_original_file_path=yaml_path)

    def process_results(history, output_path, done):
        file_path = output_path / 'log.yaml'
        if history:
            df = pd.DataFrame({'params': history})
            save_grid_search_results(df, file_path)
        if done:
            pass
    
    # Run generic grid search
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
