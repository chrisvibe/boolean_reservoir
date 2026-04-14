import pandas as pd
from project.boolean_reservoir.code.utils.utils import set_seed, generate_unique_seed
from project.boolean_reservoir.code.utils.load_save import save_grid_search_results 
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model import train_and_evaluate, DatasetInit
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations 
from project.boolean_reservoir.code.visualization import plot_grid_search
from project.parallel_grid_search.code.train_model_parallel import generic_parallel_grid_search
from project.parallel_grid_search.code.parallel_utils import JobInterface
from project.boolean_reservoir.code.kq_and_gr_metric import compute_rank
from project.boolean_reservoir.code.graph import calc_spectral_radius
from copy import deepcopy
from torch import compile
import logging
logger = logging.getLogger(__name__)


class BooleanReservoirJob(JobInterface):
    """Grid search job - runs KQGR and/or training based on config"""

    def __init__(self, i: int, j: int, total_configs: int, total_samples: int,
                 locks: dict, P: Params, compile_model: bool = False):
        super().__init__(i, j, total_configs, total_samples, locks)
        self.P: Params = P
        self.dataset_init = P.dataset_init_obj
        self.accuracy = P.accuracy_obj
        self.compile_model = compile_model

    def _init_dataset(self, init_fn, device, *args, **kwargs):
        with self.locks['dataset_lock']:
            dataset = init_fn(*args, **kwargs)
        return dataset.to(device)

    def _run_kqgr(self, model: BooleanReservoir, device, P_universe: Params, name: str = 'kqgr'):
        """Compute KQGR metrics on untrained reservoir using the given universe Params."""
        logger.debug(f"Config {self.i}, Sample {self.j}: initialising KQGR datasets")
        universe_dataset_init = P_universe.dataset_init_obj
        kq_dataset = self._init_dataset(universe_dataset_init.kqgr, device, P_universe, kq=True)
        gr_dataset = self._init_dataset(universe_dataset_init.kqgr, device, P_universe, kq=False)
        logger.debug(f"Config {self.i}, Sample {self.j}: computing KQGR ranks")

        spectral_radius = float(calc_spectral_radius(model.graph))
        kq_rank = compute_rank(model, kq_dataset.data['x'], 'kq')
        gr_rank = compute_rank(model, gr_dataset.data['x'], 'gr')

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
        logger.debug(f"Config {self.i}, Sample {self.j}: initialising training dataset")
        dataset = self._init_dataset(self.dataset_init.train, device, self.P)
        logger.debug(f"Config {self.i}, Sample {self.j}: starting train_and_evaluate")
        if self.compile_model:
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
        logger.debug(f"Config {self.i}, Sample {self.j}: building model")
        model = BooleanReservoir(self.P).to(device)

        # Record which universe this config belongs to (None for Mother configs).
        # Derived from multiverse_overrides, which has exactly one key per expanded config.
        universe_name = next(iter(self.P.multiverse_overrides or {}), None)
        self.P.L.universe = universe_name

        run = self.P.L.grid_search.run if self.P.L.grid_search else None
        if run is None:
            run = ['kqgr'] if universe_name else ['train']

        if 'kqgr' in run and universe_name:
            P_universe = getattr(self.P.U, universe_name)
            assert P_universe.M.R == self.P.M.R, (
                f"Universe '{universe_name}' overrides reservoir_layer params — not allowed: "
                "graph/lut are shared from the training model to guarantee determinism. "
                "Only input_layer, output_layer, and dataset may be overridden."
            )
            # Share graph/lut/init_state from the training model instead of re-generating.
            # Re-initialization from the same seed is not guaranteed deterministic on GPU
            # (async CUDA ops between set_seed(I.seed) and set_seed(R.seed) can corrupt
            # the random state). Passing load_dict bypasses graph/lut generation entirely.
            # lut/init_state must be on CPU so __init__ (which calls reset_reservoir
            # before .to(device)) sees a consistent device for all buffers.
            kqgr_model = BooleanReservoir(P_universe, load_dict={
                'graph': model.graph,
                'lut': model.lut.cpu(),
                'init_state': model.initial_states.cpu(),
            }).to(device)
            self._run_kqgr(kqgr_model, device, P_universe, name=universe_name)

        if 'train' in run:
            self._run_training(model, device)

        return {'status': 'completed', 'history': self.P}

    
def apply_seed(p: Params, seed: int):
    """Apply seed to all relevant param sections"""
    p.M.I.seed = p.M.R.seed = p.M.O.seed = seed
    if p.M.T:
        p.M.T.seed = seed

class BooleanReservoirJobFactory:
    def __init__(self, P: Params, param_combinations: list, compile_model: bool = False):
        self.P = P
        self.param_combinations = param_combinations
        self.compile_model = compile_model

    def __call__(self, i, j, total_configs, total_samples, locks):
        p = self._override_parameter(self.P, i, j)
        return BooleanReservoirJob(
            i=i, j=j,
            total_configs=total_configs,
            total_samples=total_samples,
            locks=locks,
            P=p,
            compile_model=self.compile_model,
        )
    
    def _override_parameter(self, p: Params, i: int, j: int):
        """Override parameters for given config/sample index"""
        p_new = deepcopy(self.param_combinations[i])
        seed = generate_unique_seed(self.P.L.grid_search.seed, i, j)
        apply_seed(p_new, seed)
        return p_new

def boolean_reservoir_grid_search(
    yaml_path: str,
    param_combinations: list = None,
    gpu_memory_per_job_gb: float = 2,
    cpu_memory_per_job_gb: float = 2,
    cpu_cores_per_job: int = 1,
    exploration_rate: float = 0.1,
    compile_model: bool = True,
):
    """Boolean Reservoir specific grid search using the generic function"""
    yaml_path = Path(yaml_path)
    P: Params = load_yaml_config(yaml_path)
    set_seed(P.L.grid_search.seed)

    if param_combinations is None:
        param_combinations = generate_param_combinations(P)

    # Create job factory
    factory = BooleanReservoirJobFactory(P, param_combinations, compile_model=compile_model)
    
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
        exploration_rate=exploration_rate,
        save_config=save_config,
        process_results=process_results,
    )
    return P
