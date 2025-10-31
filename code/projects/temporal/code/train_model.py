# Finding Optimal Random Boolean Networks for Reservoir Computing  David Snyder1, Alireza Goudarzi2, and Christof Teuscher3
from os import environ
from projects.boolean_reservoir.code.train_model import BooleanAccuracy as a, train_single_model
from projects.temporal.code.dataset_init import TemporalDatasetInit as d
from projects.temporal.code.visualizations import plot_many_things
from projects.boolean_reservoir.code.visualizations import plot_activity_trace
from projects.boolean_reservoir.code.boolean_reservoir_parallel import boolean_reservoir_grid_search 

import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(process)d - %(filename)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    pass 

    # # # debug 
    # # #####################################
    # from projects.boolean_reservoir.code.reservoir import BooleanReservoir
    # from projects.boolean_reservoir.code.utils import print_pretty_binary_matrix
    # import torch
    # from projects.boolean_reservoir.code.parameters import generate_param_combinations, load_yaml_config
    # p = load_yaml_config('config/temporal/density/grid_search/homogeneous_deterministic.yaml')
    # p.L.out_path = '/out/debug'
    # p.L.history.record_history = True
    # p.L.save_keys = ['parameters', 'w_in', 'graph', 'init_state', 'lut', 'weights']
    # p.M.I.connection = 'out-3:3:1'
    # p.M.I.n_nodes = 10
    # p.M.I.seed = p.M.R.seed = p.M.O.seed = 1
    # p.M.I.pertubation = 'override'
    # # p.M.I.pertubation = 'xor'
    # p.M.R.init = 'zeros'
    # p.M.R.n_nodes = 30
    # p.M.R.k_avg = 4
    # configs = generate_param_combinations(p)
    # model = BooleanReservoir(configs[0])
    # # model = BooleanReservoir(load_path='/out/test/temporal/density/grid_search/homogeneous-deterministic/runs/2025_06_25_125038_810586/checkpoints/last_checkpoint')
    # x = torch.tensor([[[[int(bit)]] for bit in '1001001010']], dtype=torch.uint8)
    # # model(x)
    # # model.save()
    # # model.flush_history()
    # p, model, dataset, history = train_single_model(model=model, dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])
    # pass


    # # these just add to the grid search below a 50% model and a 100% model
    # p, model, dataset, history = train_single_model('/out/test/temporal/density/grid_search/homogeneous-deterministic/runs/2025_06_25_125038_810586/checkpoints/last_checkpoint/parameters.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # p, model, dataset, history = train_single_model('/out/test/temporal/density/grid_search/homogeneous-deterministic/runs/2025_06_25_125107_553781/checkpoints/last_checkpoint/parameters.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)

    # boolean_reservoir_grid_search(
    #     'config/temporal/density/test/debug_homogeneous_deterministic.yaml',
    #     dataset_init=d().dataset_init,
    #     accuracy=a().accuracy,
    #     gpu_memory_per_job_gb = 1/2,
    #     cpu_memory_per_job_gb = 1/2,
    #     cpu_cores_per_job = 1,
    # )

    # # # Simple run
    # # #####################################

    # p, model, dataset, history = train_single_model('config/temporal/density/single_run/ok_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # p, model, dataset, history = train_single_model('config/temporal/density/single_run/sample_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # Grid search stuff 
    #####################################
    configs = [
        'config/temporal/density/grid_search/homogeneous_stochastic.yaml',
        'config/temporal/density/grid_search/homogeneous_deterministic.yaml',
        'config/temporal/density/grid_search/heterogeneous_stochastic.yaml',
        'config/temporal/density/grid_search/heterogeneous_deterministic.yaml',

        'config/temporal/parity/grid_search/homogeneous_stochastic.yaml',
        'config/temporal/parity/grid_search/homogeneous_deterministic.yaml',
        'config/temporal/parity/grid_search/heterogeneous_stochastic.yaml',
        'config/temporal/parity/grid_search/heterogeneous_deterministic.yaml',

        # 'config/temporal/density/test/heterogeneous_deterministic.yaml',
        # 'config/temporal/density/grid_search/homogeneous_stochastic.yaml',

        'config/temporal/density/grid_search/test_optimizer_and_readout_mode.yaml',
    ]

    node = environ.get("SLURMD_NODENAME") or environ.get("SLURM_NODELIST", "unknown")
    if "hpc" in node:
        logger.info(f"This is hpc node: {node}")
    else:
        logger.warning(f"Unknown node detected: {node}")

    node_job_assigments = {
        1: [0, 7],
        5: [1, 6],
        # 7: [2, 5],
        7: [-1],
        8: [3, 4],
        10: [-1],
        11: [-1],
        'unknown': [-1],
    }
    if node != 'unknown':
        id = int(node[3:])
        configs = [configs[idx] for idx in node_job_assigments[id]]
    else:
        configs = [configs[idx] for idx in node_job_assigments['unknown']]

    for c in configs:
        boolean_reservoir_grid_search(
            c,
            dataset_init=d().dataset_init,
            accuracy=a().accuracy,
            gpu_memory_per_job_gb = 1/2,
            cpu_memory_per_job_gb = 1/2,
            cpu_cores_per_job = 1,
        )