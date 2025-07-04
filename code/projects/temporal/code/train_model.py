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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(process)d - %(filename)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    pass 

    # # debug 
    # #####################################

    # from projects.boolean_reservoir.code.reservoir import BooleanReservoiear
    # from projects.boolean_reservoir.code.utils import print_pretty_binary_matrix
    # import torch
    # x = torch.tensor([[[[int(bit)]] for bit in '0001001010']], dtype=torch.uint8)
    # model = BooleanReservoir(load_path='/out/test/temporal/density/grid_search/homogeneous-deterministic/runs/2025_06_25_125038_810586/checkpoints/last_checkpoint')
    # model(x)
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

    # # Simple run
    # #####################################

    # p, model, dataset, history = train_single_model('config/temporal/density/single_run/ok_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # p, model, dataset, history = train_single_model('config/temporal/density/single_run/sample_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # # Grid search stuff 
    # #####################################
    configs = [
        # 'config/temporal/density/test/heterogeneous_deterministic.yaml',

        'config/temporal/density/grid_search/homogeneous_stochastic.yaml',
        'config/temporal/density/grid_search/homogeneous_deterministic.yaml',
        'config/temporal/density/grid_search/heterogeneous_stochastic.yaml',
        'config/temporal/density/grid_search/heterogeneous_deterministic.yaml',

        'config/temporal/parity/grid_search/homogeneous_stochastic.yaml',
        'config/temporal/parity/grid_search/homogeneous_deterministic.yaml',
        'config/temporal/parity/grid_search/heterogeneous_stochastic.yaml',
        'config/temporal/parity/grid_search/heterogeneous_deterministic.yaml',

    ]

    node = environ.get("SLURMD_NODENAME") or environ.get("SLURM_NODELIST", "unknown")
    if "hpc" in node:
        logger.info(f"This is hpc node: {node}")
    else:
        logger.warning(f"Unknown node detected: {node}")

    node_job_assigments = {
        1: [0, 2],
        2: [1, 3],
        3: [4, 6],
        5: [5, 7],
        10: [],
        'unknown': [0],
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