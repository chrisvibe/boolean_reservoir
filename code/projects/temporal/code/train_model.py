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
        'config/temporal/parity/grid_search/homogeneous_stochastic.yaml',
        'config/temporal/parity/grid_search/homogeneous_deterministic.yaml',

        'config/temporal/density/grid_search/heterogeneous_stochastic.yaml',
        'config/temporal/density/grid_search/heterogeneous_deterministic.yaml',
        'config/temporal/parity/grid_search/heterogeneous_stochastic.yaml',
        'config/temporal/parity/grid_search/heterogeneous_deterministic.yaml',

    ]

    node = environ.get("SLURMD_NODENAME") or environ.get("SLURM_NODELIST", "unknown")
    if "hpc10" in node:
        logger.info("This is the A100 node")
        configs = configs[::2]
    elif "hpc11" in node:
        configs = configs[1::2]
        logger.info("This is the H100 node")
    else:
        logger.warning(f"Unknown node detected: {node}")

    for c in configs:
        boolean_reservoir_grid_search(
            c,
            dataset_init=d().dataset_init,
            accuracy=a().accuracy,
            gpu_memory_per_job_gb = 0.5,
            cpu_memory_per_job_gb = 1,
            cpu_cores_per_job = 4,
        )