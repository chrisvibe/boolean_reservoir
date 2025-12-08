from project.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from project.boolean_reservoir.code.visualization import plot_train_history, plot_dynamics_history, plot_activity_trace
from project.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
from project.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from project.path_integration.code.visualization import plot_many_things
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search 

from os import environ
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

    # # Simple run
    # #####################################
    # p, model, dataset, history = train_single_model('config/path_integration/1D/single_run/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # p, model, dataset, history = train_single_model('config/path_integration/2D/single_run/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0], ir_subtitle=True)

    # from project.boolean_reservoir.test.profile_reservoir import profile_compile_main, profile_train_single_model_main
    # config = 'project/path_integration/test/config/2D/single_run/test_model_profiling.yaml'
    # profile_compile_main(config)
    # # profile_train_single_model_main(config)

    # Grid search stuff 
    #####################################
    configs = [
        # 'config/path_integration/1D/grid_search/heterogeneous_deterministic.yaml',
        # 'config/path_integration/1D/grid_search/heterogeneous_stochastic.yaml',
        # 'config/path_integration/1D/grid_search/homogeneous_deterministic.yaml',
        # 'config/path_integration/1D/grid_search/homogeneous_stochastic.yaml',

        # 'config/path_integration/2D/grid_search/heterogeneous_deterministic.yaml',
        # 'config/path_integration/2D/grid_search/heterogeneous_stochastic.yaml',
        # 'config/path_integration/2D/grid_search/homogeneous_deterministic.yaml',
        # 'config/path_integration/2D/grid_search/homogeneous_stochastic.yaml',

        # 'config/path_integration/1D/grid_search/no_self_loops/heterogeneous_deterministic.yaml',
        # 'config/path_integration/1D/grid_search/no_self_loops/heterogeneous_stochastic.yaml',
        # 'config/path_integration/1D/grid_search/no_self_loops/homogeneous_deterministic.yaml',
        # 'config/path_integration/1D/grid_search/no_self_loops/homogeneous_stochastic.yaml',

        # 'config/path_integration/2D/grid_search/no_self_loops/heterogeneous_deterministic.yaml',
        # 'config/path_integration/2D/grid_search/no_self_loops/heterogeneous_stochastic.yaml',
        # 'config/path_integration/2D/grid_search/no_self_loops/homogeneous_deterministic.yaml',
        # 'config/path_integration/2D/grid_search/no_self_loops/homogeneous_stochastic.yaml',

        # 'config/path_integration/1D/grid_search/3_steps/heterogeneous_deterministic.yaml',
        # 'config/path_integration/1D/grid_search/3_steps/heterogeneous_stochastic.yaml',
        # 'config/path_integration/1D/grid_search/3_steps/homogeneous_deterministic.yaml',
        # 'config/path_integration/1D/grid_search/3_steps/homogeneous_stochastic.yaml',

        # 'config/path_integration/2D/grid_search/3_steps/heterogeneous_deterministic.yaml',
        # 'config/path_integration/2D/grid_search/3_steps/heterogeneous_stochastic.yaml',
        # 'config/path_integration/2D/grid_search/3_steps/homogeneous_deterministic.yaml',
        # 'config/path_integration/2D/grid_search/3_steps/homogeneous_stochastic.yaml',

        'config/path_integration/2D/grid_search/design_choices_prep/all.yaml',
        'config/path_integration/2D/grid_search/design_choices_prep/test_optim.yaml',
        'config/path_integration/2D/grid_search/design_choices_prep/test_w_ir_redundancy_vs_bit_redundancy_reference.yaml',
        'config/path_integration/2D/grid_search/design_choices_prep/test_w_ir_redundancy_vs_bit_redundancy.yaml',
    ]

    node = environ.get("SLURMD_NODENAME") or environ.get("SLURM_NODELIST", "unknown")
    if "hpc" in node:
        logger.info(f"This is hpc node: {node}")
    else:
        logger.warning(f"Unknown node detected: {node}")

    node_job_assigments = {
        1: [0],
        6: [-4],
        7: [-4],
        8: [-1, -2],
        10: [],
        11: [],
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

