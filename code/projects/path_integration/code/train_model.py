from projects.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from projects.boolean_reservoir.code.visualizations import plot_train_history, plot_dynamics_history, plot_activity_trace
from projects.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from projects.path_integration.code.visualizations import plot_many_things
from projects.boolean_reservoir.code.boolean_reservoir_parallel import boolean_reservoir_grid_search 

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

#     # # Simple run
#     # #####################################
#     # p, model, dataset, history = train_single_model('config/path_integration/1D/single_run/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
#     # plot_many_things(model, dataset, history)
#     # p, model, dataset, history = train_single_model('config/path_integration/2D/single_run/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
#     # plot_many_things(model, dataset, history)
    p, model, dataset, history = train_single_model('config/path_integration/2D/single_run/design_choices/test2.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    plot_many_things(model, dataset, history)
    plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0], ir_subtitle=True)

#     # # # playground 
#     # # #####################################
#     # configs = [
#     #     # 'config/path_integration/1D/grid_search/initial_sweep.yaml',
#     #     # 'config/path_integration/2D/grid_search/initial_sweep.yaml',
#     #     # 'config/path_integration/1D/single_run/good_model.yaml',
#     #     'config/path_integration/2D/single_run/good_model.yaml',
#     # ]
#     # for c in configs:
#     #     from projects.boolean_reservoir.code.parameters import load_yaml_config 
#     #     from projects.boolean_reservoir.code.parameters import generate_param_combinations 
#     #     P = load_yaml_config(c)
#     #     P = generate_param_combinations(P)[0]
#     #     p, model, dataset, history = train_single_model(parameter_override=P, dataset_init=d().dataset_init, accuracy=a().accuracy)
#     #     plot_many_things(model, dataset, history)
#     #     # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0], ir_subtitle=False)

#     # # # debug 
#     # # #####################################
#     # from projects.boolean_reservoir.code.reservoir import BooleanReservoir
#     # from projects.boolean_reservoir.code.utils import print_pretty_binary_matrix
#     # import torch
#     # from projects.boolean_reservoir.code.parameters import generate_param_combinations, load_yaml_config
#     # p = load_yaml_config('config/path_integration/1D/grid_search/homogeneous_deterministic.yaml')
#     # p.M.I.pertubation = 'override'
#     # # p.M.I.pertubation = 'xor'
#     # p.M.I.w_ir = 'out-3:3:1'
#     # p.M.I.seed = p.M.R.seed = p.M.O.seed = 1
#     # p.M.I.n_nodes = 10
#     # p.M.R.n_nodes = 30
#     # p.M.R.init = 'zeros'
#     # # p.M.R.init = 'random'
#     # p.M.R.k_avg = 4
#     # p.L.out_path = f'/out/debug/{p.M.R.init}/{p.M.I.pertubation}'
#     # p.L.history.record_history = True
#     # p.L.save_keys = ['parameters', 'w_in', 'graph', 'init_state', 'lut', 'weights'] 
#     # configs = generate_param_combinations(p)
#     # model = BooleanReservoir(configs[0])
#     # x = torch.tensor([[[[int(bit)]] for bit in '1001001010']], dtype=torch.uint8)
#     # # model(x)
#     # # model.save()
#     # # model.flush_history()
#     # p, model, dataset, history = train_single_model(model=model, dataset_init=d().dataset_init, accuracy=a().accuracy)
#     # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

#     # Grid search stuff 
#     #####################################
#     configs = [
#         # 'config/path_integration/1D/grid_search/heterogeneous_deterministic.yaml',
#         # 'config/path_integration/1D/grid_search/heterogeneous_stochastic.yaml',
#         # 'config/path_integration/1D/grid_search/homogeneous_deterministic.yaml',
#         # 'config/path_integration/1D/grid_search/homogeneous_stochastic.yaml',

#         # 'config/path_integration/2D/grid_search/heterogeneous_deterministic.yaml',
#         # 'config/path_integration/2D/grid_search/heterogeneous_stochastic.yaml',
#         # 'config/path_integration/2D/grid_search/homogeneous_deterministic.yaml',
#         # 'config/path_integration/2D/grid_search/homogeneous_stochastic.yaml',

#         # 'config/path_integration/1D/grid_search/no_self_loops/heterogeneous_deterministic.yaml',
#         # 'config/path_integration/1D/grid_search/no_self_loops/heterogeneous_stochastic.yaml',
#         # 'config/path_integration/1D/grid_search/no_self_loops/homogeneous_deterministic.yaml',
#         # 'config/path_integration/1D/grid_search/no_self_loops/homogeneous_stochastic.yaml',

#         # 'config/path_integration/2D/grid_search/no_self_loops/heterogeneous_deterministic.yaml',
#         # 'config/path_integration/2D/grid_search/no_self_loops/heterogeneous_stochastic.yaml',
#         # 'config/path_integration/2D/grid_search/no_self_loops/homogeneous_deterministic.yaml',
#         # 'config/path_integration/2D/grid_search/no_self_loops/homogeneous_stochastic.yaml',

#         # 'config/path_integration/1D/grid_search/3_steps/heterogeneous_deterministic.yaml',
#         # 'config/path_integration/1D/grid_search/3_steps/heterogeneous_stochastic.yaml',
#         # 'config/path_integration/1D/grid_search/3_steps/homogeneous_deterministic.yaml',
#         # 'config/path_integration/1D/grid_search/3_steps/homogeneous_stochastic.yaml',

#         # 'config/path_integration/2D/grid_search/3_steps/heterogeneous_deterministic.yaml',
#         # 'config/path_integration/2D/grid_search/3_steps/heterogeneous_stochastic.yaml',
#         # 'config/path_integration/2D/grid_search/3_steps/homogeneous_deterministic.yaml',
#         # 'config/path_integration/2D/grid_search/3_steps/homogeneous_stochastic.yaml',

#         'config/path_integration/2D/grid_search/design_choices/test2.yaml',
#         'config/path_integration/2D/grid_search/design_choices/test.yaml',
#     ]

#     node = environ.get("SLURMD_NODENAME") or environ.get("SLURM_NODELIST", "unknown")
#     if "hpc" in node:
#         logger.info(f"This is hpc node: {node}")
#     else:
#         logger.warning(f"Unknown node detected: {node}")

#     node_job_assigments = {
#         1: [0],
#         3: [],
#         4: [],
#         6: [0, 1],
#         10: [],
#         11: [],
#         'unknown': [-1],
#     }
#     if node != 'unknown':
#         id = int(node[3:])
#         configs = [configs[idx] for idx in node_job_assigments[id]]
#     else:
#         configs = [configs[idx] for idx in node_job_assigments['unknown']]

#     for c in configs:
#         boolean_reservoir_grid_search(
#             c,
#             dataset_init=d().dataset_init,
#             accuracy=a().accuracy,
#             gpu_memory_per_job_gb = 1/2,
#             cpu_memory_per_job_gb = 1/2,
#             cpu_cores_per_job = 1,
#         )

