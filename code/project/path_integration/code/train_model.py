from project.boolean_reservoir.code.train_model import train_single_model
from project.boolean_reservoir.code.visualization import plot_train_history, plot_dynamics_history, plot_activity_trace
from project.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
from project.path_integration.code.visualization import plot_many_things
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.utils import run_on_node
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

    # Simple run
    #####################################
    # p, model, dataset, history = train_single_model('config/path_integration/1D/single_run/good_model.yaml')
    # plot_many_things(model, dataset, history)
    # p, model, dataset, history = train_single_model('config/path_integration/2D/single_run/good_model.yaml')
    # p, model, dataset, history = train_single_model('config/path_integration/2D/single_run/good_model_small.yaml')
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0], ir_subtitle=True)
    # p, model, dataset, history = train_single_model(yaml_or_checkpoint_path='/tmp/test2.yaml')

    # # adjust redundancy setting 
    # #####################################
    # configs = [
    #     'config/path_integration/1D/grid_search/design_choices/continuous_redundancy.yaml',
    #     'config/path_integration/1D/grid_search/design_choices/discrete_redundancy.yaml',
    #     'config/path_integration/2D/grid_search/design_choices/continuous_redundancy.yaml',
    #     'config/path_integration/2D/grid_search/design_choices/discrete_redundancy.yaml',
    # ]
    # for i, config in enumerate(configs):
    #     run_on_node(configs, node_job_assignments={
    #         1: [i],
    #         'unknown': [i],
    #     }, run_fn=boolean_reservoir_grid_search)

    # Grid search stuff 
    #####################################
    configs = [
        'config/path_integration/1D/grid_search/design_choices/continuous.yaml',
        'config/path_integration/1D/grid_search/design_choices/discrete.yaml',
        'config/path_integration/2D/grid_search/design_choices/continuous.yaml',
        'config/path_integration/2D/grid_search/design_choices/discrete.yaml',
    ]

    run_on_node(configs, node_job_assignments={
        1: [0],
        2: [1],
        3: [0],
        4: [1],
        5: [0],
        6: [1],
        7: [0],
        8: [1],
        9: [0],
        10: [0],
        11: [1],
        'unknown': [-1],
    }, run_fn=boolean_reservoir_grid_search)

