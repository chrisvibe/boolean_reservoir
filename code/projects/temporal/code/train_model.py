# Finding Optimal Random Boolean Networks for Reservoir Computing  David Snyder1, Alireza Goudarzi2, and Christof Teuscher3
from projects.boolean_reservoir.code.train_model import BooleanAccuracy as a, train_single_model, grid_search
from projects.temporal.code.dataset_init import TemporalDatasetInit as d
from projects.temporal.code.visualizations import plot_many_things
from projects.boolean_reservoir.code.visualizations import plot_activity_trace

if __name__ == '__main__':

    # # Simple run
    # #####################################

    # p, model, dataset, history = train_single_model('config/temporal/density/single_run/ok_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # p, model, dataset, history = train_single_model('config/temporal/density/single_run/sample_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # TODO delete
    # find . -type f -exec sed -i 's|out/single_run/temporal/density|out/temporal/density/single_run/|g' {} +
    # find . -type f -exec sed -i 's|out/single_run/temporal/parity|out/temporal/density/parity/single_run/|g' {} +
    # find . -type f -exec sed -i 's|out/grid_search/temporal/density|out/temporal/density/grid_search/|g' {} +
    # find . -type f -exec sed -i 's|out/grid_search/temporal/parity|out/temporal/density/parity/grid_search/|g' {} +
    # find . -type f -exec sed -i 's|out/temporal/kq_and_gr|out/temporal/kq_and_gr/grid_search/|g' {} +

    # # Grid search stuff 
    # #####################################
    
    configs = [

        # 'config/temporal/density/grid_search/homogeneous_stochastic.yaml',
        # 'config/temporal/density/grid_search/homogeneous_deterministic.yaml',
        # 'config/temporal/parity/grid_search/homogeneous_stochastic.yaml',
        # 'config/temporal/parity/grid_search/homogeneous_deterministic.yaml',

        'config/temporal/density/grid_search/heterogeneous_stochastic.yaml',
        'config/temporal/density/grid_search/heterogeneous_deterministic.yaml',
        'config/temporal/parity/grid_search/heterogeneous_stochastic.yaml',
        'config/temporal/parity/grid_search/heterogeneous_deterministic.yaml',

    ]
    for c in configs:
        grid_search(c, dataset_init=d().dataset_init, accuracy=a().accuracy)