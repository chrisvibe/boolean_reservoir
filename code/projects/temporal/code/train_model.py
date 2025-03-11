# Finding Optimal Random Boolean Networks for Reservoir Computing  David Snyder1, Alireza Goudarzi2, and Christof Teuscher3
from projects.boolean_reservoir.code.train_model import BooleanAccuracy as a, train_single_model, grid_search
from projects.temporal.code.dataset_init import TemporalDatasetInit as d
from projects.temporal.code.visualizations import plot_many_things
from projects.boolean_reservoir.code.visualizations import plot_activity_trace

if __name__ == '__main__':

    # # Simple run
    # #####################################

    # p, model, dataset, history = train_single_model('config/temporal/density/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_dir, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # p, model, dataset, history = train_single_model('config/temporal/parity/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # plot_activity_trace(model.save_dir, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])

    # # Grid search stuff 
    # #####################################

    grid_search('config/temporal/density/initial_heterogenous_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    grid_search('config/temporal/parity/initial_heterogenous_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)