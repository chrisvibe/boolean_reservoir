# Finding Optimal Random Boolean Networks for Reservoir Computing  David Snyder1, Alireza Goudarzi2, and Christof Teuscher3
from boolean_reservoir.train_model import BooleanAccuracy as a, train_single_model, grid_search
from projects.temporal.code.dataset_init import TemporalDatasetInit as d
from projects.temporal.code.visualizations import plot_many_things

if __name__ == '__main__':

    # # Simple run
    # #####################################
    # p, model, dataset, history = train_single_model('config/temporal/density/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # p, model, dataset, history = train_single_model('config/temporal/parity/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)

    # # Grid search stuff 
    # #####################################
    grid_search('config/temporal/density/initial_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    grid_search('config/temporal/parity/initial_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)