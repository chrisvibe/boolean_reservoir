from projects.boolean_reservoir.code.train_model import train_single_model, grid_search, EuclideanDistanceAccuracy as a
from projects.boolean_reservoir.code.visualizations import plot_train_history, plot_dynamics_history 
from projects.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from projects.path_integration.code.visualizations import plot_many_things

if __name__ == '__main__':
    # # Simple run
    # #####################################
    # p, model, dataset, history = train_single_model('config/path_integration/1D/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # p, model, dataset, history = train_single_model('config/path_integration/2D/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)

    # # Grid search stuff 
    # #####################################
    grid_search('config/path_integration/1D/initial_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    grid_search('config/path_integration/2D/initial_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    grid_search('config/path_integration/1D/initial_sweep2.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    grid_search('config/path_integration/1D/initial_sweep2.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)

