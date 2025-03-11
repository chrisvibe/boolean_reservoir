from projects.boolean_reservoir.code.train_model import train_single_model, grid_search, EuclideanDistanceAccuracy as a
from projects.boolean_reservoir.code.reservoir import BooleanReservoir, PathIntegrationVerificationModel, PathIntegrationVerificationModelBaseTwoEncoding
from projects.boolean_reservoir.code.visualizations import plot_train_history, plot_dynamics_history 
from projects.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from projects.path_integration.code.visualizations import plot_many_things

if __name__ == '__main__':

    # # Test
    # #####################################
    # test_saving_and_loading_models()
    # test_reproducibility_of_loaded_grid_search_checkpoint()

    # # Verification models
    # ############################################################
    # P = load_yaml_config('config/path_integration/1D/verification_model.yaml')
    # P = load_yaml_config('config/2D/path_integration/verification_model.yaml')
    # I = P.model.input_layer
    # model = PathIntegrationVerificationModelBaseTwoEncoding(n_dims=I.n_inputs)
    # model = PathIntegrationVerificationModel(I.bits_per_feature, I.n_inputs)
    # model.P = P
    # p, model, dataset, history = train_single_model(model=model, dataset_init=dataset_init, accuracy=euclidean_distance_accuracy)

    # # Simple run
    # #####################################
    # p, model, dataset, history = train_single_model('config/path_integration/1D/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # p, model, dataset, history = train_single_model('config/path_integration/2D/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)

    # # Grid search stuff 
    # #####################################
    # grid_search('config/path_integration/2D/test_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # grid_search('config/path_integration/1D/initial_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # grid_search('config/path_integration/2D/initial_sweep.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    grid_search('config/path_integration/1D/initial_sweep2.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    grid_search('config/path_integration/1D/initial_sweep2.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)

