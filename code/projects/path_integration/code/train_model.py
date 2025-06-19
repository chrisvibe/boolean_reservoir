from projects.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from projects.boolean_reservoir.code.visualizations import plot_train_history, plot_dynamics_history 
from projects.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from projects.path_integration.code.visualizations import plot_many_things
from projects.boolean_reservoir.code.boolean_reservoir_parallel import boolean_reservoir_grid_search 

if __name__ == '__main__':
    # # Simple run
    # #####################################
    # p, model, dataset, history = train_single_model('config/path_integration/1D/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)
    # p, model, dataset, history = train_single_model('config/path_integration/2D/good_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    # plot_many_things(model, dataset, history)

    # # Grid search stuff 
    # #####################################
    configs = [
        'config/path_integration/1D/initial_sweep.yaml',
        'config/path_integration/2D/initial_sweep.yaml',
    ]
    for c in configs:
        boolean_reservoir_grid_search(
            c,
            dataset_init=d().dataset_init,
            accuracy=a().accuracy,
            gpu_memory_per_job_gb = 0.5,
            cpu_memory_per_job_gb = 1,
            cpu_cores_per_job = 2,
        )

