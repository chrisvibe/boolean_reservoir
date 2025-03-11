from projects.boolean_reservoir.code.visualizations import plot_train_history, plot_predictions_and_labels, plot_dynamics_history
from projects.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D

def plot_many_things(model, dataset, history):
    y_test = dataset.data['y_test'][:500]
    y_hat_test = model(dataset.data['x_test'][:500])
    plot_train_history(model.save_dir, history)
    plot_predictions_and_labels(model.save_dir, y_hat_test, y_test, tolerance=model.T.accuracy_threshold, axis_limits=[0, 1])
    plot_dynamics_history(model.save_dir)
    # plot_graph_with_weight_coloring_3D(model.graph, model.readout)

if __name__ == '__main__':
    pass
    # from projects.boolean_reservoir.code.visualizations import plot_grid_search
    # plot_grid_search('out/grid_search/temporal/density/initial_sweep/log.h5')
    # plot_grid_search('out/grid_search/temporal/parity/initial_sweep/log.h5')