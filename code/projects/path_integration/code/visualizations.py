from benchmarks.path_integration.visualizations import plot_random_walk
from projects.boolean_reservoir.code.visualizations import plot_train_history, plot_predictions_and_labels, plot_dynamics_history
from projects.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
import matplotlib
matplotlib.use('Agg')

def plot_many_things(model, dataset, history):
    x_test = dataset.data['x_test'][:500]
    y_test = dataset.data['y_test'][:500]
    y_hat_test = model(x_test)
    plot_train_history(model.save_path, history)
    plot_predictions_and_labels(model.save_path, y_hat_test, y_test, tolerance=model.T.accuracy_threshold, axis_limits=[-1, 2])
    # plot_dynamics_history(model.save_path)
    # plot_graph_with_weight_coloring_3D(model.graph, model.readout)

if __name__ == '__main__':
    pass