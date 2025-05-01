import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from copy import deepcopy
from projects.boolean_reservoir.code.encoding import bin2dec
from projects.boolean_reservoir.code.visualizations import plot_train_history, plot_predictions_and_labels, plot_dynamics_history
from projects.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
from pathlib import Path
import seaborn as sns
matplotlib.use('Agg')

def plot_random_walk(dir_path, positions, strategy, boundary):
    zero_row = np.zeros_like((positions[0, :]))
    positions = np.vstack((zero_row, positions))
    steps, dimensions = positions.shape
    time = np.arange(positions.shape[0])
    
    fig = plt.figure(figsize=(10, 10))

    if dimensions == 1:
        ax = fig.add_subplot(111)
        x = positions[:, 0]
        ax.plot(x, time, label='1D Walk')
        ax.set_xlabel('Position')
        ax.set_ylabel('Time')
    
    elif dimensions == 2:
        ax = fig.add_subplot(111)
        x, y = positions[:, 0], positions[:, 1]
        ax.plot(x, y, label='2D Walk')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        polygon_points = boundary.generate_polygon_points()
        if polygon_points:
            polygon = patches.Polygon(polygon_points, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        ax.plot(x, y, z, label='3D Walk')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')

    ax.set_title('Constrained Foraging Path')
    ax.legend()

    plt.ion()
    
    # Save the plot to an image file
    path = Path(dir_path) / 'visualizations/random_walk'
    path.mkdir(parents=True, exist_ok=True)
    file_name = f'{dimensions}D-s={steps}-{strategy}-{boundary}.png'
    # TODO add strategy and boundary alias so that it prints nicely here with __str__ method or __repr___
    plt.savefig(path / file_name, bbox_inches='tight')

def plot_random_walk_model(dir_path, x: np.array, model, y: np.array):
    # TODO problems with normalization. Sum x over steps is not y if scaled differently
    m, s, d, _ = x.shape
    # incrementally consideres more steps to visualize error divergence
    # y is used to verify model correctness
    data = np.zeros((m, s, d, 2)) # add a dimension to contain both y_hat and y_ij
    for i in range(m) - 1:
        for j in range(s):
            x_ij = x[i:i+1, :j+1]
            y_ij = np.sum(x_ij, dim=1)   # TODO use cum sum instead outside of the loop!
            y_hat = model(x_ij)
            data[i, j, :, 0] = y_hat
            data[i, j, :, 1] = y_ij
        assert y_ij == y # when all steps are taken label should match sum of steps
 
    # plot two curves per path of lenth s, one with the y_hat and the other with y_ij
    # note that both y_hat and y_ij are a set of x, y coordinates when d = 2

    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(m): # plot each path
        ax.plot(data[i])

    ax.set_title('Incremental error in path integration')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.ion()
    
    path = Path(dir_path) / 'visualizations/random_walk'
    path.mkdir(parents=True, exist_ok=True)
    file_name = 'todo'
    plt.savefig(path / file_name, bbox_inches='tight')


def plot_binary_encoding_error_hist_and_boxplotplot(path, dataset, bins):
    x0 = deepcopy(dataset.data['x']).numpy()
    # distances = np.sqrt((dataset.data['y'].numpy() ** 2).sum(axis=1))
    dataset.encode_x()
    x1 = bin2dec(dataset.data['x'], dataset.data['x'].shape[-1]).numpy()
    diff = (x0.ravel() - x1.ravel())

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns
    
    # Histogram on the left
    axes[0].hist(diff, bins=bins)
    # axes[0].hist(distances, bins=bins)
    axes[0].set_title("Histogram")

    # Boxplot on the right
    axes[1].boxplot(diff, vert=False)
    axes[1].set_title("Boxplot")

    # Save the plot to an image file
    path = Path(path) / 'visualizations'
    path.mkdir(parents=True, exist_ok=True)
    file = f"binary_ecoding_error_hist_and_boxplot.png"
    plt.savefig(path / file, bbox_inches='tight')

def plot_many_things(model, dataset, history):
    y_test = dataset.data['y_test'][:500]
    y_hat_test = model(dataset.data['x_test'][:500])
    plot_train_history(model.save_path, history)
    plot_predictions_and_labels(model.save_path, y_hat_test, y_test, tolerance=model.T.accuracy_threshold, axis_limits=[-1, 2])
    plot_dynamics_history(model.save_path)
    # plot_graph_with_weight_coloring_3D(model.graph, model.readout)

if __name__ == '__main__':
    pass