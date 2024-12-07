import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from copy import deepcopy
from encoding import bin2dec
from utils import make_folders
import seaborn as sns
import pandas as pd
matplotlib.use('Agg')

make_folders('/out', ['visualizations']) 

def plot_random_walk(positions, boundary):
    zero_row = np.zeros_like((positions[0, :]))
    positions = np.vstack((zero_row, positions))
    dimensions = positions.shape[1]
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
    plt.savefig("/out/visualizations/constrained_foraging_path.png")

    # Display the plot (optional, for interactive environments)
    # plt.show()

def plot_random_walk_model(x: np.array, model, y: np.array):
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
    
    plt.savefig("/out/visualizations/incremental_error.png")


def plot_binary_encoding_error_hist_and_boxplotplot(dataset, bins):
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
    plt.savefig("/out/visualizations/binary_ecoding_error_hist_and_boxplot.png")
    plt.show()


def plot_predictions_and_labels(y_hat, y, tolerance=0.1, axis_limits=[0, 1]):
    y_hat_np = y_hat.detach().numpy()
    y_np = y.detach().numpy()
    num_dims = y_hat_np.shape[1]
    if num_dims == 1:
        sort_order = y_np[:, 0].argsort()
        y_hat_np = y_hat_np[sort_order]
        y_np = y_np[sort_order]
    
    y_correct = (abs(y_hat_np - y_np) <= tolerance).all(axis=1)

    plt.figure(figsize=(12, 6))

    # First scatter plot: y_hat
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_hat_np[:, 0], y=y_hat_np[:, 1] if num_dims > 1 else range(len(y_hat_np)), 
                    hue=y_correct, palette={True: 'black', False: 'red'}, alpha=0.3)
    plt.grid(True)
    plt.title('Predicted coordinates')
    plt.xlabel(r'$\hat{x}$')
    plt.ylabel(r'$\hat{y}$' if num_dims > 1 else 'Index')
    plt.xlim(axis_limits)
    if num_dims > 1:
        plt.ylim(axis_limits)
    
    # Second scatter plot: y
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_np[:, 0], y=y_np[:, 1] if num_dims > 1 else range(len(y_np)), 
                    hue=y_correct, palette={True: 'black', False: 'red'}, alpha=0.3)
    plt.grid(True)
    plt.title('Target coordinates')
    plt.xlabel(r'${x}$')
    plt.ylabel(r'${y}$' if num_dims > 1 else 'Index')
    plt.xlim(axis_limits)
    if num_dims > 1:
        plt.ylim(axis_limits)
    
    # Save the figure
    plt.savefig(f"/out/visualizations/{num_dims}D_predictions_versus_labels.png")