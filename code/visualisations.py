import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from copy import deepcopy
from encoding import bin2dec
matplotlib.use('Agg')

def plot_random_walk(positions, boundary):
    x, y = positions[:, 0], positions[:, 1] 
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, y)

    polygon_points = boundary.generate_polygon_points()
    if polygon_points:
        polygon = patches.Polygon(polygon_points, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)

    ax.set_title('Constrained Foraging Path')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.ion()

    # Save the plot to an image file
    plt.savefig("/out/constrained_foraging_path.png")

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
    
    plt.savefig("/out/incremental_error.png")

def plot_predictions_and_labels(y_hat, y, tolerance=0.1, scale=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Convert tensors to numpy arrays
    y_hat_np = y_hat.detach().numpy()
    y_np = y.detach().numpy()

    # Generate random colors
    num_points = y_hat_np.shape[0]
    colors = np.random.rand(num_points, 3)  # RGB colors

    # Predictions (y_hat)
    x_hat_coords = y_hat_np[:, 0]
    y_hat_coords = y_hat_np[:, 1]

    axs[0, 0].boxplot([x_hat_coords, y_hat_coords], labels=[r'$\hat{x}$', r'$\hat{y}$'])
    axs[0, 0].set_title(r'Box Plot - $\hat{y}$')

    axs[0, 1].scatter(x_hat_coords, y_hat_coords, c=colors, alpha=0.7, edgecolors='none')
    axs[0, 1].set_xlabel(r'$\hat{x}$')
    axs[0, 1].set_ylabel(r'$\hat{y}$')
    axs[0, 1].set_title(r'Scatter Plot - $\hat{y}$')
    axs[0, 1].grid(True)

    # Actual labels (y)
    x_coords = y_np[:, 0]
    y_coords = y_np[:, 1]

    axs[1, 0].boxplot([x_coords, y_coords], labels=['x', 'y'])
    axs[1, 0].set_title('Box Plot - y')

    # Check for errors within the given tolerance
    markers = np.where(np.all(np.abs(y_hat_np - y_np) <= tolerance, axis=1), 'o', 'x')

    for j, (x, y) in enumerate(zip(x_coords, y_coords)):
        axs[1, 1].scatter(x, y, color=colors[j], alpha=0.7, marker=markers[j], edgecolors='none')

    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_title('Scatter Plot - y')
    axs[1, 1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.ion()

    # Scale axes if scale is provided
    if scale:
        # Scale y-axis for box plots
        for ax in [axs[0, 0], axs[1, 0]]:
            ax.set_ylim(scale)
        
        # Scale both axes for scatter plots
        for ax in [axs[0, 1], axs[1, 1]]:
            ax.set_xlim(scale)
            ax.set_ylim(scale)

    # Save the figure
    plt.savefig("/out/predictions_versus_labels.png")

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
    plt.savefig("/out/binary_ecoding_error_hist_and_boxplot.png")
    plt.show()
