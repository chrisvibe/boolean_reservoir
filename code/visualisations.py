import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
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


def plot_predictions_and_labels(y_hat, y):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    for i, (data, label) in enumerate([(y_hat, r'$\hat{y}$'), (y, 'y')]):
        array = data.detach().numpy()
        x_coords = array[:, 0]
        y_coords = array[:, 1]

        # Create a box plot
        axs[i, 0].boxplot([x_coords, y_coords], labels=[r'$\hat{x}$' if i == 0 else 'x', r'$\hat{y}$' if i == 0 else 'y'])
        axs[i, 0].set_title(f'Box Plot - {label}')

        # Create a scatter plot
        axs[i, 1].scatter(x_coords, y_coords, alpha=0.7)
        axs[i, 1].set_xlabel(r'$\hat{x}$' if i == 0 else 'x')
        axs[i, 1].set_ylabel(r'$\hat{y}$' if i == 0 else 'y')
        axs[i, 1].set_title(f'Scatter Plot - {label}')
        axs[i, 1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.ion()
    
    # Save the figure
    plt.savefig("/out/predictions_versus_labels.png")