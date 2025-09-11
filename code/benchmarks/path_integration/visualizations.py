import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from pathlib import Path
matplotlib.use('Agg')

def plot_random_walk(dir_path, positions, strategy, boundary, file_prepend='', sub_dir='visualizations/random_walk'):
    zero_row = np.zeros_like((positions[0, :]))
    steps, dimensions = positions.shape
    positions = np.vstack((zero_row, positions))
    time = np.arange(positions.shape[0])
    fig = plt.figure(figsize=(10, 10))
    
    if dimensions == 1:
        ax = fig.add_subplot(111)
        x = positions[:, 0]
        ax.plot(x, time, label='1D Walk')
        ax.set_xlabel('Position')
        ax.set_ylabel('Time')
        
        # Add boundary visualization for 1D case using get_points()
        boundary_points = boundary.get_points()
        if boundary_points:
            # For 1D, boundary_points should be [min, max]
            ax.axvline(x=boundary_points[0], color='r', linestyle='--', linewidth=2, label='Boundary')
            ax.axvline(x=boundary_points[1], color='r', linestyle='--', linewidth=2)
    
    elif dimensions == 2:
        ax = fig.add_subplot(111)
        x, y = positions[:, 0], positions[:, 1]
        ax.plot(x, y, label='2D Walk')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Use unified get_points() method for 2D
        boundary_points = boundary.get_points()
        if boundary_points:
            polygon = patches.Polygon(boundary_points, linestyle='--', linewidth=2, edgecolor='r', facecolor='none')
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
    path = Path(dir_path) / sub_dir
    path.mkdir(parents=True, exist_ok=True)
    prepend = file_prepend + '_' if file_prepend else ''
    file_name = f'{prepend}{dimensions}D-s={steps}-{strategy}-{boundary}.svg'
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


if __name__ == '__main__':
    pass