import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from copy import deepcopy
from parameters import Params
from encoding import bin2dec
from pathlib import Path
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from reservoir import BatchedTensorHistoryWriter
from scipy.stats import zscore
import networkx as nx
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
    # TODO add strategy and boundary alias so that it pritns nicely here with __str__ method or __repr___
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


def plot_predictions_and_labels(path, y_hat, y, tolerance=0.1, axis_limits=[0, 1]):
    y_hat_np = y_hat.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
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
    path = Path(path) / 'visualizations' 
    path.mkdir(parents=True, exist_ok=True)
    file = f"{num_dims}D_predictions_versus_labels.png"
    plt.savefig(path / file, bbox_inches='tight')

def plot_train_history(path, history):
    history_df = pd.DataFrame(history)
    history_melted = history_df.melt(id_vars=['epoch'], value_vars=['loss_train', 'loss_test', 'accuracy_train', 'accuracy_test'], 
                                    var_name='metric', value_name='value')
    fig, ax1 = plt.subplots()

    loss_plot = sns.lineplot(data=history_melted[history_melted['metric'].str.contains('loss')], 
                             x='epoch', y='value', hue='metric', ax=ax1)
    loss_plot.legend(loc='upper left')

    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2 = ax1.twinx()

    accuracy_plot = sns.lineplot(data=history_melted[history_melted['metric'].str.contains('accuracy')], 
                                 x='epoch', y='value', hue='metric', ax=ax2, linestyle='--')
    accuracy_plot.legend(loc='upper right')
    ax2.set_ylabel('Accuracy')
    fig.suptitle("Loss and Accuracy")
    fig.tight_layout()
    path = Path(path) / 'visualizations' 
    path.mkdir(parents=True, exist_ok=True)
    file = f"training.png"
    print('making train history plots:', path / file)
    plt.savefig(path / file, bbox_inches='tight')

def plot_grid_search(data_file_path: Path):
    out_path = data_file_path.parent / 'visualizations'
    out_path.mkdir(exist_ok=True)
    print('making grid search plots:', out_path)
    df = pd.read_hdf(data_file_path, 'df') 
    df['model_params'] = df['params'].apply(lambda p_dict: Params(**p_dict).model)
    df['k_avg'] = df['model_params'].apply(lambda x: x.reservoir_layer.k_avg)
    df['k_max'] = df['model_params'].apply(lambda x: x.reservoir_layer.k_max)
    df['p'] = df['model_params'].apply(lambda x: x.reservoir_layer.p)
    df['self_loops'] = df['model_params'].apply(lambda x: x.reservoir_layer.self_loops)
    df['n_nodes'] = df['model_params'].apply(lambda x: x.reservoir_layer.n_nodes)
    df['init'] = df['model_params'].apply(lambda x: x.reservoir_layer.init)
    df['interleaving'] = df['model_params'].apply(lambda x: x.input_layer.interleaving)
    df = df[['accuracy', 'loss', 'k_avg', 'k_max', 'p', 'self_loops', 'n_nodes', 'init', 'interleaving']]
    df['loss'] = df['loss'].apply(lambda x: x ** .5) # MSE to RMS
    features = df.drop(columns=['accuracy', 'loss'])
    # features = df.drop(columns=['accuracy'])
    
    # Identify categorical and numerical columns
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = features.select_dtypes(exclude=['object'], include='number').columns.tolist()
    
    transformers = [('num', StandardScaler(), numerical_cols)]
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(sparse_output=False), categorical_cols))
    
    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Apply the transformations
    features_processed = preprocessor.fit_transform(features)
    
    # Applying PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_processed)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    # Visualization of PCA
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=principal_df.join(df[['loss']]), x='PC1', y='PC2', hue='loss', palette='viridis', s=100, alpha=0.7)
    plt.title('PCA of Parameters')
    plt.savefig(out_path / 'pca.png', bbox_inches='tight')
    
    # Creating a heatmap of parameter contributions
    loadings = pca.components_.T
    feature_names = deepcopy(numerical_cols)
    if categorical_cols:
        feature_names += preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    loading_df = pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2'])

    plt.figure(figsize=(3, 6))
    sns.heatmap(loading_df, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Parameter Contributions')
    plt.savefig(out_path / 'pca_legend.png', bbox_inches='tight')
    
    # Correlation matrix, including categorical variables  # TODO FIX THIS!!!!
    std = features[numerical_cols].std()
    num_columns_to_keep = std[std != 0].index.tolist()
    cat_columns_to_keep = features[categorical_cols].nunique().index.tolist()
    columns_to_keep = num_columns_to_keep + cat_columns_to_keep
    features_with_performance = pd.concat([features[columns_to_keep], df[['loss']]], axis=1)
    correlation_matrix = features_with_performance.corr(method='spearman', numeric_only=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix including Performance Metrics')
    plt.savefig(out_path / 'correlation.png', bbox_inches='tight')

    col_list = ['accuracy', 'loss', 'k_avg', 'self_loops', 'init', 'interleaving']
    df1 = df[col_list]
    num_vars = len(df1.columns) - 1
    fig, axes = plt.subplots(1, num_vars, figsize=(8*num_vars, 10))
    axes = axes.flatten()
    for i, column in enumerate(df1.columns[df1.columns != 'loss']):
        c = column.capitalize()
        if len(df1[column].unique()) > 10:
            sns.scatterplot(ax=axes[i], data=df1, x=column, y='loss')
        else:
            sns.boxplot(ax=axes[i], data=df1, x=column, y='loss')
        axes[i].set_title(f'Loss vs {c}', fontsize=16)
        axes[i].set_xlabel(c, fontsize=16)
        axes[i].set_ylabel('Loss', fontsize=16)

    plt.tight_layout()
    plt.savefig(out_path / 'loss_vs_parameters.png', bbox_inches='tight')

    df2 = df[col_list]
    df2 = df2[df2['accuracy'] >= .3]
    num_vars = len(df2.columns) - 1
    fig, axes = plt.subplots(1, num_vars, figsize=(8*num_vars, 10))
    axes = axes.flatten()
    for i, column in enumerate(df2.columns[df2.columns != 'loss']):
        c = column.capitalize()
        if len(df2[column].unique()) > 10:
            sns.scatterplot(ax=axes[i], data=df2, x=column, y='loss')
        else:
            sns.boxplot(ax=axes[i], data=df2, x=column, y='loss')
        axes[i].set_title(f'Loss vs {c}', fontsize=20)
        axes[i].set_xlabel(c, fontsize=16)
        axes[i].set_ylabel('Loss', fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path / f'loss_vs_parameters_accuracy_gt_30p_{int(len(df2)/len(df)*100):03d}.png', bbox_inches='tight')

def plot_dynamics_history(path):
    path = Path(path)
    save_path = path / 'visualizations' 
    save_path.mkdir(parents=True, exist_ok=True)
    history, expanded_meta, meta = BatchedTensorHistoryWriter(path / 'history').reload_history()
    # print(meta)
    # print('full history:', history.shape)
    expanded_meta = expanded_meta[expanded_meta['phase'] == 'reservoir_layer']
    history = history[expanded_meta.index].numpy()
    # print('filtered history:', history.shape)

    # normalize and perform dimension reduction
    history_normalized = zscore(history, axis=0)
    history_normalized = np.nan_to_num(history_normalized) # columns with all 0's or 1's will divide by zero as variance = 0
    n_components = 2

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(history_normalized)
    df = pd.DataFrame(embedding, columns=[f'PC{i+1}' for i in range(n_components)], index=expanded_meta.index)
    df = pd.concat([df, expanded_meta], axis=1)

    # print("Explained variance by each component:")
    # print(embedding.explained_variance_ratio_)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='step', palette='viridis', s=100, alpha=0.7)
    plt.title('PCA of states over time')
    file = f"pca.png"
    plt.savefig(save_path / file, bbox_inches='tight')

    tsne = TSNE(n_components=n_components, perplexity=30, learning_rate=200)
    embedding = tsne.fit_transform(history_normalized)
    df = pd.DataFrame(embedding, columns=[f'PC{i+1}' for i in range(n_components)], index=expanded_meta.index)
    df = pd.concat([df, expanded_meta], axis=1)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='step', palette='viridis', s=100, alpha=0.7)
    plt.title('tSNE of states over time')
    file = f"tsne.png"
    plt.savefig(save_path / file, bbox_inches='tight')

def plot_reconstructed_manifold(path, adjacency_matrix):
    path = Path(path)
    save_path = path / 'visualizations' 
    save_path.mkdir(parents=True, exist_ok=True)

    # Create a figure and a 3D Axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map object from Seaborn
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Plot the point cloud
    mds = MDS(n_components=3, dissimilarity="precomputed")
    reconstructed_points = mds.fit_transform(adjacency_matrix)
    x, y, z = reconstructed_points
    sc = ax.scatter(x, y, z, c=np.sqrt(x**2 + y**2 + z**2), cmap=cmap, s=20)

    # Add color bar which maps values to colors
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('Color intensity')

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Point Cloud Visualization with Seaborn colormap')
    file = f"reconstructed_manifold.png"
    plt.savefig(save_path / file, bbox_inches='tight')
        

if __name__ == '__main__':
    plot_grid_search(Path('/out/grid_search/1D/initial_sweep/log.h5'))
    plot_grid_search(Path('/out/grid_search/2D/initial_sweep/log.h5'))