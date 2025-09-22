import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from pathlib import Path
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from projects.boolean_reservoir.code.reservoir import BatchedTensorHistoryWriter
from scipy.stats import zscore
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('Agg')

def plot_train_history(path, history):
    history_df = pd.DataFrame(history)
    history_melted = history_df.melt(id_vars=['epoch'], value_vars=sorted([c for c in history_df.columns if c != 'epoch']), 
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
    data_file_path = Path(data_file_path)
    out_path = data_file_path.parent / 'visualizations'
    out_path.mkdir(exist_ok=True, parents=True)
    print('making grid search plots:', out_path)
    df = pd.read_hdf(data_file_path, 'df') 
    df = df[df['loss'] != float('inf')] # filter out error configs (if they are illegal)
    df['loss'] = df['loss'].apply(lambda x: x ** .5) # MSE to RMS
    plot_histogram_of_top_percentile_vs_config_id(out_path, df, top_percentile=0.1)
    flatten_params = lambda x: pd.concat([
        pd.Series({f"I.{k}": v for k, v in x.M.I.model_dump().items()}),
        pd.Series({f"R.{k}": v for k, v in x.M.R.model_dump().items()}),
    # pd.Series({f"O.{k}": v for k, v in x.M.O.model_dump().items()}),
    # pd.Series({f"T.{k}": v for k, v in x.M.T.model_dump().items()}),
    ])
    df_flattend_params = df['params'].apply(lambda p: flatten_params(p))
    df = pd.concat([df, df_flattend_params], axis=1)
    df = df[['accuracy', 'loss'] + list(df_flattend_params.keys())]
    df = df.loc[:, [(col == 'params' or len(df[col].unique()) > 1) for col in df.columns]]
    features = df.drop(columns=['accuracy', 'loss'])
    features = features.loc[:, ~features.columns.str.contains('seed', case=False)]
    features_pluss_loss = df.drop(columns=['accuracy'])
    features_pluss_loss = features_pluss_loss.loc[:, ~features_pluss_loss.columns.str.contains('seed', case=False)]

    # Identify categorical and numerical columns
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = features.select_dtypes(exclude=['object'], include='number').columns.tolist()

    # pre-processing
    transformers = list() 
    transformers.append(('num', StandardScaler(), numerical_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(sparse_output=False), categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers)
    features_processed = preprocessor.fit_transform(features)

    # pre-processing w loss
    numerical_cols_plus_loss = deepcopy(numerical_cols)
    numerical_cols_plus_loss.append('loss')
    transformers2 = list() 
    transformers2.append(('num', StandardScaler(), numerical_cols_plus_loss))
    if categorical_cols:
        transformers2.append(('cat', OneHotEncoder(sparse_output=False), categorical_cols))
    preprocessor2 = ColumnTransformer(transformers=transformers2)
    features_processed2 = preprocessor2.fit_transform(features_pluss_loss)
    
    # PCA
    def pca_helper(out_path, file_name, features_processed, df, numerical_cols, categorical_cols, preprocessor):
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(features_processed)
        principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        
        # Visualization of PCA
        nested_out_path = out_path / 'pca' 
        nested_out_path.mkdir(exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=principal_df.join(df[['loss']]), x='PC1', y='PC2', hue='loss', palette='viridis', s=100, alpha=0.7)
        plt.title('PCA of Parameters')
        plt.savefig(nested_out_path / (file_name + '.png'), bbox_inches='tight')
        
        # Creating a heatmap of parameter contributions
        loadings = pca.components_.T
        feature_names = deepcopy(numerical_cols)
        if categorical_cols:
            feature_names += preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
        loading_df = pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2'])

        plt.figure(figsize=(3, 6))
        sns.heatmap(loading_df, annot=True, cmap='coolwarm', cbar=True)
        plt.title('Parameter Contributions')
        plt.savefig(nested_out_path / (file_name + '_legend.png'), bbox_inches='tight')
    
    pca_helper(out_path, 'pca', features_processed, df, numerical_cols, categorical_cols, preprocessor)
    pca_helper(out_path, 'pca_w_loss', features_processed2, df, numerical_cols_plus_loss, categorical_cols, preprocessor2)

    # Correlation matrix, including categorical variables
    std = features[numerical_cols].std()
    num_columns_to_keep = std[std != 0].index.tolist()
    cat_columns_to_keep = features[categorical_cols].nunique().index.tolist()
    columns_to_keep = num_columns_to_keep + cat_columns_to_keep
    features_with_performance = pd.concat([features[columns_to_keep], df[['loss']]], axis=1)
    correlation_matrix = features_with_performance.corr(method='spearman', numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix - Numerical')
    plt.savefig(out_path / 'correlation_num.png', bbox_inches='tight')

    if categorical_cols:
        num_bins = 10
        features_with_performance['loss_bin'] = pd.qcut(features_with_performance['loss'], q=num_bins)
        crosstab_result = pd.crosstab(
            [features_with_performance[col] for col in categorical_cols],
            features_with_performance['loss_bin']
        )
        normalized_crosstab_result = crosstab_result.div(crosstab_result.sum(axis=1), axis=0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(normalized_crosstab_result, annot=True, cmap='coolwarm', cbar=True)
        plt.title('Normalized Correlation Matrix - Quantile Bins')
        plt.savefig(out_path / 'correlation_cat.png', bbox_inches='tight')

    def loss_vs_parameter(path, df):
        for column in df.columns[df.columns != 'loss']:
            c = column.capitalize()
            n_unique = len(df[column].unique())
            if not (n_unique > 1):
                continue
            plt.figure(figsize=(8, 10))
            if n_unique > 10:
                sns.scatterplot(data=df, x=column, y='loss')
            else:
                sns.boxplot(data=df, x=column, y='loss')

            plt.title(f'Loss vs {c}', fontsize=16)
            plt.xlabel(c, fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            
            plt.tight_layout()
            plot_path = path/f'{column}.png'
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    col_list = df.columns 

    df1 = df[col_list]
    nested_out_path = out_path / 'loss_vs_parameter' 
    nested_out_path.mkdir(exist_ok=True)
    loss_vs_parameter(nested_out_path, df1)

    df2 = df[col_list]
    df2 = df2[df2['accuracy'] >= .3]
    nested_out_path = out_path / 'loss_vs_parameter_accuracy_lt_30' 
    nested_out_path.mkdir(exist_ok=True)
    loss_vs_parameter(nested_out_path, df2)

def plot_histogram_of_top_percentile_vs_config_id(path, df, top_percentile=0.1):
    threshold = df['accuracy'].quantile(1-top_percentile)
    config_ids = df[df['accuracy'] >= threshold]['config']
    config_counts = config_ids.value_counts().sort_index()
    top_config_id = config_counts.idxmax()
    print(f"Config ID with highest frequency: {top_config_id}, Count: {config_counts[top_config_id]}")
    print(f"Checkpoint: {str(df.iloc[top_config_id].params.L.last_checkpoint)}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(config_counts.index, config_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Config ID')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Config IDs in Top {int(top_percentile*100)}% of Accuracy (≥{threshold:.4f})')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(rotation=45 if len(config_counts) > 10 else 0)
    plt.tight_layout()
    plt.savefig(path / 'histogram_accuracy_vs_config.png', bbox_inches='tight')
    plt.close()


def plot_dynamics_history(path):
    path = Path(path)
    save_path = path / 'visualizations' 
    save_path.mkdir(parents=True, exist_ok=True)
    load_dict, history, expanded_meta, meta = BatchedTensorHistoryWriter(path / 'history').reload_history()
    # print(meta)
    # print('full history:', history.shape)
    expanded_meta = expanded_meta[expanded_meta['phase'].isin(['reservoir_layer', 'output_layer'])]
    history = history[expanded_meta.index].numpy()
    # print('filtered history:', history.shape)

    # normalize and perform dimension reduction
    history_normalized = zscore(history, axis=0)
    history_normalized = np.nan_to_num(history_normalized) # columns with all 0's or 1's will divide by zero as variance = 0
    n_components = 2

    # TODO transpose for space and time pca and break up into smaller parts
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(history_normalized)
    df = pd.DataFrame(embedding, columns=[f'PC{i+1}' for i in range(n_components)], index=expanded_meta.index)
    df = pd.concat([df, expanded_meta], axis=1)
    df['hue_tuple'] = df[['phase', 's', 'f']].apply(tuple, axis=1)

    # print("Explained variance by each component:")
    # print(embedding.explained_variance_ratio_)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='hue_tuple', palette='viridis', s=100, alpha=0.7)
    plt.legend(title='(phase, step, feature)')
    plt.title('PCA of states over time')
    file = f"pca.png"
    plt.savefig(save_path / file, bbox_inches='tight')

    tsne = TSNE(n_components=n_components, perplexity=30, learning_rate=200)
    embedding = tsne.fit_transform(history_normalized)
    df = pd.DataFrame(embedding, columns=[f'PC{i+1}' for i in range(n_components)], index=expanded_meta.index)
    df = pd.concat([df, expanded_meta], axis=1)
    df['hue_tuple'] = df[['phase', 's', 'f']].apply(tuple, axis=1)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='hue_tuple', palette='viridis', s=100, alpha=0.7)
    plt.legend(title='(phase, step, feature)')
    plt.title('tSNE of states over time')
    file = f"tsne.png"
    plt.savefig(save_path / file, bbox_inches='tight')

def plot_activity_trace(path, save_path=None, file_name="activity_trace_with_phase.svg", 
                                    highlight_input_nodes=True, data_filter=lambda df: df, 
                                    aggregation_handle=lambda df: df[df['sample_id'] == 0], figsize=(15,15), cell_aspect_ratio=1, ir_subtitle=True):
    """
    Plot activity trace with phase band underneath - using constrained_layout.
    """
    path = Path(path)
    save_path = save_path if save_path else path / 'visualizations'
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load history and metadata
    load_dict, history, expanded_meta, meta = BatchedTensorHistoryWriter(path / 'history').reload_history(
        include={'parameters', 'w_in', 'graph'}
    )
    expanded_meta = data_filter(expanded_meta)
    expanded_meta = aggregation_handle(expanded_meta)
    history = history[expanded_meta.index]
    
    # Get dimensions
    n_times = len(history)
    n_nodes = history.shape[1]
    
    # Create phase mapping
    unique_phases = expanded_meta['phase'].unique()
    phase_to_int = {phase: idx for idx, phase in enumerate(unique_phases)}
    phase_array = expanded_meta['phase'].map(phase_to_int).values
    
    # Setup colors
    phase_colors = sns.color_palette("husl", len(unique_phases))
    
    # Create combined data array
    combined_data = np.zeros((n_nodes + 1, n_times))
    combined_data[0, :] = phase_array
    combined_data[1:, :] = history.T
    
    # Create figure with constrained_layout for automatic centering
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Prepare subtitle if needed
    ax.set_title('Activity trace of node states over time', fontsize=16, pad=40)
    if highlight_input_nodes:
        p = load_dict['parameters']
        w_in = load_dict['w_in']
        g = load_dict['graph']
        ir_edges = [(u, v) for u, v, data in g.edges(data=True) if data.get('quadrant') == 'IR']
        ir_map = {u: v for u, v in ir_edges}
        ir_subtitle = 'I→R: ' + str(ir_map) if ir_subtitle else ''
        ax.text(0.5, 1.02, ir_subtitle, transform=ax.transAxes, fontsize=10, ha='center', va='bottom')
    
    # Plot node data
    node_data = combined_data[1:, :]
    im_nodes = ax.imshow(node_data, cmap='Greys', aspect=cell_aspect_ratio, origin='lower',
                         interpolation='nearest', extent=[0, n_times, 0, n_nodes])
    
    # Plot phase data as colored rectangles
    for t in range(n_times):
        phase_idx = int(phase_array[t])
        color = phase_colors[phase_idx]
        rect = plt.Rectangle((t, -1), 1, 1, facecolor=color, edgecolor='none')
        ax.add_patch(rect)
    
    # Set axis limits and labels
    ax.set_xlim(0, n_times)
    ax.set_ylim(-1, n_nodes)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Nodes', fontsize=12)
    
    # Adjust ticks - offset by 0.5 to center on boxes
    y_tick_interval = 1 if n_nodes <= 50 else max(1, n_nodes//10)
    y_ticks = [-0.5] + [i + 0.5 for i in range(0, n_nodes, y_tick_interval)]
    y_labels = ['Phase'] + [str(i) for i in range(0, n_nodes, y_tick_interval)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    x_tick_interval = 1 if n_times <= 50 else max(1, n_times//20)
    x_ticks = [i + 0.5 for i in range(0, n_times, x_tick_interval)]
    x_labels = [str(i) for i in range(0, n_times, x_tick_interval)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    # Add horizontal line
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Create colorbar
    cbar_grey = plt.colorbar(im_nodes, ax=ax, label='Activity', pad=0.02)
    
    # Create phase legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=phase_colors[i], label=phase) 
                       for i, phase in enumerate(unique_phases)]
    
    phase_legend = ax.legend(handles=legend_elements, 
                            loc='upper center', 
                            bbox_to_anchor=(0.5, -0.08),
                            ncol=min(len(unique_phases), 5),
                            frameon=True,
                            title='Phases',
                            title_fontsize=10,
                            fontsize=9)
    
    # Highlight input nodes if requested
    if highlight_input_nodes:
        alpha_value = 0.8
        a = b = 0
        I = p.M.I
        
        input_indices = expanded_meta[expanded_meta['phase'] == 'input_layer'].index
        time_positions = {idx: pos for pos, idx in enumerate(expanded_meta.index)}
        
        for idx in input_indices:
            b += I.chunk_size
            w_in_i = w_in[a:b]
            selected_input_indices = w_in_i.sum(axis=0).nonzero(as_tuple=True)[0]
            b %= I.n_nodes # steps re-use w_in so we reset 
            a = b
            
            time_pos = time_positions[idx]
            
            for node_idx in selected_input_indices:
                ax.add_patch(
                    plt.Rectangle((time_pos, node_idx), 1, 1, fill=False,
                                edgecolor=(1, 0, 0, alpha_value), lw=2)
                )
    
    # Save figure
    plt.savefig(save_path / file_name, bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig

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



if __name__ == '__main__':
    pass
    # plot_grid_search(Path('out/grid_search/path_integation/1D/initial_sweep/log.h5'))
    # plot_grid_search(Path('out/grid_search/path_integation/2D/initial_sweep/log.h5'))
    # plot_grid_search('out/temporal/density/grid_search/initial_sweep/log.h5')
    # plot_grid_search('out/grid_search/temporal/parity/initial_sweep/log.h5')

    plot_grid_search('out/temporal/density/grid_search/homogeneous-deterministic/log.h5')
    # plot_grid_search('out/temporal/density/grid_search/homogeneous-stochastic/log.h5')
    # plot_grid_search('out/temporal/density/grid_search/heterogeneous-deterministic/log.h5')
    # plot_grid_search('out/temporal/density/grid_search/heterogeneous-stochastic/log.h5')

    # plot_grid_search('out/grid_search/temporal/parity/homogeneous-deterministic/log.h5')
    # plot_grid_search('out/grid_search/temporal/parity/homogeneous-stochastic/log.h5')
    # plot_grid_search('out/grid_search/temporal/parity/heterogeneous-deterministic/log.h5')
    # plot_grid_search('out/grid_search/temporal/parity/heterogeneous-stochastic/log.h5')
