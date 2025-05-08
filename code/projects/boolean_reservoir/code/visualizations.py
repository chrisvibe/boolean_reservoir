import matplotlib.pyplot as plt
import matplotlib
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
    out_path.mkdir(exist_ok=True)
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
    
    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_processed)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    # Visualization of PCA
    nested_out_path = out_path / 'pca' 
    nested_out_path.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=principal_df.join(df[['loss']]), x='PC1', y='PC2', hue='loss', palette='viridis', s=100, alpha=0.7)
    plt.title('PCA of Parameters')
    plt.savefig(nested_out_path / 'pca.png', bbox_inches='tight')
    
    # Creating a heatmap of parameter contributions
    loadings = pca.components_.T
    feature_names = deepcopy(numerical_cols)
    if categorical_cols:
        feature_names += preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    loading_df = pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2'])

    plt.figure(figsize=(3, 6))
    sns.heatmap(loading_df, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Parameter Contributions')
    plt.savefig(nested_out_path / 'pca_legend.png', bbox_inches='tight')
    
    # Correlation matrix, including categorical variables
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

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(history_normalized)
    df = pd.DataFrame(embedding, columns=[f'PC{i+1}' for i in range(n_components)], index=expanded_meta.index)
    df = pd.concat([df, expanded_meta], axis=1)

    # print("Explained variance by each component:")
    # print(embedding.explained_variance_ratio_)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=df[['phase', 's', 'f']].apply(tuple, axis=1), palette='viridis', s=100, alpha=0.7)
    plt.legend(title='phase, step, feature')
    plt.title('PCA of states over time')
    file = f"pca.png"
    plt.savefig(save_path / file, bbox_inches='tight')

    tsne = TSNE(n_components=n_components, perplexity=30, learning_rate=200)
    embedding = tsne.fit_transform(history_normalized)
    df = pd.DataFrame(embedding, columns=[f'PC{i+1}' for i in range(n_components)], index=expanded_meta.index)
    df = pd.concat([df, expanded_meta], axis=1)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=df[['phase', 's', 'f']].apply(tuple, axis=1), palette='viridis', s=100, alpha=0.7)
    plt.legend(title='phase, step, feature')
    plt.title('tSNE of states over time')
    file = f"tsne.png"
    plt.savefig(save_path / file, bbox_inches='tight')


def plot_activity_trace(path, save_path=None, file_name="activity_trace_with_phase.png", highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0]):
    path = Path(path)
    save_path = save_path if save_path else path / 'visualizations'
    save_path.mkdir(parents=True, exist_ok=True)

    # Load history and metadata
    load_dict, history, expanded_meta, meta = BatchedTensorHistoryWriter(path / 'history').reload_history(include={'parameters', 'w_in'})
    expanded_meta = data_filter(expanded_meta)
    expanded_meta = aggregation_handle(expanded_meta) 
    history = history[expanded_meta.index]
    expanded_meta = expanded_meta.set_index(['time'], drop=False)

    # Create a DataFrame for phase and map to integer representations for plotting
    phase_df = expanded_meta[['time', 'phase']].copy()
    unique_phases = phase_df['phase'].unique()
    phase_to_int = {phase: idx for idx, phase in enumerate(unique_phases)}
    phase_df['phase'] = phase_df['phase'].map(phase_to_int)

    # Plotting heatmaps
    phase_colors = sns.color_palette("husl", len(unique_phases))
    cmap_phase = ListedColormap(phase_colors)
    colorbar_labels = {v: k for k, v in phase_to_int.items()}
    fig, (ax_heatmap, ax_phase) = plt.subplots(nrows=2, figsize=(15, 12), gridspec_kw={'height_ratios': [10, 0.5], 'hspace': 0.2})

    # Main activity trace heatmap
    sns.heatmap(history.T, cmap='viridis', cbar=True, ax=ax_heatmap)
    expanded_meta = expanded_meta.set_index(['time'], drop=False)
    ax_heatmap.set_title('Activity trace of node states over time', pad=50)
    ax_heatmap.set_xlabel('Time')
    ax_heatmap.set_ylabel('Nodes')

    # Highlight the input nodes on the y-axis with transparency
    # since recording captures each pertubation round various parts of the reservoir are highlighted per time step
    if highlight_input_nodes:
        p = load_dict['parameters']
        w_in = load_dict['w_in']
        alpha_value = 0.8
        a = b = 0
        I = p.M.I
        input_times = (t for t in expanded_meta[expanded_meta['phase'] == 'input_layer']['time'])
        for i in input_times:
            b += I.bits_per_feature
            w_in_i = w_in[a:b]
            selected_input_indices = w_in_i.sum(axis=0).nonzero(as_tuple=True)[0]
            a = b
            for idx in selected_input_indices:
                ax_heatmap.add_patch(
                    plt.Rectangle((i, idx), 1, 1, fill=False,
                                edgecolor=(1, 0, 0, alpha_value), lw=1)
                )
        subtitle = 'mapping I→R: ' + str({i: np.argwhere(w_in[i])[0].tolist() for i in range(w_in.shape[0])})
        plt.text(0.5, 1.05, subtitle, ha='center', va='center', transform=ax_heatmap.transAxes, fontsize=10)

    # Phase heatmap using integer mapping for consistent color representation
    sns.heatmap(phase_df[['phase']].T, cmap=cmap_phase, cbar=True, ax=ax_phase, xticklabels=False, yticklabels=False, cbar_kws={'ticks': list(colorbar_labels.keys())})
    cbar = ax_phase.collections[0].colorbar
    cbar.ax.set_yticklabels([colorbar_labels[v] for v in cbar.get_ticks()])
    ax_phase.set_title('Phases over Time')
    ax_phase.set_xlabel('')
    ax_phase.set_ylabel('')
    plt.savefig(save_path / file_name, bbox_inches='tight')
    plt.close()

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
    plot_grid_search('out/grid_search/temporal/density/initial_sweep/log.h5')
    plot_grid_search('out/grid_search/temporal/parity/initial_sweep/log.h5')

