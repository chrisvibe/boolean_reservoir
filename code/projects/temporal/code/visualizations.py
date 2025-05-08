from projects.boolean_reservoir.code.visualizations import plot_train_history, plot_predictions_and_labels, plot_dynamics_history
from projects.boolean_reservoir.code.graph_visualizations_dash import plot_graph_with_weight_coloring_3D
from projects.boolean_reservoir.code.parameters import Params 
import matplotlib.pyplot as plt
import seaborn as sns
from labellines import labelLines
import json
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
            

def plot_many_things(model, dataset, history):
    y_test = dataset.data['y_test'][:500]
    y_hat_test = model(dataset.data['x_test'][:500])
    plot_train_history(model.save_path, history)
    plot_predictions_and_labels(model.save_path, y_hat_test, y_test, tolerance=model.T.accuracy_threshold, axis_limits=[0, 1])
    plot_dynamics_history(model.save_path)
    # plot_graph_with_weight_coloring_3D(model.graph, model.readout)

def group_df_data_by_parameters(df):
    def set_to_none(p):
        p.M.R.k_min = None
        p.M.R.k_avg = None
        p.M.R.k_max = None
        p.M.R.seed = None
        p.M.I.seed = None
        p.M.O.seed = None
        p.D.seed = None
        p.D.path = None
        p.logging = None
        return p
    df = df.sort_values(by='config')
    df['group_params_str'] = df['params'].apply(set_to_none)
    df['group_params_str'] = df['group_params_str'].apply(lambda p: json.dumps(p.model_dump(), sort_keys=True, default=str))
    grouped = df.groupby(df['group_params_str'])
    return grouped

def plot_kq_and_gr(df, P: Params, filename: str):
    samples_per_config = df['sample'].max() + 1
    subtitle = f"Mode: {df.iloc[0]['params'].M.R.mode}, Nodes: {P.M.R.n_nodes}, Bit Stream Length: {P.D.bit_stream_length}, Tao: {P.D.tao}, Samples per config: {samples_per_config}, Configs: {len(df['group_params_str'].unique())}"
    
    # Create the figure and axis with extra space for legends
    fig, ax = plt.subplots(figsize=(18, 8))  # Increased width to accommodate legends
    
    # Create a color mapper for spectral radius
    norm = plt.Normalize(df['spectral_radius'].min(), df['spectral_radius'].max())
    
    # Predefined marker styles
    markers = ['<', '>', '^']
    
    # Store handles and labels for manual legend creation
    scatter_handles = []
    trend_lines = []
    
    # Scatter plot with a custom color scheme
    for i, metric in enumerate(sorted(df['metric'].unique())):
        # Subset for this metric
        subset = df[df['metric'] == metric]
        
        # Choose color
        color = plt.cm.tab10(i % 10)
        
        # Scatter plot
        scatter = ax.scatter(
            subset['k_avg'],
            subset['value'],
            label=metric,
            marker=markers[i % len(markers)],
            c=subset['spectral_radius'],
            cmap='viridis',
            norm=norm,
            edgecolors='black',
            linewidth=1,
            alpha=0.7,
            s=30,
        )
        scatter_handles.append(scatter)
        
        # Regression line
        trend = sns.regplot(
            x='k_avg',
            y='value',
            data=subset,
            scatter=False,
            lowess=True,
            ax=ax,
            color=color,
            line_kws={'linestyle':'--', 'linewidth':2}
        )
        trend_lines.append(trend.lines[0])
    ax.set_ylabel('Rank')
    ax.set_xlabel('Average K')
    
    # Add colorbar for spectral radius
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02, label='Spectral Radius')
    
    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(right=0.8)  # This leaves room for legends
    
    # Custom legend with scatter points
    first_legend = ax.legend(
        scatter_handles,
        [h.get_label() for h in scatter_handles],
        title='Points',
        loc='upper left',
        bbox_to_anchor=(0.0, 1.0)  # Positioned slightly inside the top left corner, above the plot
    )
    
    # Add the first legend to the plot
    ax.add_artist(first_legend)
    
    # Trend lines legend
    trend_legend = ax.legend(
        [plt.Line2D([0], [0], color=plt.cm.tab10(i % 10), linestyle='--', linewidth=2) for i in range(len(trend_lines))],
        [f'{h.get_label()}' for h in scatter_handles],
        title='Lines (lowess)',
        loc='upper left',
        bbox_to_anchor=(0.0, 0.8)  # Positioned below the first legend
    )
    
    plt.title('Reservoir Metrics: Kernel Quality, Generalization Rank, Delta', fontsize=16)
    
    # Add subtitle at the bottom of the plot
    fig.text(0.5, 0.01, subtitle, ha='right', fontsize=12)
    
    # Save the figure
    save_path = P.L.out_path / 'visualizations'
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close(fig)

def plot_kq_and_gr_many_config(grouped_df, P: Params, filename: str):
    # Create the figure and axis with extra space for legends
    fig, ax = plt.subplots(figsize=(18, 8))  # Increased width to accommodate legends
    
    color_idx = 0
    xvals = list()
    shift = 5  # Lines are too close at k_avg < 4 (subtract 1 as we use 1 for jitter in k_avg)
    g = len(grouped_df)
    val_range = grouped_df['k_avg'].max().max() - shift
    for name, subset in grouped_df:
        # print(color_idx, ':')
        # p = subset.iloc[0]['params']
        # print(p.M.I)
        # print(p.M.R)
        # print(p.M.O)
        # print(p.D)
        metrics = sorted(subset['metric'].unique())
        n_metrics = len(metrics)
        for i, metric in enumerate(metrics):
            # Filter data for the current metric
            data = subset[subset['metric'] == metric]
            color = plt.cm.tab10(color_idx % 10)
            
            # Get x and y values for plotting
            data = data.groupby('k_avg').agg({'value': 'mean'}).reset_index() # only needed for certain smoothing
            data.sort_values(by='k_avg', inplace=True)
            x_sorted = data['k_avg'].values
            y_sorted = data['value'].values
            
            # # Generate points on a smooth spline line - Quadratic spline interpolation (k=3 for cubic)
            cubic_spline = UnivariateSpline(x_sorted, y_sorted, k=3, s=0.0) # requires aggregation beforhand!
            x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            y_smooth = cubic_spline(x_fine)

            # # Generate points w linear smoothing
            # linear_interpolator = interp1d(x_sorted, y_sorted, kind='linear', fill_value="extrapolate") # requires aggregation beforhand!
            # x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            # y_smooth = linear_interpolator(x_fine)

            # # Generate points on a smooth lowess line
            # xy = lowess(y_sorted, x_sorted, frac=2./3., it=3, delta=0.0, is_sorted=True, missing='none', return_sorted=True)
            # x_fine, y_smooth = xy[:, 0], xy[:, 1]
            
            # Plot the lines
            ax.plot(x_fine, y_smooth, linestyle='--', linewidth=2, color=color, label=f'{color_idx}-{metric}')

            # trend = sns.regplot(
            #     x='k_avg',
            #     y='value',
            #     data=data,
            #     scatter=False,
            #     lowess=True,
            #     ax=ax,
            #     color=color,
            #     line_kws={'linestyle':'--', 'linewidth':2},
            #     label=f'{color_idx}-{metric}'
            # )

            xvals.append(color_idx * val_range / g + ((i + 1) / n_metrics) - 1 + shift)
        color_idx += 1

    lines = plt.gca().get_lines()
    labelLines(lines, align=False, xvals=xvals, fontsize=10)
    ax.set_ylabel('Rank')
    ax.set_xlabel('Average K')
    
    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(right=0.8)  # This leaves room for legends
    plt.title('Reservoir Metrics: Kernel Quality, Generalization Rank, Delta', fontsize=16)

    # Save the figure
    save_path = P.L.out_path / 'visualizations'
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / filename, bbox_inches='tight')
    plt.close(fig)

def plot_optimal_k_vs_n(df):
    # GR changes with TAO!!! now what???
    filtered_df = df[df['metric'] == 'delta']
    grouped_df = group_df_data_by_parameters(filtered_df)
    # cant group the group...
    grouped_df = grouped_df.groupby('k_avg', 'tao').agg({'value': 'max'}).reset_index()
    pass


if __name__ == '__main__':
    pass
    # from projects.boolean_reservoir.code.visualizations import plot_grid_search
    # plot_grid_search('out/grid_search/temporal/density/initial_sweep/log.h5')
    # plot_grid_search('out/grid_search/temporal/parity/initial_sweep/log.h5')