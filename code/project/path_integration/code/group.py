from project.boolean_reservoir.code.utils.utils import load_grid_search_data_from_yaml
from project.temporal.code.stat import polar_design_plot
import pandas as pd
from enum import Enum
from pathlib import Path

def params_col_to_fields(df, extractions):
    """
    Args:
        df: DataFrame with 'params' column
        extractions: List of (lambda, field_set) tuples
                    e.g., [
                        (lambda p: p.D, d_set),
                        (lambda p: p.M.I, i_set),
                        (lambda p: p.M.T.optim, t_subset),
                    ]
    
    Returns:
        (df, factors): DataFrame with flattened params, and list of factors 
    """
    factors = []
    
    def flatten_params(params):
        series_dict = {}
        
        for get_source, field_set in extractions:
            source = get_source(params)
            attr_names = get_source.__code__.co_names[1:]
            prefix = "_".join(attr_names)
            
            for k, v in source.model_dump().items():
                if k in field_set:
                    field_name = f"{prefix}_{k}"
                    series_dict[field_name] = str(v) if isinstance(v, Enum) else v
                    if field_name not in factors:
                        factors.append(field_name)
        
        return pd.Series(series_dict)
    
    df = pd.concat([df, df['params'].apply(flatten_params)], axis=1)
    
    df['grid_search'] = df['params'].apply(lambda p: p.L.out_path.name)
    factors.append('grid_search')
    
    params_sample = df.iloc[0]['params']
    if params_sample.L.train_log.loss:
        df['loss'] = df['params'].apply(lambda p: p.L.train_log.loss)
    if params_sample.L.train_log.accuracy:
        df['accuracy'] = df['params'].apply(lambda p: p.L.train_log.accuracy)
    
    df.drop('params', axis=1, inplace=True)
    return df, factors

def load_custom_data(response, paths, extractions, factor_score=[]):
    data = list()
    factors = None
    for path in paths: # concat data
        _, df_i = load_grid_search_data_from_yaml(path)
        df_i, factors = params_col_to_fields(df_i, extractions)
        data.append(df_i)
    df = pd.concat(data, ignore_index=True)

    if factor_score:
        factors = [x for _, x in sorted(zip(factor_score, factors), reverse=True)]
    df, factors = fix_factors_and_combo(df, factors)

    groups_dict = {k: v[response].values for k, v in df.groupby('combo')}
    return df, factors, groups_dict

def fix_factors_and_combo(df, factors=list()):
    # need more than one to be considered a factor (and be part of a set)
    factors = [f for f in factors if f in df.columns]
    factors = list(df[factors].nunique()[df[factors].nunique() > 1].index)
    df['combo'] = df.apply(lambda row: tuple(row[feature] for feature in factors), axis=1)
    df['combo_str'] = df['combo'].apply(lambda t: "_".join(map(str, t)))
    return df, factors

def filter_combo(combo_col, factors, keep=None, exclude=None, return_as_str=True):
    """
    Generate a Series of 'design' strings or tuples from a combo column by keeping or excluding certain factors.
    
    Parameters:
    - combo_col: pd.Series of tuples
    - factors: list of factor names corresponding to tuple positions
    - keep: set of factor names to keep (optional)
    - exclude: set of factor names to exclude (optional)
    - return_as_str: if True, returns joined string; if False, returns tuple
    
    Returns:
    - pd.Series of design strings or tuples
    """
    if keep is not None:
        keep_indices = [i for i, k in enumerate(factors) if k in keep]
    elif exclude is not None:
        keep_indices = [i for i, k in enumerate(factors) if k not in exclude]
    else:
        keep_indices = list(range(len(factors)))  # keep all if nothing specified

    if return_as_str:
        return combo_col.apply(lambda t: "_".join(str(t[i]) for i in keep_indices))
    else:
        return combo_col.apply(lambda t: tuple(t[i] for i in keep_indices))


import numpy as np
import plotly.express as px
import plotly.io as pio
def graph_accuracy_vs_k_avg(out_path: Path, df: pd.DataFrame, factors: list[str]):
    out_path.mkdir(parents=True, exist_ok=True)
    df['R_k_avg_w_jitter'] = df['R_k_avg'] + np.random.uniform(-0.5, 0.5, size=len(df))
    df['design'] = filter_combo(df['combo'], factors, exclude='R_k_avg') 
    
    fig = px.scatter(
        df, 
        x='R_k_avg_w_jitter', 
        y='accuracy',
        color='design',
        opacity=0.7,
        labels={
            'R_k_avg_w_jitter': 'R_k_avg',
            'accuracy': 'Accuracy'
        }
    )
    fig.update_traces(marker=dict(size=5))
    
    x_min = int(df['R_k_avg'].min())
    x_max = int(df['R_k_avg'].max())
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(x_min, x_max + 1, 1))
    )
    fig.update_layout(
        title=f"Accuracy vs R_k_avg: {[f for f in factors if f != 'R_k_avg']}",
        title_x=0.5,
    )
    # fig.update_layout(showlegend=False)
    fig.write_html(out_path / 'scatter_accuracy_vs_k_avg.html')
    pio.write_image(fig, out_path / 'scatter_accuracy_vs_k_avg.svg', format='svg', width=1200, height=1600)
    return fig


import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Input, Output
import pandas as pd

def create_accuracy_vs_k_avg_dashboard(df: pd.DataFrame, factors: list[str]):
    df['R_k_avg_w_jitter'] = df['R_k_avg'] + np.random.uniform(-0.5, 0.5, size=len(df))
    df['design'] = filter_combo(df['combo'], factors, exclude='R_k_avg')
    
    filter_factors = [f for f in factors if f != 'R_k_avg']
    factor_values = {f: [v for v in sorted(df[f].unique().tolist()) if pd.notna(v)] for f in filter_factors}
    
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Label(factor),
                dcc.Dropdown(
                    id=f'dropdown-{factor}',
                    options=[{'label': 'All', 'value': 'All'}] + [{'label': str(v), 'value': v} for v in factor_values[factor]],
                    value='All'
                )
            ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'})
            for factor in filter_factors
        ], style={'marginBottom': '20px'}),
        dcc.Graph(id='scatter-plot')
    ])
    
    @callback(
        Output('scatter-plot', 'figure'),
        [Input(f'dropdown-{factor}', 'value') for factor in filter_factors]
    )
    def update_figure(*selected_values):
        filtered_df = df.copy()
        for factor, value in zip(filter_factors, selected_values):
            if value != 'All':
                filtered_df = filtered_df[filtered_df[factor] == value]
        
        fig = go.Figure()
        
        # Only add traces for combos actually in filtered data
        for combo in filtered_df['combo'].unique():
            group = filtered_df[filtered_df['combo'] == combo]
            fig.add_trace(go.Scatter(
                x=group['R_k_avg_w_jitter'],
                y=group['accuracy'],
                mode='markers',
                marker=dict(size=5, opacity=0.7),
                name=str(combo)
            ))
        
        x_min = int(df['R_k_avg'].min())
        x_max = int(df['R_k_avg'].max())
        
        fig.update_layout(
            title=f"Accuracy vs R_k_avg: {selected_values}",
            title_x=0.5,
            xaxis_title='R_k_avg',
            yaxis_title='Accuracy',
        )
        fig.update_xaxes(tickmode='array', tickvals=list(range(x_min, x_max + 1, 1)))
        fig.update_yaxes(range=[0, 1])
        
        return fig
    
    return app


if __name__ == '__main__':
    response = 'accuracy'
    out_path = Path('/out/path_integration/stats/design_evaluation/test_optim')
    path = out_path / ''
    print(path)
    extractions = [
        (lambda p: p.M.T.optim, {'name'}),
        (lambda p: p.M.T.optim.params, {'lr', 'weight_decay'}),
        (lambda p: p.M.T, {'batch_size'}),
        (lambda p: p.M.R, {'mode', 'k_avg', 'init'}),
        (lambda p: p.M.I, {'chunks', 'interleaving'}),
    ]
    paths = list()
    paths.append(f'config/path_integration/2D/grid_search/design_choices_prep/test_optim.yaml')
    df, factors, groups_dict = load_custom_data(response, paths, extractions)
    # chucks = 2 is terrible
    # graph_accuracy_vs_k_avg(path, df, factors)
    factors = [f for f in factors if f != 'I_chunks']
    df = df[df['I_chunks'] != 2]
    graph_accuracy_vs_k_avg(path, df, factors)
    app = create_accuracy_vs_k_avg_dashboard(df, factors)
    app.run_server(debug=True)