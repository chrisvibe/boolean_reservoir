from project.boolean_reservoir.code.utils.utils import load_grid_search_data_from_yaml
from enum import Enum
from pathlib import Path
import plotly.io as pio

from dash import Dash, dcc, html, callback, Input, Output, State, ALL, ctx, MATCH
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from hashlib import md5
    
    

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

def load_custom_data(paths, extractions, factor_score=[], df_filter=None, response_variable: str=None):
    data = list()
    factors = None
    for path in paths: # concat data
        _, df_i = load_grid_search_data_from_yaml(path)
        df_i, factors = params_col_to_fields(df_i, extractions)
        if df_filter:
            df_i = df_i[df_filter(df_i)]
        data.append(df_i)
    df = pd.concat(data, ignore_index=True)

    if factor_score:
        factors = [x for _, x in sorted(zip(factor_score, factors), reverse=True)]
    df, factors = fix_factors_and_combo(df, factors)
    groups_dict = {k: v[response_variable].values for k, v in df.groupby('combo')} if response_variable else dict()
    return df, factors, groups_dict

def fix_factors_and_combo(df, factors=list()):
    # need more than one to be considered a factor (and be part of a set)
    factors = [f for f in factors if f in df.columns]
    factors = list(df[factors].nunique()[df[factors].nunique() > 1].index)
    df['combo'], _ = make_combo_column(df, factors, return_as_str=False)
    df['combo_str'] = df['combo'].apply(lambda t: "_".join(map(str, t)))
    return df, factors

def make_combo_column(df, factors, keep=None, exclude=None, return_as_str=True):
    """
    Extract combo from actual DataFrame columns instead of tuple positions.
    
    Parameters:
    - df: DataFrame containing the factor columns
    - factors: list of factor column names
    - keep: set/list of factor names to keep (optional)
    - exclude: set/list of factor names to exclude (optional)
    - return_as_str: if True, returns joined string; if False, returns tuple
    
    Returns:
    - combo: Series of design strings or tuples
    - factors_subset: list of factors used
    """
    if keep is not None:
        factors_subset = [f for f in factors if f in keep]
    elif exclude is not None:
        factors_subset = [f for f in factors if f not in exclude]
    else:
        factors_subset = factors
    
    if return_as_str:
        combo = df[factors_subset].astype(str).agg('_'.join, axis=1)
    else:
        combo = df[factors_subset].apply(tuple, axis=1)
    
    return combo, factors_subset

def graph_accuracy_vs_k_avg(out_path: Path, df: pd.DataFrame, factors: list[str]):
    out_path.mkdir(parents=True, exist_ok=True)
    df['R_k_avg_w_jitter'] = df['R_k_avg'] + np.random.uniform(-0.5, 0.5, size=len(df))
    df['design'], _ = make_combo_column(df, factors, exclude='R_k_avg') 
    
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

def create_accuracy_vs_k_avg_dashboard(df: pd.DataFrame, factors: list[str], jitter={'R_k_avg': [-0.5, 0.5]}):
    # ---- jitter --------------------------------------------
    df = df.copy()
    new_cols = list()
    for col, jitt in jitter.items():
        new_col = col + '_w_jitter'
        new_cols.append(new_col)
        df[new_col] = (
            df[col] + np.random.uniform(jitt[0], jitt[1], size=len(df))
        )
    
    # ---- Factor handling -----------------------------------------------------
    df['design'], factors_subset = make_combo_column(
        df, factors, exclude='R_k_avg'
    )
    filter_factors = list(factors_subset) + ['R_k_avg']
    
    # Convert all factor values to strings for consistency
    factor_values = {
        f: [str(v) for v in sorted(df[f].unique()) if pd.notna(v)]
        for f in filter_factors
    }
    
    # ---- Available axis options ----------------------------------------------
    axis_options = [*new_cols, 'accuracy', 'loss'] + filter_factors
    
    # ---- Stable color mapping ------------------------------------------------
    # Use a large color palette for consistent coloring
    color_palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    
    def get_stable_color(combo_str, offset=0):
        """Get a consistent color for a combination string using hash and offset"""
        if combo_str == "Other":
            return '#CCCCCC'  # Gray for "Other"
        # Add offset to get different colors
        hash_input = f"{combo_str}_{offset}"
        hash_value = int(md5(hash_input.encode()).hexdigest(), 16)
        return color_palette[hash_value % len(color_palette)]
    
    # ---- Helper to create button style ---------------------------------------
    def button_style(is_active=False):
        return {
            'margin': '2px',
            'padding': '2px 8px',
            'fontSize': '11px',
            'border': '1px solid #ccc',
            'borderRadius': '3px',
            'background': '#e3f2fd' if is_active else 'white',
            'color': '#1976d2' if is_active else 'black',
            'fontWeight': 'bold' if is_active else 'normal',
            'cursor': 'pointer',
        }
    
    # ---- Helper to create figure ---------------------------------------------
    def create_figure(filtered_df, x_col, y_col, color_group_col=None, color_offsets=None):
        if filtered_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No data for selected filters",
                xaxis_title=x_col,
                yaxis_title=y_col,
            )
            return fig
        
        filtered_df = filtered_df.copy()
        filtered_df['combo_str'] = filtered_df['design'].astype(str)
        
        use_color = color_group_col and filtered_df[color_group_col].nunique() > 1
        
        if use_color:
            # Create stable color mapping with offsets
            unique_combos = sorted(filtered_df[color_group_col].unique())
            color_map = {}
            for combo in unique_combos:
                # Calculate offset based on all values in this combo
                offset = 0
                if color_offsets and combo != "Other":
                    # Parse the combo string to extract keys
                    for part in combo.split(', '):
                        if ':' in part:
                            # Key is already in "factor:value" format
                            offset += color_offsets.get(part, 0)
                
                # IMPORTANT: Use the full combo string + offset for unique colors
                color_map[combo] = get_stable_color(combo, offset)
            
            fig = px.scatter(
                filtered_df,
                x=x_col,
                y=y_col,
                color=color_group_col,
                color_discrete_map=color_map,  # Use our stable color map
                labels={color_group_col: 'Combination'},
                custom_data=['combo_str'],
                category_orders={color_group_col: unique_combos}  # Stable ordering
            )
            hover_template = f'<b>Combo:</b> %{{customdata[0]}}<br><b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>'
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df[x_col],
                y=filtered_df[y_col],
                mode='markers',
                marker=dict(color='steelblue'),
                name='Data',
                customdata=filtered_df['combo_str'],
            ))
            hover_template = f'<b>Combo:</b> %{{customdata}}<br><b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>'
        
        fig.update_traces(
            marker=dict(size=5, opacity=0.7),
            hovertemplate=hover_template,
        )
        
        fig.update_layout(
            title=f"{y_col} vs {x_col}",
            title_x=0.5,
            xaxis_title=x_col,
            yaxis_title=y_col,
            uirevision="scatter-plot",
        )
        
        # Special handling for R_k_avg x-axis
        if x_col == 'R_k_avg_w_jitter':
            x_min = int(df['R_k_avg'].min())
            x_max = int(df['R_k_avg'].max())
            x_ticks = list(range(x_min, x_max + 1))
            fig.update_xaxes(tickmode='array', tickvals=x_ticks)
        
        # Special handling for accuracy y-axis
        if y_col == 'accuracy':
            fig.update_yaxes(range=[0, 1], fixedrange=True)
        
        return fig
    
    # ---- Dash app ------------------------------------------------------------
    app = Dash(__name__)
    app.layout = html.Div([
        # Collapse/expand button
        html.Button(
            '▼ Hide Controls',
            id='toggle-controls',
            n_clicks=0,
            style={
                'marginBottom': '10px',
                'padding': '8px 16px',
                'fontSize': '14px',
                'cursor': 'pointer',
                'border': '1px solid #ccc',
                'borderRadius': '4px',
                'background': '#f5f5f5',
            }
        ),
        
        # Controls container
        html.Div([
            # Axis selection
            html.Div([
                html.Div([
                    html.Label('X-axis:'),
                    dcc.Dropdown(
                        id='x-axis',
                        options=[{'label': col, 'value': col} for col in axis_options],
                        value='R_k_avg_w_jitter',
                        clearable=False,
                    ),
                ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label('Y-axis:'),
                    dcc.Dropdown(
                        id='y-axis',
                        options=[{'label': col, 'value': col} for col in axis_options],
                        value='accuracy',
                        clearable=False,
                    ),
                ], style={'width': '20%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Button(
                        '🎨 Change Color Mode (OFF)',
                        id='color-mode-toggle',
                        n_clicks=0,
                        style={
                            'padding': '6px 12px',
                            'fontSize': '12px',
                            'cursor': 'pointer',
                            'border': '2px solid #ccc',
                            'borderRadius': '4px',
                            'background': 'white',
                        }
                    ),
                ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '2%'}),
            ], style={'marginBottom': '20px'}),
            
            # Filter dropdowns
            html.Div([
                html.Div([
                    "Select values to filter. Click buttons below to color by specific values."
                ], style={'marginBottom': '10px', 'fontStyle': 'italic', 'color': '#666'}),
                
                html.Div([
                    html.Div([
                        html.Label(factor),
                        html.Div([
                            html.Button(
                                'Select All',
                                id={'type': 'select-all', 'factor': factor},
                                n_clicks=0,
                                style={
                                    'fontSize': '10px',
                                    'padding': '2px 6px',
                                    'marginRight': '4px',
                                    'cursor': 'pointer',
                                }
                            ),
                            html.Button(
                                'Clear All',
                                id={'type': 'clear-all', 'factor': factor},
                                n_clicks=0,
                                style={
                                    'fontSize': '10px',
                                    'padding': '2px 6px',
                                    'cursor': 'pointer',
                                }
                            ),
                        ], style={'marginBottom': '4px'}),
                        dcc.Dropdown(
                            id={'type': 'dropdown', 'factor': factor},
                            options=[{'label': v, 'value': v} for v in factor_values[factor]],
                            value=[],  # Start with nothing selected (= all data shown)
                            multi=True,
                        ),
                        html.Div([
                            html.Button(
                                str(v),
                                id={'type': 'color-btn', 'factor': factor, 'value': v},
                                n_clicks=0,
                                style=button_style(),
                            )
                            for v in factor_values[factor]
                        ], style={'marginTop': '5px'}),
                    ], style={
                        'width': '18%',
                        'display': 'inline-block',
                        'marginRight': '2%',
                        'verticalAlign': 'top',
                    })
                    for factor in filter_factors
                ]),
            ], style={'marginBottom': '20px'}),
        ], id='controls-container', style={'display': 'block'}),
        
        dcc.Store(id='color-state', data={}),
        dcc.Store(id='value-color-offsets', data={}),
        dcc.Store(id='color-mode-active', data=False),
        dcc.Graph(id='scatter-plot', style={'height': '85vh'}),
    ])
    
    # ---- Callback for collapsing controls ------------------------------------
    @callback(
        Output('controls-container', 'style'),
        Output('toggle-controls', 'children'),
        Input('toggle-controls', 'n_clicks'),
    )
    def toggle_controls(n_clicks):
        if n_clicks % 2 == 1:  # Collapsed
            return {'display': 'none'}, '▶ Show Controls'
        else:  # Expanded
            return {'display': 'block'}, '▼ Hide Controls'
    
    # ---- Callback to toggle color mode ---------------------------------------
    @callback(
        Output('color-mode-active', 'data'),
        Output('color-mode-toggle', 'children'),
        Output('color-mode-toggle', 'style'),
        Input('color-mode-toggle', 'n_clicks'),
        State('color-mode-active', 'data'),
        prevent_initial_call=True,
    )
    def toggle_color_mode(n_clicks, is_active):
        new_state = not is_active
        label = '🎨 Change Color Mode (ON)' if new_state else '🎨 Change Color Mode (OFF)'
        style = {
            'padding': '6px 12px',
            'fontSize': '12px',
            'cursor': 'pointer',
            'border': '2px solid #ccc',
            'borderRadius': '4px',
            'background': '#e3f2fd' if new_state else 'white',
            'fontWeight': 'bold' if new_state else 'normal',
        }
        return new_state, label, style
    
    # ---- Callback for select all / clear all ---------------------------------
    @callback(
        Output({'type': 'dropdown', 'factor': MATCH}, 'value'),
        Input({'type': 'select-all', 'factor': MATCH}, 'n_clicks'),
        Input({'type': 'clear-all', 'factor': MATCH}, 'n_clicks'),
        State({'type': 'dropdown', 'factor': MATCH}, 'id'),
        prevent_initial_call=True,
    )
    def update_dropdown_selection(select_clicks, clear_clicks, dropdown_id):
        triggered_id = ctx.triggered_id
        if triggered_id is None:
            raise PreventUpdate
        
        factor = triggered_id['factor']
        
        if triggered_id['type'] == 'select-all':
            return factor_values[factor]
        else:  # clear-all
            return []
    
    # ---- Callback for button toggles -----------------------------------------
    @callback(
        Output('color-state', 'data'),
        Output({'type': 'color-btn', 'factor': ALL, 'value': ALL}, 'style'),
        Output('value-color-offsets', 'data'),
        Input({'type': 'color-btn', 'factor': ALL, 'value': ALL}, 'n_clicks'),
        State('color-state', 'data'),
        State('value-color-offsets', 'data'),
        State('color-mode-active', 'data'),
        prevent_initial_call=True,
    )
    def toggle_color_buttons(n_clicks_list, color_state, color_offsets, color_mode_active):
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered_id
        if triggered_id is None:
            raise PreventUpdate
        
        factor = triggered_id['factor']
        value = triggered_id['value']
        key = f"{factor}:{value}"
        
        if color_mode_active:
            # Change color mode - cycle the color offset
            color_offsets[key] = color_offsets.get(key, 0) + 1
            print(f"Color offset updated: {key} = {color_offsets[key]}")  # Debug
        else:
            # Normal mode - toggle selection
            if factor not in color_state:
                color_state[factor] = []
            
            if value in color_state[factor]:
                color_state[factor].remove(value)
            else:
                color_state[factor].append(value)
        
        # Generate button styles
        styles = [
            button_style(
                is_active=(f in color_state and v in color_state[f])
            )
            for f in filter_factors
            for v in factor_values[f]
        ]
        
        return color_state, styles, color_offsets
    
    # ---- Main plot callback --------------------------------------------------
    @callback(
        Output('scatter-plot', 'figure'),
        Input('x-axis', 'value'),
        Input('y-axis', 'value'),
        Input({'type': 'dropdown', 'factor': ALL}, 'value'),
        Input('color-state', 'data'),
        Input('value-color-offsets', 'data'),
    )
    def update_figure(x_col, y_col, filter_values, color_state, color_offsets):
        print(f"Color offsets in update_figure: {color_offsets}")  # Debug
        
        # Apply filters (empty list = show all)
        mask = np.ones(len(df), dtype=bool)
        for factor, values in zip(filter_factors, filter_values):
            if values:  # Only filter if something is selected
                mask &= df[factor].astype(str).isin(values)
        
        filtered_df = df[mask]
        
        # Determine if we need color grouping
        color_factors = [f for f in filter_factors if color_state.get(f, [])]
        
        if not color_factors:
            return create_figure(filtered_df, x_col, y_col)

        # Create color grouping - DON'T filter, just label as "Other"
        def make_color_key(row):
            parts = []
            has_all_color_factors = True
            for factor in filter_factors:
                if factor in color_state and color_state[factor]:
                    val_str = str(row[factor])
                    if val_str in color_state[factor]:
                        parts.append(f"{factor}:{val_str}")
                    else:
                        has_all_color_factors = False
            
            if not has_all_color_factors or not parts:
                return "Other"
            return ", ".join(parts)

        filtered_df = filtered_df.copy()
        filtered_df['color_group'] = filtered_df.apply(make_color_key, axis=1)

        return create_figure(filtered_df, x_col, y_col, 'color_group', color_offsets)
    
    return app