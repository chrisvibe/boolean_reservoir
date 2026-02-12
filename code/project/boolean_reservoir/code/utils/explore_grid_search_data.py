from pathlib import Path
import plotly.io as pio
from dash import Dash, dcc, html, callback, Input, Output, State, ALL, ctx, MATCH
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from hashlib import md5
from dash.dependencies import Input, Output, State

    
def fix_factors_and_combo(df, factors=list(), keep=None, exclude=None):
    # need more than one to be considered a factor (and be part of a set)
    factors = [f for f in factors if f in df.columns]
    factors = list(df[factors].nunique()[df[factors].nunique() > 1].index)
    df['combo'], _ = make_combo_column(df, factors, return_as_str=False, keep=keep, exclude=exclude)
    df['combo_str'] = df['combo'].apply(lambda t: "_".join(map(str, t)))
    df['combo_id'] = df['combo_str'].astype('category').cat.codes
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
        y='T_accuracy',
        color='design',
        opacity=0.7,
        labels={
            'R_k_avg_w_jitter': 'R_k_avg',
            'T_accuracy': 'Accuracy'
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

def _detect_gpu():
    """Detect if GPU-accelerated WebGL is likely available on the client.

    Since Dash renders plots client-side via Plotly.js, true GPU detection
    happens in the browser. This server-side heuristic checks whether the
    machine has a GPU (useful for remote/headless setups where the browser
    runs on the same host). Returns True if any GPU is detected.
    """
    import subprocess
    # Check NVIDIA GPU
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Check if any GPU-like device is listed (AMD, Intel integrated, etc.)
    try:
        result = subprocess.run(
            ['lspci'], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.lower().split('\n'):
                if 'vga' in line or '3d controller' in line or 'display' in line:
                    return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def create_scatter_dashboard(
    df: pd.DataFrame,
    factors: list[str],
    jitter: dict = {'R_k_avg': [-0.5, 0.5]},
    discrete_threshold: int = 20,
    renderer: str = 'auto',
):
    # ---- Resolve renderer ----------------------------------------------------
    if renderer == 'auto':
        use_webgl = _detect_gpu()
    elif renderer == 'webgl':
        use_webgl = True
    else:
        use_webgl = False

    print(f"Scatter dashboard renderer: {'WebGL (Scattergl)' if use_webgl else 'SVG (Scatter)'}")

    df = df.copy()

    # ---- Factor handling -----------------------------------------------------
    exclude_cols = list(jitter.keys())
    df['design'], factors_subset = make_combo_column(
        df, factors, exclude=exclude_cols, return_as_str=True
    )
    candidate_factors = list(factors_subset) + [c for c in jitter.keys() if c in df.columns]

    discrete_factors = [f for f in candidate_factors if df[f].nunique() < discrete_threshold]
    continuous_factors = [f for f in candidate_factors if df[f].nunique() >= discrete_threshold]

    factor_values = {
        f: [str(v) for v in sorted(df[f].unique()) if pd.notna(v)]
        for f in discrete_factors
    }

    # ---- Available axis options ----------------------------------------------
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    axis_options = list(dict.fromkeys(numeric_cols))

    default_x = 'R_k_avg' if 'R_k_avg' in axis_options else (axis_options[0] if axis_options else None)
    default_y = 'T_accuracy' if 'T_accuracy' in axis_options else (axis_options[1] if len(axis_options) > 1 else None)

    jitter_defaults = {col: vals for col, vals in jitter.items() if col in df.columns}

    # ---- Stable color mapping for discrete -----------------------------------
    color_palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

    def get_stable_color(combo_str, offset=0):
        if combo_str == "Other":
            return '#CCCCCC'
        hash_input = f"{combo_str}_{offset}"
        hash_value = int(md5(hash_input.encode()).hexdigest(), 16)
        return color_palette[hash_value % len(color_palette)]

    # ---- Color-by dropdown options -------------------------------------------
    SELECTION_VALUE = '_selection_'

    color_by_options = [
        {'label': '\u2b21 Filter Selection', 'value': SELECTION_VALUE},
        {'label': '\u2500\u2500 Discrete \u2500\u2500', 'value': '_header_discrete', 'disabled': True},
    ]
    for f in discrete_factors:
        color_by_options.append({'label': f'  {f}', 'value': f})
    if continuous_factors:
        color_by_options.append({'label': '\u2500\u2500 Continuous \u2500\u2500', 'value': '_header_continuous', 'disabled': True})
        for f in continuous_factors:
            color_by_options.append({'label': f'  {f}', 'value': f})

    # ---- Helper: selection-based color groups --------------------------------
    def make_selection_color_groups(filtered_df, filter_selections):
        active = {}
        for factor, values in zip(discrete_factors, filter_selections):
            if values and factor in filtered_df.columns:
                active[factor] = set(values)
        if not active:
            return None

        def row_key(row):
            parts = []
            for factor, selected in active.items():
                val_str = str(row[factor])
                if val_str in selected:
                    parts.append(f"{factor}:{val_str}")
                else:
                    return "Other"
            return ", ".join(parts) if parts else "Other"

        filtered_df['color_group'] = filtered_df.apply(row_key, axis=1)
        return 'color_group'

    # ---- WebGL helper --------------------------------------------------------
    def _apply_webgl(fig):
        if use_webgl:
            for trace in fig.data:
                if trace.type == 'scatter':
                    trace.type = 'scattergl'
        return fig

    # ---- Helper: build figure ------------------------------------------------
    ScatterType = go.Scattergl if use_webgl else go.Scatter

    def _make_discrete_figure(filtered_df, x_col, y_col, col_name, label, hover_base, color_offsets):
        unique_groups = sorted(
            [g for g in filtered_df[col_name].unique() if g != 'Other']
        )
        if 'Other' in filtered_df[col_name].values:
            unique_groups.append('Other')
        color_map = {
            g: get_stable_color(g, (color_offsets or {}).get(g, 0))
            for g in unique_groups
        }
        fig = px.scatter(
            filtered_df, x=x_col, y=y_col,
            color=col_name,
            color_discrete_map=color_map,
            custom_data=['combo_str'],
            category_orders={col_name: unique_groups},
            labels={col_name: label},
        )
        fig.update_traces(hovertemplate=hover_base)
        return _apply_webgl(fig)

    def _make_uniform_figure(filtered_df, x_col, y_col, hover_base):
        return go.Figure(ScatterType(
            x=filtered_df[x_col], y=filtered_df[y_col],
            mode='markers',
            marker=dict(color='steelblue'),
            name='Data',
            customdata=filtered_df[['combo_str']].values,
            hovertemplate=hover_base,
        ))

    def create_figure(filtered_df, x_col, y_col, color_cfg=None, color_offsets=None,
                      x_jitter=None, y_jitter=None, y_agg_label=None):
        if not x_col or not y_col:
            fig = go.Figure()
            fig.update_layout(title="Select X and Y axes to display data", xaxis_title="X", yaxis_title="Y")
            return fig

        if filtered_df is None or filtered_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data for selected filters", xaxis_title=x_col, yaxis_title=y_col)
            return fig

        filtered_df = filtered_df.copy()
        if 'design' in filtered_df.columns:
            filtered_df['combo_str'] = filtered_df['design'].astype(str)
        else:
            filtered_df['combo_str'] = ''

        plot_x_col = x_col
        plot_y_col = y_col
        if x_jitter:
            jit_col = x_col + '_jit'
            filtered_df[jit_col] = filtered_df[x_col] + np.random.uniform(x_jitter[0], x_jitter[1], size=len(filtered_df))
            plot_x_col = jit_col
        if y_jitter:
            jit_col = y_col + '_jit'
            filtered_df[jit_col] = filtered_df[y_col] + np.random.uniform(y_jitter[0], y_jitter[1], size=len(filtered_df))
            plot_y_col = jit_col

        hover_base = f'<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra>%{{fullData.name}}</extra>'

        if color_cfg and color_cfg['type'] == 'continuous':
            col = color_cfg['column']
            fig = px.scatter(
                filtered_df, x=plot_x_col, y=plot_y_col,
                color=col,
                color_continuous_scale=color_cfg.get('scale', 'Viridis'),
                custom_data=['combo_str'],
            )
            fig.update_traces(hovertemplate=hover_base)
            fig = _apply_webgl(fig)

        elif color_cfg and color_cfg['type'] == 'selection':
            col_name = make_selection_color_groups(filtered_df, color_cfg['filter_selections'])
            if col_name:
                fig = _make_discrete_figure(filtered_df, plot_x_col, plot_y_col, col_name, 'Selection', hover_base, color_offsets)
            else:
                fig = _make_uniform_figure(filtered_df, plot_x_col, plot_y_col, hover_base)

        elif color_cfg and color_cfg['type'] == 'discrete':
            col = color_cfg['column']
            selected = color_cfg['values']
            filtered_df['color_group'] = filtered_df[col].astype(str).where(
                filtered_df[col].astype(str).isin(selected), other='Other'
            )
            fig = _make_discrete_figure(filtered_df, plot_x_col, plot_y_col, 'color_group', col, hover_base, color_offsets)

        else:
            fig = _make_uniform_figure(filtered_df, plot_x_col, plot_y_col, hover_base)

        fig.update_traces(marker=dict(size=5, opacity=0.7))

        y_label = y_col
        if y_agg_label and y_agg_label != 'raw':
            y_label = f"{y_agg_label}({y_col})"

        fig.update_layout(
            title=f"{y_label} vs {x_col}", title_x=0.5,
            xaxis_title=x_col + (' (jittered)' if x_jitter else ''),
            yaxis_title=y_label + (' (jittered)' if y_jitter else ''),
            uirevision="scatter-plot",
            clickmode='event+select',
        )

        if x_jitter and x_col in df.columns:
            try:
                x_min, x_max = int(df[x_col].min()), int(df[x_col].max())
                fig.update_xaxes(tickmode='array', tickvals=list(range(x_min, x_max + 1)))
            except (ValueError, TypeError):
                pass
        if y_jitter and y_col in df.columns:
            try:
                y_min, y_max = int(df[y_col].min()), int(df[y_col].max())
                fig.update_yaxes(tickmode='array', tickvals=list(range(y_min, y_max + 1)))
            except (ValueError, TypeError):
                pass

        if not y_jitter and y_col in df.columns and y_agg_label != 'std':
            y_min, y_max = df[y_col].min(), df[y_col].max()
            if y_min >= 0 and y_max <= 1:
                fig.update_yaxes(range=[0, 1], fixedrange=True)

        return fig

    # ---- Aggregation helper --------------------------------------------------
    def aggregate_df(plot_df, x_col, y_col, group_by_cols, y_agg):
        if y_agg == 'raw' or not group_by_cols:
            return plot_df

        group_cols = list(dict.fromkeys(group_by_cols + [x_col]))
        group_cols = [c for c in group_cols if c in plot_df.columns]

        if y_col in group_cols:
            return plot_df

        agged = plot_df.groupby(group_cols, dropna=False)[y_col].agg(y_agg).reset_index()

        if 'design' not in agged.columns:
            agged['design'] = agged[group_cols].astype(str).agg(' | '.join, axis=1)

        return agged

    # ---- Jitter control builder ----------------------------------------------
    input_style = {
        'width': '60px', 'display': 'inline-block', 'fontSize': '11px',
        'padding': '2px 4px', 'marginLeft': '4px',
    }
    label_style = {'fontSize': '11px', 'color': '#666', 'marginLeft': '4px'}

    def make_jitter_controls(axis):
        default_col = default_x if axis == 'x' else default_y
        has_default = default_col in jitter_defaults
        default_min = jitter_defaults[default_col][0] if has_default else -0.5
        default_max = jitter_defaults[default_col][1] if has_default else 0.5

        return html.Div([
            dcc.Checklist(
                id=f'{axis}-jitter-toggle',
                options=[{'label': f' Jitter {axis.upper()}', 'value': 'on'}],
                value=['on'] if has_default else [],
                style={'display': 'inline-block', 'fontSize': '11px'},
                inputStyle={'marginRight': '3px'},
            ),
            html.Span('min:', style=label_style),
            dcc.Input(
                id=f'{axis}-jitter-min', type='number', value=default_min,
                step=0.1, style=input_style,
            ),
            html.Span('max:', style=label_style),
            dcc.Input(
                id=f'{axis}-jitter-max', type='number', value=default_max,
                step=0.1, style=input_style,
            ),
        ], style={'marginTop': '4px'})

    # ---- Shared button styles ------------------------------------------------
    btn_sm = {'fontSize': '10px', 'padding': '2px 6px', 'cursor': 'pointer'}

    # ---- Dash app ------------------------------------------------------------
    app = Dash(__name__)
    app.layout = html.Div([
        html.Button(
            '\u25bc Hide Controls', id='toggle-controls', n_clicks=0,
            style={
                'marginBottom': '10px', 'padding': '8px 16px',
                'fontSize': '14px', 'cursor': 'pointer',
                'border': '1px solid #ccc', 'borderRadius': '4px',
                'background': '#f5f5f5',
            }
        ),

        html.Div([
            # Row 1: Axis selection + jitter + color + lazy mode
            html.Div([
                html.Div([
                    html.Label('X-axis:'),
                    dcc.Dropdown(
                        id='x-axis',
                        options=[{'label': c, 'value': c} for c in axis_options],
                        value=default_x, clearable=True,
                        placeholder='Select X axis...',
                    ),
                    make_jitter_controls('x'),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label('Y-axis:'),
                    dcc.Dropdown(
                        id='y-axis',
                        options=[{'label': c, 'value': c} for c in axis_options],
                        value=default_y, clearable=True,
                        placeholder='Select Y axis...',
                    ),
                    make_jitter_controls('y'),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label('Color by:'),
                    dcc.Dropdown(
                        id='color-by-factor',
                        options=color_by_options,
                        value=SELECTION_VALUE, clearable=True,
                        placeholder='None (uniform color)',
                    ),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Color values:'),
                    dcc.Dropdown(
                        id='color-by-values',
                        options=[], value=[], multi=True,
                        placeholder='Select a color factor first...',
                    ),
                ], id='color-values-container',
                   style={'width': '18%', 'display': 'none', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Color scale:'),
                    dcc.Dropdown(
                        id='color-scale-picker',
                        options=[
                            {'label': 'Viridis', 'value': 'Viridis'},
                            {'label': 'Plasma', 'value': 'Plasma'},
                            {'label': 'RdBu', 'value': 'RdBu'},
                            {'label': 'Hot', 'value': 'Hot'},
                            {'label': 'Turbo', 'value': 'Turbo'},
                        ],
                        value='Viridis', clearable=False,
                    ),
                ], id='color-scale-container',
                   style={'width': '18%', 'display': 'none', 'marginRight': '2%'}),

                html.Div([
                    html.Button(
                        'Lazy Mode (ON)', id='lazy-mode-toggle', n_clicks=0,
                        style={
                            'padding': '6px 12px', 'fontSize': '12px',
                            'cursor': 'pointer', 'border': '2px solid #ccc',
                            'borderRadius': '4px', 'background': '#e3f2fd',
                            'fontWeight': 'bold', 'marginRight': '8px',
                        }
                    ),
                    html.Button(
                        'Refresh', id='refresh-btn', n_clicks=0,
                        style={
                            'padding': '6px 12px', 'fontSize': '12px',
                            'cursor': 'pointer', 'border': '2px solid #4CAF50',
                            'borderRadius': '4px', 'background': '#4CAF50',
                            'color': 'white', 'fontWeight': 'bold',
                        }
                    ),
                ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'bottom'}),
            ], style={'marginBottom': '20px'}),

            # Row 2: Aggregation controls
            html.Div([
                html.Div(["Aggregation:"], style={'marginBottom': '10px', 'fontStyle': 'italic', 'color': '#666'}),
                html.Div([
                    html.Div([
                        html.Label('Group by:'),
                        html.Div([
                            html.Button('Select All', id='agg-group-select-all', n_clicks=0,
                                        style={**btn_sm, 'marginRight': '4px'}),
                            html.Button('Clear All', id='agg-group-clear-all', n_clicks=0, style=btn_sm),
                        ], style={'marginBottom': '4px'}),
                        dcc.Dropdown(
                            id='agg-group-by',
                            options=[{'label': f, 'value': f} for f in discrete_factors],
                            value=[], multi=True,
                            placeholder='Select columns to group by...',
                        ),
                    ], style={'width': '36%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                    html.Div([
                        html.Label('Y Aggregation:'),
                        dcc.Dropdown(
                            id='agg-y-mode',
                            options=[
                                {'label': 'None (raw)', 'value': 'raw'},
                                {'label': 'Mean', 'value': 'mean'},
                                {'label': 'Median', 'value': 'median'},
                                {'label': 'Min', 'value': 'min'},
                                {'label': 'Max', 'value': 'max'},
                                {'label': 'Std', 'value': 'std'},
                            ],
                            value='raw', clearable=False,
                        ),
                    ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                ]),
            ], style={'marginBottom': '20px'}),

            # Row 3: Filter dropdowns (discrete factors only)
            html.Div([
                html.Div(["Filter by factor values:"], style={'marginBottom': '10px', 'fontStyle': 'italic', 'color': '#666'}),
                html.Div([
                    html.Div([
                        html.Label(factor),
                        html.Div([
                            html.Button('Select All', id={'type': 'select-all', 'factor': factor},
                                        n_clicks=0, style={**btn_sm, 'marginRight': '4px'}),
                            html.Button('Clear All', id={'type': 'clear-all', 'factor': factor},
                                        n_clicks=0, style=btn_sm),
                        ], style={'marginBottom': '4px'}),
                        dcc.Dropdown(
                            id={'type': 'dropdown', 'factor': factor},
                            options=[{'label': v, 'value': v} for v in factor_values[factor]],
                            value=[], multi=True,
                        ),
                    ], style={
                        'width': '18%', 'display': 'inline-block',
                        'marginRight': '2%', 'verticalAlign': 'top',
                    })
                    for factor in discrete_factors
                ]),
            ], style={'marginBottom': '20px'}),
        ], id='controls-container', style={'display': 'block'}),

        dcc.Store(id='lazy-mode-active', data=True),
        dcc.Store(id='color-offsets', data={}),
        dcc.Store(id='legend-shift-click', data=''),
        dcc.Graph(id='scatter-plot', style={'height': '60vh'}),

        html.Pre(id='point-details', children='Click a point to see details.', style={
            'padding': '10px', 'background': '#f8f8f8', 'border': '1px solid #ddd',
            'borderRadius': '4px', 'fontSize': '12px', 'maxHeight': '150px',
            'overflowY': 'auto', 'whiteSpace': 'pre-wrap', 'marginTop': '10px',
        }),
    ])

    # ---- Shift+click legend handler ------------------------------------------
    app.clientside_callback(
        """
        function(figure) {
            if (!window._shiftKeyState) {
                window._shiftKeyState = {pressed: false};
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Shift') window._shiftKeyState.pressed = true;
                });
                document.addEventListener('keyup', function(e) {
                    if (e.key === 'Shift') window._shiftKeyState.pressed = false;
                });
            }

            setTimeout(function() {
                var gd = document.querySelector('#scatter-plot > .js-plotly-plot');
                if (!gd || gd._shiftClickAttached) return;
                gd._shiftClickAttached = true;

                gd.on('plotly_legendclick', function(eventData) {
                    if (window._shiftKeyState.pressed) {
                        var traceName = eventData.data[eventData.curveNumber].name;
                        dash_clientside.set_props('legend-shift-click', {
                            data: traceName + '|' + Date.now()
                        });
                        return false;
                    }
                });
            }, 300);
            return window.dash_clientside.no_update;
        }
        """,
        Output('scatter-plot', 'id'),
        Input('scatter-plot', 'figure'),
    )

    # ---- Callbacks -----------------------------------------------------------

    @callback(
        Output('controls-container', 'style'),
        Output('toggle-controls', 'children'),
        Input('toggle-controls', 'n_clicks'),
    )
    def toggle_controls(n_clicks):
        if n_clicks % 2 == 1:
            return {'display': 'none'}, '\u25b6 Show Controls'
        return {'display': 'block'}, '\u25bc Hide Controls'

    @callback(
        Output('lazy-mode-active', 'data'),
        Output('lazy-mode-toggle', 'children'),
        Output('lazy-mode-toggle', 'style'),
        Input('lazy-mode-toggle', 'n_clicks'),
        State('lazy-mode-active', 'data'),
        prevent_initial_call=True,
    )
    def toggle_lazy_mode(n_clicks, is_active):
        new_state = not is_active
        label = 'Lazy Mode (ON)' if new_state else 'Lazy Mode (OFF)'
        style = {
            'padding': '6px 12px', 'fontSize': '12px',
            'cursor': 'pointer', 'border': '2px solid #ccc',
            'borderRadius': '4px',
            'background': '#e3f2fd' if new_state else 'white',
            'fontWeight': 'bold' if new_state else 'normal',
            'marginRight': '8px',
        }
        return new_state, label, style

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
        return []

    @callback(
        Output('agg-group-by', 'value'),
        Input('agg-group-select-all', 'n_clicks'),
        Input('agg-group-clear-all', 'n_clicks'),
        prevent_initial_call=True,
    )
    def update_agg_group_selection(select_clicks, clear_clicks):
        if ctx.triggered_id == 'agg-group-select-all':
            return discrete_factors
        return []

    @callback(
        Output('color-by-values', 'options'),
        Output('color-by-values', 'value'),
        Output('color-values-container', 'style'),
        Output('color-scale-container', 'style'),
        Input('color-by-factor', 'value'),
    )
    def update_color_controls(color_factor):
        show = {'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}
        hidden = {'width': '18%', 'display': 'none', 'marginRight': '2%'}

        if not color_factor or color_factor == SELECTION_VALUE:
            return [], [], hidden, hidden

        if color_factor in continuous_factors:
            return [], [], hidden, show

        opts = [{'label': v, 'value': v} for v in factor_values[color_factor]]
        return opts, factor_values[color_factor], show, hidden

    @callback(
        Output('x-jitter-toggle', 'value'),
        Output('x-jitter-min', 'value'),
        Output('x-jitter-max', 'value'),
        Input('x-axis', 'value'),
        prevent_initial_call=True,
    )
    def auto_jitter_x(x_col):
        if x_col in jitter_defaults:
            return ['on'], jitter_defaults[x_col][0], jitter_defaults[x_col][1]
        return [], -0.5, 0.5

    @callback(
        Output('y-jitter-toggle', 'value'),
        Output('y-jitter-min', 'value'),
        Output('y-jitter-max', 'value'),
        Input('y-axis', 'value'),
        prevent_initial_call=True,
    )
    def auto_jitter_y(y_col):
        if y_col in jitter_defaults:
            return ['on'], jitter_defaults[y_col][0], jitter_defaults[y_col][1]
        return [], -0.5, 0.5

    @callback(
        Output('color-offsets', 'data'),
        Input('legend-shift-click', 'data'),
        State('color-offsets', 'data'),
        prevent_initial_call=True,
    )
    def bump_color_offset(shift_click_data, offsets):
        if not shift_click_data:
            raise PreventUpdate
        trace_name = shift_click_data.rsplit('|', 1)[0]
        offsets[trace_name] = offsets.get(trace_name, 0) + 1
        return offsets

    @callback(
        Output('point-details', 'children'),
        Input('scatter-plot', 'clickData'),
        State('x-axis', 'value'),
        State('y-axis', 'value'),
        State('agg-group-by', 'value'),
        State('agg-y-mode', 'value'),
        prevent_initial_call=True,
    )
    def show_point_details(click_data, x_col, y_col, agg_group_by, agg_y_mode):
        if not click_data or not click_data['points']:
            raise PreventUpdate

        pt = click_data['points'][0]
        combo = pt.get('customdata', [None])[0]
        lines = []

        if combo:
            values = [v.strip() for v in str(combo).split('_')]
            for name, val in zip(factors_subset, values):
                lines.append(f"{name}: {val}")

        # Add x/y only if not already shown via factors
        y_label = f"{agg_y_mode}({y_col})" if agg_y_mode and agg_y_mode != 'raw' and agg_group_by else y_col
        if x_col not in factors_subset:
            lines.append(f"{x_col}: {pt.get('x')}")
        if y_col not in factors_subset:
            lines.append(f"{y_label}: {pt.get('y')}")

        return '\n'.join(lines)

    @callback(
        Output('scatter-plot', 'figure'),
        Input('refresh-btn', 'n_clicks'),
        Input('x-axis', 'value'),
        Input('y-axis', 'value'),
        Input('x-jitter-toggle', 'value'),
        Input('x-jitter-min', 'value'),
        Input('x-jitter-max', 'value'),
        Input('y-jitter-toggle', 'value'),
        Input('y-jitter-min', 'value'),
        Input('y-jitter-max', 'value'),
        Input('color-by-factor', 'value'),
        Input('color-by-values', 'value'),
        Input('color-scale-picker', 'value'),
        Input({'type': 'dropdown', 'factor': ALL}, 'value'),
        Input('color-offsets', 'data'),
        Input('agg-group-by', 'value'),
        Input('agg-y-mode', 'value'),
        State('lazy-mode-active', 'data'),
    )
    def update_figure(refresh_clicks, x_col, y_col,
                      x_jit_on, x_jit_min, x_jit_max,
                      y_jit_on, y_jit_min, y_jit_max,
                      color_factor, color_values, color_scale,
                      filter_values, color_offsets,
                      agg_group_by, agg_y_mode,
                      lazy_mode):
        if lazy_mode and ctx.triggered_id not in ('refresh-btn', 'color-offsets'):
            raise PreventUpdate

        if not x_col or not y_col:
            return create_figure(None, x_col, y_col)

        # Apply filters
        mask = np.ones(len(df), dtype=bool)
        for factor, values in zip(discrete_factors, filter_values):
            if values:
                mask &= df[factor].astype(str).isin(values)
        filtered_df = df[mask]

        # Aggregation
        if agg_y_mode and agg_y_mode != 'raw' and agg_group_by:
            effective_group = list(agg_group_by)
            if color_factor and color_factor not in (SELECTION_VALUE, None) and color_factor not in effective_group:
                effective_group.append(color_factor)
            filtered_df = aggregate_df(filtered_df, x_col, y_col, effective_group, agg_y_mode)

        # Parse jitter
        x_jitter = None
        if x_jit_on and 'on' in x_jit_on and x_jit_min is not None and x_jit_max is not None:
            x_jitter = (float(x_jit_min), float(x_jit_max))
        y_jitter = None
        if y_jit_on and 'on' in y_jit_on and y_jit_min is not None and y_jit_max is not None:
            y_jitter = (float(y_jit_min), float(y_jit_max))

        # Build color config
        color_cfg = None
        if color_factor == SELECTION_VALUE:
            color_cfg = {'type': 'selection', 'filter_selections': filter_values}
        elif color_factor:
            if color_factor in continuous_factors:
                color_cfg = {'type': 'continuous', 'column': color_factor, 'scale': color_scale}
            elif color_values:
                color_cfg = {'type': 'discrete', 'column': color_factor, 'values': color_values}

        y_agg_label = agg_y_mode if (agg_y_mode and agg_y_mode != 'raw' and agg_group_by) else None

        return create_figure(filtered_df, x_col, y_col, color_cfg, color_offsets,
                             x_jitter=x_jitter, y_jitter=y_jitter, y_agg_label=y_agg_label)

    return app