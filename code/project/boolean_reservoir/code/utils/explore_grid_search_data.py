from pathlib import Path
import plotly.io as pio
from dash import Dash, dcc, html, callback, Input, Output, State, ALL, ctx, MATCH, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from hashlib import md5
from dash.dependencies import Input, Output, State
import yaml
import orjson

    
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
    import os, subprocess
    # No local display → headless server accessed remotely (e.g. over SSH).
    # The remote browser almost certainly supports WebGL, so return True
    # without running hardware detection tools.
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        return True
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

# ── Module-level style constants ──────────────────────────────────────────────
_DARK = '#1e1e1e'
_DARK2 = '#2a2a2a'
_DARK3 = '#333333'
_BORDER = '#444'
_TEXT = '#ddd'
_TEXT_DIM = '#888'

input_style = {
    'width': '60px', 'display': 'inline-block', 'fontSize': '11px',
    'padding': '2px 4px', 'marginLeft': '4px',
    'background': _DARK2, 'color': _TEXT, 'border': f'1px solid {_BORDER}', 'borderRadius': '3px',
}
label_style = {'fontSize': '11px', 'color': _TEXT_DIM, 'marginLeft': '4px'}

_collapse_btn_style = {
    'padding': '4px 10px', 'fontSize': '12px',
    'cursor': 'pointer', 'border': f'1px solid {_BORDER}',
    'borderRadius': '4px', 'background': _DARK3, 'color': _TEXT,
}

btn_sm = {
    'fontSize': '10px', 'padding': '2px 6px', 'cursor': 'pointer',
    'background': _DARK3, 'color': _TEXT, 'border': f'1px solid {_BORDER}', 'borderRadius': '3px',
}

_COLOR_SCALE_OPTIONS = [
    {'label': 'Viridis', 'value': 'Viridis'},
    {'label': 'Plasma', 'value': 'Plasma'},
    {'label': 'RdBu', 'value': 'RdBu'},
    {'label': 'Hot', 'value': 'Hot'},
    {'label': 'Turbo', 'value': 'Turbo'},
]


def aggregate_df(plot_df, x_col, y_col, group_by_cols, y_agg, x_is_binned=False, extra_agg_cols=None):
    if y_agg == 'raw' or (not group_by_cols and not x_is_binned):
        return plot_df
    group_cols = list(dict.fromkeys(group_by_cols + [x_col]))
    group_cols = [c for c in group_cols if c in plot_df.columns]
    if y_col in group_cols:
        return plot_df
    cols_to_agg = [c for c in ([y_col] + (extra_agg_cols or []))
                   if c in plot_df.columns and c not in group_cols]
    agged = plot_df.groupby(group_cols, dropna=False)[cols_to_agg].agg(y_agg).reset_index()
    if 'design' not in agged.columns:
        agged['design'] = agged[group_cols].astype(str).apply(' | '.join, axis=1)
    return agged


def make_collapsible_panel(panel_id, label, children, start_open=False):
    """DRY collapsible panel used for both controls and custom filter."""
    arrow_open, arrow_closed = '\u25bc', '\u25b6'
    return html.Div([
        html.Button(
            f'{arrow_open if start_open else arrow_closed} {label}',
            id=f'{panel_id}-toggle',
            n_clicks=0,
            style=_collapse_btn_style,
        ),
        html.Div(
            children,
            id=f'{panel_id}-body',
            style={'display': 'block' if start_open else 'none'},
        ),
    ], style={'marginBottom': '10px'})


def parse_and_apply_fields(working_df, fields_text):
    """Parse 'name = expr' lines and eval onto df. Returns (df, info, error)."""
    if not fields_text or not fields_text.strip():
        return working_df, '', ''
    errors = []
    added = []
    for i, line in enumerate(fields_text.strip().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            errors.append(f'L{i}: missing "=" in "{line}"')
            continue
        name, expr = line.split('=', 1)
        name = name.strip()
        expr = expr.strip()
        if not name or not expr:
            errors.append(f'L{i}: empty name or expression')
            continue
        try:
            working_df[name] = working_df.eval(expr)
            added.append(name)
        except Exception as e:
            errors.append(f'L{i} ({name}): {e}')
    error_str = '\n'.join(errors) if errors else ''
    info_str = f'Added: {", ".join(added)}' if added else ''
    return working_df, info_str, error_str


def _views_dir(path_str):
    return Path(path_str.strip()) if path_str and path_str.strip() else Path('/tmp/dashboard')


def _list_views(views_dir):
    if not views_dir.exists():
        return []
    return sorted(p.stem for p in views_dir.glob('*.json'))


def _view_dropdown_options(views_dir):
    return [{'label': name, 'value': name} for name in _list_views(views_dir)]


def _is_safe_path(p: Path, safe_roots: list) -> bool:
    resolved = p.resolve()
    return any(resolved == r or r in resolved.parents for r in safe_roots)


def _fmt(n: int) -> str:
    if n >= 1_000_000: return f'{n/1_000_000:.1f}M'
    if n >= 10_000:    return f'{n/1_000:.0f}k'
    if n >= 1_000:     return f'{n/1_000:.1f}k'
    return str(n)


def create_scatter_dashboard(
    df: pd.DataFrame,
    factors: list[str],
    jitter: dict = {'R_k_avg': [-0.5, 0.5]},
    discrete_threshold: int = 20,
    renderer: str = 'auto',
    initial_display_points: int = 10_000,
    safe_roots: list[str] = ['/out', '/tmp'],
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
    has_params_json = 'params_json' in df.columns

    # ---- Factor handling -----------------------------------------------------
    df['design'], factors_subset = make_combo_column(
        df, factors, return_as_str=True
    )
    candidate_factors = list(factors_subset)

    discrete_factors = [f for f in candidate_factors if df[f].nunique() < discrete_threshold]
    continuous_factors = [f for f in candidate_factors if df[f].nunique() >= discrete_threshold]
    numeric_discrete_factors = [f for f in discrete_factors if pd.api.types.is_numeric_dtype(df[f])]

    factor_values = {
        f: [str(v) for v in sorted(v for v in df[f].unique() if pd.notna(v))]
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
        _cd_cols = list(factors_subset) if factors_subset else ['combo_str']
        if has_params_json and '_row_idx' in filtered_df.columns:
            _cd_cols = _cd_cols + ['_row_idx']
        fig = px.scatter(
            filtered_df, x=x_col, y=y_col,
            color=col_name,
            color_discrete_map=color_map,
            custom_data=_cd_cols,
            category_orders={col_name: unique_groups},
            labels={col_name: label},
            render_mode='webgl' if use_webgl else 'svg',
        )
        fig.update_traces(hovertemplate=hover_base)
        return fig

    def _make_uniform_figure(filtered_df, x_col, y_col, hover_base):
        cd_cols = list(factors_subset) if factors_subset else ['combo_str']
        if has_params_json and '_row_idx' in filtered_df.columns:
            cd_cols = cd_cols + ['_row_idx']
        return go.Figure(ScatterType(
            x=filtered_df[x_col], y=filtered_df[y_col],
            mode='markers',
            marker=dict(color='steelblue'),
            name='Data',
            customdata=filtered_df[cd_cols].values,
            hovertemplate=hover_base,
        ))

    def create_figure(filtered_df, x_col, y_col, color_cfg=None, color_offsets=None,
                      x_jitter=None, y_jitter=None, y_agg_label=None,
                      x_label=None, marker_size=5, marker_opacity=0.7):
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

        # Add row index for YAML export; -1 when aggregated (no meaningful row mapping).
        # Skip if already present (set by data stage and preserved through store).
        if has_params_json and '_row_idx' not in filtered_df.columns:
            filtered_df['_row_idx'] = -1 if y_agg_label else filtered_df.index

        # Ensure all factor columns exist in filtered_df (may be missing after aggregation)
        for f in factors_subset:
            if f not in filtered_df.columns:
                filtered_df[f] = None

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
            _cd_cols = list(factors_subset) if factors_subset else ['combo_str']
            if has_params_json and '_row_idx' in filtered_df.columns:
                _cd_cols = _cd_cols + ['_row_idx']
            fig = px.scatter(
                filtered_df, x=plot_x_col, y=plot_y_col,
                color=col,
                color_continuous_scale=color_cfg.get('scale', 'Viridis'),
                custom_data=_cd_cols,
                render_mode='webgl' if use_webgl else 'svg',
            )
            fig.update_traces(hovertemplate=hover_base)

        elif color_cfg and color_cfg['type'] == 'precomputed':
            col = color_cfg['column']
            label = color_cfg.get('label', 'Group')
            if col in filtered_df.columns:
                fig = _make_discrete_figure(filtered_df, plot_x_col, plot_y_col, col, label, hover_base, color_offsets)
            else:
                fig = _make_uniform_figure(filtered_df, plot_x_col, plot_y_col, hover_base)

        elif color_cfg and color_cfg['type'] == 'columns':
            cols = [c for c in color_cfg['columns'] if c in filtered_df.columns]
            if cols:
                filtered_df['color_group'] = (
                    filtered_df[cols].astype(str).apply(' | '.join, axis=1)
                    if len(cols) > 1
                    else filtered_df[cols[0]].astype(str)
                )
                fig = _make_discrete_figure(filtered_df, plot_x_col, plot_y_col, 'color_group',
                                            ' | '.join(cols), hover_base, color_offsets)
            else:
                fig = _make_uniform_figure(filtered_df, plot_x_col, plot_y_col, hover_base)

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

        fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

        y_label = y_col
        if y_agg_label and y_agg_label != 'raw':
            y_label = f"{y_agg_label}({y_col})"

        _x_display = x_label or x_col
        fig.update_layout(
            template='plotly_dark',
            title=f"{y_label} vs {_x_display}", title_x=0.5,
            xaxis_title=_x_display + (' (jittered)' if x_jitter else ''),
            yaxis_title=y_label + (' (jittered)' if y_jitter else ''),
            uirevision="scatter-plot",
            clickmode='event+select',
        )

        if x_jitter and x_col in filtered_df.columns:
            try:
                x_min, x_max = int(filtered_df[x_col].min()), int(filtered_df[x_col].max())
                fig.update_xaxes(tickmode='array', tickvals=list(range(x_min, x_max + 1)))
            except (ValueError, TypeError):
                pass
        if y_jitter and y_col in filtered_df.columns:
            try:
                y_min, y_max = int(filtered_df[y_col].min()), int(filtered_df[y_col].max())
                fig.update_yaxes(tickmode='array', tickvals=list(range(y_min, y_max + 1)))
            except (ValueError, TypeError):
                pass

        if not y_jitter and y_col in filtered_df.columns and y_agg_label != 'std':
            y_min, y_max = filtered_df[y_col].min(), filtered_df[y_col].max()
            if y_min >= 0 and y_max <= 1:
                fig.update_yaxes(range=[0, 1], fixedrange=True)

        return fig

    # ---- Jitter control builder ----------------------------------------------
    def make_jitter_controls(axis):
        default_col = default_x if axis == 'x' else default_y
        has_default = default_col in jitter_defaults
        default_min = jitter_defaults[default_col][0] if has_default else -0.5
        default_max = jitter_defaults[default_col][1] if has_default else 0.5

        jitter_row = html.Div([
            dcc.Checklist(
                id=f'{axis}-jitter-toggle',
                options=[{'label': f' Jitter {axis.upper()}', 'value': 'on'}],
                value=['on'] if has_default else [],
                style={'display': 'inline-block', 'fontSize': '11px'},
                inputStyle={'marginRight': '3px'},
                labelStyle={'color': _TEXT_DIM},
            ),
            html.Span('min:', style=label_style),
            dcc.Input(
                id=f'{axis}-jitter-min', type='number', value=default_min,
                step='any', style=input_style,
            ),
            html.Span('max:', style=label_style),
            dcc.Input(
                id=f'{axis}-jitter-max', type='number', value=default_max,
                step='any', style=input_style,
            ),
        ])
        if axis == 'x':
            bin_row = html.Div([
                dcc.Checklist(
                    id='x-bin-toggle',
                    options=[{'label': ' Bin X', 'value': 'on'}],
                    value=[],
                    style={'display': 'inline-block', 'fontSize': '11px'},
                    inputStyle={'marginRight': '3px'},
                    labelStyle={'color': _TEXT_DIM},
                ),
                html.Span('width:', style=label_style),
                dcc.Input(
                    id='x-bin-width', type='number', value=1,
                    min=0, step='any', style=input_style,
                ),
            ], style={'marginTop': '2px'})
            return html.Div([jitter_row, bin_row], style={'marginTop': '4px'})
        return html.Div([jitter_row], style={'marginTop': '4px'})

    # ---- Build available columns hint for filter placeholder -----------------
    sample_cols = axis_options[:6]
    cols_hint = ', '.join(sample_cols)
    if len(axis_options) > 6:
        cols_hint += ', ...'

    # ---- Dash app ------------------------------------------------------------
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace(
        '<body>',
        '<body style="background:#1e1e1e;color:#ddd;margin:0;">'
    ).replace(
        '</head>',
        '''<style>
        :root {
            --dd-bg: #2a2a2a;
            --dd-bg-hover: #3a3a3a;
            --dd-bg-selected: #1a3a5c;
            --dd-border: #444;
            --dd-text: #ddd;
            --dd-text-dim: #888;
        }
        /* Closed state: wrapper + ALL children (catches button browser defaults) */
        .dash-dropdown-wrapper,
        .dash-dropdown-wrapper * {
            background-color: var(--dd-bg) !important;
            color: var(--dd-text) !important;
            border-color: var(--dd-border) !important;
        }
        /* Open panel */
        .dash-dropdown-content {
            background-color: var(--dd-bg) !important;
            border-color: var(--dd-border) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.6) !important;
        }
        .dash-dropdown-option {
            background-color: var(--dd-bg) !important;
            color: var(--dd-text) !important;
        }
        .dash-dropdown-option:hover { background-color: var(--dd-bg-hover) !important; color: #fff !important; }
        .dash-dropdown-option.selected { background-color: var(--dd-bg-selected) !important; color: #fff !important; }
        .dash-dropdown-option.disabled { color: var(--dd-text-dim) !important; }
        input, textarea { color-scheme: dark; }
        </style>
        </head>'''
    )
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        # Main controls panel (collapsible)
        make_collapsible_panel('controls', 'Controls', start_open=False, children=[
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
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label('Y-axis:'),
                    dcc.Dropdown(
                        id='y-axis',
                        options=[{'label': c, 'value': c} for c in axis_options],
                        value=default_y, clearable=True,
                        placeholder='Select Y axis...',
                    ),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                html.Div([
                    html.Label('Color by:'),
                    dcc.Dropdown(
                        id='color-by-factor',
                        options=color_by_options,
                        value=[SELECTION_VALUE], clearable=True,
                        multi=True,
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
                    dcc.Checklist(
                        id='color-as-continuous',
                        options=[{'label': ' Continuous', 'value': 'on'}],
                        value=[],
                        inputStyle={'marginRight': '3px'},
                        labelStyle={'color': _TEXT_DIM, 'fontSize': '11px'},
                    ),
                ], id='color-continuous-toggle-container',
                   style={'display': 'none', 'width': '9%', 'marginRight': '1%', 'verticalAlign': 'bottom'}),

                html.Div([
                    html.Label('Color scale:'),
                    dcc.Dropdown(
                        id='color-scale-picker',
                        options=_COLOR_SCALE_OPTIONS,
                        value='Viridis', clearable=False,
                    ),
                ], id='color-scale-container',
                   style={'width': '18%', 'display': 'none', 'marginRight': '2%'}),

                html.Div([
                    html.Button(
                        'Lazy Mode (ON)', id='lazy-mode-toggle', n_clicks=0,
                        style={
                            'padding': '6px 12px', 'fontSize': '12px',
                            'cursor': 'pointer', 'border': f'2px solid {_BORDER}',
                            'borderRadius': '4px', 'background': '#1a3a5c',
                            'color': _TEXT, 'fontWeight': 'bold', 'marginRight': '8px',
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
                html.Div(["Aggregation:"], style={'marginBottom': '10px', 'fontStyle': 'italic', 'color': _TEXT_DIM}),
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
                html.Div(["Filter by factor values:"], style={'marginBottom': '10px', 'fontStyle': 'italic', 'color': _TEXT_DIM}),
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
        ]),

        # Custom filter & computed fields panel (collapsible, separate from main controls)
        make_collapsible_panel('custom', 'Custom Filter & Fields', start_open=False, children=[
            # Custom filter
            html.Div([
                html.Div(["Filter (df.query):"], style={'fontStyle': 'italic', 'color': _TEXT_DIM, 'fontSize': '12px'}),
                dcc.Textarea(
                    id='filter-expr',
                    placeholder=f'e.g. (T_loss > 0.1) & (T_accuracy <= 0.95)\n'
                                f'Columns: {cols_hint}',
                    style={
                        'width': '100%', 'fontFamily': 'monospace',
                        'fontSize': '12px', 'minHeight': '40px',
                        'marginTop': '4px', 'padding': '6px',
                        'border': f'1px solid {_BORDER}', 'borderRadius': '4px',
                        'background': _DARK2, 'color': _TEXT,
                    },
                    value='',
                ),
                html.Div(id='filter-error', style={
                    'color': '#ff6b6b', 'fontSize': '11px',
                    'marginTop': '4px', 'fontFamily': 'monospace',
                }),
                html.Div(id='filter-count', style={
                    'color': _TEXT_DIM, 'fontSize': '11px', 'marginTop': '2px',
                }),
            ], style={'marginTop': '6px', 'marginBottom': '12px'}),

            # Computed fields
            html.Div([
                html.Div(["Computed fields (name = expr, one per line):"],
                         style={'fontStyle': 'italic', 'color': _TEXT_DIM, 'fontSize': '12px'}),
                dcc.Textarea(
                    id='fields-expr',
                    placeholder='e.g.:\n'
                                '  loss_sq = T_loss ** 2\n'
                                '  acc_pct = T_accuracy * 100\n'
                                '  ratio = T_loss / (T_accuracy + 1e-8)',
                    style={
                        'width': '100%', 'fontFamily': 'monospace',
                        'fontSize': '12px', 'minHeight': '50px',
                        'marginTop': '4px', 'padding': '6px',
                        'border': f'1px solid {_BORDER}', 'borderRadius': '4px',
                        'background': _DARK2, 'color': _TEXT,
                    },
                    value='',
                ),
                html.Div(id='fields-error', style={
                    'color': '#ff6b6b', 'fontSize': '11px',
                    'marginTop': '4px', 'fontFamily': 'monospace',
                }),
                html.Div(id='fields-info', style={
                    'color': _TEXT_DIM, 'fontSize': '11px', 'marginTop': '2px',
                }),
            ]),
        ]),

        # Views panel (collapsible)
        make_collapsible_panel('views', 'Views', start_open=False, children=[
            html.Div([
                html.Div([
                    dcc.Input(
                        id='view-name', type='text', placeholder='View name...',
                        style={'fontSize': '12px', 'padding': '4px 8px', 'width': '180px',
                               'marginRight': '8px', 'border': f'1px solid {_BORDER}', 'borderRadius': '4px',
                               'background': _DARK2, 'color': _TEXT},
                    ),
                    html.Button('Save', id='view-save-btn', n_clicks=0,
                                style={**btn_sm, 'marginRight': '8px', 'background': '#4CAF50',
                                       'color': 'white', 'border': '1px solid #4CAF50', 'borderRadius': '4px'}),
                    html.Button('Delete', id='view-delete-btn', n_clicks=0,
                                style={**btn_sm, 'marginRight': '16px', 'background': '#f44336',
                                       'color': 'white', 'border': '1px solid #f44336', 'borderRadius': '4px'}),
                    dcc.Dropdown(
                        id='view-picker',
                        options=[], value=None, clearable=True,
                        placeholder='Load a saved view...',
                        style={'width': '220px', 'display': 'inline-block', 'verticalAlign': 'middle',
                               'fontSize': '12px'},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '4px'}),
                html.Div([
                    html.Span('Dir:', style={'fontSize': '11px', 'color': _TEXT_DIM, 'marginRight': '4px'}),
                    dcc.Input(
                        id='views-dir', type='text', value='/out/dashboard',
                        style={'fontSize': '11px', 'padding': '2px 6px', 'width': '300px',
                               'border': f'1px solid {_BORDER}', 'borderRadius': '4px', 'fontFamily': 'monospace',
                               'background': _DARK2, 'color': _TEXT},
                        debounce=True,
                    ),
                ], style={'marginTop': '4px'}),
                html.Div(id='view-status', style={
                    'color': _TEXT_DIM, 'fontSize': '11px', 'marginTop': '4px',
                }),
            ], style={'marginTop': '6px'}),
        ]),

        # Settings panel (collapsible, opened via ⚙ cog) — session-only defaults
        make_collapsible_panel('settings', '\u2699 Settings', start_open=False, children=[
            html.Div([
                # ── Sampling ──
                html.Div('Sampling', style={'fontWeight': 'bold', 'color': _TEXT_DIM, 'fontSize': '11px', 'marginBottom': '4px', 'textTransform': 'uppercase', 'letterSpacing': '0.05em'}),
                html.Div([
                    html.Label('Max display points:', style={'fontSize': '12px', 'marginRight': '6px'}),
                    dcc.Input(
                        id='setting-display-points', type='number', value=initial_display_points,
                        min=1000, step=1000,
                        style={**input_style, 'width': '80px'},
                    ),
                    html.Span('sampled when data exceeds this', style={**label_style, 'marginLeft': '8px'}),
                ], style={'marginBottom': '4px', 'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.Label('Discrete threshold:', style={'fontSize': '12px', 'marginRight': '6px'}),
                    dcc.Input(
                        id='setting-discrete-threshold', type='number', value=discrete_threshold,
                        min=2, step=1,
                        style={**input_style, 'width': '60px'},
                    ),
                    html.Span('unique values below this → discrete (restart required)', style={**label_style, 'marginLeft': '8px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),

                # ── Jitter & Binning ──
                html.Div('Jitter & Binning', style={'fontWeight': 'bold', 'color': _TEXT_DIM, 'fontSize': '11px', 'marginBottom': '4px', 'textTransform': 'uppercase', 'letterSpacing': '0.05em'}),
                make_jitter_controls('x'),
                make_jitter_controls('y'),
                html.Div(style={'marginBottom': '12px'}),

                # ── Markers ──
                html.Div('Markers', style={'fontWeight': 'bold', 'color': _TEXT_DIM, 'fontSize': '11px', 'marginBottom': '4px', 'textTransform': 'uppercase', 'letterSpacing': '0.05em'}),
                html.Div([
                    html.Label('Size:', style={'fontSize': '12px', 'marginRight': '6px'}),
                    dcc.Input(
                        id='setting-marker-size', type='number', value=5,
                        min=1, max=30, step=1,
                        style={**input_style, 'width': '55px'},
                    ),
                    html.Label('Opacity:', style={'fontSize': '12px', 'marginLeft': '12px', 'marginRight': '6px'}),
                    dcc.Input(
                        id='setting-marker-opacity', type='number', value=0.7,
                        min=0.05, max=1.0, step=0.05,
                        style={**input_style, 'width': '55px'},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={'marginTop': '6px'}),
        ]),

        dcc.Store(id='lazy-mode-active', data=True),
        dcc.Store(id='color-offsets', data={}),
        dcc.Store(id='legend-shift-click', data=''),
        dcc.Store(id='view-loading', data=False),
        dcc.Store(id='active-view-name', data=None),
        dcc.Store(id='processed-data', data=None),
        dcc.Graph(id='scatter-plot', style={'height': '60vh'}),

        html.Pre(id='point-details', children='Click a point to see details.', style={
            'padding': '10px', 'background': _DARK2, 'border': f'1px solid {_BORDER}',
            'borderRadius': '4px', 'fontSize': '12px', 'maxHeight': '150px',
            'overflowY': 'auto', 'whiteSpace': 'pre-wrap', 'marginTop': '10px',
            'color': _TEXT,
        }),
        html.Div([
            html.Div([
                html.Span('Dir:', style={'fontSize': '11px', 'color': _TEXT_DIM, 'marginRight': '4px'}),
                dcc.Input(
                    id='yaml-export-dir', type='text', placeholder='Leave blank for Downloads folder',
                    style={'fontSize': '11px', 'padding': '2px 6px', 'width': '300px',
                           'border': f'1px solid {_BORDER}', 'borderRadius': '4px', 'fontFamily': 'monospace',
                           'background': _DARK2, 'color': _TEXT},
                ),
                html.Button('Export YAML', id='export-yaml-btn', n_clicks=0, disabled=True, style={
                    'marginLeft': '8px', 'padding': '2px 12px', 'fontSize': '12px',
                    'background': _DARK3, 'color': _TEXT, 'border': f'1px solid {_BORDER}',
                    'borderRadius': '4px', 'cursor': 'pointer',
                }),
            ], style={'marginTop': '6px', 'display': 'flex', 'alignItems': 'center'}),
            html.Div(id='export-status', style={'fontSize': '11px', 'color': _TEXT_DIM, 'marginTop': '4px'}),
        ], hidden=not has_params_json),
        dcc.Download(id='yaml-download'),
        dcc.Store(id='selected-row-idx', data=None),
    ], style={'background': _DARK, 'color': _TEXT, 'padding': '16px', 'minHeight': '100vh'})

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

    # Generic collapsible panel toggle (DRY for both controls and filter panels)
    def _register_panel_toggle(panel_id, label):
        @callback(
            Output(f'{panel_id}-body', 'style'),
            Output(f'{panel_id}-toggle', 'children'),
            Input(f'{panel_id}-toggle', 'n_clicks'),
        )
        def toggle(n_clicks):
            if n_clicks % 2 == 1:
                return {'display': 'none'}, f'\u25b6 {label}'
            return {'display': 'block'}, f'\u25bc {label}'

    _register_panel_toggle('controls', 'Controls')
    _register_panel_toggle('custom', 'Custom Filter & Fields')
    _register_panel_toggle('views', 'Views')
    _register_panel_toggle('settings', '\u2699 Settings')

    # ---- View save/load system -----------------------------------------------
    import json

    # Single definition of which controls make up a "view"
    # (id, property) pairs — order matters for the load callback outputs
    _view_fields = [
        ('x-axis', 'value'),
        ('y-axis', 'value'),
        ('x-jitter-toggle', 'value'),
        ('x-jitter-min', 'value'),
        ('x-jitter-max', 'value'),
        ('x-bin-toggle', 'value'),
        ('x-bin-width', 'value'),
        ('y-jitter-toggle', 'value'),
        ('y-jitter-min', 'value'),
        ('y-jitter-max', 'value'),
        ('color-by-factor', 'value'),
        ('color-by-values', 'value'),
        ('color-scale-picker', 'value'),
        ('agg-group-by', 'value'),
        ('agg-y-mode', 'value'),
        ('filter-expr', 'value'),
        ('fields-expr', 'value'),
        ('yaml-export-dir', 'value'),
        ('setting-display-points', 'value'),
        ('setting-discrete-threshold', 'value'),
        ('setting-marker-size', 'value'),
        ('setting-marker-opacity', 'value'),
        ('color-as-continuous', 'value'),
    ]

    _safe_roots = [Path(r) for r in safe_roots]

    # Save view
    @callback(
        Output('view-status', 'children', allow_duplicate=True),
        Output('view-picker', 'options', allow_duplicate=True),
        Input('view-save-btn', 'n_clicks'),
        State('view-name', 'value'),
        State('views-dir', 'value'),
        *[State(fid, prop) for fid, prop in _view_fields],
        *[State({'type': 'dropdown', 'factor': f}, 'value') for f in discrete_factors],
        prevent_initial_call=True,
    )
    def save_view(n_clicks, view_name, views_dir_str, *args):
        if not view_name or not view_name.strip():
            return '\u26a0 Enter a view name first.', no_update
        view_name = view_name.strip()
        vdir = _views_dir(views_dir_str)

        n_fields = len(_view_fields)
        field_vals = args[:n_fields]
        factor_vals = args[n_fields:]

        view_data = {fid: val for (fid, _), val in zip(_view_fields, field_vals)}
        view_data['_factor_filters'] = {f: v for f, v in zip(discrete_factors, factor_vals)}

        if not _is_safe_path(vdir, _safe_roots):
            return f'\u26a0 Directory must be under {[str(r) for r in _safe_roots]}', no_update

        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / f'{view_name}.json').write_text(json.dumps(view_data, indent=2, default=str))

        return f'\u2713 Saved "{view_name}"', _view_dropdown_options(vdir)

    # Load view
    @callback(
        Output('view-loading', 'data', allow_duplicate=True),
        *[Output(fid, prop, allow_duplicate=True) for fid, prop in _view_fields],
        *[Output({'type': 'dropdown', 'factor': f}, 'value', allow_duplicate=True) for f in discrete_factors],
        Output('view-status', 'children', allow_duplicate=True),
        Output('active-view-name', 'data', allow_duplicate=True),
        Input('view-picker', 'value'),
        State('views-dir', 'value'),
        prevent_initial_call=True,
    )
    def load_view(view_name, views_dir_str):
        if not view_name:
            raise PreventUpdate

        path = _views_dir(views_dir_str) / f'{view_name}.json'
        if not path.exists():
            return no_update, *([no_update] * (len(_view_fields) + len(discrete_factors))), f'\u26a0 View "{view_name}" not found.', no_update

        view_data = json.loads(path.read_text())

        outputs = [True]  # view-loading flag
        for fid, _ in _view_fields:
            outputs.append(view_data.get(fid, no_update))

        factor_filters = view_data.get('_factor_filters', {})
        for f in discrete_factors:
            outputs.append(factor_filters.get(f, no_update))

        outputs.append(f'\u2713 Loaded "{view_name}"')
        outputs.append(view_name)
        return tuple(outputs)

    # Delete view
    @callback(
        Output('view-status', 'children', allow_duplicate=True),
        Output('view-picker', 'options', allow_duplicate=True),
        Output('view-picker', 'value', allow_duplicate=True),
        Input('view-delete-btn', 'n_clicks'),
        State('view-picker', 'value'),
        State('views-dir', 'value'),
        prevent_initial_call=True,
    )
    def delete_view(n_clicks, view_name, views_dir_str):
        if not view_name:
            return '\u26a0 Select a view to delete.', no_update, no_update
        vdir = _views_dir(views_dir_str)
        path = vdir / f'{view_name}.json'
        if path.exists():
            path.unlink()
        return f'\u2713 Deleted "{view_name}"', _view_dropdown_options(vdir), None

    # Refresh view list when dir changes or on init
    @callback(
        Output('view-picker', 'options'),
        Input('views-dir', 'value'),
    )
    def refresh_view_options(views_dir_str):
        return _view_dropdown_options(_views_dir(views_dir_str))

    # Load view from URL query string — fires on page load if ?view_name is present
    @callback(
        Output('view-loading', 'data', allow_duplicate=True),
        *[Output(fid, prop, allow_duplicate=True) for fid, prop in _view_fields],
        *[Output({'type': 'dropdown', 'factor': f}, 'value', allow_duplicate=True) for f in discrete_factors],
        Output('view-status', 'children', allow_duplicate=True),
        Output('active-view-name', 'data', allow_duplicate=True),
        Input('url', 'search'),
        State('views-dir', 'value'),
        prevent_initial_call='initial_duplicate',
    )
    def load_view_from_url(search, views_dir_str):
        view_name = (search or '').lstrip('?').strip()
        if not view_name:
            raise PreventUpdate
        path = _views_dir(views_dir_str) / f'{view_name}.json'
        if not path.exists():
            return no_update, *([no_update] * (len(_view_fields) + len(discrete_factors))), f'\u26a0 View "{view_name}" not found.', no_update
        view_data = json.loads(path.read_text())
        outputs = [True]
        for fid, _ in _view_fields:
            outputs.append(view_data.get(fid, no_update))
        factor_filters = view_data.get('_factor_filters', {})
        for f in discrete_factors:
            outputs.append(factor_filters.get(f, no_update))
        outputs.append(f'\u2713 Loaded "{view_name}" from URL')
        outputs.append(view_name)
        return tuple(outputs)

    # Update URL query string when a view is loaded (without page reload)
    app.clientside_callback(
        """
        function(view_name) {
            if (!view_name) return window.dash_clientside.no_update;
            var new_url = window.location.pathname + '?' + view_name;
            window.history.replaceState(null, '', new_url);
            return window.dash_clientside.no_update;
        }
        """,
        Output('url', 'search'),
        Input('active-view-name', 'data'),
        prevent_initial_call=True,
    )

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
            'cursor': 'pointer', 'border': f'2px solid {_BORDER}',
            'borderRadius': '4px', 'color': _TEXT,
            'background': '#1a3a5c' if new_state else _DARK3,
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
        Output('color-continuous-toggle-container', 'style'),
        Input('color-by-factor', 'value'),
        Input('color-as-continuous', 'value'),
    )
    def update_color_controls(color_factor, as_continuous):
        show_vals = {'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}
        hide_vals = {'width': '18%', 'display': 'none', 'marginRight': '2%'}
        show_scale = {'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}
        hide_scale = {'width': '18%', 'display': 'none', 'marginRight': '2%'}
        show_toggle = {'display': 'inline-block', 'width': '9%', 'marginRight': '1%', 'verticalAlign': 'bottom'}
        hide_toggle = {'display': 'none', 'width': '9%', 'marginRight': '1%', 'verticalAlign': 'bottom'}

        continuous_override = as_continuous and 'on' in as_continuous

        factors = [f for f in (color_factor or []) if f and f != SELECTION_VALUE]
        if len(factors) == 1:
            f = factors[0]
            if f in continuous_factors:
                # Always continuous — no toggle needed
                return [], [], hide_vals, show_scale, hide_toggle
            # Discrete factor
            is_numeric = f in numeric_discrete_factors
            opts = [{'label': v, 'value': v} for v in factor_values[f]]
            if is_numeric and continuous_override:
                # User switched to continuous rendering for this numeric factor
                return opts, factor_values[f], hide_vals, show_scale, show_toggle
            return opts, factor_values[f], show_vals, hide_scale, (show_toggle if is_numeric else hide_toggle)
        return [], [], hide_vals, hide_scale, hide_toggle

    @callback(
        Output('x-jitter-toggle', 'value'),
        Output('x-jitter-min', 'value'),
        Output('x-jitter-max', 'value'),
        Output('view-loading', 'data', allow_duplicate=True),
        Input('x-axis', 'value'),
        State('view-loading', 'data'),
        prevent_initial_call=True,
    )
    def auto_jitter_x(x_col, view_loading):
        if view_loading:
            return no_update, no_update, no_update, False
        if x_col in jitter_defaults:
            return ['on'], jitter_defaults[x_col][0], jitter_defaults[x_col][1], False
        return [], -0.5, 0.5, False

    @callback(
        Output('y-jitter-toggle', 'value'),
        Output('y-jitter-min', 'value'),
        Output('y-jitter-max', 'value'),
        Output('view-loading', 'data', allow_duplicate=True),
        Input('y-axis', 'value'),
        State('view-loading', 'data'),
        prevent_initial_call=True,
    )
    def auto_jitter_y(y_col, view_loading):
        if view_loading:
            return no_update, no_update, no_update, False
        if y_col in jitter_defaults:
            return ['on'], jitter_defaults[y_col][0], jitter_defaults[y_col][1], False
        return [], -0.5, 0.5, False

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
        Output('selected-row-idx', 'data'),
        Output('export-yaml-btn', 'disabled'),
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
        customdata = pt.get('customdata', [])
        lines = []

        for name, val in zip(factors_subset, customdata):
            if val is not None and val == val:  # skip None and NaN
                lines.append(f"{name}: {val}")

        # Add x/y values if not already shown via factors
        y_label = f"{agg_y_mode}({y_col})" if agg_y_mode and agg_y_mode != 'raw' and agg_group_by else y_col
        if x_col not in factors_subset:
            lines.append(f"{x_col}: {pt.get('x')}")
        if y_col not in factors_subset:
            lines.append(f"{y_label}: {pt.get('y')}")

        # Extract row index for YAML export (last element of customdata when has_params_json)
        row_idx = None
        btn_disabled = True
        if has_params_json and len(customdata) > len(factors_subset):
            raw_idx = customdata[len(factors_subset)]
            if raw_idx is not None and int(raw_idx) != -1:
                row_idx = int(raw_idx)
                btn_disabled = False

        return '\n'.join(lines), row_idx, btn_disabled

    # ---- YAML export ---------------------------------------------------------
    @callback(
        Output('yaml-download', 'data'),
        Output('export-status', 'children'),
        Input('export-yaml-btn', 'n_clicks'),
        State('selected-row-idx', 'data'),
        State('yaml-export-dir', 'value'),
        prevent_initial_call=True,
    )
    def export_yaml(n_clicks, row_idx, export_dir_str):
        if not n_clicks or row_idx is None:
            raise PreventUpdate
        params_data = orjson.loads(df.at[row_idx, 'params_json'])
        if 'logging' in params_data and params_data['logging'] and 'out_path' in params_data['logging']:
            params_data['logging']['out_path'] = str(params_data['logging']['out_path']).replace('grid_search', 'single_run')
        yaml_str = yaml.dump(params_data, default_flow_style=False, sort_keys=False)
        if export_dir_str and export_dir_str.strip():
            p = Path(export_dir_str.strip())
            out_path = p if p.suffix else p / 'params.yaml'
            if not _is_safe_path(out_path.parent):
                return no_update, f'\u26a0 Directory must be under {[str(r) for r in _safe_roots]}'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml_str)
            return no_update, f'Saved to {out_path}'
        return dcc.send_string(yaml_str, filename='params.yaml'), ''

    # ── Stage 1: Data pipeline callback ─────────────────────────────────────────
    # Triggered by anything that changes the underlying data subset/shape.
    # Outputs serialized processed DataFrame to dcc.Store for the figure callback.
    @callback(
        Output('processed-data', 'data'),
        Output('filter-error', 'children'),
        Output('filter-count', 'children'),
        Output('fields-error', 'children'),
        Output('fields-info', 'children'),
        Output('x-axis', 'options'),
        Output('y-axis', 'options'),
        Input('refresh-btn', 'n_clicks'),
        Input({'type': 'dropdown', 'factor': ALL}, 'value'),
        Input('agg-group-by', 'value'),
        Input('agg-y-mode', 'value'),
        Input('x-bin-toggle', 'value'),
        Input('x-bin-width', 'value'),
        Input('setting-display-points', 'value'),
        Input('x-axis', 'value'),
        Input('y-axis', 'value'),
        Input('color-by-factor', 'value'),
        State('lazy-mode-active', 'data'),
        State('filter-expr', 'value'),
        State('fields-expr', 'value'),
        State('scatter-plot', 'relayoutData'),
    )
    def process_data(refresh_clicks, filter_values, agg_group_by, agg_y_mode,
                     x_bin_on, x_bin_width, display_pts,
                     x_col, y_col, color_factor,
                     lazy_mode, filter_expr, fields_expr, relayout_data):
        if lazy_mode and ctx.triggered_id is not None and ctx.triggered_id != 'refresh-btn':
            raise PreventUpdate

        filter_error = ''
        fields_error = ''
        fields_info = ''
        current_axis_opts = [{'label': c, 'value': c} for c in axis_options]
        _pipeline = []

        def _empty_store(title, x=None, y=None):
            payload = {'empty': True, 'title': title, 'x': x, 'y': y, 'pipeline': ''}
            return orjson.dumps(payload).decode(), filter_error, '', fields_error, fields_info, current_axis_opts, current_axis_opts

        if not x_col or not y_col:
            return _empty_store('Select X and Y axes to display data')

        # 1. Discrete factor filters
        mask = np.ones(len(df), dtype=bool)
        for factor, values in zip(discrete_factors, filter_values):
            if values:
                mask &= df[factor].astype(str).isin(values)
        filtered_df = df[mask].copy()
        _pipeline.append((len(df), ''))
        if len(filtered_df) != len(df):
            _pipeline.append((len(filtered_df), 'filter'))

        # 2. Computed fields
        if fields_expr and fields_expr.strip():
            filtered_df, fields_info, fields_error = parse_and_apply_fields(filtered_df, fields_expr)
            new_cols = [c for c in filtered_df.select_dtypes(include='number').columns
                        if c not in axis_options]
            if new_cols:
                current_axis_opts = (
                    [{'label': c, 'value': c} for c in axis_options]
                    + [{'label': '\u2500\u2500 computed \u2500\u2500', 'value': '_h_computed', 'disabled': True}]
                    + [{'label': f'  {c}', 'value': c} for c in new_cols]
                )

        # 3. Validate axes
        missing = [c for c in (x_col, y_col) if c not in filtered_df.columns]
        if missing:
            if not fields_error:
                fields_error = f'\u26a0 Column(s) not in data: {", ".join(missing)}. Define in Computed Fields.'
            return _empty_store(f"Column(s) not found: {', '.join(missing)}", x_col, y_col)

        # 4. Custom query
        if filter_expr and filter_expr.strip():
            try:
                filtered_df = filtered_df.query(filter_expr)
                _pipeline.append((len(filtered_df), 'query'))
            except Exception as e:
                filter_error = f'\u26a0 {e}'

        # 4.5. X Binning
        x_plot_col = x_col
        x_bin_label = None
        if x_bin_on and 'on' in x_bin_on and x_bin_width and float(x_bin_width) > 0:
            bw = float(x_bin_width)
            _bin_col = x_col + '_binned'
            filtered_df[_bin_col] = (filtered_df[x_col] / bw).round() * bw
            x_plot_col = _bin_col
            x_bin_label = f'{x_col} (bin={bw})'

        # 5. Aggregation
        x_is_binned = x_plot_col != x_col
        y_agg_label = agg_y_mode if (agg_y_mode and agg_y_mode != 'raw' and (agg_group_by or x_is_binned)) else None
        if y_agg_label:
            effective_group = list(agg_group_by)
            extra_agg_cols = []
            str_collapsed = []  # non-numeric color vars not in group_by → collapse to "Other"
            for cf in (color_factor or []):
                if not cf or cf in (SELECTION_VALUE, None) or cf == y_col:
                    continue
                if cf in effective_group or cf not in filtered_df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(df[cf]):
                    extra_agg_cols.append(cf)
                else:
                    str_collapsed.append(cf)
            filtered_df = aggregate_df(filtered_df, x_plot_col, y_col, effective_group, agg_y_mode,
                                       x_is_binned=x_is_binned, extra_agg_cols=extra_agg_cols)
            for cf in str_collapsed:
                if cf in filtered_df.columns:
                    filtered_df[cf] = 'Other'
            _pipeline.append((len(filtered_df), 'agg'))

        # 5.5. Viewport-aware sampling
        total_points = len(filtered_df)
        if relayout_data and x_plot_col and y_col:
            x0 = relayout_data.get('xaxis.range[0]')
            x1 = relayout_data.get('xaxis.range[1]')
            y0 = relayout_data.get('yaxis.range[0]')
            y1 = relayout_data.get('yaxis.range[1]')
            if x0 is not None or y0 is not None:
                vp_mask = np.ones(len(filtered_df), dtype=bool)
                if x0 is not None and x1 is not None and x_plot_col in filtered_df.columns:
                    vp_mask &= (filtered_df[x_plot_col] >= x0) & (filtered_df[x_plot_col] <= x1)
                if y0 is not None and y1 is not None and y_col in filtered_df.columns:
                    vp_mask &= (filtered_df[y_col] >= y0) & (filtered_df[y_col] <= y1)
                viewport_df = filtered_df[vp_mask]
                if len(viewport_df) > 0:
                    filtered_df = viewport_df
                    _pipeline.append((len(filtered_df), 'viewport'))
        _display_limit = display_pts if display_pts is not None else initial_display_points
        if len(filtered_df) == total_points and _display_limit and total_points > _display_limit:
            filtered_df = filtered_df.sample(_display_limit, random_state=None)
            _pipeline.append((len(filtered_df), 'sample'))

        # 6. Row index for YAML export
        if has_params_json:
            filtered_df['_row_idx'] = -1 if y_agg_label else filtered_df.index

        # Build pipeline string
        pipeline_parts = [_fmt(_pipeline[0][0])] if _pipeline else []
        for n, label in (_pipeline[1:] if _pipeline else []):
            pipeline_parts.append(f'{_fmt(n)} ({label})')
        pipeline_str = ' → '.join(pipeline_parts)

        payload = {
            'records': filtered_df.to_dict('records'),
            'x_col': x_col,
            'x_plot_col': x_plot_col,
            'x_label': x_bin_label,
            'y_col': y_col,
            'y_agg_label': y_agg_label,
            'x_is_binned': x_is_binned,
        }
        return (orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY).decode(),
                filter_error, pipeline_str, fields_error, fields_info,
                current_axis_opts, current_axis_opts)

    # ── Stage 2: Figure render callback ──────────────────────────────────────────
    # Triggered by store changes, color settings, and jitter settings.
    # No df filtering/agg — all heavy data work is done in process_data.
    @callback(
        Output('scatter-plot', 'figure'),
        Input('processed-data', 'data'),
        Input('color-by-factor', 'value'),
        Input('color-by-values', 'value'),
        Input('color-scale-picker', 'value'),
        Input('color-offsets', 'data'),
        Input('color-as-continuous', 'value'),
        Input('x-jitter-toggle', 'value'),
        Input('x-jitter-min', 'value'),
        Input('x-jitter-max', 'value'),
        Input('y-jitter-toggle', 'value'),
        Input('y-jitter-min', 'value'),
        Input('y-jitter-max', 'value'),
        State({'type': 'dropdown', 'factor': ALL}, 'value'),
        State('setting-marker-size', 'value'),
        State('setting-marker-opacity', 'value'),
    )
    def render_figure(store_json, color_factor, color_values, color_scale, color_offsets,
                      as_continuous,
                      x_jit_on, x_jit_min, x_jit_max,
                      y_jit_on, y_jit_min, y_jit_max,
                      filter_values, marker_size, marker_opacity):
        if not store_json:
            raise PreventUpdate

        payload = orjson.loads(store_json)

        def _empty_fig(title, x=None, y=None):
            fig = go.Figure()
            fig.update_layout(title=title, xaxis_title=x or 'X', yaxis_title=y or 'Y')
            return fig

        if payload.get('empty'):
            return _empty_fig(payload['title'], payload.get('x'), payload.get('y'))

        filtered_df = pd.DataFrame(payload['records'])
        x_plot_col = payload['x_plot_col']
        x_col = payload['x_col']
        y_col = payload['y_col']
        x_bin_label = payload.get('x_label')
        y_agg_label = payload.get('y_agg_label')
        x_is_binned = payload.get('x_is_binned', False)

        if filtered_df.empty:
            return _empty_fig('No data for selected filters', x_col, y_col)

        # Color config
        continuous_override = as_continuous and 'on' in as_continuous
        color_cfg = None
        _non_special = [f for f in (color_factor or []) if f and f not in (SELECTION_VALUE, None, '_header_discrete', '_header_continuous')]
        if SELECTION_VALUE in (color_factor or []):
            color_cfg = {'type': 'selection', 'filter_selections': filter_values}
        elif len(_non_special) == 1:
            f = _non_special[0]
            is_continuous = f in continuous_factors or (f in numeric_discrete_factors and continuous_override)
            if is_continuous:
                color_cfg = {'type': 'continuous', 'column': f, 'scale': color_scale}
            elif color_values:
                color_cfg = {'type': 'discrete', 'column': f, 'values': color_values}
        elif _non_special:
            color_cfg = {'type': 'columns', 'columns': _non_special}

        x_jitter = None
        if x_jit_on and 'on' in x_jit_on and x_jit_min is not None and x_jit_max is not None:
            x_jitter = (float(x_jit_min), float(x_jit_max))
        y_jitter = None
        if y_jit_on and 'on' in y_jit_on and y_jit_min is not None and y_jit_max is not None:
            y_jitter = (float(y_jit_min), float(y_jit_max))

        fig = create_figure(filtered_df, x_plot_col, y_col, color_cfg, color_offsets,
                            x_jitter=x_jitter, y_jitter=y_jitter,
                            y_agg_label=y_agg_label,
                            x_label=x_bin_label,
                            marker_size=marker_size if marker_size is not None else 5,
                            marker_opacity=marker_opacity if marker_opacity is not None else 0.7)
        return fig

    # ── Stage 3: Clientside marker size/opacity — zero server roundtrip ───────
    app.clientside_callback(
        """
        function(size, opacity, figure) {
            if (!figure || !figure.data || !figure.data.length) return window.dash_clientside.no_update;
            var traces = figure.data.map(function(t) {
                return Object.assign({}, t, {marker: Object.assign({}, t.marker || {}, {size: size, opacity: opacity})});
            });
            return Object.assign({}, figure, {data: traces});
        }
        """,
        Output('scatter-plot', 'figure', allow_duplicate=True),
        Input('setting-marker-size', 'value'),
        Input('setting-marker-opacity', 'value'),
        State('scatter-plot', 'figure'),
        prevent_initial_call=True,
    )

    return app