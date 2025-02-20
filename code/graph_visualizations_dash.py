import networkx as nx
import plotly.graph_objs as go
from pathlib import Path
from torch.nn import Linear
import numpy as np
from graphs import remove_isolated_nodes
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

def plot_graph_with_weight_coloring_3D(graph: nx.Graph, readout: Linear, layout=lambda g: nx.spring_layout(g, dim=3), metadata=None, draw_edges=True):
    def id_generator():
        seen_elements = dict()
        current_id = 0
        while True:
            element = yield
            if element not in seen_elements:
                seen_elements[element] = current_id
                current_id += 1
            yield seen_elements[element]
    
    def convert_to_numeric(id_gen, value):
        if isinstance(value, str) and len(value) == 1:
            return ord(value)
        elif isinstance(value, (int, float, np.number, np.ndarray)):
            return value
        else:
            next(id_gen)
            return id_gen.send(value)

    def normalize_df(df):
        df2 = pd.DataFrame()
        for col in df:
            id_gen = id_generator()
            numeric_series = df[col].apply(lambda x: convert_to_numeric(id_gen, x))
            min_val = numeric_series.min(axis=0)
            max_val = numeric_series.max(axis=0)
            df2[col + "_normalized_"] = (numeric_series - min_val) / (max_val - min_val)
        return df2
    
    def split_coordinates(d):
        attrs = {}
        for key, arr in d.items():
            shp = arr.shape
            arr = np.resize(arr, 3)
            arr[shp[0]:] = 0
            attrs[key] = {
                "x": arr[0],
                "y": arr[1],
                "z": arr[2]
            }
        return attrs

    # Add node metadata
    weights = readout.weight.data.numpy()
    weight_set_1 = weights[0, :]
    weight_set_2 = weights[1, :]
    nx.set_node_attributes(graph, dict(enumerate(range(len(weight_set_1)))), 'id')
    nx.set_node_attributes(graph, dict(enumerate(list(np.around(weight_set_1, decimals=3)))), 'weight_0')
    if weights.shape[0] > 1:
        nx.set_node_attributes(graph, dict(enumerate(list(np.around(weight_set_2, decimals=3)))), 'weight_1')
    if metadata:
        for label, dict_data in metadata:
            nx.set_node_attributes(graph, dict_data, label)
    pos_d = layout(graph)
    nx.set_node_attributes(graph, split_coordinates(pos_d))

    # collect all data in df
    df = pd.DataFrame.from_dict(dict(graph.nodes(data=True))).T.set_index('id', drop=False)
    df = df.infer_objects()
    df_normalized = normalize_df(df)
    available_attributes = sorted(df.columns)

    def get_colors(attr, index):
        if 'color' in df:
            return df_normalized.loc[index][attr + '_normalized_'].where(df['color'].isna(), df['color'])
        else:
            return df_normalized.loc[index][attr + '_normalized_']
    
    def create_figure(query="id >= 0", color_attr='', relayout_data=None):
        try:
            filtered_df = df.query(query)
        except Exception:
            filtered_df = df.query("id >= 0")
            
        if not color_attr:
            color_attr = available_attributes[0]
        
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in graph.edges():
            if edge[0] in filtered_df.index and edge[1] in filtered_df.index:
                x0, y0, z0 = df.loc[edge[0]][['x', 'y', 'z']]
                x1, y1, z1 = df.loc[edge[1]][['x', 'y', 'z']]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=0.5, color='black') if draw_edges else dict(width=0, color='rgba(0,0,0,0)'),
            hoverinfo='none',
            mode='lines')
        
        node_trace = go.Scatter3d(
            x=filtered_df['x'], y=filtered_df['y'], z=filtered_df['z'],
            mode='markers',
            marker=dict(
                size=5,
                colorscale='Viridis',
                colorbar=dict(
                    title='Node Attribute'
                ),
                color=get_colors(color_attr, filtered_df.index),
                line=dict(width=2)
            ),
            hoverinfo='text'
        )
        node_trace.text = [f"Node {i}: " + ', '.join([f"{col}: {val}" for col, val in row.items()]) for i, row in filtered_df.astype(str).iterrows()]
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False)
        fig.update_layout(scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ))
        
        if relayout_data:
            fig.update_layout(scene_camera=relayout_data.get('scene.camera', {}))
        
        return fig

    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Store(id='relayout_data_store'),
        html.Div([
            html.Label('Enter Pandas Query to Filter Nodes:'),
            dcc.Input(
                id='query_input',
                type='text',
                placeholder='Enter Pandas query to filter nodes...',
                value='id >= 0'
            ),
            html.Br(),
            html.Label('Select Node Attribute for Coloring:'),
            dcc.Dropdown(
                id='color_selector',
                options=[{'label': attr, 'value': attr} for attr in available_attributes],
                value=available_attributes[0],
                clearable=False
            ),
            html.Button('Apply Changes', id='apply_filter_button', n_clicks=0)
        ], style={'position': 'fixed', 'top': '10px', 'left': '10px', 'width': '300px', 'background-color': 'rgba(255,255,255,0.8)', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)', 'z-index': '1000'}),
        dcc.Graph(id='graph', style={'height': '100vh', 'width': '100vw', 'position': 'absolute', 'top': '0', 'left': '0', 'z-index': '1'}, figure=create_figure())
    ], style={'position': 'relative', 'height': '100vh', 'width': '100vw', 'overflow': 'hidden'})

    @app.callback(
        Output('relayout_data_store', 'data'),
        [Input('graph', 'relayoutData')]
    )
    def store_relayout_data(relayoutData):
        return relayoutData or {}

    @app.callback(
        Output('graph', 'figure'),
        [Input('apply_filter_button', 'n_clicks')],
        [State('query_input', 'value'),
         State('color_selector', 'value'),
         State('relayout_data_store', 'data')]
    )
    def update_graph(n_clicks, query, color_attr, relayout_data):
        return create_figure(query=query, color_attr=color_attr, relayout_data=relayout_data)

    app.run_server(debug=True, use_reloader=False)

if __name__ == '__main__':
    from reservoir import BooleanReservoir
    model = BooleanReservoir(load_path='/out/single_run/2D/good_model/2025_02_17_113551_406400/checkpoints/2025_02_17_114011_225303')
    metadata = list()
    metadata.append(['in_degree', dict(model.graph.in_degree)])
    d = {node: 'normal' for node in model.graph.nodes()}
    for idx, nodes in enumerate(model.input_nodes):
        d.update({v: f'input_{idx}' for v in nodes.tolist()})
        # metadata.append(['color', {k: 'red' for k in input_nodes.tolist()}])  # override node colors!
    metadata.append(['type', d])
    graph = remove_isolated_nodes(model.graph)
    plot_graph_with_weight_coloring_3D(graph, model.readout, layout=lambda g: nx.random_layout(g, dim=3), metadata=metadata, draw_edges=True)
    # plot_graph_with_weight_coloring_3D(graph, model.readout, layout=lambda g: nx.spring_layout(g, dim=3), metadata=metadata, draw_edges=True)
    # plot_graph_with_weight_coloring_3D(graph, model.readout, layout=lambda g: nx.kamada_kawai_layout(g, dim=3), metadata=metadata, draw_edges=True)
    # example query: abs(weight_0) > 0.1 or abs(weight_1) > 0.1 or type.str.startswith('input')