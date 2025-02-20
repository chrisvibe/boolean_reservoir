import networkx as nx
import plotly.graph_objs as go
from pathlib import Path
from torch.nn import Linear
import numpy as np
from graphs import remove_isolated_nodes
import pandas as pd

def plot_graph_with_weight_coloring_3D(path, graph: nx.Graph, readout: Linear, layout=lambda g: nx.spring_layout(g, dim=3), metadata=None, draw_edges=True):
    path = Path(path) / 'visualizations'
    path.mkdir(parents=True, exist_ok=True)

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
    nx.set_node_attributes(graph, dict(enumerate(list(np.around(weight_set_1, decimals=3)))), 'weight_x')
    if weights.shape[0] > 1:
        nx.set_node_attributes(graph, dict(enumerate(list(np.around(weight_set_2, decimals=3)))), 'weight_y')
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

    edge_x = []
    edge_y = []
    edge_z = []
    for edge in graph.edges():
        x0, y0, z0 = df.loc[edge[0]][['x', 'y', 'z']]
        x1, y1, z1 = df.loc[edge[1]][['x', 'y', 'z']]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    line_formating = dict(width=0.5, color='black')
    edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=line_formating if draw_edges else dict(width=0, color='rgba(0,0,0,0)'),
            hoverinfo='none',
            mode='lines')

    # Helper function to create normalized color list based on selected attribute
    def get_colors(df, df_normalized, attribute):
        if 'color' in df:
            return df_normalized[attribute + '_normalized_'].where(df['color'].isna(), df['color'])
        else:
            return df_normalized[attribute + '_normalized_']

    node_trace = go.Scatter3d(
        x=df['x'], y=df['y'], z=df['z'],
        mode='markers',
        marker=dict(
            size=5,
            colorscale='Viridis',
            colorbar=dict(
                title='Node Attribute'
            ),
            color=get_colors(df, df_normalized, available_attributes[0]),
            line=dict(width=2)
        ),
        hoverinfo='text'
    )

    node_trace.text = [f"Node {i}: " + ', '.join([f"{col}: {val}" for col, val in row.items()]) for i, row in df.astype(str).iterrows()]
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    fig.update_layout(scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    ))
    
    node_trace.marker.color = get_colors(df, df_normalized, available_attributes[0])
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label=attribute,
                         method="restyle",
                         args=[{"marker.color": [get_colors(df, df_normalized, attribute)]}])
                    for attribute in available_attributes
                ],
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.12,
                yanchor="top"
            )
        ]
    )
    
    file_path = path / 'readout_3d.html'
    fig.write_html(file_path)
    print(f"Graph rendered at: {file_path}")

if __name__ == '__main__':
    from reservoir import BooleanReservoir
    model = BooleanReservoir(load_path='/out/single_run/2D/good_model/2025_02_17_113551_406400/checkpoints/2025_02_17_114011_225303')
    metadata = list()
    metadata.append(['in_degree', dict(model.graph.in_degree)])
    d = {node: 'normal' for node in model.graph.nodes()}
    for idx, input_nodes in enumerate(model.input_nodes):
        d.update({v: f'input_{idx}' for v in input_nodes.tolist()})
        metadata.append(['color', {k: 'red' for k in input_nodes.tolist()}])  # override node colors!
    metadata.append(['type', d])
    graph = remove_isolated_nodes(model.graph)
    # plot_graph_with_weight_coloring_3D('/out/test', graph, model.readout, layout=lambda g: nx.random_layout(g, dim=3), metadata=metadata, draw_edges=False)
    # plot_graph_with_weight_coloring_3D('/out/test', graph, model.readout, layout=lambda g: nx.random_layout(g, dim=2), metadata=metadata, draw_edges=True)
    plot_graph_with_weight_coloring_3D('/out/test', graph, model.readout, layout=lambda g: nx.spring_layout(g, dim=3), metadata=metadata, draw_edges=True)
    # plot_graph_with_weight_coloring_3D('/out/test', graph, model.readout, layout=lambda g: nx.kamada_kawai_layout(g, dim=3), metadata=metadata, draw_edges=True)
    # plot_graph_with_weight_coloring_3D('/out/test', graph, model.readout, layout=lambda g: nx.kamada_kawai_layout(g, dim=2), metadata=metadata, draw_edges=True)