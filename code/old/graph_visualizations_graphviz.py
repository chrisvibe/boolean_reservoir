import graphviz
import networkx as nx
from pathlib import Path
from copy import deepcopy
from reservoir import BooleanReservoir
from graphs import remove_isolated_nodes

def normalize(x, min_val, max_val):
    assert min_val < max_val
    return (x - min_val) / (max_val - min_val)

def unnormalize(x, min_val, max_val):
    assert min_val < max_val
    return (x * (max_val - min_val)) + min_val

def hsv_temp_map(v):
    return 0.5 + 0.5 * v

def color_nodes_by_property(f, graph: graphviz.Graph, property_extractor, value_highlight=1, min_val=0, max_val=1, **node_kwargs):
    d = deepcopy(node_kwargs)
    for node, data in graph.nodes(data=True):
        v = property_extractor((node, data))
        v = normalize(v, min_val, max_val)
        v = v ** value_highlight
        d['fillcolor'] = f'{hsv_temp_map(v)} 1 1'
        f.node(str(node), **d)
    return f

def add_legend(f: graphviz.Graph, n_nodes, hsv_map, min_val=0, max_val=1, node_kwargs=dict(), edge_kwargs=dict()):
    node_defaults = {'fontcolor': 'black', 'height': '0.25', 'width': '0.25', 'margin': '0,0', 'fontsize': '8', 'style': 'filled'}
    edge_defaults = {'color': 'black', 'style': 'solid'}
    
    # Merge default and custom kwargs
    node_kwargs = {**node_defaults, **node_kwargs}
    edge_kwargs = {**edge_defaults, **edge_kwargs}
    
    with f.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Legend", fontsize="10", labelloc="b", style="rounded,dashed", rankdir="LR", ordering="out", rank="same")
        
        # Add nodes for legend
        spacing = .5
        for i in range(n_nodes):
            pos_x = i * spacing
            node_kwargs['pos'] = f'{pos_x},0!'
            normalized_color_val = i / (n_nodes - 1) 
            node_kwargs['fillcolor'] = '{} 1 1'.format(hsv_map(normalized_color_val))
            node_kwargs['label'] = '{:.2f}'.format(unnormalize(normalized_color_val, min_val, max_val))
            legend.node(f'legend_{i}', **node_kwargs)
        
        # Add edges between legend nodes to form a line
        for i in range(n_nodes - 1):
            node_i = f'legend_{i}'
            node_j = f'legend_{i+1}'
            legend.edge(node_i, node_j, **edge_kwargs)
    
    return f

def plot_graph_with_weight_coloring_1D(path, graph: nx.Graph, readout, layout='circo', node_kwargs=dict(), edge_kwargs=dict()):
    node_defaults = {'label': '', 'shape': 'circle', 'fillcolor': 'lightgrey', 'fontcolor': 'black', 'style': 'filled', 'root': 'False', 'height': '.1', 'width': '.1'}
    edge_defaults = {'color': 'black', 'style': 'filled', 'arrowhead': 'ornormal'}
    node_kwargs = {**node_defaults, **node_kwargs}
    edge_kwargs = {**edge_defaults, **edge_kwargs}

    path = Path(path) / 'visualizations' 
    path.mkdir(parents=True, exist_ok=True)

    weights = readout.weight.data.numpy()
    min_val = min(weights[0, :])
    max_val = max(weights[0, :])

    for node in graph.nodes():
        graph.nodes[node]['weight'] = float(weights[0, node])

    f = graphviz.Digraph(comment='Boolean Network Graph')
    f.attr(rankdir='TB', size="8.3,11.7!", page="8.3,11.7", ratio="fill", margin="0", layout=layout)
    f.attr('node', **node_kwargs)
    
    # add nodes
    f = color_nodes_by_property(f, graph, lambda n: n[1]['weight'], min_val=min_val, max_val=max_val, **node_kwargs)
    
    # add edges
    for u, v in graph.edges():
        f.edge(str(u), str(v), **edge_kwargs)
    
    # add legend
    add_legend(f, 11, hsv_temp_map, min_val=min_val, max_val=max_val, node_kwargs=node_kwargs, edge_kwargs=edge_kwargs)

    file = f'readout_{layout}'
    f.render(path / file, format='svg', view=False)
    f.render(path / file, format='pdf', view=False)
    print(f"Graph rendered at: {path / file}")

if __name__ == '__main__':
    model = BooleanReservoir(load_path='/out/single_run/1D/good_model/2025_02_17_131747_965842/checkpoints/2025_02_17_131824_009874')
    graph = remove_isolated_nodes(model.graph)
    readout = model.readout
    plot_graph_with_weight_coloring_1D('/out/test', graph, readout, layout='sfdp')
    # plot_graph_with_weight_coloring_1D('/out/test', graph, readout, layout='neato')
    # plot_graph_with_weight_coloring_1D('/out/test', graph, readout, layout='fdp')
    # plot_graph_with_weight_coloring_1D('/out/test', graph, readout, layout='circo')
    # plot_graph_with_weight_coloring_1D('/out/test', graph, readout, layout='dot')