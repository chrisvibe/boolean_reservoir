import graphviz
import networkx as nx

def normalize(x, min_val, max_val):
    assert min_val < max_val
    return (x - min_val) / (max_val - min_val)

def unnormalize(x, min_val, max_val):
    assert min_val < max_val
    return (x * (max_val - min_val)) + min_val

def hsv_temp_map(v):
    return 0.5 + 0.5 * v

def color_nodes_by_property(f, graph, property_extractor, value_highlight=1, min_val=0, max_val=1):
    for node, data in graph.nodes(data=True):
        v = property_extractor((node, data))
        v = normalize(v, min_val, max_val)
        v = v ** value_highlight
        f.node(str(node), fillcolor=f'{hsv_temp_map(v)} 1 1', fontcolor='black')
    return f

def add_legend(f, n_nodes, hsv_map, min_val=0, max_val=1):
    # make legend as nodes and edges
    for i in range(n_nodes):
        val = unnormalize(i / (n_nodes - 1), min_val, max_val)
        node_i = '{:.2f}'.format(val)
        f.node(node_i, label=node_i, fillcolor='{} 1 1'.format(hsv_map(i / (n_nodes - 1))), fontcolor='black')
    for i in range(n_nodes - 1):
        val = unnormalize(i / (n_nodes - 1), min_val, max_val)
        node_i = '{:.2f}'.format(val)
        val = unnormalize((i+1) / (n_nodes - 1), min_val, max_val)
        node_j = '{:.2f}'.format(val)
        f.edge(node_i, node_j)
    return f

def plot_graph_with_weight_coloring_1D(model, output_path='/out/visualizations/output_graph', layout='circo'):
    weights = model.readout.weight.data.numpy()
    min_val = min(weights[0, :])
    max_val = max(weights[0, :])
    graph = model.graph
    assert isinstance(graph, nx.Graph)

    for node in graph.nodes():
        graph.nodes[node]['weight'] = float(weights[0, node])

    f = graphviz.Digraph(comment='Boolean Network Graph')
    f.attr(rankdir='LR', size='8.5', pack='False', layout=layout)
    f.attr('node', label='', shape='circle', fillcolor='lightgrey', style='filled', root='False')
    # add nodes
    f = color_nodes_by_property(f, graph, lambda n: n[1]['weight'], min_val=min_val, max_val=max_val)
    # add edges
    for u, v in graph.edges():
        f.edge(str(u), str(v))
    add_legend(f, 11, hsv_temp_map, min_val, max_val)
    f.render(output_path, format='svg', view=False)
    f.render(output_path, format='pdf', view=False)
    print(f"Graph SVG rendered at: {output_path}")

