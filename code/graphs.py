import networkx as nx
import numpy as np

def adjacency_matrix_average_k_incoming_edges_w_self_loops(n_nodes, avg_k):
    assert 0 <= avg_k <= n_nodes
    total_edges = round(n_nodes * avg_k)
    adj_matrix_flat = np.zeros(n_nodes * n_nodes, dtype=bool)
    adj_matrix_flat[:total_edges] = True
    np.random.shuffle(adj_matrix_flat)
    adj_matrix = adj_matrix_flat.reshape((n_nodes, n_nodes))
    return adj_matrix

def graph_average_k_incoming_edges_w_self_loops(n_nodes, avg_k):
    assert avg_k <= n_nodes
    adj_matrix = adjacency_matrix_average_k_incoming_edges_w_self_loops(n_nodes, avg_k)
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return graph

def graph2adjacency_list(graph: nx.Graph):
    adj_list = [[] for _ in range(graph.number_of_nodes())]
    for node, neighbors in graph.adjacency():
        adj_list[node] = list(neighbors) # good to do it this way in case not all nodes are in dictionary! ;)
    return adj_list


if __name__ == '__main__':
    reservoir_size = 20
    avg_k = 2
    g = graph_average_k_incoming_edges_w_self_loops(reservoir_size, avg_k)
    print(graph2adjacency_list(g))