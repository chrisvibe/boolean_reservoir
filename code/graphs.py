import networkx as nx
import numpy as np
import random

def generate_graph_w_k_avg_incoming_edges(n_nodes, k_avg, k_max=None, self_loops=None):
    adj_matrix = generate_adjacency_matrix(n_nodes, k_avg, k_max=k_max, self_loops=self_loops)
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return graph

def graph2adjacency_list_outgoing(graph: nx.Graph):
    # convention is typically outgoing edges
    adj_list = [[] for _ in range(graph.number_of_nodes())]
    for node, neighbors in graph.adjacency():
        adj_list[node] = list(neighbors)
    return adj_list

def graph2adjacency_list_incoming(graph: nx.DiGraph):
    # convention is typically outgoing edges
    adj_list = [[] for _ in range(graph.number_of_nodes())]
    for node in graph.nodes():
        adj_list[node] = list(graph.predecessors(node))
    return adj_list

def generate_adjacency_matrix(n_nodes, k_avg, k_max=None, self_loops=None, adj_matrix=None):
    # n_nodes is the number of nodes in the graphk
    # k_avg is the average incoming edges
    # k_max is the max incoming edges
    # self_loops is the number of edges where i==j, this is normalized by n_nodes [0, 1]
    # convention is i → j, so sum of rows is the number of incoming edges per node
    # note that avk_k for incoming or outcoming edges is the same here...
    k_max = n_nodes if k_max is None else k_max
    self_loops = random.uniform(0, 1) if self_loops is None else self_loops
    assert 0 <= k_avg <= n_nodes
    assert 0 <= k_avg <= k_max
    assert 0 <= self_loops <= 1
    idx2col = lambda idx: idx % n_nodes
    diagonal = lambda x: x[::n_nodes+1]
    indices = np.arange(n_nodes ** 2)
    total_edges = round(n_nodes * k_avg)
    self_loops = round(self_loops * n_nodes)
    if adj_matrix is None:
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
    flat_adj_matrix = adj_matrix.ravel()
    k_per_node = adj_matrix.sum(axis=0) 
    n_edges = k_per_node.sum()
    n_loops = diagonal(flat_adj_matrix).sum()
    visited_mask = flat_adj_matrix.copy()

    # set self-loops
    # implemented as per "CONST-RND" tactic in "The Impact of Self-Loops on Boolean Networks  Attractor Landscape and Implications for  Cell Differentiation Modelling"
    add_edges = n_loops <= self_loops
    adjustment = 1 if add_edges else -1
    visited_mask = visited_mask if add_edges else ~visited_mask
    while (n_loops - self_loops) != 0:
        pool = ~diagonal(visited_mask)
        idx = np.random.choice(diagonal(indices)[pool])
        col_idx = idx2col(idx)
        flat_adj_matrix[idx] ^= True 
        k_per_node[col_idx] += adjustment 
        n_edges += adjustment 
        n_loops += adjustment 
        visited_mask[idx] = True 
    visited_mask ^= not add_edges # undo mask inversion
    # print('1')
    # print(flat_adj_matrix.reshape(n_nodes, -1).astype(int), flat_adj_matrix.sum(), diagonal(flat_adj_matrix).sum())

    # enforce k_max
    visited_mask[diagonal(indices)] = False 
    while (k_per_node > k_max).any():
        pool = visited_mask
        idx = np.random.choice(indices[pool])
        col_idx = idx2col(idx)
        if k_per_node[col_idx] <= k_max:
            continue
        flat_adj_matrix[idx] ^= True 
        k_per_node[col_idx] += -1
        n_edges += -1
        visited_mask[idx] = True 
    # print('2')
    # print(flat_adj_matrix.reshape(n_nodes, -1).astype(int), flat_adj_matrix.sum(), diagonal(flat_adj_matrix).sum())

    # set rest of edges
    add_edges = n_edges <= total_edges
    adjustment = 1 if add_edges else -1
    visited_mask = visited_mask if add_edges else ~visited_mask
    visited_mask[diagonal(indices)] = True 
    while (n_edges - total_edges) != 0:
        pool = ~visited_mask
        idx = np.random.choice(indices[pool])
        col_idx = idx2col(idx)
        if k_per_node[col_idx] == k_max:
            continue
        flat_adj_matrix[idx] ^= True 
        k_per_node[col_idx] += adjustment
        n_edges += adjustment
        visited_mask[idx] = True 
    # print('3')
    # print(flat_adj_matrix.reshape(n_nodes, -1).astype(int), flat_adj_matrix.sum(), diagonal(flat_adj_matrix).sum())

    adj_matrix = flat_adj_matrix.reshape(n_nodes, n_nodes)
    assert adj_matrix.sum() == total_edges
    assert diagonal(adj_matrix.ravel()).sum() == self_loops
    assert (adj_matrix.sum(axis=0) <= k_max).all()
    return adj_matrix


if __name__ == '__main__':
    n_nodes = 10
    k_avg = 3
    k_max = 5
    self_loops = 0.3
    # g = graph_average_k_incoming_edges_w_self_loops(reservoir_size, k_avg, k_max=k_max, self_loops=self_loops)
    # print(graph2adjacency_list_incoming(g))
    matrix = generate_adjacency_matrix(n_nodes, k_avg, k_max=k_max, self_loops=self_loops, adj_matrix=None)
    matrix = generate_adjacency_matrix(n_nodes, k_avg, k_max=k_max-1, self_loops=self_loops, adj_matrix=matrix)
    print(matrix.astype(int), matrix.sum(), matrix.ravel()[::n_nodes+1].sum())
