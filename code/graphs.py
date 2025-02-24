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

def generate_adjacency_matrix_old(n_nodes, k_avg, k_max=None, self_loops=None, adj_matrix=None):
    # n_nodes is the number of nodes in the graph
    # k_avg is the average incoming edges
    # k_max is the max incoming edges
    # self_loops is the number of edges where i==j, this is normalized by n_nodes [0, 1]
    # convention is i → j, so sum of rows is the number of incoming edges per node
    # note that avk_k for incoming or outcoming edges is the same here...
    k_max = n_nodes if k_max is None else k_max
    self_loops = random.uniform(0, 1) if self_loops is None else self_loops
    assert 0 <= k_avg <= n_nodes, '0 <= k_avg <= n_nodes'
    assert 0 <= k_avg <= k_max, '0 <= k_avg <= k_max'
    assert 0 <= self_loops <= 1, '0 <= self_loops <= 1'
    idx2col = lambda idx: idx % n_nodes
    diagonal = lambda x: x[::n_nodes+1]
    indices = np.arange(n_nodes ** 2)
    total_edges = round(n_nodes * k_avg)
    self_loops = round(self_loops * n_nodes)
    if adj_matrix is None:
        adj_matrix = np.zeros(n_nodes * n_nodes, dtype=bool)
        rand_indices = np.random.permutation(indices)[:total_edges]
        adj_matrix[rand_indices] = True
        adj_matrix = adj_matrix.reshape((n_nodes, n_nodes))
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

    # enforce k_max # TODO better to iterate through cols
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

    # set rest of edges # TODO better to iterate through cols
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

def generate_adjacency_matrix(n_nodes, k_avg, k_max=None, self_loops=None):
    # init adjacency matrix but may not respect requested properties
    k_max = n_nodes if k_max is None else k_max
    self_loops = random.uniform(0, 1) if self_loops is None else self_loops
    assert 0 <= k_avg <= n_nodes
    assert 0 <= k_avg <= k_max
    assert 0 <= self_loops <= 1
    diagonal = lambda x: x[::n_nodes+1]
    total_edges = round(n_nodes * k_avg)
    self_loops = round(self_loops * n_nodes)
    flat_adj_matrix = np.zeros(n_nodes * n_nodes, dtype=bool)
    indices = np.arange(n_nodes)

    # calculate k for self-loops
    k_self_loop = np.zeros(n_nodes, dtype=bool)
    pool = indices
    random_indices = np.random.permutation(indices)[:self_loops]
    k_self_loop[random_indices] = True

    # calculate k 
    k = np.random.rand(n_nodes)
    k *= total_edges / k.sum()
    k = np.floor(k).astype(int)
    k = np.minimum(k, k_max)

    # make sure k - k_self_loop > 0
    mask = k - k_self_loop < 0
    k[mask] = 1 

    # Note errors introduced above: from rounding, self-loops, and k_max is not respected
    # Correct errors so the sum to be exactly total_edges

    # adjust for overshoot
    difference = total_edges - k.sum()
    if difference < 0:
        pool = indices[~mask]
        selected_indices = np.random.choice(pool, difference, replace=False)
        k[selected_indices] -= 1
        
    # adjust for undershoot
    difference = total_edges - k.sum()
    if difference > 0:
        mask = k < k_max
        pool = indices[mask]
        selected_indices = np.random.choice(pool, difference, replace=False)
        k[selected_indices] += 1
    
    # set k
    k_eff = k - k_self_loop
    adj_matrix = flat_adj_matrix.reshape(n_nodes, n_nodes)
    for i in range(adj_matrix.shape[1]):
        adj_matrix[:k_eff[i], i] = True
        np.random.shuffle(adj_matrix[:, i])
        if adj_matrix[i, i]:
            mask = adj_matrix[:, i] == False 
            mask[i] = False
            pool = indices[mask]
            idx = np.random.choice(pool)
            adj_matrix[idx, i] = True 
        adj_matrix[i, i] = k_self_loop[i]

    assert adj_matrix.sum() == total_edges
    assert diagonal(adj_matrix.ravel()).sum() == self_loops
    assert (adj_matrix.sum(axis=0) <= k_max).all()
    return adj_matrix

def calc_spectral_radius(graph: nx.DiGraph):
    adj_matrix = nx.adjacency_matrix(graph).todense()
    eigenvalues = np.linalg.eigvals(adj_matrix)
    rho = max(abs(eigenvalues))
    return rho

def remove_isolated_nodes(graph: nx.Graph, remove_connected_to_self_only=False):
    in_degree = graph.in_degree
    non_isolated_nodes = {node for node in graph.nodes() if in_degree[node] > 0}
    if remove_connected_to_self_only:
        self_loops = {node for node in graph.nodes() if in_degree[node] == 1 and graph.has_edge(node, node)}
        non_isolated_nodes = non_isolated_nodes - self_loops
    return graph.subgraph(non_isolated_nodes).copy()

if __name__ == '__main__':
    n_nodes = 10
    k_avg = 3
    k_max = 5
    self_loops = 0.3
    # g = graph_average_k_incoming_edges_w_self_loops(reservoir_size, k_avg, k_max=k_max, self_loops=self_loops)
    # print(graph2adjacency_list_incoming(g))
    # matrix = generate_adjacency_matrix_old(n_nodes, k_avg, k_max=k_max, self_loops=self_loops, adj_matrix=None)
    # matrix = generate_adjacency_matrix_old(n_nodes, k_avg, k_max=k_max-1, self_loops=self_loops, adj_matrix=matrix)
    # print(matrix.astype(int), matrix.sum(), matrix.ravel()[::n_nodes+1].sum())

    # generate_graph_w_k_avg_incoming_edges(1000, 2)

    G = generate_graph_w_k_avg_incoming_edges(n_nodes=1000, k_avg=1, k_max=6, self_loops=0)
    adj_matrix = nx.adjacency_matrix(G).todense()
    eigenvalues = np.linalg.eigvals(adj_matrix)
    rho = max(abs(eigenvalues))
    print(rho)
    eigenvalue_magnitudes = np.abs(eigenvalues)
    eigenvalue_magnitudes = eigenvalue_magnitudes[np.argsort(-eigenvalue_magnitudes,)]
    print("Eigenvalues (first 10):\n", eigenvalue_magnitudes[:10])