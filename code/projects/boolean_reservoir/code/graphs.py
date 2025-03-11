import networkx as nx
import numpy as np
import random

def generate_graph_w_k_avg_incoming_edges(n_nodes, k_min=None, k_avg=None, k_max=None, self_loops=None):
    adj_matrix = generate_adjacency_matrix(n_nodes=n_nodes, k_min=k_min, k_avg=k_avg, k_max=k_max, self_loops=self_loops)
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

def gen_boolean_array(n):
    return np.random.randint(0, 2, size=n, dtype=bool)

def generate_adjacency_matrix(n_nodes, k_min: int=0, k_avg: float=None, k_max: int=None, self_loops: float=None):
    """
    Generate a random boolean directed adjacency matrix with optional min and max in-degree constraints.
    Each entry in the adjacency matrix is set with uniform probability.
    Additionally a optional self_loops sets the self-connections
    1. Set a 1D array tracking in-degree by a constrained random pidgeon distribution to holes (constrained normal distribution by CLT [k_min, k_max])
    2. Convert 1D in degree array to a 2D adjacency matrix with random assingment avoiding the diagonal
    3. Set diagonal according to self_loops input parameter
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes in the graph
    k_min : int
        Minimum number of neighbors per node (default: 0)
    k_avg : float
        Average number of neighbors per node (default: random between 0 and k_max)
    k_max : int or None
        Maximum number of neighbors per node (default: n_nodes)
    self_loops : float or None
        Proportion of nodes with self-loops (default: random between 0 and 1)
    
    Returns:
    --------
    adj_matrix : numpy.ndarray
        Boolean adjacency matrix
    """
    k_max = n_nodes if k_max is None else k_max
    k_avg = random.uniform(k_min, k_max) if k_avg is None else float(k_avg)
    self_loops = random.uniform(0, 1) if self_loops is None else float(self_loops)

    # Calculate required number of edges and self-loops
    n_self_loops = round(self_loops * n_nodes)
    total_edges = round(k_avg * n_nodes)

    # Parameter validation
    assert all(isinstance(var, expected_type) for var, expected_type, name in [
        (k_min, int, "k_min"),
        (k_avg, float, "k_avg"),
        (k_max, int, "k_max"),
        (self_loops, float, "self_loops")
    ]), "Parameters must be of correct types"
    assert 0 <= k_min <= k_avg <= k_max <= n_nodes, "Invalid k parameters: must have 0 ≤ k_min ≤ k_avg ≤ k_max ≤ n_nodes"
    assert 0 <= self_loops <= 1, "self_loops must be between 0 and 1"
    assert total_edges <= (n_nodes - 1) * n_nodes + n_self_loops , "Theoretical limit on edges exceeded - can happen when self_loops < 1"

    # Initialize k reserving space for k_min
    capacity_range = k_max - k_min
    edge_range = total_edges - k_min * n_nodes
    k = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, n_nodes, capacity_range)
    k += k_min
    # guaranteed k_min <= k <= k_max

    # adjust k to prepare for self-loops
    # spread could be concentrated so that we cant add self-loops (k_max)
    # cant have a self-loop if there are no edges in the column
    self_loop_potential = (k > 0).sum()
    self_loop_potential_diff = self_loop_potential - n_self_loops
    if self_loop_potential_diff < 0:
        for _ in range(abs(self_loop_potential_diff)): #  this may take many rounds if one node has all the edges
            # remove edge
            mask = k > 1
            pool = np.where(mask)[0]
            indices = np.random.choice(pool, 1)
            k[indices] -= 1
            # add edge
            mask = k == 0
            pool = np.where(mask)[0]
            indices = np.random.choice(pool, 1)
            k[indices] += 1

    # project 1D in-degree sequence to 2D adjacency matrix
    adj_matrix = randomly_project_1d_to_2d(k)

    # assign self-loops along diagonal
    # projecting to 2D may have added edges to the diagonal
    # check how many self_loops we have, move remaining to diagonal if requiring more self-loops or reverse 
    # the strategy is a vertical move of the edge so that k_min and k_max are invariant
    diagonal = np.diag(adj_matrix)
    self_loop_diff = n_self_loops - diagonal.sum()
    if self_loop_diff != 0:
        add_edge_to_diagonal = self_loop_diff >= 0
        change_mask = (k > 0) * (diagonal ^ add_edge_to_diagonal)
        change_indices = np.where(change_mask)[0]
        col_indices = np.random.choice(change_indices, abs(self_loop_diff), replace=False)
        for c in col_indices:
            col = adj_matrix[:, c]
            col_change_mask = col ^ (~add_edge_to_diagonal)
            rows = np.where(col_change_mask)[0]
            r = np.random.choice(rows, 1)
            adj_matrix[r, c] = ~add_edge_to_diagonal # invariant change with regards to k_min & k_max
            adj_matrix[c, c] = add_edge_to_diagonal # changes self-loop only

    # Final verification
    assert adj_matrix.sum() == total_edges, 'total edge test failed'
    assert np.diagonal(adj_matrix).sum() == n_self_loops, 'self-loop test failed'
    assert (adj_matrix.sum(axis=0) <= k_max).all(), 'k_max test failed'
    assert (adj_matrix.sum(axis=0) >= k_min).all(), 'k_min test failed'
    return adj_matrix

def randomly_project_1d_to_2d(k):
    n_nodes = k.shape[-1]
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
    for i in range(adj_matrix.shape[1]):
        adj_matrix[:k[i], i] = True 
        np.random.shuffle(adj_matrix[:, i])
    return adj_matrix

def randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(pigeons, holes, capacity):
    # this returns a normally distributed hole occupance by CLT constrained by capacity [0, capacity]
    max_occupance = holes * capacity
    if max_occupance == pigeons:
        return np.full((holes,), capacity, dtype=int)
    assert pigeons <= max_occupance, "Too many pigeons for the given number of holes and capacity"
    worst_case_capacity = min(capacity, pigeons)
    possible_hole_assignments = np.zeros((holes * worst_case_capacity), dtype=bool)
    possible_hole_assignments[:pigeons] = True
    np.random.shuffle(possible_hole_assignments)
    hole_occupance = possible_hole_assignments.reshape(worst_case_capacity, holes).sum(axis=0)
    return hole_occupance

if __name__ == '__main__':
    G = generate_graph_w_k_avg_incoming_edges(n_nodes=1000, k_avg=3, k_max=6, k_min=2, self_loops=0)
    adj_matrix = nx.adjacency_matrix(G).todense()
    k = adj_matrix.sum(axis=0)
    print(k)
    print(k.min(), k.max())
    eigenvalues = np.linalg.eigvals(adj_matrix)
    rho = max(abs(eigenvalues))
    print(rho)
    eigenvalue_magnitudes = np.abs(eigenvalues)
    eigenvalue_magnitudes = eigenvalue_magnitudes[np.argsort(-eigenvalue_magnitudes,)]
    print("Eigenvalues (first 10):\n", eigenvalue_magnitudes[:10])