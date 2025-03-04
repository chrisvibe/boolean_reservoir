import networkx as nx
import numpy as np
import random

def generate_graph_w_k_avg_incoming_edges(n_nodes, k_avg, k_max=None, k_min=0, self_loops=None):
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
    total_edges = round(k_avg * n_nodes)
    n_self_loops = round(self_loops * n_nodes)

    # Parameter validation
    assert all(isinstance(var, expected_type) for var, expected_type, name in [
        (k_min, int, "k_min"),
        (k_avg, float, "k_avg"),
        (k_max, int, "k_max"),
        (self_loops, float, "self_loops")
    ]), "Parameters must be of correct types"
    assert 0 <= k_min <= k_avg <= k_max <= n_nodes, "Invalid k parameters: must have 0 ≤ k_min ≤ k_avg ≤ k_max ≤ n_nodes"
    assert 0 <= self_loops <= 1, "self_loops must be between 0 and 1"
    assert k_min + n_self_loops / n_nodes <= k_max, "Constraint violation: k_min + n_self_loops / n_nodes <= k_max"
    assert total_edges <= n_nodes * n_nodes, "total_edges exceeds maximum possible edges in directed adjacency graph"
    assert 0 < k_max, "Invalid k_max: must have 0 < k_max"

    # Initialize k
    edge_range = total_edges - k_min*n_nodes - n_self_loops
    capacity_range = k_max - k_min
    # reserve space for k_min and subtract self_loops 
    k = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, n_nodes, capacity_range)
    k += k_min
    # guaranteed k_min <= k <= k_max

    # 1D to 2D matrix
    assert k.sum() == total_edges - n_self_loops
    adj_matrix = randomly_project_1d_to_2d_avoiding_diagonal(k)
    assert adj_matrix.sum() == total_edges - n_self_loops

    # assign self-loops along diagonal
    k_self_loops = np.zeros((n_nodes,), dtype=bool)
    k_self_loops[:n_self_loops] = True
    np.random.shuffle(k_self_loops)
    np.fill_diagonal(adj_matrix, k_self_loops)

    # Final verification
    assert adj_matrix.sum() == total_edges
    assert np.diagonal(adj_matrix).sum() == n_self_loops
    assert (adj_matrix.sum(axis=0) <= k_max).all()
    assert (adj_matrix.sum(axis=0) >= k_min).all()
    return adj_matrix

def randomly_project_1d_to_2d_avoiding_diagonal(k):
    n_nodes = k.shape[-1]
    assert k.sum() + n_nodes <= n_nodes ** 2, 'cant place along diagonal, too many'
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
    for i in range(adj_matrix.shape[1]):
        adj_matrix[:k[i], i] = True 
        np.random.shuffle(adj_matrix[:, i])
        if adj_matrix[i, i]: # dont allow placement along diagonal
            mask = ~adj_matrix[:, i]
            adj_matrix[i, i] = False
            indices = np.where(mask)[0]
            idx = np.random.choice(indices)
            adj_matrix[idx, i] = True
    return adj_matrix

# TODO delete
def randomly_distribute_pigeons_to_holes_with_capacity_n_rows_at_a_time(pigeons, holes, capacity):
    # this returns a normally distributed hole occupance by CLT constrained by capacity [0, capacity]
    max_occupance = holes * capacity
    assert pigeons <= max_occupance, "Too many pigeons for the given number of holes and capacity"
    hole_occupance = np.zeros((holes,), dtype=int) 
    remaining_pigeons = pigeons
    while remaining_pigeons != 0: # parallel increment holes_with_capacity but with 50% chance of success (like rain accumulation)
        hole_slack = capacity - hole_occupance
        holes_with_capacity = hole_slack > 0
        hole_slack_with_capacity = hole_slack[holes_with_capacity] 
        n_rows = max(1, min(remaining_pigeons // holes, hole_slack.min())) # worst case expected number of pigeon rows we can add without breaking capacity
        n_cols = hole_slack_with_capacity.shape[0]
        probability_mask = gen_boolean_array(n_rows * n_cols) # stack many pidgeon rows with 50% 1/0
        probability_mask = probability_mask.reshape(n_rows, n_cols)
        if remaining_pigeons < n_cols: # make sure we dont add more pigeons in parallel than needed
            pigeon_mask = np.zeros((n_cols,), dtype=bool)
            true_indices = np.random.choice(n_cols, remaining_pigeons, replace=False)
            pigeon_mask[true_indices] = True
            probability_mask *= pigeon_mask 
        potential_new_increment = probability_mask.cumsum(axis=0)
        diff = hole_slack_with_capacity - potential_new_increment[n_rows-1]
        min_diff = diff.min()
        while min_diff < 0: # required multiple iterations as each row is not necesarilly an increment
            n_rows += min_diff # conservative assumed all rows are increments
            diff = hole_slack_with_capacity - potential_new_increment[n_rows-1]
            min_diff = diff.min()
        increment = potential_new_increment[n_rows-1]
        hole_occupance[holes_with_capacity] += increment
        remaining_pigeons -= increment.sum() 
    return hole_occupance

# TODO delete
def randomly_distribute_pigeons_to_holes_with_capacity_1_row_at_a_time(pigeons, holes, capacity):
    # this returns a normally distributed hole occupance by CLT constrained by capacity [0, capacity]
    max_occupance = holes * capacity
    assert pigeons <= max_occupance, "Too many pigeons for the given number of holes and capacity"
    hole_occupance = np.zeros((holes,), dtype=int) 
    remaining_pigeons = pigeons
    while remaining_pigeons != 0:
        holes_with_capacity = capacity > hole_occupance
        probability_mask = gen_boolean_array(holes) 
        if remaining_pigeons < holes: # make sure we dont add more pigeons in parallel than needed
            pigeon_mask = np.zeros((holes,), dtype=bool)
            true_indices = np.random.choice(holes, remaining_pigeons, replace=False)
            pigeon_mask[true_indices] = True
            probability_mask *= pigeon_mask 
        increment = holes_with_capacity * probability_mask
        hole_occupance += increment 
        remaining_pigeons -= increment.sum() 
    return hole_occupance

def randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(pigeons, holes, capacity):
    # this returns a normally distributed hole occupance by CLT constrained by capacity [0, capacity]
    max_occupance = holes * capacity
    if max_occupance == pigeons:
        return np.full((n_nodes,), capacity, dtype=int)
    assert pigeons <= max_occupance, "Too many pigeons for the given number of holes and capacity"
    worst_case_capacity = min(capacity, pigeons)
    possible_hole_assignments = np.zeros((holes * worst_case_capacity), dtype=bool)
    possible_hole_assignments[:pigeons] = True
    np.random.shuffle(possible_hole_assignments)
    hole_occupance = possible_hole_assignments.reshape(worst_case_capacity, holes).sum(axis=0)
    return hole_occupance

# TODO delete
def old_generate_adjacency_matrix(n_nodes, k_min=0, k_avg=None, k_max=None, self_loops=None):
    # init adjacency matrix but may not respect requested properties
    k_max = n_nodes if k_max is None else k_max
    self_loops = random.uniform(0, 1) if self_loops is None else float(self_loops)
    assert 0 <= k_avg <= n_nodes
    assert 0 <= k_avg <= k_max
    assert 0 <= self_loops <= 1
    diagonal = lambda x: x[::n_nodes+1]
    total_edges = round(n_nodes * k_avg)
    self_loops = round(self_loops * n_nodes)
    flat_adj_matrix = np.zeros((n_nodes * n_nodes,), dtype=bool)
    indices = np.arange(n_nodes)

    # calculate k for self-loops
    k_self_loop = np.zeros((n_nodes,), dtype=bool)
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

if __name__ == '__main__':
    n_nodes = 10
    k_avg = 3
    k_max = 5
    self_loops = 0.3
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
