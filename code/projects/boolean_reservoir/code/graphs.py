import networkx as nx
import numpy as np
import random
from projects.boolean_reservoir.code.utils.utils import print_pretty_binary_matrix

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

def constrain_degree_of_bipartite_mapping(a, b, min_degree, max_degree, p, in_degree=True):
    '''
    a and b repesent a bipartite mapping a→b
    to build adjacency matrices for this we use an analogy of pigeons finding pigeon holes of a certain capacity
    in_degree: sets control of in-degree vs out-degree
    constraint: sets min max constrain on a or b
    p: probability of connection from max - min

    if in_degree = False we constrain out degree of a not b....
    '''
    capacity_range = max_degree - min_degree
    constrained_set = b if in_degree else a
    free_set = a if in_degree else b
    edge_range = capacity_range * constrained_set 
    edge_range = (np.random.random(edge_range) <= p).sum()
    k = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, constrained_set, capacity_range) # probabilistic connections
    k += min_degree # deterimistic connections

    # project 1D in-degree sequence to random 2D adjacency matrix
    w = random_projection_1d_to_2d(k, m=free_set)
    if in_degree:
        return w
    else:
        return w.T

def random_constrained_stub_matching(a, b, a_min, a_max, b_min, b_max, p):
    # make ka:
    capacity_range_a = a_max - a_min
    capacity_range_b = b_max - b_min
    edge_range = min(capacity_range_a * b, capacity_range_b * a)
    edge_range = (np.random.random(edge_range) <= p).sum()
    ka_out = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, a, capacity_range_a) # probabilistic connections
    ka_out += a_min # deteriministic connections

    # make kb:
    edge_range = ka_out.sum() 
    kb_in = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, b, capacity_range_b) # probabilistic connections
    kb_in += b_min # deteriministic connections
    
    # project 1D in-degree sequence to random 2D adjacency matrix
    w = random_boolean_adjancency_matrix_from_two_degree_sets(ka_out, kb_in)
    return w

def gen_boolean_array(n):
    return np.random.randint(0, 2, size=n, dtype=bool)

def generate_adjacency_matrix(n_nodes, k_min: int=0, k_avg: float=None, k_max: int=None, self_loops: float=None, rows=None):
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
    total_edges = round(k_avg * n_nodes)

    if self_loops is None:
        required_self_loops = max(0, total_edges - (n_nodes * (n_nodes - 1)))
        optional_loops = n_nodes - required_self_loops
        random_self_loops = random.randint(0, optional_loops)
        n_self_loops = required_self_loops + random_self_loops
        self_loops = n_self_loops / n_nodes
    else:
        self_loops = float(self_loops)
    n_self_loops = round(self_loops * n_nodes)

    # Parameter validation
    assert all(isinstance(var, expected_type) for var, expected_type, name in [
        (k_min, int, "k_min"),
        (k_avg, float, "k_avg"),
        (k_max, int, "k_max"),
        (self_loops, float, "self_loops")
    ]), "Parameters must be of correct types"
    assert 0 <= k_min <= k_avg <= k_max <= n_nodes, f"Invalid k parameters: must have 0 ≤ k_min ({k_min}) ≤ k_avg ({k_avg}) ≤ k_max ({k_max}) ≤ n_nodes ({n_nodes})"
    assert 0 <= self_loops <= 1, f"self_loops ({self_loops}) must be between 0 and 1"
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
    rows = n_nodes if rows is None else rows
    adj_matrix = random_projection_1d_to_2d(k, m=rows)

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

def random_projection_1d_to_2d(k: np.ndarray, m: int=None):
    """
    Projects a 1D array k into a 2D boolean array where each column i 
    has k[i] randomly positioned True values.
    
    Parameters:
    ----------
    k : np.ndarray
        1D array where each value k[i] represents the number of True values in column i
    m : int, optional
        Number of rows in output array. Defaults to length of k
        
    Returns:
    -------
    np.ndarray
        A 2D boolean array of shape (m, len(k))
    """
    # Handle 0D array (scalar)
    if k.ndim == 0:
        k = np.array([k])
        
    n = len(k)
    m = n if m is None else m
    
    # Initialize output array
    adj_matrix = np.zeros((m, n), dtype=bool)
    
    # For each column
    for i in range(n):
        true_indices = np.random.choice(m, k[i], replace=False)
        adj_matrix[true_indices, i] = True
            
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

def random_boolean_adjancency_matrix_from_p(n: int, m: int, p: float) -> np.ndarray:
    adj_matrix = np.random.rand(n, m)
    adj_matrix = adj_matrix <= p
    return adj_matrix

def random_boolean_adjancency_matrix_from_two_degree_sets(ka: np.ndarray, kb: np.ndarray, max_tries=100) -> np.ndarray:
    """
    Generate a boolean adjacency matrix from two degree sequences ka and kb.
    
    Why not do this?
    G = nx.bipartite.configuration_model(ka, kb, create_using=nx.Graph())
    Exact degree sequences may not be realized due to the rejection of non-simple elements
    creat_using=nx.graph() removes the multi-edges and self-loops, but then the edge count may be wrong

    Ok... and this?
    G = nx.bipartite.havel_hakimi_graph(ka, kb, create_using=nx.Graph())
    This is deterministic. I need random.

    ... And this?
    G = nx.bipartite.random_graph(len(ka), len(kb), p)
    Doesnt give fine grained control over degrees :(

    Conclusion:
    1. configuration_model
    2. attempt repair remaining after drop of non-simple elements

    Alternative: 
    1. deterministic solve: havel-hakimi
    2. scramble: double edge swap (problem: makes connections within sets) or curveball algorithm (problem: not available)

    :param ka: A 1D integer ndarray of degree sequence for set A (length n).
    :param kb: A 1D integer ndarray of degree sequence for set B (length n).
    :return: A boolean ndarray representing the adjacency matrix.
    """
    raise NotImplementedError('This is a hard problem and is not yet needed...')
    assert sum(ka) == sum(kb), "The sum of degrees in ka and kb must be equal (handshake lemma)."
    assert ka.max() <= len(kb), "No node in set A should have a degree greater than the number of nodes in set B."
    assert kb.max() <= len(ka), "No node in set B should have a degree greater than the number of nodes in set A."
    assert (ka >= 0).all(), "All elements of ka must be > 0 (nodes should have non-negative degree)."
    assert (kb >= 0).all(), "All elements of kb must be > 0 (nodes should have non-negative degree)."

    # Create initial probabilisitic graph
    A = range(len(ka))
    B = range(len(ka), len(ka) + len(kb))
    for i in range(max_tries):
        G = nx.bipartite.configuration_model(ka, kb, create_using=nx.Graph())
        adj_matrix = nx.bipartite.biadjacency_matrix(G, row_order=A, column_order=B).toarray()
        currrent_ka = adj_matrix.sum(axis=1)
        currrent_kb = adj_matrix.sum(axis=0)
        if (currrent_ka == ka).all() and (currrent_kb == kb).all():
            return adj_matrix 

    # # Brute Force Repair (wont always work...)
    # currrent_ka = adj_matrix.sum(axis=1)
    # deficit_a = ka - currrent_ka
    # missing = deficit_a.sum()
    # if missing: # handshake lemma implies kb is satisdied when ka is
    #     currrent_kb = adj_matrix.sum(axis=0)
    #     deficit_b = kb - currrent_kb
    #     candidates_a = np.where(deficit_a > 0)[0]
    #     candidates_b = np.where(deficit_b > 0)[0]
    #     # make all possible edges (ignore edges that are already 1)
    #     candidate_edges = [(i, j) for i in candidates_a for j in candidates_b if adj_matrix[i, j] == 0]
    #     if not candidate_edges:
    #         if recursion <= max_recursion:
    #             return random_boolean_adjancency_matrix_from_two_degree_sets(ka, kb, recursion=recursion+1)
    #         UserWarning('random_boolean_adjancency_matrix_from_two_degree_sets: Could not satisfy degree set constraints')
    #         return adj_matrix
    #     # randomly choose missing
    #     candidate_edges = np.stack(candidate_edges, axis=0)
    #     idx = np.random.choice(len(candidate_edges), size=missing, replace=False)
    #     candidate_edges = candidate_edges[idx]
    #     # set missing (without checking)
    #     adj_matrix[candidate_edges[:, 0], candidate_edges[:, 1]] = 1

    # assert (ka == adj_matrix.sum(axis=1)).all(), "row sum mismatch (ka)"
    # assert (kb == adj_matrix.sum(axis=0)).all(), "column sum mismatch (kb)"
    # return adj_matrix
   

if __name__ == '__main__':
    w = generate_adjacency_matrix(n_nodes=1000, k_avg=3, k_max=6, k_min=2, self_loops=0)
    k = w.sum(axis=0)
    print(k)
    print(k.min(), k.max())
    eigenvalues = np.linalg.eigvals(w)
    rho = max(abs(eigenvalues))
    print(rho)
    eigenvalue_magnitudes = np.abs(eigenvalues)
    eigenvalue_magnitudes = eigenvalue_magnitudes[np.argsort(-eigenvalue_magnitudes,)]
    print("Eigenvalues (first 10):\n", eigenvalue_magnitudes[:10])