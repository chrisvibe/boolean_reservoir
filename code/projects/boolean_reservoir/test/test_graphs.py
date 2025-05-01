import pytest
import numpy as np
from projects.boolean_reservoir.code.graphs import generate_adjacency_matrix_w_k_avg_incoming_edges
from scipy.stats import shapiro

def calculate_worst_case_parameters_for_expecting_normal_distribution(k_avg):
    n = 30  # Assumed minimum number of nodes to ensure normal distribution
    variance_per_node = k_avg * (1 - (1/n))  # Variance for the binomial distribution approximation
    std_dev = np.sqrt(variance_per_node)

    k_min = max(0, k_avg - std_dev)
    k_max = k_avg + std_dev
    
    return {
        "n_nodes": n,
        "k_min": k_min,
        "k_max": k_max,
    }

@pytest.mark.parametrize(
    "n_nodes, k_min, k_avg, k_max, self_loops",
    [
        # Test case with no self-loops with min and max close to avg
        (1000, 2, 3, 4, 0),

        # Test case with no self-loops with min equal to avg
        (1000, 3, 3, 4, 0),

        # Test case with no self-loops with max equal to avg
        (1000, 2, 3, 3, 0),

        # Test case with no self-loops with min and max equal to avg
        (1000, 3, 3, 3, 0),

        # Test case with 100% self-loops with min and max close to avg
        (1000, 2, 3, 4, 1),

        # Test case with 100% self-loops with min equal to avg
        (1000, 3, 3, 4, 1),

        # Test case with 100% self-loops with max equal to avg
        (1000, 2, 3, 3, 1),

        # Test case with 100% self-loops with min and max equal to avg
        (1000, 3, 3, 3, 1),
        
        # Lower average edges no self-loops
        (1000, 1, 2, 3, 0),

        # 10% self-loops with higher average edges
        (1000, 2, 3, 4, 0.1),
        
        # 10% self-loops with lower average edges
        (1000, 1, 2, 3, 0.1),

        # Minimal self-loops (1%) with higher average edges
        (1000, 2, 3, 4, 0.01),
        
        # Minimal self-loops (1%) with lower average edges
        (1000, 1, 2, 3, 0.01),

        # High percentage of self-loops (90%) with higher average edges
        (1000, 2, 3, 4, 0.9),
        
        # High percentage of self-loops (90%) with lower average edges
        (1000, 1, 2, 3, 0.9),
        
        # Lower average edges, 100% self-loops
        (1000, 1, 2, 3, 1),

        # Medium average edges, 10% self-loops
        (1000, 1, 500, 1000, 0.1),

        # Maximum average edges, 100% self-loops
        (1000, 1, 1000, 1000, 1),

        # Should have normal distribution
        (1000, 0, 100, 500, 0),

        # Should have normal distribution
        (1000, 0, 100, 500, 1),

        # many self-loops relative to edges 
        (500, 0, 1, 10, 0.95)
    ]
)
def test_generate_adjacency_matrix(n_nodes, k_min, k_avg, k_max, self_loops):
    adj_matrix = generate_adjacency_matrix_w_k_avg_incoming_edges(n_nodes, k_min, k_avg, k_max, self_loops)
    assert adj_matrix is not None
    assert adj_matrix.shape == (n_nodes, n_nodes), "Adjacency matrix shape mismatch"

    total_edges = round(n_nodes * k_avg)
    n_self_loops = round(n_nodes * self_loops)

    assert adj_matrix.sum() == total_edges, 'Total edge test failed'
    assert np.diagonal(adj_matrix).sum() == n_self_loops, 'Self-loop test failed'
    assert (adj_matrix.sum(axis=0) <= k_max).all(), 'k_max test failed'
    assert (adj_matrix.sum(axis=0) >= k_min).all(), 'k_min test failed'

    # Test for normality of in-degree distribution
    p = calculate_worst_case_parameters_for_expecting_normal_distribution(k_avg)
    if n_nodes >= p["n_nodes"] and p["k_min"] <= k_min and p["k_max"] >= k_max:
        in_degrees = adj_matrix.sum(axis=0)
        in_degree_range = in_degrees.max() - in_degrees.min()
        unique_values = len(np.unique(in_degrees))
        samples = len(in_degrees)
        if in_degree_range >= 3 and unique_values >= 3 and samples >= 8:
            stat, p_value = shapiro(in_degrees)  # Use the Shapiro-Wilk test
            assert p_value > 0.05, "In-degree distribution does not follow normal distribution (p-value < 0.05)"

if __name__ == '__main__':
    print("run: pytest /code/projects/boolean_reservoir/test/test_graphs.py")