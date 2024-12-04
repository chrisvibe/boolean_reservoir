import torch


def graph_average_k_income_edges_w_self_loops(reservoir_size, avg_k):
    # TODO this is not a lut but a adjacency matrix... whoops!
    # incoming edge average is k, self-loops allowed
    total_edges = round(reservoir_size * avg_k)
    adj_matrix = torch.zeros(reservoir_size**2, dtype=torch.bool)
    adj_matrix[:total_edges] = 1
    adj_matrix = adj_matrix[torch.randperm(reservoir_size**2)]
    adj_matrix = adj_matrix.view(reservoir_size, reservoir_size)
    return adj_matrix
