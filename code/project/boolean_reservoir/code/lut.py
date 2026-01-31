import torch

def lut_random(n_nodes, max_incoming_edges, p=0.5):
    # make a independent lut for each node
    # lut[0] is the lut for node 0
    # let idx represent the state of the reservoir: lut[0][idx] is the next state of node 0 with probability p
    assert 0 <= p <= 1
    lut = torch.rand((n_nodes, 2 ** max_incoming_edges)) < p
    return lut.to(dtype=torch.uint8)