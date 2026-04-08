import torch
import numpy as np

def lut_random(n_nodes, max_incoming_edges, p=0.5):
    # make a independent lut for each node
    # lut[0] is the lut for node 0
    # let idx represent the state of the reservoir: lut[0][idx] is the next state of node 0 with probability p
    # Uses numpy RNG (not torch) so output is identical across CPU and GPU given the same seed.
    assert 0 <= p <= 1
    lut = np.random.rand(n_nodes, 2 ** max_incoming_edges) < p
    return torch.from_numpy(lut.astype(np.uint8))