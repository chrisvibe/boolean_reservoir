import torch


def lut_random(n_nodes, max_connectivity, p=0.5):
    assert 0 <= p <= 1
    expected_sum = round(n_nodes * p)
    
    # Generate a random boolean tensor for the lut matrix
    lut = torch.rand((n_nodes, max_connectivity)) < 0.5
    
    # Initialize the out_col tensor with zeros 
    out_col = torch.zeros(n_nodes, dtype=torch.bool)
    
    # Set the first 'expected_sum' elements to True
    out_col[:expected_sum] = True
    
    # Shuffle the out_col tensor
    out_col = out_col[torch.randperm(n_nodes)]
    
    # Concatenate along the correct dimension
    lut = torch.cat((lut, out_col.unsqueeze(1)), dim=1)
    
    return lut

if __name__ == '__main__':
    n = 100
    k = 3
    p = 0.5
    lut = lut_random(n, k, p=p)
    print(torch.sum(lut, dim=0) / n)
    print(lut[:10].to(torch.int).numpy())