import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from encoding import bin2dec
from graphs import graph2adjacency_list
import networkx as nx
from utils import make_folders


class PathIntegrationVerificationModelBinaryEncoding(nn.Module):
    # Linear model for sanity check to verify:
    # a) Base 2 binary encoding is relatively lossless with a decent number of bits
    # b) Path integration task can be computed by summing steps
    # Note that x values should be in the range [0, 1] for use of bin2dec
    # Encoding assumed to be binary base 2
    def __init__(self, n_dims):
        super(PathIntegrationVerificationModelBinaryEncoding, self).__init__()
        self.scale = nn.Linear(n_dims, n_dims)

    def forward(self, x):
        m, s, d, b = x.shape
        x = x.to(dtype=torch.float32)
        x = x.view(m * s * d, -1)          # role out dims
        x = bin2dec(x, b)                  # undo bit encoding 
        x = x.view(m, s, d)                # recover dimensions
        x = self.scale(x)                  # scale to y range
        x = torch.sum(x, dim=1)            # sum over s time steps
        return x

    def flush_history(self):
        pass


class PathIntegrationVerificationModel(nn.Module):
    # Linear model for sanity check to verify:
    # a) Any reasonable binary encoding (not just base 2)
    # b) Path integration task can be computed by summing steps
    # Encoding assumed to be a generalized linear transformation
    def __init__(self, bits_per_feature, n_inputs):
        super(PathIntegrationVerificationModel, self).__init__()
        self.decoder = nn.Linear(bits_per_feature, 1)
        self.scale = nn.Linear(n_inputs, n_inputs)

    def forward(self, x):
        m, s, d, b = x.shape
        x = x.to(dtype=torch.float32)
        x = x.view(m * s * d, -1)          # role out dims
        x = self.decoder(x)                # undo bit encoding 
        x = x.view(m, s, d)                # recover dimensions
        x = self.scale(x)                  # scale to y range
        x = torch.sum(x, dim=1)            # sum over s time steps
        return x

    def flush_history(self):
        pass



class BooleanReservoir(nn.Module):
    def __init__(self, graph: nx.Graph, lut, batch_size, max_connectivity, n_inputs, bits_per_feature, n_outputs, out_path='/out', record=False, max_history_buffer_size=10000):
        super(BooleanReservoir, self).__init__()
        self.graph = graph
        self.adj_list = graph2adjacency_list(graph)
        self.lut = lut

        self.n_nodes = graph.number_of_nodes()
        self.n_parallel = batch_size
        self.max_connectivity = max_connectivity
        self.bits_per_feature = bits_per_feature
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Initialize state
        # self.states_paralell = torch.randint(0, 2, (self.n_parallel, self.n_nodes), dtype=torch.bool) # random init per sample
        self.states_paralell = torch.randint(0, 2, (1, self.n_nodes), dtype=torch.bool).repeat(self.n_parallel, 1) # same init per sample
        self.initial_states_paralell = self.states_paralell.clone()

        # Dense readout layer
        self.readout = nn.Linear(self.n_nodes, self.n_inputs)

        # Preselect which reservoir nodes will be perturbed for input
        # TODO confine features to certain areas of the reservoir??? Now they are mixing...
        self.input_nodes = torch.randperm(self.n_nodes)[:self.n_inputs * self.bits_per_feature]
        # self.input_nodes2 = torch.randperm(self.n_nodes)[:self.n_inputs * self.bits_per_feature] # TODO delete
        
        # Precompute adj_list and expand it to the batch size
        self.adj_list, self.adj_list_mask = self.homogenize_adj_list(self.adj_list, max_length=self.max_connectivity) 
        self.adj_list_expanded = self.adj_list.unsqueeze(0).expand(self.n_parallel, -1, -1)

        # other precomputations
        self.node_indices = torch.arange(self.n_nodes)
        bits = self.adj_list.shape[1]
        self.mask = 2 ** torch.arange(bits) # left endian

        # Logging
        self.out_path = Path(out_path)
        self.folders = ['reservoir_history']
        make_folders(self.out_path, self.folders) 
        self.record = record
        self.max_history_buffer_size = max_history_buffer_size
        self.history_buffer = list() 
        self.history_buffer_file_count = 0

    def flush_history(self):
        if self.record and self.history_buffer:
            time = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M")
            f = str(self.history_buffer_file_count) + '_' + time + '.csv'
            df = pd.DataFrame(self.history_buffer)
            df.to_csv(self.out_path / 'reservoir_history' / f, index=False) # TODO consider np.savez_compressed
            self.history_buffer = list()
            self.history_buffer_file_count += 1
    
    @staticmethod
    def homogenize_adj_list(adj_list, max_length):
        adj_list_tensors = [torch.tensor(sublist, dtype=torch.long) for sublist in adj_list]
        padded_tensor = pad_sequence(adj_list_tensors, batch_first=True, padding_value=-1)
        padded_tensor = padded_tensor[:, :max_length]
        mask = padded_tensor != -1
        padded_tensor[padded_tensor == -1] = 0
        return padded_tensor, mask

        
    def bin2int(self, x):
        # left endian
        vals = torch.sum(self.mask * x, -1).long()
        return vals

    def reset_reservoir(self):
        # TODO reset output layer too?
        self.states_paralell = self.initial_states_paralell.clone()
        # TODO warmup reservoir?
    
    def forward(self, x):
        '''
        Accepts the decomposed velocities encoded in boolean format. In 2d: x = dx, dy
        Since each series of velocities corresponds to a single coordinate label the shape of x is: mxsxd
        m is the number of samples
        s is the number of steps
        d is the number of dimensions
        b is the number of bits used for the boolean encoding
        the output is then mxdxb

        1. input the encoded velocity data in s steps
        2. the reservoir should is hopefully able to represent the integral of these steps
        3. readout interprets the reservoir and outputs the integral; the final position coordinate

        TODO order of operations below: input step vs state index calculation!! good?
        '''
        m, s, d, b = x.shape
        self.reset_reservoir()

        # handle large batch size
        # note that the :m is to handle batch sizes smaller than the number of parallel reservoirs handy
        if m > self.n_parallel:
            outputs_list = list()
            kb = 0
            for i in range(m // self.n_parallel):
                ka = i*self.n_parallel
                kb = ka + self.n_parallel
                outputs = self.forward(x[ka:kb])
                outputs_list.append(outputs)
            if kb < m:
                outputs = self.forward(x[kb:])
                outputs_list.append(outputs)
            return torch.cat(outputs_list, dim=0)

        # Record states # TODO what granularity do we want?
        # if self.record:
        #     record = dict()
        #     record['sample'] = i 
        #     record['step'] = j + 1 
        #     record['reservoir_states'] = (torch.clone(self.node_states).to(torch.int).numpy())
        #     self.save_record(record)

        # INPUT LAYER
        # ----------------------------------------------------

        for j in range(s):
            # Perturb specific reservoir nodes with input
            self.states_paralell[:m, self.input_nodes] ^= x[:, j].view(m, -1)
            # self.states_paralell[:m, self.input_nodes2] ^= x[:, j].view(m, -1)

            # RESERVOIR LAYER
            # ----------------------------------------------------
            # Gather the states based on the expanded adj_list
            state_expanded = self.states_paralell[:m].unsqueeze(-1).expand(-1, -1, self.max_connectivity)
            states_paralell = torch.gather(state_expanded, 1, self.adj_list_expanded[:m])

            # Apply mask as the adj_list has invalid connections due to homogenized tensor
            states_paralell &= self.adj_list_mask

            # Convert binary to integer index
            idx = self.bin2int(states_paralell)

            # Update the state with LUT
            self.states_paralell[:m] = self.lut[self.node_indices, idx].clone()

        # Record states # TODO what granularity do we want?
        # if self.record:
        #     record = dict()
        #     record['sample'] = i 
        #     record['step'] = j + 1 
        #     record['reservoir_states'] = (torch.clone(self.node_states).to(torch.int).numpy())
        #     self.save_record(record)
    
        # READOUT LAYER
        # ----------------------------------------------------
        outputs = self.readout(self.states_paralell[:m].float())
        return outputs 


if __name__ == '__main__':
    from graphs import graph_average_k_incoming_edges_w_self_loops
    from luts import lut_random

    batch_size = 5
    n_nodes = 10
    max_connectivity = 2
    avg_k = 2
    n_inputs = 2
    bits_per_feature = 2
    n_outputs = 2

    graph = graph_average_k_incoming_edges_w_self_loops(n_nodes, avg_k)
    lut = lut_random(n_nodes, 2**max_connectivity)

    model = BooleanReservoir(graph, lut, batch_size, max_connectivity, n_inputs, bits_per_feature, n_outputs)

    # data with s steps per sample
    s = 3
    x = torch.randint(0, 2, (batch_size, s, n_inputs, bits_per_feature,), dtype=torch.bool)
    print(model(x).detach().numpy())