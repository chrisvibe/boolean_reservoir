import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from encoding import bin2dec
import networkx as nx
import gzip
from parameters import * 
from luts import lut_random 
from graphs import generate_graph_w_k_avg_incoming_edges, graph2adjacency_list_incoming
from utils import set_seed


class PathIntegrationVerificationModelBaseTwoEncoding(nn.Module):
    # Linear model for sanity check to verify:
    # a) Base 2 binary encoding is relatively lossless with a decent number of bits
    # b) Path integration task can be computed by summing steps
    # Note that x values should be in the range [0, 1] for use of bin2dec
    # Encoding assumed to be binary base 2
    def __init__(self, n_dims):
        super(PathIntegrationVerificationModelBaseTwoEncoding, self).__init__()
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
    # load_path can be a yaml file or a checkpoint directory
    # a yaml file doesnt load any parameters while a checkpoint does
    # params can be used to override stuff in conjunction with load_path
    def __init__(self, params: Params=None, load_path=None, load_dict=dict()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if load_path:
            load_path = Path(load_path)
            if load_path.suffix in ['.yaml', '.yml']:
                P = load_yaml_config(load_path)
                self.__init__(params=P)
            elif load_path.is_dir():
                paths = self.make_load_paths(load_path)
                self.load(paths=paths, parameter_override=params)
            else:
                raise AssertionError('load_path error')
        else:
            super(BooleanReservoir, self).__init__()
            self.P = params
            self.I = self.P.model.input_layer
            self.R = self.P.model.reservoir_layer
            self.O = self.P.model.output_layer
            self.T = self.P.model.training
            set_seed(self.R.seed)

            self.graph = self.optional_load('graph', load_dict, 
                generate_graph_w_k_avg_incoming_edges(self.R.n_nodes, self.R.k_avg, k_max=self.R.k_max, self_loops=self.R.self_loops)
            )
            self.adj_list = graph2adjacency_list_incoming(self.graph)
            self.n_nodes = self.graph.number_of_nodes()
            self.node_indices = torch.arange(self.n_nodes).to(self.device)
            self.n_parallel = self.T.batch_size
            self.bits_per_feature = self.I.bits_per_feature
            self.n_inputs = self.I.n_inputs
            self.n_outputs = self.O.n_outputs

            # Precompute adj_list and expand it to the batch size
            self.adj_list, self.adj_list_mask = self.homogenize_adj_list(self.adj_list, max_length=self.R.k_max) 
            self.max_connectivity = self.adj_list.shape[-1] # Note that k_max may be larger than max_connectivity
            self.adj_list_expanded = self.adj_list.unsqueeze(0).expand(self.n_parallel, -1, -1).to(self.device)
            self.adj_list_mask = self.adj_list_mask.to(self.device)

            # Each node as a LUT of length 2**k where next state is looked up by index determined by neighbour state
            self.lut = self.optional_load('lut', load_dict,
                lut_random(self.R.n_nodes, self.max_connectivity, p=self.R.p)
            ).to(self.device)

            # Precompute bins2int conversion mask 
            bits = self.max_connectivity 
            self.powers_of_2 = 2 ** torch.arange(bits).flip(dims=(0,)).to(self.device)
            assert bits <= 24, 'too many bits! (overflow)' # bin2int overflows if too large

            # Initialize state
            self.states_paralell = None
            self.initial_states = self.optional_load('init_state', load_dict,
                self.initialization_strategy(self.R.init)
            ).to(self.device)
            self.reset_reservoir()
            self.states_paralell.to(self.device)
            
            # Preselect which reservoir nodes will be perturbed for input
            input_bits = self.n_inputs * self.bits_per_feature
            assert input_bits <= self.n_nodes, 'more inputs bits than nodes in graph!'
            self.input_nodes = self.optional_load('input_nodes', load_dict,
                # assumes input nodes are dedicated to their feature (no repeats)
                torch.randperm(self.n_nodes)[:input_bits].reshape(self.n_inputs, self.bits_per_feature)
            ).to(self.device)

            # Dense readout layer
            set_seed(self.O.seed)
            self.readout = nn.Linear(self.n_nodes, self.n_inputs).to(self.device)
            if 'weights' in load_dict:
                self.load_state_dict(load_dict['weights'])
            set_seed(self.R.seed)

            # Logging
            logging_params = self.P.logging
            self.out_path = Path(logging_params.out_path)
            self.record_history = logging_params.history.record_history
            self.history_buffer_size = logging_params.history.history_buffer_size
            self.history = list() 
            self.history_file_count = 0
            self.timestamp_utc = self.get_timestamp_utc()
            self.P.logging.train_log.timestamp_utc = self.timestamp_utc
            self.save_folder = self.out_path / 'models' / self.timestamp_utc
            
    def flush_history(self):
        if self.record_history and self.history:
            df = pd.DataFrame(self.history)
            f = str(self.history_file_count) + '.csv'
            path = self.out_path / 'history' / self.timestamp_utc / f
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False) # TODO consider np.savez_compressed
            self.history = list()
            self.history_file_count += 1
    
    @staticmethod 
    def optional_load(load_key: str, load_dict: dict, default):
        if load_key in load_dict:
            return load_dict[load_key]
        else:
            return default

    @staticmethod
    def homogenize_adj_list(adj_list, max_length): # tensor may have less than max_length columns if not needed
        adj_list_tensors = [torch.tensor(sublist, dtype=torch.long) for sublist in adj_list]
        padded_tensor = pad_sequence(adj_list_tensors, batch_first=True, padding_value=-1)
        padded_tensor = padded_tensor[:, :max_length]
        valid_mask = padded_tensor != -1
        padded_tensor[~valid_mask] = 0
        return padded_tensor, valid_mask.to(torch.uint8)
    
    def initialization_strategy(self, strategy:str):
        if strategy == 'random':
            return torch.randint(0, 2, (1, self.n_nodes), dtype=torch.uint8)
        elif strategy == 'zeros':
            return torch.zeros((1, self.n_nodes), dtype=torch.uint8)
        elif strategy == 'ones':
            return torch.ones((1, self.n_nodes), dtype=torch.uint8)
        elif strategy == 'every_other':
            states = self.initialization_strategy('zeros')
            states[::2] = 1
            return states
        elif strategy == 'first_lut_state': # warmup reservoir with first state from LUT 
            return self.lut[self.node_indices, 0]
        elif strategy == 'random_lut_state': # warmup reservoir with random states from LUT
            idx = torch.randint(low=0, high=2**self.max_connectivity, size=(self.n_nodes,))
            return self.lut[self.node_indices, idx]
        raise ValueError
    
    @staticmethod
    def get_timestamp_utc():
        return datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S_%f")

    def save(self):
        self.save_folder.mkdir(parents=True, exist_ok=True)
        paths = self.make_load_paths(self.save_folder)
        save_yaml_config(self.P, paths['parameters'])
        with gzip.open(paths['graph'], 'wb') as f:
            nx.write_graphml(self.graph, f)
        torch.save(self.lut, paths['lut'])
        torch.save(self.input_nodes, paths['input_nodes'])
        torch.save(self.initial_states, paths['init_state'])
        torch.save(self.state_dict(), paths['weights'])
        self.P.logging.checkpoint_path = self.save_folder
        return paths 

    def load(self, paths=None, parameter_override:Params=None):
        if paths is None:
            paths = self.make_load_paths(self.save_folder)
        p = parameter_override
        if p is None:
            p = load_yaml_config(paths['parameters'])
        d = dict() 
        with gzip.open(paths['graph'], 'rb') as f:
            d['graph'] = nx.read_graphml(f) 
            d['graph'] = nx.relabel_nodes(d['graph'], lambda x: int(x)) 
        d['lut'] = torch.load(paths['lut'], weights_only=True, map_location=self.device)
        d['input_nodes'] = torch.load(paths['input_nodes'], weights_only=True, map_location=self.device)
        d['init_state'] = torch.load(paths['init_state'], weights_only=True, map_location=self.device)
        d['weights'] = torch.load(paths['weights'], weights_only=True, map_location=self.device)
        self.__init__(params=p, load_dict=d)

    @staticmethod
    def make_load_paths(folder_path):
        folder_path = Path(folder_path)
        paths = dict()
        files = []
        files.append(('parameters', 'yaml'))
        files.append(('graph', 'graphml.gz'))
        files.append(('lut', 'pt'))
        files.append(('input_nodes', 'pt'))
        files.append(('init_state', 'pt'))
        files.append(('weights', 'pt'))
        for file, filetype in files:
            paths[file] = folder_path / f'{file}.{filetype}'
        return paths
    
    def bin2int(self, x):  # TODO check if GPU has data
        vals = (x * self.powers_of_2).sum(dim=-1)
        return vals

    def reset_reservoir(self):
        self.states_paralell = self.initial_states.repeat(self.n_parallel, 1)
   
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
            self.states_paralell[:m, self.input_nodes] ^= x[:m, j]

            # RESERVOIR LAYER
            # ----------------------------------------------------
            # Gather the states based on the expanded adj_list (note that this is maybe not efficient...)
            state_expanded = self.states_paralell[:m].unsqueeze(-1).expand(-1, -1, self.max_connectivity)
            states_paralell = torch.gather(state_expanded, dim=1, index=self.adj_list_expanded[:m])

            # Apply mask as the adj_list has invalid connections due to homogenized tensor
            states_paralell &= self.adj_list_mask

            # Convert binary to integer index
            idx = self.bin2int(states_paralell)

            # Update the state with LUT
            self.states_paralell[:m] = self.lut[self.node_indices, idx]

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
    I = InputParams(encoding='binary', 
                        n_inputs=1,
                        bits_per_feature=10,
                        redundancy=2
            )
    R = ReservoirParams(
                        n_nodes=100,
                        k_avg=2,
                        k_max=5,
                        p=0.5,
                        self_loops=0.1
                )
    O = OutputParams(n_outputs=1)
    T = TrainingParams(batch_size=64,
                       epochs=10,
                       radius_threshold=0.05,
                       learning_rate=0.001)
    model_params = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O, training=T)
    params = Params(model=model_params)
    model = BooleanReservoir(params)

    # data with s steps per sample
    s = 3
    x = torch.randint(0, 2, (T.batch_size, s, I.n_inputs, I.bits_per_feature,), dtype=torch.uint8)
    print(model(x).detach().numpy())
    # TODO fix bugg: make sure this has the same result, seeds in model should guarantee this...