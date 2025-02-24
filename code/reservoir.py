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
            self.L = self.P.logging
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
            self.input_pertubation_function = self.input_pertubation_strategy(self.I.pertubation_strategy)

            # Dense readout layer
            set_seed(self.O.seed)
            self.readout = nn.Linear(self.n_nodes, self.n_inputs).to(self.device)
            if 'weights' in load_dict:
                self.load_state_dict(load_dict['weights'])
            set_seed(self.R.seed)

            # Logging
            self.out_path = self.L.out_path
            self.timestamp_utc = self.get_timestamp_utc()
            self.save_dir = self.out_path / self.timestamp_utc 
            self.checkpoint_folder = self.L.last_checkpoint
            self.L.save_dir = self.save_dir
            self.L.timestamp_utc = self.timestamp_utc
            self.record_history = self.L.history.record_history
            self.history = BatchedTensorHistoryWriter(folderpath=self.save_dir / 'history', buffer_size=self.L.history.buffer_size) if self.record_history else None
    
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
    
    def input_pertubation_strategy(self, strategy:str):
        if strategy == 'override':
            return lambda old, new: new
        elif strategy == 'xor':
                return lambda old, new: new ^ old
        raise ValueError

    
    @staticmethod
    def get_timestamp_utc():
        return datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S_%f")

    def save(self):
        self.checkpoint_folder = self.save_dir / 'checkpoints' / self.get_timestamp_utc()
        self.checkpoint_folder.mkdir(parents=True, exist_ok=False)
        paths = self.make_load_paths(self.checkpoint_folder)
        save_yaml_config(self.P, paths['parameters'])
        with gzip.open(paths['graph'], 'wb') as f:
            nx.write_graphml(self.graph, f)
        torch.save(self.lut, paths['lut'])
        torch.save(self.input_nodes, paths['input_nodes'])
        torch.save(self.initial_states, paths['init_state'])
        torch.save(self.state_dict(), paths['weights'])
        self.L.last_checkpoint = self.checkpoint_folder 
        return paths 

    def load(self, paths=None, parameter_override:Params=None):
        if paths is None:
            paths = self.make_load_paths(self.checkpoint_folder)
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
    
    def flush_history(self):
        if self.history:
            self.history.flush()

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
    
    def bin2int(self, x):
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

        # handle small batch size
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

        self.batch_record(phase='init', step=0)

        # INPUT LAYER
        # ----------------------------------------------------
        for j in range(s):
            # Perturb specific reservoir nodes with input
            self.states_paralell[:m, self.input_nodes] = self.input_pertubation_function(self.states_paralell[:m, self.input_nodes], x[:m, j])

            self.batch_record(phase='input_layer', step=j)

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

            self.batch_record(phase='reservoir_layer', step=j)

        self.batch_record(phase='output_layer', step=0)

        # READOUT LAYER
        # ----------------------------------------------------
        outputs = self.readout(self.states_paralell[:m].float())
        return outputs 

    def batch_record(self, **meta_data):
        if self.record_history and meta_data:
            self.history.append_batch(self.states_paralell, meta_data)


class BatchedTensorHistoryWriter:
    def __init__(self, folderpath='history', buffer_size=64):
        self.dir_path = Path(folderpath)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.file_index = 0
        self.buffer_size = buffer_size
        self.buffer = []
        self.meta_buffer = []

    def append_batch(self, batch_tensor, meta_data):
        self.buffer.append(batch_tensor.clone().cpu())
        self.meta_buffer.append(meta_data)
        meta_data['file_idx'] = self.file_index 
        meta_data['batch_number'] = len(self.meta_buffer)
        meta_data['samples'] = len(batch_tensor)
        if len(self.buffer) >= self.buffer_size:
            self._write_buffer()

    def _write_buffer(self):
        if not self.buffer:
            return
        data = torch.cat(self.buffer, dim=0)
        tensor_path = self.dir_path / f'tensor_{self.file_index}.pt'
        torch.save(data, tensor_path)
        meta_path = self.dir_path / f'meta_{self.file_index}.csv'
        pd.DataFrame(self.meta_buffer).to_csv(meta_path, index=False)
        self.buffer = []
        self.meta_buffer = []
        self.file_index += 1

    def flush(self):
        self._write_buffer()

    def reload_history(self, history_dir=None):
        dir_path = Path(history_dir) if history_dir else self.dir_path
        all_data = []
        all_meta_data = []
        idx = 0
        for _ in dir_path.glob('*.pt'):
            tensor_path = dir_path / f'tensor_{idx}.pt'
            tensor_data = torch.load(tensor_path, weights_only=True)
            meta_path = dir_path / f'meta_{idx}.csv'
            meta_data = pd.read_csv(meta_path)
            all_data.append(tensor_data)
            all_meta_data.append(meta_data)
            idx += 1
        combined_data = torch.cat(all_data, dim=0)
        combined_meta_data = pd.concat(all_meta_data, ignore_index=True, axis=0)

        df = combined_meta_data
        expanded_meta_data = df.loc[df.index.repeat(df['samples'])].reset_index(drop=True)
        expanded_meta_data.drop(columns=['samples'], inplace=True)
        return combined_data, expanded_meta_data, combined_meta_data


if __name__ == '__main__':
    I = InputParams(
        pertubation_strategy='override', 
        encoding='binary', 
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
    T = TrainingParams(batch_size=32,
        epochs=10,
        radius_threshold=0.05,
        learning_rate=0.001)
    L = LoggingParams(out_path='/tmp/boolean_reservoir/out/test/', history=HistoryParams(record_history=True, buffer_size=10))

    model_params = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O, training=T)
    params = Params(model=model_params, logging=L)
    model = BooleanReservoir(params)

    # data with s steps per sample
    s = 3
    x = torch.randint(0, 2, (T.batch_size, s, I.n_inputs, I.bits_per_feature,), dtype=torch.uint8)
    model(x)
    print(model(x).detach().numpy())
    model.flush_history()
    history, meta, expanded_meta = BatchedTensorHistoryWriter(L.save_dir / 'history').reload_history()
    print(history[expanded_meta[expanded_meta['phase'] == 'init'].index].shape)
    print(history.shape)
    print(meta)
