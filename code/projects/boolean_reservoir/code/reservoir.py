import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
import networkx as nx
import gzip
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.luts import lut_random 
from projects.boolean_reservoir.code.graphs import generate_graph_w_k_avg_incoming_edges, graph2adjacency_list_incoming, random_boolean_adjancency_matrix_from_two_degree_sets, random_projection_1d_to_2d, randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick
from projects.boolean_reservoir.code.utils import set_seed, override_symlink
import numpy as np
import sympy

class ExpressionEvaluator:
    def __init__(self, params: Params):
        self._setup_sympy_env()
        self.P = params
        self.L = self.P.logging
        self.I = self.P.model.input_layer
        self.R = self.P.model.reservoir_layer
        self.O = self.P.model.output_layer
        self.T = self.P.model.training
    
    def _setup_sympy_env(self):
        """Set up the sympy environment with symbols and their mappings."""
        # Define symbols
        self.sympy_symbols = {
            'a_f': sympy.Symbol('a_f'),
            'f': sympy.Symbol('f'),
            'a': sympy.Symbol('a'),
            'b': sympy.Symbol('b')
        }
        
    def _get_symbol_values(self):
        """Get current values for the symbols based on instance state."""
        return {
            self.sympy_symbols['a_f']: self.I.bits_per_feature,
            self.sympy_symbols['f']: self.I.n_inputs,
            self.sympy_symbols['a']: self.I.bits_per_feature * self.I.n_inputs,
            self.sympy_symbols['b']: self.R.n_nodes
        }
    
    def to_float(self, expr: str):
        """Convert a string expression to a float using sympy."""
        try:
            # Parse the expression
            parsed_expr = sympy.sympify(expr)
            
            # Get current values and substitute
            symbol_values = self._get_symbol_values()
            result = parsed_expr.subs(symbol_values)
            
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr}': {e}")


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
                self.load(checkpoint_path=load_path, parameter_override=params)
            else:
                raise AssertionError('load_path error')
        else:
            super(BooleanReservoir, self).__init__()
            self.P = params if params else load_dict['parameters']
            self.L = self.P.logging
            self.I = self.P.model.input_layer
            self.R = self.P.model.reservoir_layer
            self.O = self.P.model.output_layer
            self.T = self.P.model.training

            # How to perturb reservoir and which nodes to perturb 
            set_seed(self.I.seed)
            self.input_bits = self.I.n_inputs * self.I.bits_per_feature 
            assert self.input_bits <= self.R.n_nodes, 'more inputs bits than nodes in graph!'
            self.input_pertubation = self.input_pertubation_strategy(self.I.pertubation)
            self.w_in = self.optional_load('w_in', load_dict,
                self.load_torch_tensor(self.I.w_in, self.device) if self.I.w_in
                  else self.input_node_distribution_strategy(self.I.distribution)
            )
            set_seed(self.R.seed)
            self.graph = self.optional_load('graph', load_dict, 
                generate_graph_w_k_avg_incoming_edges(self.R.n_nodes, k_min=self.R.k_min, k_avg=self.R.k_avg, k_max=self.R.k_max, self_loops=self.R.self_loops)
            )
            assert self.R.n_nodes == self.graph.number_of_nodes()
            self.adj_list = graph2adjacency_list_incoming(self.graph)
            self.node_indices = torch.arange(self.R.n_nodes).to(self.device)

            # Precompute adj_list and expand it to the batch size
            self.adj_list, self.adj_list_mask = self.homogenize_adj_list(self.adj_list, max_length=self.R.k_max) 
            self.max_connectivity = self.adj_list.shape[-1] # Note that k_max may be larger than max_connectivity
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
            self.reset_reservoir(self.T.batch_size)
            self.states_paralell.to(self.device)
            
            # Dense readout layer
            set_seed(self.O.seed)
            self.readout = nn.Linear(self.R.n_nodes, self.O.n_outputs).to(self.device)
            if 'weights' in load_dict:
                self.load_state_dict(load_dict['weights'])
            self.output_activation = self.output_activation_strategy(self.O.activation)
            set_seed(self.R.seed)

            # Logging
            self.out_path = self.L.out_path
            self.timestamp_utc = self.get_timestamp_utc()
            self.save_dir = self.out_path / 'runs' / self.timestamp_utc 
            self.checkpoint_folder = self.L.last_checkpoint
            self.L.save_dir = self.save_dir
            self.L.timestamp_utc = self.timestamp_utc
            self.L.history.save_dir = self.save_dir / 'history'
            self.record_history = self.L.history.record_history
            self.history = BatchedTensorHistoryWriter(save_dir=self.L.history.save_dir, buffer_size=self.L.history.buffer_size) if self.record_history else None

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
            return torch.randint(0, 2, (1, self.R.n_nodes), dtype=torch.uint8)
        elif strategy == 'zeros':
            return torch.zeros((1, self.R.n_nodes), dtype=torch.uint8)
        elif strategy == 'ones':
            return torch.ones((1, self.R.n_nodes), dtype=torch.uint8)
        elif strategy == 'every_other':
            states = self.initialization_strategy('zeros')
            states[::2] = 1
            return states
        elif strategy == 'first_lut_state': # warmup reservoir with first state from LUT 
            return self.lut[self.node_indices, 0]
        elif strategy == 'random_lut_state': # warmup reservoir with random states from LUT
            idx = torch.randint(low=0, high=2**self.max_connectivity, size=(self.R.n_nodes,))
            return self.lut[self.node_indices, idx]
        raise ValueError

    def input_node_distribution_strategy(self, strategy: str):
        # produce w_in, a [dxb]xn matrix → bipartite map of how inputs bits perturb reservoir nodes
        # idea: make a determinisitic bipartite graph from input to reservoir + a probabilitic mapping
        # the deterministic mapping is from a_min:b_min, the probabilistic increments this to a_max, b_max by probability p
        # notation I: a and b get their names from normal mapping notation; number of nodes on the left and right side respectively in a bipartite map
        # notation II: k is the edge count per node (array)
        input_bits = self.I.n_inputs * self.I.bits_per_feature
        assert input_bits <= self.R.n_nodes, 'more inputs bits than nodes in graph!'
        
        # Parse the strategy string
        expression_evaluator = ExpressionEvaluator(self.P)
        parts = strategy.split('-')
        if parts[0] == 'min_max_degree':
            parts = parts[1].split(':')
            assert len(parts) == 5, "Strategy must have format 'min_max_degree-a_min:a_max:b_min:b_max:p'"
            a_min_expr, a_max_expr, b_min_expr, b_max_expr, p_expr = parts

            # Parse a_min, a_max, b_min, b_max, p 
            a_min = int(expression_evaluator.to_float(a_min_expr))
            a_max = int(expression_evaluator.to_float(a_max_expr))
            assert 0 <= a_min <= a_max <= self.R.n_nodes, f'outgoing connections per input node: 0 <= {a_min} (a_min) <= {a_max} (a_max) <= {self.R.n_nodes} (nodes in graph)'

            b_min = int(expression_evaluator.to_float(b_min_expr))
            b_max = int(expression_evaluator.to_float(b_max_expr))
            assert 0 <= b_min <= b_max <= self.R.n_nodes, f'incoming connections per reservoir node: 0 <= {b_min} (b_min) <= {b_max} (b_max) <= {input_bits} (nodes in graph)'

            p = expression_evaluator.to_float(p_expr)
            assert 0 <= p <= 1, f"Probability must be between 0 and 1, got {p} from expression '{p_expr}'"

            # make ka:
            capacity_range_a = a_max - a_min
            capacity_range_b = b_max - b_min
            edge_range = min(capacity_range_a * input_bits, capacity_range_b * self.R.n_nodes)
            edge_range = (np.random.random(edge_range) <= p).sum()
            ka = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, input_bits, capacity_range_a) # probabilistic connections
            ka += a_min # deterimistic connections

            # make kb:
            edge_range = ka.sum() 
            kb = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, self.R.n_nodes, capacity_range_b) # probabilistic connections
            kb += b_min # deterimistic connections
            
            # project 1D in-degree sequence to random 2D adjacency matrix
            w_in = random_boolean_adjancency_matrix_from_two_degree_sets(ka, kb)
            return torch.tensor(w_in, dtype=torch.uint8)
        elif parts[0] == 'min_max_random':
            parts = parts[1].split(':')
            assert len(parts) == 3, "Strategy must have format 'min_max_random-min:max:p'"
            a_min_expr, a_max_expr, p_expr = parts

            # Parse a_min, a_max, b_min, b_max, p 
            a_min = int(expression_evaluator.to_float(a_min_expr))
            a_max = int(expression_evaluator.to_float(a_max_expr))
            assert 0 <= a_min <= a_max <= self.R.n_nodes, f'outgoing connections per input node: 0 <= {a_min} (a_min) <= {a_max} (a_max) <= {self.R.n_nodes} (nodes in graph)'

            # Parse p 
            p = expression_evaluator.to_float(p_expr)
            assert 0 <= p <= 1, f"Probability must be between 0 and 1, got {p} from expression '{p_expr}'"

            # make ka:
            capacity_range_a = a_max - a_min
            edge_range = capacity_range_a * input_bits
            edge_range = (np.random.random(edge_range) <= p).sum()
            ka = randomly_distribute_pigeons_to_holes_with_capacity_dimension_trick(edge_range, input_bits, capacity_range_a) # probabilistic connections
            ka += a_min # deterimistic connections

            # project 1D in-degree sequence to random 2D adjacency matrix
            w_in = random_projection_1d_to_2d(ka, m=self.R.n_nodes).T
            return torch.tensor(w_in, dtype=torch.uint8)
 

    def input_pertubation_strategy(self, strategy:str):
        if strategy == 'xor': # Apply XOR only where input_mask is True, otherwise keep the old value
            return lambda states, perturbations, input_mask: (states * ~input_mask) | ((states ^ perturbations) & input_mask)
        elif strategy == 'and': # Apply AND only where input_mask is True, otherwise keep the old value
            return lambda states, perturbations, input_mask: (states * ~input_mask) | ((states & perturbations) & input_mask)
        elif strategy == 'or': # Apply OR only where input_mask is True, otherwise keep the old value
            return lambda states, perturbations, input_mask: (states * ~input_mask) | ((states | perturbations) & input_mask)
        elif strategy == 'override': 
            return lambda states, perturbations, input_mask: (states * ~input_mask) | (perturbations & input_mask)
        raise ValueError('Unknown perturbation strategy: {}'.format(strategy))
    
    def output_activation_strategy(self, strategy:str):
        if strategy is None:
            return lambda x: x
        elif strategy == 'sigmoid':
                return lambda x: torch.sigmoid(x)
        raise ValueError
    
    @staticmethod
    def get_timestamp_utc():
        return datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S_%f")

    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        self.checkpoint_folder = save_dir / 'checkpoints' / self.get_timestamp_utc()
        self.checkpoint_folder.mkdir(parents=True, exist_ok=False)
        override_symlink(save_dir.name, save_dir.parent / 'last_run')
        self.L.last_checkpoint = self.checkpoint_folder 
        paths = self.make_load_path_dict(self.checkpoint_folder)
        save_yaml_config(self.P, paths['parameters'])
        torch.save(self.w_in, paths['w_in'])
        with gzip.open(paths['graph'], 'wb') as f:
            nx.write_graphml(self.graph, f)
        torch.save(self.initial_states, paths['init_state'])
        torch.save(self.lut, paths['lut'])
        torch.save(self.state_dict(), paths['weights'])
        override_symlink(self.checkpoint_folder.name, self.checkpoint_folder.parent / 'last_checkpoint')
        if self.L.history.record_history:
            override_symlink(Path('../checkpoints') / self.L.last_checkpoint.name, self.L.history.save_dir / 'checkpoint')
        return paths 

    @staticmethod
    def load_from_path_dict_or_checkpoint_folder(
        path_dict: dict = None,
        checkpoint_path = None,
        load_key_include_set: set = None,
        load_key_exclude_set: set = None
    ):
        """
        path_dict takes precedence over optional checkpoint_path.
        load_key_include_set (None loads all) and load_key_exclude_set filter which keys to include/exclude.
        Inclusion is applied before exclusion.
        """
        if path_dict is None:
            if checkpoint_path is None:
                raise ValueError("Either path_dict or checkpoint_path must be provided.")
            path_dict = BooleanReservoir.make_load_path_dict(checkpoint_path)

        if load_key_include_set is not None:
            path_dict = {k: v for k, v in path_dict.items() if k in load_key_include_set}

        if load_key_exclude_set is not None:
            path_dict = {k: v for k, v in path_dict.items() if k not in load_key_exclude_set}

        d = dict() 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'parameters' in path_dict:
            d['parameters'] = load_yaml_config(path_dict['parameters'])
        if 'w_in' in path_dict:
            d['w_in'] = BooleanReservoir.load_torch_tensor(path_dict['w_in'], device)
        if 'graph' in path_dict:
            d['graph'] = BooleanReservoir.load_graph(path_dict['graph'])
        if 'init_state' in path_dict:
            d['init_state'] = BooleanReservoir.load_torch_tensor(path_dict['init_state'], device)
        if 'lut' in path_dict:
            d['lut'] = BooleanReservoir.load_torch_tensor(path_dict['lut'], device)
        if 'weights' in path_dict:
            d['weights'] = BooleanReservoir.load_torch_tensor(path_dict['weights'], device)
        return d
    
    def load(self, checkpoint_path:Path=None, paths:dict=None, parameter_override:Params=None):
        load_dict = self.load_from_path_dict_or_checkpoint_folder(path_dict=paths, checkpoint_path=checkpoint_path)
        if parameter_override:
            load_dict['parameters'] = parameter_override
        self.__init__(load_dict=load_dict)
        return load_dict
    
    @staticmethod
    def load_graph(path):
        with gzip.open(path, 'rb') as f:
            graph = nx.read_graphml(f) 
            graph = nx.relabel_nodes(graph, lambda x: int(x)) 
        return graph

    @staticmethod
    def load_torch_tensor(path, device):
        tensor = torch.load(path, weights_only=True, map_location=device)
        return tensor
    
    def flush_history(self):
        if self.history:
            self.history.flush()

    @staticmethod
    def make_load_path_dict(folder_path):
        folder_path = Path(folder_path)
        paths = dict()
        files = []
        files.append(('parameters', 'yaml'))
        files.append(('w_in', 'pt'))
        files.append(('graph', 'graphml.gz'))
        files.append(('init_state', 'pt'))
        files.append(('lut', 'pt'))
        files.append(('weights', 'pt'))
        paths = {file: folder_path / f'{file}.{filetype}' for file, filetype in files}
        return paths
    
    def bin2int(self, x):
        vals = (x * self.powers_of_2).sum(dim=-1)
        return vals

    def reset_reservoir(self, batches):
        self.states_paralell = self.initial_states.repeat(batches, 1)
   
    def forward(self, x):
        '''
        input is reshaped to fit number of input bits in w_in
        ie.
        assume x.shape == mxsxdxb
        m: samples
        s: steps
        d: n_inputs
        b: bits_per_feature

        how they perturb the reservoir if self.input_bits == 2:
        1. m paralell samples 
        2. s sequential step sets of inputs re-using w_in
        3. d sequential input sets as per w_in[a_i:b_i, :] (w_in is partitioned per input so two parts in the example)
        4. b simultaneous bits as per w_in[a_i:b_i, :]
        
        note that x only controls m and s (assume d and b dimensions match self.I.n_inputs and seld.I.bits_per_feature)
        thus one can change pertubations behavious via input shape or model configuration
        '''

        # handle batch size greater than paralell reservoirs
        m = x.shape[0]
        if m > self.T.batch_size:
            outputs_list = list()
            kb = 0
            for i in range(m // self.T.batch_size):
                ka = i*self.T.batch_size
                kb = ka + self.T.batch_size
                outputs = self.forward(x[ka:kb])
                outputs_list.append(outputs)
            if kb < m:
                outputs = self.forward(x[kb:])
                outputs_list.append(outputs)
            return torch.cat(outputs_list, dim=0)

        if self.R.reset:
            self.reset_reservoir(m)
        self.batch_record(m, phase='init', s=0, d=0)

        # INPUT LAYER
        # ----------------------------------------------------
        x = x.view(m, -1, self.I.n_inputs, self.I.bits_per_feature) # if input has more input bits than model expects: loop through these re-using w_in for each pertubation (samples kept in parallel)
        s = x.shape[1]
        k = self.w_in.shape[0]//self.I.n_inputs
        d = self.I.n_inputs
        for si in range(s):
            a = b = 0
            for di in range(d):
                # Perturb reservoir nodes with partial input depending on d dimension
                x_i = x[:, si, di]
                b += k
                w_in_i = self.w_in[a:b]
                a = b
                input_mask = w_in_i.sum(axis=0).to(torch.bool)
                perturbations = x_i @ w_in_i > 0 # some inputs bits may overlap which nodes are perturbed → counts as a single perturbation
                self.states_paralell[:m] = self.input_pertubation(self.states_paralell[:m], perturbations, input_mask)

                self.batch_record(m, phase='input_layer', s=si+1, d=di+1)

                # RESERVOIR LAYER
                # ----------------------------------------------------
                # Gather the states based on the adj_list
                neighbour_states_paralell = self.states_paralell[:m, self.adj_list]

                # Apply mask as the adj_list has invalid connections due to homogenized tensor
                neighbour_states_paralell &= self.adj_list_mask

                # Convert binary to integer index
                idx = self.bin2int(neighbour_states_paralell)

                # Update the state with LUT for each node
                self.states_paralell[:m] = self.lut[self.node_indices, idx]

                if not ((si == s - 1) and (di == d - 1)): # skip last recording, as this is output_layer
                    self.batch_record(m, phase='reservoir_layer', s=si+1, d=di+1)
        self.batch_record(m, phase='output_layer', s=s, d=d)

        # READOUT LAYER
        # ----------------------------------------------------
        outputs = self.readout(self.states_paralell[:m].float())
        if self.output_activation:
            outputs = self.output_activation(outputs)
        return outputs 

    def batch_record(self, m, **meta_data):
        if self.record_history and meta_data:
            self.history.append_batch(self.states_paralell[:m], meta_data)


class BatchedTensorHistoryWriter:
    def __init__(self, save_dir='history', buffer_size=64):
        self.save_dir = Path(save_dir)
        self.file_index = 0
        self.time = 0
        self.buffer_size = buffer_size
        self.buffer = []
        self.meta_buffer = []

    def append_batch(self, batch_tensor, meta_data):
        self.buffer.append(batch_tensor.clone().cpu())
        self.meta_buffer.append(meta_data)
        meta_data['file_idx'] = self.file_index 
        meta_data['batch_number'] = len(self.meta_buffer)
        meta_data['samples'] = len(batch_tensor)
        meta_data['time'] = self.time
        self.time += 1
        if len(self.buffer) >= self.buffer_size:
            self._write_buffer()

    def _write_buffer(self):
        if not self.buffer:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        data = torch.cat(self.buffer, dim=0)
        tensor_path = self.save_dir / f'tensor_{self.file_index}.pt'
        torch.save(data, tensor_path)
        meta_path = self.save_dir / f'meta_{self.file_index}.csv'
        pd.DataFrame(self.meta_buffer).to_csv(meta_path, index=False)
        self.buffer = []
        self.meta_buffer = []
        self.file_index += 1

    def flush(self):
        self._write_buffer()

    def reload_history(self, history_dir=None):
        history_dir = Path(history_dir) if history_dir else self.save_dir
        all_data = []
        all_meta_data = []
        idx = 0
        assert any(history_dir.glob('*.pt')), f"No files found. Try Recording the data? Maybe the path is wrong"
        for _ in history_dir.glob('*.pt'):
            tensor_path = history_dir / f'tensor_{idx}.pt'
            tensor_data = torch.load(tensor_path, weights_only=True)
            meta_path = history_dir / f'meta_{idx}.csv'
            meta_data = pd.read_csv(meta_path)
            all_data.append(tensor_data)
            all_meta_data.append(meta_data)
            idx += 1
        combined_data = torch.cat(all_data, dim=0)
        combined_meta_data = pd.concat(all_meta_data, ignore_index=True, axis=0)

        df = combined_meta_data
        expanded_meta_data = df.loc[df.index.repeat(df['samples'])].reset_index(drop=True)
        expanded_meta_data['sample_id'] = expanded_meta_data.groupby(['phase', 's', 'd']).cumcount()
        expanded_meta_data.drop(columns=['samples'], inplace=True)
        return combined_data, expanded_meta_data, combined_meta_data
    

if __name__ == '__main__':
    I = InputParams(
        distribution='1:n:1/n:o', 
        pertubation='override', 
        encoding='base2', 
        features=2,
        bits_per_feature=4,
        redundancy=2
        )
    R = ReservoirParams(
        n_nodes=10,
        k_min=0,
        k_avg=2,
        k_max=5,
        p=0.5,
        self_loops=0.1
        )
    O = OutputParams(features=2)
    T = TrainingParams(batch_size=1,
        epochs=10,
        accuracy_threshold=0.05,
        learning_rate=0.001)
    L = LoggingParams(out_path='/tmp/boolean_reservoir/out/test/', history=HistoryParams(record_history=True, buffer_size=10))

    model_params = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O, training=T)
    params = Params(model=model_params, logging=L)
    model = BooleanReservoir(params)

    # data with s steps per sample
    s = 1
    x = torch.randint(0, 2, (T.batch_size, s, I.features, I.bits_per_feature,), dtype=torch.uint8)
    model(x)
    print(model(x).detach().numpy())
    model.flush_history()
    history, meta, expanded_meta = BatchedTensorHistoryWriter(L.save_dir / 'history').reload_history()
    print(history[expanded_meta[expanded_meta['phase'] == 'init'].index].shape)
    print(history.shape)
    print(meta)
