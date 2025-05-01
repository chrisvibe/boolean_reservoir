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
from projects.boolean_reservoir.code.graphs import generate_adjacency_matrix, graph2adjacency_list_incoming, random_constrained_stub_matching, constrain_degree_of_bipartite_mapping
from projects.boolean_reservoir.code.utils import set_seed, override_symlink
import numpy as np
import sympy

class ExpressionEvaluator:
    def __init__(self, params: Params, symbols: dict=dict()):
        self._setup_sympy_env()
        self.P = params
        self.M = self.P.M
        self.L = self.P.L
        self.I = self.P.M.I
        self.R = self.P.M.R
        self.O = self.P.M.O
        self.T = self.P.M.T
        self.symbols = symbols

    def _setup_sympy_env(self):
        """Set up the sympy environment with symbols and their mappings."""
        # Define symbols
        self.sympy_symbols = {
            'a': sympy.Symbol('a'),
            'b': sympy.Symbol('b')
        }
        
    def _get_symbol_values(self):
        """Get current values for the symbols based on instance state."""
        return {self.sympy_symbols[k]: v for k, v in self.symbols.items()}
    
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


class BatchedTensorHistoryWriter:
    def __init__(self, save_path='history', buffer_size=64):
        self.save_path = Path(save_path)
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
        self.save_path.mkdir(parents=True, exist_ok=True)
        data = torch.cat(self.buffer, dim=0)
        tensor_path = self.save_path / f'tensor_{self.file_index}.pt'
        torch.save(data, tensor_path)
        meta_path = self.save_path / f'meta_{self.file_index}.csv'
        pd.DataFrame(self.meta_buffer).to_csv(meta_path, index=False)
        self.buffer = []
        self.meta_buffer = []
        self.file_index += 1

    def flush(self):
        self._write_buffer()

    def reload_history(self, history_path=None, checkpoint_path=None, include={''}, exclude={}):
        history_path = Path(history_path) if history_path else self.save_path
        all_data = []
        all_meta_data = []
        idx = 0
        assert any(history_path.glob('*.pt')), f"No files found at {history_path}. Try Recording the data? Maybe the path is wrong"
        for _ in history_path.glob('*.pt'):
            tensor_path = history_path / f'tensor_{idx}.pt'
            tensor_data = torch.load(tensor_path, weights_only=True)
            meta_path = history_path / f'meta_{idx}.csv'
            meta_data = pd.read_csv(meta_path)
            all_data.append(tensor_data)
            all_meta_data.append(meta_data)
            idx += 1
        combined_data = torch.cat(all_data, dim=0)
        combined_meta_data = pd.concat(all_meta_data, ignore_index=True, axis=0)

        df = combined_meta_data
        expanded_meta_data = df.loc[df.index.repeat(df['samples'])].reset_index(drop=True)
        expanded_meta_data['sample_id'] = expanded_meta_data.groupby(['phase', 's', 'f']).cumcount()
        expanded_meta_data.drop(columns=['samples'], inplace=True)

        checkpoint_path = checkpoint_path if checkpoint_path else history_path / 'checkpoint'
        load_dict = dict()
        if checkpoint_path.exists():
            load_dict = BooleanReservoir.load_from_path_dict_or_checkpoint_folder(checkpoint_path=checkpoint_path, load_key_include_set=include, load_key_exclude_set=exclude)

        return load_dict, combined_data, expanded_meta_data, combined_meta_data
 

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
            self.M = self.P.M
            self.L = self.P.L
            self.I = self.P.M.I
            self.R = self.P.M.R
            self.O = self.P.M.O
            self.T = self.P.M.T

            # INPUT LAYER
            # How to perturb reservoir nodes that receive input (I) - w_in maps input bits to input_nodes 
            set_seed(self.I.seed)
            self.input_pertubation = self.input_pertubation_strategy(self.I.pertubation)
            self.w_in = self.load_or_generate('w_in', load_dict, lambda:
                self.load_torch_tensor(self.I.w_in, self.device) if self.I.w_in
                  else self.bipartite_mapping_strategy(self.I.distribution, self.I.bits, self.I.n_nodes)
            )
            # TODO optionally control each quadrant individually
            # w_ii = w_ir = None
            # if 'graph' not in load_dict:
                # w_ii = torch.zeros((self.I.n_nodes, self.I.n_nodes))  # TODO no effect atm - add flexibility for connections betwen input nodes (I→I)
                # w_ir = self.bipartite_mapping_strategy(self.I.connection, self.I.n_nodes, self.R.n_nodes)
            self.node_indices = torch.arange(self.M.n_nodes).to(self.device)
            self.input_nodes_mask = torch.zeros(self.M.n_nodes, dtype=bool, device=self.device)
            self.input_nodes_mask[:self.I.n_nodes] = True

            # RESERVOIR LAYER
            # Properties of reservoir nodes that dont receive input (R)
            set_seed(self.R.seed)
            # Graph is divided into nodes that recieve input (I) and nodes that dont (R)
            # This forms four partitions when constructing the adjacency matrix that can be fine-tuned
            # TODO optionally control each quadrant individually
            # w_ri = w_rr = None
            # if 'graph' not in load_dict:
                # w_ri = torch.zeros((self.R.n_nodes, self.I.n_nodes))
                # w_rr = torch.tensor(generate_adjacency_matrix(self.R.n_nodes, k_min=self.R.k_min, k_avg=self.R.k_avg, k_max=self.R.k_max, self_loops=self.R.self_loops))
            # self.graph = self.load_or_generate('graph', load_dict, lambda:
                # self.build_graph_from_quadrants(w_ii, w_ir, w_ri, w_rr)
            # )
            self.graph = self.load_or_generate('graph', load_dict,
                self.build_graph
            )
            assert self.M.n_nodes == self.graph.number_of_nodes()
            self.adj_list = graph2adjacency_list_incoming(self.graph)

            # Precompute adj_list and expand it to the batch size
            self.adj_list, self.adj_list_mask, self.no_neighbours_mask = map(lambda x: x.to(self.device),
                self.homogenize_adj_list(self.adj_list, max_length=self.R.k_max)
            )
            self.max_connectivity = self.adj_list.shape[-1] # Note that k_max may be larger than max_connectivity

            # Each node as a LUT of length 2**k where next state is looked up by index determined by neighbour state
            # LUT allows incoming edges from I + R for each node
            self.lut = self.load_or_generate('lut', load_dict, lambda:
                lut_random(self.M.n_nodes, self.max_connectivity, p=self.R.p).to(torch.uint8)
            )

            # Precompute bin2int conversion mask 
            self.powers_of_2 = self.make_minimal_powers_of_two(self.max_connectivity).to(self.device)

            # Initialize state (R + I)
            self.states_parallel = None
            self.initial_states = self.load_or_generate('init_state', load_dict, lambda:
                self.initialization_strategy(self.R.init)
            ).to(self.device)
            self.reset_reservoir(self.T.batch_size)
            self.states_parallel.to(self.device)
            
            # OUTPUT LAYER
            set_seed(self.O.seed)
            self.readout = nn.Linear(self.R.n_nodes, self.O.n_outputs).to(self.device) # readout of R only TODO add option to choose
            if 'weights' in load_dict:
                self.load_state_dict(load_dict['weights'])
            self.output_activation = self.output_activation_strategy(self.O.activation)
            self.output_nodes_mask = ~self.input_nodes_mask # TODO add option to choose with w_out (bit mask or mapping)
            self.output_nodes_mask.to(self.device)
            set_seed(self.R.seed)

            # LOGGING
            self.out_path = self.L.out_path
            self.timestamp_utc = self.get_timestamp_utc()
            self.save_path = self.out_path / 'runs' / self.timestamp_utc 
            self.checkpoint_path = self.L.last_checkpoint
            self.L.save_path = self.save_path
            self.L.timestamp_utc = self.timestamp_utc
            self.L.history.save_path = self.save_path / 'history'
            self.record_history = self.L.history.record_history
            self.history = BatchedTensorHistoryWriter(save_path=self.L.history.save_path, buffer_size=self.L.history.buffer_size) if self.record_history else None

    def add_graph_labels(self, graph):
        labels_mapping = {node: node for node in graph.nodes()}
        nx.set_node_attributes(graph, labels_mapping, 'id')
        labels_mapping = {k: v.item() for k, v in zip(graph.nodes(), self.input_nodes_mask)}
        nx.set_node_attributes(graph, labels_mapping, 'I')
        labels_mapping = {k: v.item() for k, v in zip(graph.nodes(), self.output_nodes_mask)}
        nx.set_node_attributes(graph, labels_mapping, 'bipartite')
        nx.set_node_attributes(graph, labels_mapping, 'R')
        nx.set_node_attributes(graph, labels_mapping, 'O') # TODO might not be true in future: all reservoir nodes are output nodes

        # Assign quadrant labels to each edge based on source and target nodes
        parts_size = [self.I.n_nodes, self.R.n_nodes]
        boundary = parts_size[0]
        edge_quadrants = {}
        for u, v in graph.edges():
            if u < boundary and v < boundary:
                quadrant = 'II'  # Both nodes in first partition (I-I)
            elif u < boundary and v >= boundary:
                quadrant = 'IR'  # Source in first partition, target in second (I-R)
            elif u >= boundary and v < boundary:
                quadrant = 'RI'  # Source in second partition, target in first (R-I)
            else:  # u >= boundary and v >= boundary
                quadrant = 'RR'  # Both nodes in second partition (R-R)
            edge_quadrants[(u, v)] = quadrant
        nx.set_edge_attributes(graph, edge_quadrants, 'quadrant')

    def build_graph(self):
        # TODO this is wrong as it mixes model levels settings with reservoir level
        w = generate_adjacency_matrix(self.M.n_nodes, k_min=self.R.k_min, k_avg=self.R.k_avg, k_max=self.R.k_max, self_loops=self.R.self_loops)
        graph = nx.from_numpy_array(w, create_using=nx.DiGraph)
        return graph
    
    def build_graph_from_quadrants(self, w_ii, w_ir, w_ri, w_rr):
        w_i = torch.cat((w_ii, w_ir), dim=1)
        w_r = torch.cat((w_ri, w_rr), dim=1)
        w = torch.cat((w_i, w_r), dim=0)
        graph = nx.from_numpy_array(w.numpy(), create_using=nx.DiGraph)
        return graph
    
    @staticmethod
    def make_minimal_powers_of_two(bits: int):
        assert bits < 64, 'Too many bits! (bin2int may overflow)'
        if bits < 8:
            dtype = torch.uint8
        elif bits < 16:
            dtype = torch.int16
        elif bits < 32:
            dtype = torch.int32
        else:
            dtype = torch.int64
        return (2 ** torch.arange(bits, dtype=dtype).flip(0))

    @staticmethod 
    def load_or_generate(load_key: str, load_dict: dict, generator: callable):
        if load_key in load_dict:
            return load_dict[load_key]
        else:
            return generator()

    @staticmethod
    def homogenize_adj_list(adj_list, max_length): # tensor may have less than max_length columns if not needed
        adj_list_tensors = [torch.tensor(sublist, dtype=torch.long) for sublist in adj_list]
        padded_tensor = pad_sequence(adj_list_tensors, batch_first=True, padding_value=-1)
        padded_tensor = padded_tensor[:, :max_length]
        valid_mask = padded_tensor != -1
        padded_tensor[~valid_mask] = 0
        no_neighbours_mask = ~valid_mask.any(dim=1) # if all neigbours are off its not the same as having no neighbours
        return padded_tensor, valid_mask.to(torch.uint8), no_neighbours_mask
    
    def initialization_strategy(self, strategy:str):
        if strategy == 'random':
            return torch.randint(0, 2, (1, self.M.n_nodes), dtype=torch.uint8)
        elif strategy == 'zeros':
            return torch.zeros((1, self.M.n_nodes), dtype=torch.uint8)
        elif strategy == 'ones':
            return torch.ones((1, self.M.n_nodes), dtype=torch.uint8)
        elif strategy == 'every_other':
            states = self.initialization_strategy('zeros')
            states[::2] = 1
            return states
        elif strategy == 'first_lut_state': # warmup reservoir with first state from LUT 
            return self.lut[self.node_indices, 0]
        elif strategy == 'random_lut_state': # warmup reservoir with random states from LUT
            idx = torch.randint(low=0, high=2**self.max_connectivity, size=(self.M.n_nodes,))
            return self.lut[self.node_indices, idx]
        raise ValueError

    def bipartite_mapping_strategy(self, strategy: str, a: int, b:int):
        # produce w [axb] matrix → bipartite map, by default a maps to b but this can be inverted by swaping a and b in the input string
        # notation I: a and b get their names from mapping notation; number of nodes on the left and right side respectively in a bipartite map
        # notation II: k is the edge count per node (array)
        expression_evaluator = ExpressionEvaluator(self.P, {'a': a, 'b': b})
        parts = strategy.split('-')
        w = None
        if parts[0] == 'identity':
            w = np.eye(a, b)
        elif parts[0] == 'stub':
            # idea: make a determinisitic bipartite graph + a probabilitic mapping
            # constrain in and out degree simultaneously
            # the deterministic mapping is from a_min:b_min, the probabilistic increments this to a_max, b_max by probability p

            # Parse a_min, a_max, b_min, b_max, p 
            parts = parts[1].split(':')
            assert len(parts) == 5, "Strategy must have format 'stub-a_min:a_max:b_min:b_max:p'"
            a_min_expr, a_max_expr, b_min_expr, b_max_expr, p_expr = parts
            a_min = int(expression_evaluator.to_float(a_min_expr))
            a_max = int(expression_evaluator.to_float(a_max_expr))
            b_min = int(expression_evaluator.to_float(b_min_expr))
            b_max = int(expression_evaluator.to_float(b_max_expr))
            p = expression_evaluator.to_float(p_expr)
            assert 0 <= p <= 1, f"Probability must be between 0 and 1, got {p} from expression '{p_expr}'"
            assert 0 <= b_min <= b_max <= a, f'incoming connections per node in b: 0 <= {b_min} (b_min) <= {b_max} (b_max) <= {a} (a)'
            assert 0 <= a_min <= a_max <= b, f'outgoing connections per node in a: 0 <= {a_min} (a_min) <= {a_max} (a_max) <= {b} (b)'

            w = random_constrained_stub_matching(a, b, a_min, a_max, b_min, b_max, p)
        elif parts[0] in {'in', 'out'}:
            # idea: make a determinisitic bipartite graph + a probabilitic mapping
            # constrain either in or out degree
            # the deterministic mapping is to satisfy min_degree, the probabilistic increments this to max_degree by probability p

            # Parse b_min, b_max, p 
            parts = parts[1].split(':')
            assert len(parts) == 3, "Strategy must have format 'in-min_degree:max_degree:p'"
            min_degree_expr, max_degree_expr, p_expr = parts
            min_degree = int(expression_evaluator.to_float(min_degree_expr))
            max_degree = int(expression_evaluator.to_float(max_degree_expr))
            p = expression_evaluator.to_float(p_expr)
            assert 0 <= p <= 1, f"Probability must be between 0 and 1, got {p} from expression '{p_expr}'"

            in_degree = parts[0] == 'in'
            if in_degree: # in-degree of b
                assert 0 <= min_degree <= max_degree <= a, f'incoming connections per node in b: 0 <= {min_degree} (b_min) <= {max_degree} (b_max) <= {a} (a)'
            else: # out-degree of a
                assert 0 <= min_degree <= max_degree <= b, f'outgoing connections per node in a: 0 <= {min_degree} (a_min) <= {max_degree} (a_max) <= {b} (b)'
            w = constrain_degree_of_bipartite_mapping(a, b, min_degree, max_degree, p, in_degree=in_degree)
        if w is not None:
            return torch.tensor(w, dtype=torch.uint8)
        raise ValueError
 
    @staticmethod
    def input_pertubation_strategy(strategy:str):
        # assumes states is only input nodes
        if strategy == 'xor':
            return lambda states, perturbations: states ^ perturbations
        elif strategy == 'and':
            return lambda states, perturbations: states & perturbations
        elif strategy == 'or':
            return lambda states, perturbations: states | perturbations
        elif strategy == 'override': 
            return lambda states, perturbations: perturbations
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

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        self.checkpoint_path = save_path / 'checkpoints' / self.get_timestamp_utc()
        self.checkpoint_path.mkdir(parents=True, exist_ok=False)
        override_symlink(save_path.name, save_path.parent / 'last_run')
        self.L.last_checkpoint = self.checkpoint_path 
        paths = self.make_load_path_dict(self.checkpoint_path)
        save_yaml_config(self.P, paths['parameters'])
        torch.save(self.w_in, paths['w_in'])
        with gzip.open(paths['graph'], 'wb') as f:
            nx.write_graphml(self.graph, f)
        torch.save(self.initial_states, paths['init_state'])
        torch.save(self.lut, paths['lut'])
        torch.save(self.state_dict(), paths['weights'])
        override_symlink(self.checkpoint_path.name, self.checkpoint_path.parent / 'last_checkpoint')
        if self.L.history.record_history:
            self.L.history.save_path.mkdir(parents=True, exist_ok=True) # in case model is saved before history is recorded
            override_symlink(Path('../checkpoints') / self.L.last_checkpoint.name, self.L.history.save_path / 'checkpoint')
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
        self.states_parallel = self.initial_states.repeat(batches, 1)
   
    def forward(self, x):
        '''
        input is reshaped to fit number of input bits in w_in
        ie.
        assume x.shape == mxsxdxb
        m: samples
        s: steps
        f: features
        b: bits_per_feature

        how they perturb the reservoir if self.I.features == 2:
        1. m paralell samples 
        2. s sequential step sets of inputs re-using w_in
        3. f sequential input sets as per w_in[a_i:b_i, :] (w_in is partitioned per input so two parts in the example)
        4. b simultaneous bits as per w_in[a_i:b_i, :] (if input is the bit string abcd then we perturb with ab first, then cd)
        
        note that x only controls m and s (assume f and b dimensions match self.I.features and self.I.bits_per_feature)
        thus one can change pertubations behaviour via input shape or model configuration
        w_in arbitratily maps bits to input_nodes, so there is no requirement for a 1:1 mapping
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
        self.batch_record(m, phase='init', s=0, f=0)

        # INPUT LAYER
        # ----------------------------------------------------
        x = x.view(m, -1, self.I.features, self.I.bits_per_feature) # if input has more input bits than model expects: loop through these re-using w_in for each pertubation (samples kept in parallel)
        s = x.shape[1]
        f = self.I.features
        for si in range(s):
            a = b = 0
            for fi in range(f):

                # Perturb reservoir nodes with partial input depending on f dimension
                x_i = x[:, si, fi]
                b += self.I.bits_per_feature
                w_in_i = self.w_in[a:b]
                a = b
                input_mask = w_in_i.any(dim=0)
                perturbations = x_i @ w_in_i > 0 # some inputs bits may overlap which nodes are perturbed → counts as a single perturbation
                self.states_parallel[:m, self.input_nodes_mask] = (self.input_pertubation(self.states_parallel[:m, self.input_nodes_mask], perturbations) & input_mask) | (self.states_parallel[:m, self.input_nodes_mask] & ~input_mask)

                # # TODO consider indices instead 
                # # in outer loop: perturbations = x @ self.w_in # not sure this takes away stepping through features...
                # b += self.I.bits_per_feature
                # w_in_i = self.w_in[a:b]
                # selected_input_indices = w_in_i.any(dim=0).nonzero(as_tuple=True)[0]
                # perturbations_i = perturbations[a:b]
                # a = b
                # x_i = x[:, si, fi, selected_input_indices]
                # input_nodes[selected_input_indices] = self.input_pertubation(input_nodes[selected_input_indices], perturbations_i)

                self.batch_record(m, phase='input_layer', s=si+1, f=fi+1)

                # RESERVOIR LAYER
                # ----------------------------------------------------
                # Gather the states based on the adj_list
                neighbour_states_paralell = self.states_parallel[:m, self.adj_list]

                # Apply mask as the adj_list has invalid connections due to homogenized tensor
                neighbour_states_paralell &= self.adj_list_mask 

                idx = self.bin2int(neighbour_states_paralell)

                # Update the state with LUT for each node
                # TODO more complicated than necesarry - no neighbours defaults in the first LUT entry → fix by no_neighbours_mask
                states_parallel = self.lut[self.node_indices, idx]
                states_parallel[:, self.no_neighbours_mask] = self.states_parallel[:m, self.no_neighbours_mask]
                self.states_parallel[:m] = states_parallel

                if not ((si == s - 1) and (fi == f - 1)): # skip last recording, as this is output_layer
                    self.batch_record(m, phase='reservoir_layer', s=si+1, f=fi+1)
        self.batch_record(m, phase='output_layer', s=s, f=f)

        # READOUT LAYER
        # ----------------------------------------------------
        outputs = self.readout(self.states_parallel[:m, self.output_nodes_mask].float())
        if self.output_activation:
            outputs = self.output_activation(outputs)
        return outputs 

    def batch_record(self, m, **meta_data):
        if self.record_history and meta_data:
            self.history.append_batch(self.states_parallel[:m], meta_data)
   

if __name__ == '__main__':
    I = InputParams(
        pertubation='override', 
        encoding='base2', 
        features=2,
        bits_per_feature=4,
        redundancy=2, # no effect here since we are not using encoding (mapping floats to binary)
        n_nodes=8,
        )
    R = ReservoirParams(
        n_nodes=10,
        k_min=0,
        k_avg=2,
        k_max=5,
        p=0.5,
        self_loops=0.1,
        )
    O = OutputParams(features=2)
    T = TrainingParams(batch_size=3,
        epochs=10,
        accuracy_threshold=0.05,
        learning_rate=0.001)
    L = LoggingParams(out_path='/tmp/boolean_reservoir/out/test/', history=HistoryParams(record_history=True, buffer_size=10))

    model_params = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O, training=T)
    params = Params(model=model_params, logging=L)
    model = BooleanReservoir(params)

    # test forward pass w. fake data. s steps per sample
    s = 2
    x = torch.randint(0, 2, (T.batch_size, s, I.features, I.bits_per_feature,), dtype=torch.uint8)
    model(x)
    print(model(x).detach().numpy())
    model.flush_history()
    model.save()
    load_dict, history, meta, expanded_meta = BatchedTensorHistoryWriter(L.save_path / 'history').reload_history()
    print(history[expanded_meta[expanded_meta['phase'] == 'init'].index].shape)
    print(history.shape)
    print(meta)
