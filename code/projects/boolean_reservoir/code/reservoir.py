import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import networkx as nx
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.luts import lut_random 
from projects.boolean_reservoir.code.graphs import generate_adjacency_matrix, graph2adjacency_list_incoming, random_constrained_stub_matching, constrain_degree_of_bipartite_mapping
from projects.boolean_reservoir.code.utils import set_seed
from projects.boolean_reservoir.code.reservoir_utils import ExpressionEvaluator, InputPerturbationStrategy, OutputActivationStrategy, BatchedTensorHistoryWriter, SaveAndLoadModel
import numpy as np


class BooleanReservoir(nn.Module):
    # load_path can be a yaml file or a checkpoint directory
    # a yaml file doesnt load any parameters while a checkpoint does
    # params can be used to override stuff in conjunction with load_path
    def __init__(self, params: Params=None, load_path=None, load_dict=dict()):
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
            self.w_in = SaveAndLoadModel.load_or_generate('w_in', load_dict, lambda:
                self.load_torch_tensor(self.I.w_in) if self.I.w_in
                  else self.bipartite_mapping_strategy(self.I.distribution, self.I.bits, self.I.n_nodes)
            )
            # TODO optionally control each quadrant individually
            w_ii = w_ir = None
            if 'graph' not in load_dict: # generated here due to random seed
                w_ii = torch.zeros((self.I.n_nodes, self.I.n_nodes))  # TODO no effect atm - add flexibility for connections betwen input nodes (I→I)
                # w_ii = self.bipartite_mapping_strategy(self.I.connection, self.I.n_nodes, self.I.n_nodes)
                w_ir = self.bipartite_mapping_strategy(self.I.connection, self.I.n_nodes, self.R.n_nodes)
            self.node_indices = torch.arange(self.M.n_nodes)
            self.input_nodes_mask = torch.zeros(self.M.n_nodes, dtype=bool)
            self.input_nodes_mask[:self.I.n_nodes] = True

            # RESERVOIR LAYER
            # Properties of reservoir nodes that dont receive input (R)
            set_seed(self.R.seed)
            # Graph is divided into nodes that recieve input (I) and nodes that dont (R)
            # This forms four partitions when constructing the adjacency matrix that can be fine-tuned
            # TODO optionally control each quadrant individually
            w_ri = w_rr = None
            if 'graph' not in load_dict: # generated here due to random seed
                w_ri = torch.zeros((self.R.n_nodes, self.I.n_nodes))
                # w_ri = self.bipartite_mapping_strategy(self.I.connection, self.R.n_nodes, self.I.n_nodes)
                w_rr = torch.tensor(generate_adjacency_matrix(self.R.n_nodes, k_min=self.R.k_min, k_avg=self.R.k_avg, k_max=self.R.k_max, self_loops=self.R.self_loops))
            self.graph = SaveAndLoadModel.load_or_generate('graph', load_dict, lambda:
                self.build_graph_from_quadrants(w_ii, w_ir, w_ri, w_rr)
            )
            assert self.M.n_nodes == self.graph.number_of_nodes()
            self.adj_list = graph2adjacency_list_incoming(self.graph)

            # Precompute adj_list and expand it to the batch size
            self.adj_list, self.adj_list_mask, self.no_neighbours_mask = self.homogenize_adj_list(self.adj_list, max_length=self.R.k_max)
            self.max_connectivity = self.adj_list.shape[-1] # Note that k_max may be larger than max_connectivity
            self.no_neighbours_indices = self.no_neighbours_mask.nonzero(as_tuple=True)[0]

            # Each node as a LUT of length 2**k where next state is looked up by index determined by neighbour state
            # LUT allows incoming edges from I + R for each node
            self.lut = SaveAndLoadModel.load_or_generate('lut', load_dict, lambda:
                lut_random(self.M.n_nodes, self.max_connectivity, p=self.R.p).to(torch.uint8)
            )

            # Precompute bin2int conversion mask 
            self.powers_of_2 = self.make_minimal_powers_of_two(self.max_connectivity)

            # Initialize state (R + I)
            self.states_parallel = None
            self.initial_states = SaveAndLoadModel.load_or_generate('init_state', load_dict, lambda:
                self.initialization_strategy(self.R.init)
            )
            self.reset_reservoir(self.T.batch_size)
            
            # OUTPUT LAYER
            set_seed(self.O.seed)
            self.readout = nn.Linear(self.R.n_nodes, self.O.n_outputs) # readout of R only TODO add option to choose
            if 'weights' in load_dict:
                self.load_state_dict(load_dict['weights'])
            self.output_activation = self.output_activation_strategy(self.O.activation)
            self.output_nodes_mask = ~self.input_nodes_mask # TODO add option to choose with w_out (bit mask or mapping)
            set_seed(self.R.seed)

            # LOGGING
            self.out_path = self.L.out_path
            self.timestamp_utc = SaveAndLoadModel.get_timestamp_utc()
            self.save_path = self.out_path / 'runs' / self.timestamp_utc 
            self.L.save_path = self.save_path
            self.L.timestamp_utc = self.timestamp_utc
            self.L.history.save_path = self.save_path / 'history'
            self.record_history = self.L.history.record_history
            self.history = BatchedTensorHistoryWriter(save_path=self.L.history.save_path, buffer_size=self.L.history.buffer_size) if self.record_history else None
            self.device = None
    
    def to(self, device=None, dtype=None, non_blocking=False):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        if device is not None:
            self.device = device
            self.move_to_device(device)
        return self
    
    def move_to_device(self, device):
        attributes = [
            'w_in',
            'node_indices',
            'input_nodes_mask',
            'adj_list',
            'adj_list_mask',
            'no_neighbours_indices',
            'lut',
            'powers_of_2',
            'initial_states',
            'output_nodes_mask',
            'states_parallel'
        ]
        list(map(lambda attr: setattr(self, attr, getattr(self, attr).to(device)), attributes))

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

    def bipartite_mapping_strategy(self, strategy_parameters_str: str, a: int, b:int):
        # produce w [axb] matrix → bipartite map, by default a maps to b but this can be inverted by swaping a and b in the input string
        # notation I: a and b get their names from mapping notation; number of nodes on the left and right side respectively in a bipartite map
        # notation II: k is the edge count per node (array)
        expression_evaluator = ExpressionEvaluator(self.P, {'a': a, 'b': b})
        if '-' in strategy_parameters_str:
            strategy, parameters = strategy_parameters_str.split('-', 1)
            parameters = parameters.split(':')
        else:
            strategy = strategy_parameters_str
            parameters = None
        w = None
        if strategy == 'identity':
            w = np.eye(a, b)
        elif strategy == 'stub':
            # idea: make a determinisitic bipartite graph + a probabilitic mapping
            # constrain in and out degree simultaneously
            # the deterministic mapping is from a_min:b_min, the probabilistic increments this to a_max, b_max by probability p

            # Parse a_min, a_max, b_min, b_max, p 
            assert len(parameters) == 5, "a→b [{a}x{b}] - Strategy must have format 'stub-a_min:a_max:b_min:b_max:p'"
            a_min_expr, a_max_expr, b_min_expr, b_max_expr, p_expr = parameters
            a_min = int(expression_evaluator.to_float(a_min_expr))
            a_max = int(expression_evaluator.to_float(a_max_expr))
            b_min = int(expression_evaluator.to_float(b_min_expr))
            b_max = int(expression_evaluator.to_float(b_max_expr))
            p = expression_evaluator.to_float(p_expr)
            assert 0 <= p <= 1, f"Probability must be between 0 and 1, got {p} from expression '{p_expr}'"
            assert 0 <= b_min <= b_max <= a, f'a→b [{a}x{b}] - incoming connections per node in b: 0 <= {b_min} (b_min) <= {b_max} (b_max) <= {a} (a)'
            assert 0 <= a_min <= a_max <= b, f'a→b [{a}x{b}] - outgoing connections per node in a: 0 <= {a_min} (a_min) <= {a_max} (a_max) <= {b} (b)'

            w = random_constrained_stub_matching(a, b, a_min, a_max, b_min, b_max, p)
        elif strategy in {'in', 'out'}:
            # idea: make a determinisitic bipartite graph + a probabilitic mapping
            # constrain either in or out degree
            # the deterministic mapping is to satisfy min_degree, the probabilistic increments this to max_degree by probability p

            # Parse b_min, b_max, p 
            assert len(parameters) == 3, "Strategy must have format 'in-min_degree:max_degree:p'"
            min_degree_expr, max_degree_expr, p_expr = parameters
            min_degree = int(expression_evaluator.to_float(min_degree_expr))
            max_degree = int(expression_evaluator.to_float(max_degree_expr))
            p = expression_evaluator.to_float(p_expr)
            assert 0 <= p <= 1, f"Probability must be between 0 and 1, got {p} from expression '{p_expr}'"

            if strategy == 'in':
                assert 0 <= min_degree <= max_degree <= a, f'a→b [{a}x{b}] - incoming connections per node in b: 0 <= {min_degree} (b_min) <= {max_degree} (b_max) <= {a} (a)'
            elif strategy == 'out':
                assert 0 <= min_degree <= max_degree <= b, f'a→b [{a}x{b}] - outgoing connections per node in a: 0 <= {min_degree} (a_min) <= {max_degree} (a_max) <= {b} (b)'
            in_degree = strategy == 'in'
            w = constrain_degree_of_bipartite_mapping(a, b, min_degree, max_degree, p, in_degree=in_degree)
        if w is not None:
            return torch.tensor(w, dtype=torch.uint8)
        raise ValueError
 
    @staticmethod
    def input_pertubation_strategy(strategy:str):
        return InputPerturbationStrategy.get(strategy)
    
    @staticmethod
    def output_activation_strategy(strategy:str):
        return OutputActivationStrategy.get(strategy)
    
    @staticmethod
    def get_timestamp_utc():
        return SaveAndLoadModel.get_timestamp_utc()

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.save_path

        paths, checkpoint_path = SaveAndLoadModel.save_model({
            'P': self.P,
            'w_in': self.w_in,
            'graph': self.graph,
            'initial_states': self.initial_states,
            'lut': self.lut,
            'state_dict': self.state_dict,
            'save_path': save_path,
            'history': self.L.history,
        })

        self.L.last_checkpoint = checkpoint_path
        return paths

    def load(self, checkpoint_path:Path=None, paths:dict=None, parameter_override:Params=None):
        load_dict = SaveAndLoadModel.load(checkpoint_path=checkpoint_path, paths=paths, parameter_override=parameter_override)
        self.__init__(load_dict=load_dict)
        return load_dict

    def flush_history(self):
        if self.history:
            self.history.flush()
    
    def bin2int(self, x): # TODO consider bit packig?
        vals = (x * self.powers_of_2).sum(dim=-1)
        return vals

    def reset_reservoir(self, samples): # TODO could lead to bug if batch has say 1 samples then 100. you need a state to continue from.
        self.states_parallel = self.initial_states.repeat(samples, 1)
   
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
        2. s sequential step sets of inputs re-using w_in. Ie. if s>1, then w_in is re-used s times.
        3. f sequential input sets as per w_in[a_i:b_i, :] (w_in is partitioned per input so two parts in the example)
        4. b simultaneous bits as per w_in[a_i:b_i, :] (if input is the bit string abcd then we perturb with ab first, then cd)
        
        note that x only controls m and s (assume f and b dimensions match self.I.features and self.I.bits_per_feature)
        thus one can change pertubations behaviour via input shape or model configuration
        w_in can arbitratily map bits to input_nodes, so there is no requirement for a 1:1 mapping
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

                # # Perturb reservoir nodes with partial input depending on f dimension
                x_i = x[:m, si, fi]
                b += self.I.bits_per_feature
                w_in_i = self.w_in[a:b]
                a = b
                selected_input_indices = w_in_i.any(dim=0).nonzero(as_tuple=True)[0]
                perturbations_i = (x_i.to(torch.float16) @ w_in_i.to(torch.float16)) > 0 # some inputs bits may overlap which nodes are perturbed → counts as a single perturbation, TODO gpu doesnt like uint8...
                self.states_parallel[:m, selected_input_indices] = self.input_pertubation(self.states_parallel[:m, selected_input_indices], perturbations_i[:, selected_input_indices]).to(torch.uint8)

                self.batch_record(m, phase='input_layer', s=si+1, f=fi+1)

                # RESERVOIR LAYER
                # ----------------------------------------------------
                # Gather the states based on the adj_list
                neighbour_states_paralell = self.states_parallel[:m, self.adj_list]

                # Apply mask as the adj_list has invalid connections due to homogenized tensor
                neighbour_states_paralell &= self.adj_list_mask 

                idx = self.bin2int(neighbour_states_paralell)

                # Update the state with LUT for each node
                # no neighbours defaults in the first LUT entry → fix by no_neighbours_indices
                states_parallel = self.lut[self.node_indices, idx]
                states_parallel[:, self.no_neighbours_indices] = self.states_parallel[:m, self.no_neighbours_indices]
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
