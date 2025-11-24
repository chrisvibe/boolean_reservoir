import torch
import torch.nn as nn
from pathlib import Path
import networkx as nx
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.luts import lut_random 
from projects.boolean_reservoir.code.graphs import generate_adjacency_matrix, graph2adjacency_list_incoming
from projects.boolean_reservoir.code.utils.utils import set_seed
from projects.boolean_reservoir.code.utils.reservoir_utils import InputPerturbationStrategy, InitializationStrategy, OutputActivationStrategy, BatchedTensorHistoryWriter, SaveAndLoadModel, ChainedSelector, BipartiteMappingStrategy, homogenize_adj_list

class BooleanReservoir(nn.Module):
    # load_path can be a yaml file or a checkpoint directory
    # a yaml file doesnt load any parameters while a checkpoint does
    # params can be used to override stuff in conjunction with load_path
    def __init__(self, params: Params=None, load_path=None, load_dict=dict(), ignore_gpu=False):
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
            self.input_pertubation = self.input_pertubation_strategy(self.P)
            w_in = SaveAndLoadModel.load_or_generate('w_in', load_dict, lambda:
                self.load_torch_tensor(self.I.w_in) if self.I.w_in
                  else self.bipartite_mapping_strategy(self.P, self.I.w_ir, self.I.bits, self.I.n_nodes)
            )
            # TODO optionally control each quadrant individually
            w_ii = w_ir = None
            if 'graph' not in load_dict: # generated here due to random seed
                w_ii = torch.zeros((self.I.n_nodes, self.I.n_nodes))  # TODO no effect atm - add flexibility for connections betwen input nodes (I→I)
                # w_ii = self.bipartite_mapping_strategy(self.P, self.I.connection, self.I.n_nodes, self.I.n_nodes)
                w_ir = self.bipartite_mapping_strategy(self.P, self.I.w_ir, self.I.n_nodes, self.R.n_nodes)

            node_indices = torch.arange(self.M.n_nodes)
            cs = ChainedSelector(self.M.n_nodes, parameters={'I': self.M.I.n_nodes})
            input_nodes = cs.eval(self.M.I.selector)
            input_nodes_mask = torch.zeros(self.M.n_nodes, dtype=torch.bool)
            input_nodes_mask[input_nodes] = True
            ticks = torch.tensor([int(c) for c in self.I.ticks])

            # RESERVOIR LAYER
            # Properties of reservoir nodes that dont receive input (R)
            set_seed(self.R.seed)
            # Graph is divided into nodes that recieve input (I) and nodes that dont (R)
            # This forms four partitions when constructing the adjacency matrix that can be fine-tuned
            # TODO optionally control each quadrant individually
            w_ri = w_rr = None
            if 'graph' not in load_dict: # generated here due to random seed
                w_ri = torch.zeros((self.R.n_nodes, self.I.n_nodes))
                # w_ri = self.bipartite_mapping_strategy(self.P, self.I.connection, self.R.n_nodes, self.I.n_nodes)
                w_rr = torch.tensor(generate_adjacency_matrix(self.R.n_nodes, k_min=self.R.k_min, k_avg=self.R.k_avg, k_max=self.R.k_max, self_loops=self.R.self_loops))
            self.graph = SaveAndLoadModel.load_or_generate('graph', load_dict, lambda:
                self.build_graph_from_quadrants(w_ii, w_ir, w_ri, w_rr)
            )
            assert self.M.n_nodes == self.graph.number_of_nodes()

            # Precompute adj_list and expand it to the batch size
            adj_list = graph2adjacency_list_incoming(self.graph)
            adj_list, adj_list_mask, no_neighbours_mask = homogenize_adj_list(adj_list, max_length=self.R.k_max)
            no_neighbours_indices = no_neighbours_mask.nonzero(as_tuple=True)[0]

            # Each node as a LUT of length 2**k where next state is looked up by index determined by neighbour state
            # LUT allows incoming edges from I + R for each node
            self.max_connectivity = adj_list.shape[-1] # Note that k_max may be larger than max_connectivity
            lut = SaveAndLoadModel.load_or_generate('lut', load_dict, lambda:
                lut_random(self.M.n_nodes, self.max_connectivity, p=self.R.p)
            )

            # Precompute bin2int conversion mask 
            powers_of_2 = self.precompute_minimal_powers_of_2(self.max_connectivity) 

            # Initialize state (R + I)
            initial_states = SaveAndLoadModel.load_or_generate('init_state', load_dict, lambda:
                self.initialization_strategy(self.P)
            )
            states_parallel = initial_states.repeat(self.T.batch_size, 1)
            
            # OUTPUT LAYER
            set_seed(self.O.seed)
            self.readout = nn.Linear(self.R.n_nodes, self.O.n_outputs) # readout of R only TODO add option to choose
            if 'weights' in load_dict:
                self.load_state_dict(load_dict['weights'])
            self.output_activation = self.output_activation_strategy(self.P)
            output_nodes_mask = ~input_nodes_mask # TODO add option to choose with w_out (bit mask or mapping)
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
            self.device = torch.device("cuda" if torch.cuda.is_available() and not ignore_gpu else "cpu")

            # TYPE OPTIMIZATION + BUFFER REGISTRATION
            '''
            torch.bool     # masks (logical conditions)
            torch.uint8    # binary states (0/1 values & for boolean arithmetic)
            torch.float32  # arithmetic (profiling atm says this is best) 
            torch.int64    # indices for tensor indexing
            '''

            type_arithmetic = torch.float16 if self.device.type == 'cuda' else torch.float32 # fast according to profile
            type_states = type_arithmetic 
            self.register_buffer("node_indices", node_indices.to(torch.int64)) # indices
            self.register_buffer("input_nodes_mask", input_nodes_mask.to(torch.bool)) # mask
            self.register_buffer("ticks", ticks.to(torch.uint8)) # pure lookup
            self.register_buffer("w_in", w_in.to(torch.uint8)) # arithmetic with input x
            self.register_buffer("adj_list", adj_list.to(torch.int64)) # indices
            self.register_buffer("adj_list_mask", adj_list_mask.to(torch.bool)) # mask
            self.register_buffer("no_neighbours_indices", no_neighbours_indices.to(torch.int64)) # indices
            self.register_buffer("lut", lut.to(type_states)) # assigns to states
            self.register_buffer("powers_of_2", powers_of_2.to(type_arithmetic)) # arithmetic with states
            self.register_buffer("initial_states", initial_states.to(type_states)) # arithmetic with states
            self.register_buffer("states_parallel", states_parallel.to(type_states)) # arithmetic with states
            self.register_buffer("output_nodes_mask", output_nodes_mask.to(torch.bool)) # mask

            # OTHER
            self.add_graph_labels(self.graph)
    
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
        # to get subgraph: ir_subgraph = g.edge_subgraph([(u, v) for u, v, data in g.edges(data=True) if data.get('quadrant') == 'IR'])

    def build_graph_from_quadrants(self, w_ii, w_ir, w_ri, w_rr):
        w_i = torch.cat((w_ii, w_ir), dim=1)
        w_r = torch.cat((w_ri, w_rr), dim=1)
        w = torch.cat((w_i, w_r), dim=0)
        graph = nx.from_numpy_array(w.numpy(), create_using=nx.DiGraph)
        return graph
    
    @staticmethod
    def bipartite_mapping_strategy(p: Params, strategy: str, a: int, b: int):
        strategy_fn = BipartiteMappingStrategy.get(strategy)
        return strategy_fn(p, a, b)

    @staticmethod
    def input_pertubation_strategy(p: Params):
        return InputPerturbationStrategy.get(p.M.I.pertubation)
    
    @staticmethod
    def initialization_strategy(p: Params):
        return InitializationStrategy.get(p.M.R.init)(p.M.n_nodes)

    @staticmethod
    def output_activation_strategy(p: Params):
        return OutputActivationStrategy.get(p.M.O.activation)
    
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

    @staticmethod
    def precompute_minimal_powers_of_2(bits):
        return (2 ** torch.arange(bits, dtype=torch.float32).flip(0))
    
    def bin2int(self, x): # consider making x (states) float32?
        return torch.matmul(x.to(self.powers_of_2.dtype), self.powers_of_2).to(torch.int64)
  
    def reset_reservoir(self, samples):
        self.states_parallel.resize_(samples, self.initial_states.size(-1))
        self.states_parallel.copy_(self.initial_states.repeat(samples, 1))
   
    def forward(self, x):
        '''
        input is reshaped to fit number of input bits in w_in
        ie.
        assume x.shape == mxsxcxb
        m: samples
        s: steps
        c: chunks
        b: chunk size

        how they perturb the reservoir if self.I.chunks == 2:
        1. m paralell samples 
        2. s sequential step sets of inputs re-using w_in. Ie. if s>1, then w_in is re-used s times.
        3. c sequential input sets as per w_in[a_i:b_i, :] (w_in is partitioned per input so two parts in the example)
        4. b simultaneous bits as per w_in[a_i:b_i, :] (if input is the bit string abcd then we perturb with ab first, then cd)
        
        sxcxb should match data after encoding mxsxf (f are floats to be turned into bits)
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
        x = x.view(m, -1, self.I.chunks, self.I.chunk_size) # if input has more input bits than model expects: loop through these re-using w_in for each pertubation (samples kept in parallel)
        s = x.shape[1]
        c = self.I.chunks
        k = self.I.bits // self.I.chunks
        for si in range(s):
            a = b = 0
            for ci in range(c):
                # Perturb reservoir nodes with partial input depending on chunks dimension
                x_i = x[:m, si, ci]
                b += k 
                w_in_i = self.w_in[a:b]
                a = b
                selected_input_indices = w_in_i.any(dim=0).nonzero(as_tuple=True)[0] # TODO can be pre-computed per chunk
                perturbations_i = (x_i.unsqueeze(-1) & w_in_i).any(dim=1).to(torch.uint8) # some inputs bits may overlap which nodes are perturbed → counts as a single perturbation
                # perturbations_i = ((x_i @ w_in_i) > 0).to(torch.uint8)
                input_slice = self.states_parallel[:m, selected_input_indices]
                pert_slice = perturbations_i[:, selected_input_indices]
                self.states_parallel[:m, selected_input_indices] = self.input_pertubation(input_slice, pert_slice).to(self.states_parallel.dtype)

                self.batch_record(m, phase='input_layer', s=si+1, f=ci+1)

                # RESERVOIR LAYER
                # ----------------------------------------------------
                # dynamics loops to digest input
                t = self.ticks[ci]
                for ti in range(t):
                    # Gather the states based on the adj_list
                    neighbour_states_paralell = self.states_parallel[:m, self.adj_list]

                    # Apply mask as the adj_list has invalid connections due to homogenized tensor
                    # neighbour_states_paralell &= self.adj_list_mask 
                    neighbour_states_paralell *= self.adj_list_mask
                    idx = self.bin2int(neighbour_states_paralell)

                    # Update the state with LUT for each node I + R
                    # no neighbours defaults in the first LUT entry → fix by no_neighbours_indices
                    states_parallel = self.lut[self.node_indices, idx]
                    states_parallel[:, self.no_neighbours_indices] = self.states_parallel[:m, self.no_neighbours_indices]
                    self.states_parallel[:m] = states_parallel

                    if si == s - 1 and ci == c - 1 and ti == t - 1:
                        continue  # skip last recording, as this is output_layer

                    self.batch_record(m, phase='reservoir_layer', s=si+1, f=ci+1, t=ti+1)
        self.batch_record(m, phase='output_layer', s=s, f=c)

        # READOUT LAYER
        # ----------------------------------------------------
        if self.O.readout_mode == 'bipolar':
            o = self.states_parallel[:m, self.output_nodes_mask] * 2 - 1 # convert to {-1, 1}
        else:
            o = self.states_parallel[:m, self.output_nodes_mask]
        outputs = self.readout(o.float())
        if self.output_activation:
            outputs = self.output_activation(outputs)
        return outputs 

    def batch_record(self, m, **meta_data):
        if self.record_history:
            self.history.append_batch(self.states_parallel[:m], meta_data)
   

if __name__ == '__main__':
    I = InputParams(
        pertubation='override', 
        encoding='base2', 
        features=2,
        chunk_size=4,
        bits=4, 
        n_nodes=8,
        ticks='2',
        seed=0
        )
    R = ReservoirParams(
        n_nodes=10,
        k_min=0,
        k_avg=7,
        k_max=7,
        p=0.5,
        self_loops=0.1,
        seed=0
        )
    O = OutputParams(features=2, seed=0)
    T = TrainingParams(
        batch_size=3,
        epochs=10,
        accuracy_threshold=0.05,
        learning_rate=0.001, 
        seed=0,
        )
    L = LoggingParams(out_path='/tmp/boolean_reservoir/out/test/', history=HistoryParams(record_history=True, buffer_size=10))
    L = LoggingParams(out_path='/out/delete/', history=HistoryParams(record_history=True, buffer_size=10))

    model_params = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O, training=T)
    params = Params(model=model_params, logging=L)
    model = BooleanReservoir(params)

    # test forward pass w. fake data. s steps per sample
    s = 1
    x = torch.randint(0, 2, (T.batch_size, s, I.features, I.bits // I.features,), dtype=torch.uint8)
    model(x)
    print(model(x).detach().numpy())
    model.flush_history()
    model.save()
    load_dict, history, expanded_meta, meta = BatchedTensorHistoryWriter(L.save_path / 'history').reload_history()
    print(history[meta[meta['phase'] == 'init'].index].shape)
    print(history.shape)
    print(meta)

    from projects.boolean_reservoir.code.visualizations import plot_activity_trace 
    model.flush_history()
    plot_activity_trace(model.save_path, highlight_input_nodes=True, data_filter=lambda df: df, aggregation_handle=lambda df: df[df['sample_id'] == 0])