import pandas as pd
from project.boolean_reservoir.code.parameter import * 
import torch
from torch.nn.utils.rnn import pad_sequence
from project.boolean_reservoir.code.utils.utils import print_pretty_binary_matrix, override_symlink
from project.boolean_reservoir.code.utils.param_utils import ExpressionEvaluator 
from project.boolean_reservoir.code.graph import random_constrained_stub_matching, constrain_degree_of_bipartite_mapping
from datetime import datetime, timezone
import pickle
import re
import numpy as np

class BatchedTensorHistoryWriter:
    def __init__(self, save_path='history', buffer_size=64, persist_to_disk=True):
        self.save_path = Path(save_path)
        self.file_index = 0
        self.time = 0
        self.buffer_size = buffer_size
        self.persist_to_disk = persist_to_disk
        self.buffer = []
        self.meta_buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def append_batch(self, batch_tensor, meta_data):
        self.buffer.append(batch_tensor.clone().cpu())
        self.meta_buffer.append(meta_data)
        meta_data['file_idx'] = self.file_index 
        meta_data['batch_number'] = len(self.meta_buffer)
        meta_data['samples'] = len(batch_tensor)
        meta_data['time'] = self.time
        self.time += 1
        if self.persist_to_disk and len(self.buffer) >= self.buffer_size:
            self._write_buffer()
    
    def _write_buffer(self):
        if not self.buffer or not self.persist_to_disk:
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
        if self.persist_to_disk:
            self._write_buffer()
    
    def reload_history(self, history_path=None, checkpoint_path=None, include={}, exclude={}):
        if self.persist_to_disk:
            # Read from disk
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
            combined_data = torch.cat(all_data, dim=0).to(self.device)
            combined_meta_data = pd.concat(all_meta_data, ignore_index=True, axis=0)
        else:
            # Read from memory
            combined_data = torch.cat(self.buffer, dim=0).to(self.device)
            combined_meta_data = pd.DataFrame(self.meta_buffer)

        # Shared logic
        df = combined_meta_data
        expanded_meta_data = df.loc[df.index.repeat(df['samples'])].reset_index(drop=True)
        expanded_meta_data['sample_id'] = expanded_meta_data.groupby(['phase', 's', 'f']).cumcount()
        expanded_meta_data.drop(columns=['samples'], inplace=True)
        
        checkpoint_path = checkpoint_path if checkpoint_path else self.save_path / 'checkpoint'
        load_dict = dict()
        if self.persist_to_disk and checkpoint_path.exists():
            load_dict = SaveAndLoadModel.load_from_path_dict_or_checkpoint_folder(checkpoint_path=checkpoint_path, load_key_include_set=include, load_key_exclude_set=exclude)
        
        return load_dict, combined_data, expanded_meta_data, combined_meta_data

class SaveAndLoadModel:
    @staticmethod
    def get_timestamp_utc():
        return datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S_%f")

    @staticmethod 
    def load_or_generate(load_key: str, load_dict: dict, generator: callable):
        if load_key in load_dict:
            return load_dict[load_key]
        else:
            return generator()

    @staticmethod
    def save_model(config):
        P = config['P']
        if P.L.save_keys is None:
            return dict(), None
        save_path = config['save_path']
        checkpoint_path = save_path / 'checkpoints' / SaveAndLoadModel.get_timestamp_utc() 
        checkpoint_path.mkdir(parents=True, exist_ok=False)
        override_symlink(save_path.name, save_path.parent / 'last_run')

        all_paths = SaveAndLoadModel.make_load_path_dict(checkpoint_path)
        paths = {k: all_paths[k] for k in P.L.save_keys if k in all_paths}

        save_map = {
            'parameters': lambda path: save_yaml_config(P, path),
            'w_in': lambda path: torch.save(config['w_in'], path),
            'graph': lambda path: SaveAndLoadModel.save_graph(path, config['graph']),
            'init_state': lambda path: torch.save(config['initial_states'], path),
            'lut': lambda path: torch.save(config['lut'], path),
            'weights': lambda path: torch.save(  # TODO this can be expanded to save any state params also registered buffers 
                {k: v for k, v in config['state_dict']().items() 
                if k.startswith('readout.')},
                path
            )
        }

        for key in P.L.save_keys:
            if key not in save_map:
                raise KeyError(f"Unsupported key in save_keys: '{key}'")
            save_map[key](paths[key])

        override_symlink(checkpoint_path.name, checkpoint_path.parent / 'last_checkpoint')

        history = config.get('history')
        if history and history.record:
            history.save_path.mkdir(parents=True, exist_ok=True)
            override_symlink(Path('../checkpoints') / checkpoint_path.name, history.save_path / 'checkpoint')

        return paths, checkpoint_path


    @staticmethod
    def save_graph(path, graph):
        with open(path, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

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
            path_dict = SaveAndLoadModel.make_load_path_dict(checkpoint_path)

        if load_key_include_set is not None:
            path_dict = {k: path_dict[k] for k in load_key_include_set}

        if load_key_exclude_set is not None:
            path_dict = {k: v for k, v in path_dict.items() if k not in load_key_exclude_set}

        d = dict() 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_map = {
            'parameters': lambda path: load_yaml_config(path),
            'w_in': lambda path: SaveAndLoadModel.load_torch_tensor(path, device),
            'graph': SaveAndLoadModel.load_graph,
            'init_state': lambda path: SaveAndLoadModel.load_torch_tensor(path, device),
            'lut': lambda path: SaveAndLoadModel.load_torch_tensor(path, device),
            'weights': lambda path: SaveAndLoadModel.load_torch_tensor(path, device),
        }
        for key, path in path_dict.items():
            if key not in load_map:
                raise KeyError(f"Unsupported key in path_dict: '{key}'")
            if path.exists():
                loader = load_map[key]
                d[key] = loader(path)
            else:
                print(f"Warning: Model object key '{key}' does not exist at '{path}' (A replacement may be generated)")
        return d
    
    def load(checkpoint_path:Path=None, paths:dict=None, parameter_override:Params=None):
        load_dict = SaveAndLoadModel.load_from_path_dict_or_checkpoint_folder(path_dict=paths, checkpoint_path=checkpoint_path)
        if parameter_override:
            load_dict['parameters'] = parameter_override
        return load_dict

    @staticmethod
    def load_graph(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_torch_tensor(path, device=None):
        tensor = torch.load(path, weights_only=True, map_location=device)
        return tensor

    @staticmethod
    def make_load_path_dict(folder_path):
        folder_path = Path(folder_path)
        paths = dict()
        files = []
        files.append(('parameters', 'yaml'))
        files.append(('w_in', 'pt'))
        files.append(('graph', 'pkl'))
        files.append(('init_state', 'pt'))
        files.append(('lut', 'pt'))
        files.append(('weights', 'pt'))
        paths = {file: folder_path / f'{file}.{filetype}' for file, filetype in files}
        return paths


class InputPerturbationStrategy: # assumes states is only input nodes
    @staticmethod
    def xor(states, perturbations):
        return states ^ perturbations

    @staticmethod
    def and_(states, perturbations):  # 'and' is a keyword in Python
        return states & perturbations

    @staticmethod
    def or_(states, perturbations):  # 'or' is also a keyword
        return states | perturbations

    @staticmethod
    def override(states, perturbations):
        return perturbations

    @staticmethod
    def get(strategy: str):
        strategies = {
            'xor': InputPerturbationStrategy.xor,
            'and': InputPerturbationStrategy.and_,
            'or': InputPerturbationStrategy.or_,
            'override': InputPerturbationStrategy.override
        }
        if strategy not in strategies:
            raise ValueError(f'Unknown perturbation strategy: {strategy}')
        return strategies[strategy] 

class InitializationStrategy:
    @staticmethod
    def random(n_nodes):
        return torch.randint(0, 2, (1, n_nodes), dtype=torch.uint8)
    
    @staticmethod
    def zeros(n_nodes):
        return torch.zeros((1, n_nodes), dtype=torch.uint8)
    
    @staticmethod
    def ones(n_nodes):
        return torch.ones((1, n_nodes), dtype=torch.uint8)
    
    @staticmethod
    def every_other(n_nodes):
        states = InitializationStrategy.zeros(n_nodes)
        states[0, ::2] = 1
        return states
    
    @staticmethod
    def get(strategy: str):
        strategies = {
            'random': InitializationStrategy.random,
            'zeros': InitializationStrategy.zeros,
            'ones': InitializationStrategy.ones,
            'every_other': InitializationStrategy.every_other
        }
        if strategy not in strategies:
            raise ValueError(f'Unknown initialization strategy: {strategy}')
        return strategies[strategy]

class OutputActivationStrategy:
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def sigmoid(x):
        return torch.sigmoid(x)

    @staticmethod
    def get(strategy: str):
        strategies = {
            None: OutputActivationStrategy.identity,
            'sigmoid': OutputActivationStrategy.sigmoid
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown output activation strategy: {strategy}")
        return strategies[strategy]

class ChainedSelector:
    """
    Flexible selection of a range via chainable operations.
    
    Examples:
        ChainedSelector(100).eval('F i<20 -> S -5:')  # Filter then slice last 5
        ChainedSelector(100, min_val=20).eval('S 20:80 -> F i%2 == 0')  # Slice then filter evens
        ChainedSelector(100).eval('F i>10 -> F i<50 -> F i%3 == 0')  # Chain filters
        ChainedSelector(100).eval('F i<50 -> R 5')  # Filter then random sample 5
    """
    
    def __init__(self, max_val: int, min_val: int = 0, var: str = 'i', parameters: dict | None = None):
        self.max_val = max_val
        self.min_val = min_val
        self.var = var
        self.parameters = parameters or {}
        if var in self.parameters:
            raise ValueError(f"Variable '{var}' cannot be in parameters dict")
        self.parameters['min'] = min_val
        self.parameters['max'] = max_val
    
    def eval(self, chain: str) -> torch.Tensor:
        s = pd.Series(range(self.min_val, self.max_val), name=self.var)
        chain = chain.strip()
        if not chain:
            return torch.from_numpy(s.values).long()
        
        def substitute(expr: str) -> str:
            for key, val in self.parameters.items():
                expr = expr.replace(key, str(val))
            return expr
        
        for link in chain.split('->'):
            link = link.strip()
            if not link:
                continue
            match = re.match(r'^([A-Za-z]+)\s+(.*)$', link)
            if not match:
                raise ValueError(f"Invalid step syntax: {link}")
            tag, expr = match.groups()
            expr = substitute(expr.strip())
            tag = tag.upper()
            if tag == 'F':
                s = s[s.eval(expr)]
            elif tag == 'S':
                parts = [int(p) if p else None for p in expr.split(':')]
                slc = slice(*parts)
                s = s.iloc[slc]
            elif tag == 'R':
                n = int(expr) if expr else len(s)
                s = s.sample(n=min(n, len(s)))
            else:
                raise ValueError(f"Unknown step tag: {tag}")
        return torch.from_numpy(s.values).long()

class BipartiteMappingStrategy:
    """Produce w [axb] matrix → bipartite map.
    
    By default a maps to b, but this can be inverted by swapping a and b in the input string.
    
    Notation I: a and b get their names from mapping notation; number of nodes on the 
    left and right side respectively in a bipartite map.
    
    Notation II: k is the edge count per node (array).
    """
    
    @staticmethod
    def get(strategy_str: str):
        """Returns a callable that handles parameter parsing internally.
        
        Strategy string formats:
        - 'identity': no parameters
        - 'stub-a_min:a_max:b_min:b_max:p'
        - 'in-min_degree:max_degree:p'
        - 'out-min_degree:max_degree:p'
        
        Usage:
            strategy_fn = BipartiteMappingStrategy.get('stub-1:5:2:3:0.5')
            result = strategy_fn(p, a, b)
        """
        # Parse strategy name and parameters
        if '-' in strategy_str:
            strategy_name, parameters_str = strategy_str.split('-', 1)
        else:
            strategy_name = strategy_str
            parameters_str = None
        
        # Map strategy name to its implementation
        strategy_map = {
            'identity': BipartiteMappingStrategy._identity,
            'stub': BipartiteMappingStrategy._stub,
            'in': BipartiteMappingStrategy._in_degree,
            'out': BipartiteMappingStrategy._out_degree,
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown bipartite mapping strategy: {strategy_name}")
        
        strategy_fn = strategy_map[strategy_name]
        
        # Return a wrapper that includes the parsed parameters
        def wrapped(p: Params, a: int, b: int):
            return strategy_fn(p, a, b, parameters_str)
        
        return wrapped

    @staticmethod
    def _identity(p: Params, a: int, b: int, parameters_str: str = None):
        """Identity strategy: repeating eye matrix."""
        def repeating_eye(a, b):
            w = np.zeros((a, b))
            w[np.arange(a), np.arange(a) % b] = 1
            return w
        return torch.tensor(repeating_eye(a, b), dtype=torch.uint8)

    @staticmethod
    def _stub(p: Params, a: int, b: int, parameters_str: str):
        """Stub strategy: deterministic + probabilistic bipartite graph.
        
        Parameters format: 'a_min:a_max:b_min:b_max:p'
        """
        expression_evaluator = ExpressionEvaluator({'a': a, 'b': b})
        params = parameters_str.split(':')
        
        assert len(params) == 5, "Stub strategy must have format 'a_min:a_max:b_min:b_max:p'"
        a_min_expr, a_max_expr, b_min_expr, b_max_expr, p_expr = params
        
        a_min = int(expression_evaluator.eval(a_min_expr))
        a_max = int(expression_evaluator.eval(a_max_expr))
        b_min = int(expression_evaluator.eval(b_min_expr))
        b_max = int(expression_evaluator.eval(b_max_expr))
        p = expression_evaluator.eval(p_expr)
        
        assert 0 <= p <= 1, f"Probability must be between 0 and 1, got {p} from expression '{p_expr}'"
        assert 0 <= b_min <= b_max <= a, f'a→b [{a}x{b}] - incoming connections per node in b: 0 <= {b_min} (b_min) <= {b_max} (b_max) <= {a} (a)'
        assert 0 <= a_min <= a_max <= b, f'a→b [{a}x{b}] - outgoing connections per node in a: 0 <= {a_min} (a_min) <= {a_max} (a_max) <= {b} (b)'
        
        w = random_constrained_stub_matching(a, b, a_min, a_max, b_min, b_max, p)
        return torch.tensor(w, dtype=torch.uint8)

    @staticmethod
    def _constrain_degree(p: Params, a: int, b: int, parameters_str: str, in_degree: bool):
        """Shared logic for in-degree and out-degree strategies.
        
        Parameters format: 'min_degree:max_degree:p'
        """
        expression_evaluator = ExpressionEvaluator({'a': a, 'b': b})
        params = parameters_str.split(':')
        
        assert len(params) == 3, "Degree strategies must have format 'min_degree:max_degree:p'"
        min_degree_expr, max_degree_expr, p_expr = params
        
        min_degree = int(expression_evaluator.eval(min_degree_expr))
        max_degree = int(expression_evaluator.eval(max_degree_expr))
        p = expression_evaluator.eval(p_expr)
        
        assert 0 <= p <= 1, f"Probability must be between 0 and 1, got {p} from expression '{p_expr}'"
        
        if in_degree:
            assert 0 <= min_degree <= max_degree <= a, f'a→b [{a}x{b}] - incoming connections per node in b: 0 <= {min_degree} (min) <= {max_degree} (max) <= {a} (a)'
        else:
            assert 0 <= min_degree <= max_degree <= b, f'a→b [{a}x{b}] - outgoing connections per node in a: 0 <= {min_degree} (min) <= {max_degree} (max) <= {b} (b)'
        
        w = constrain_degree_of_bipartite_mapping(a, b, min_degree, max_degree, p, in_degree=in_degree)
        return torch.tensor(w, dtype=torch.uint8)

    @staticmethod
    def _in_degree(p: Params, a: int, b: int, parameters_str: str):
        """In-degree strategy: constrain incoming connections per node."""
        return BipartiteMappingStrategy._constrain_degree(p, a, b, parameters_str, in_degree=True)

    @staticmethod
    def _out_degree(p: Params, a: int, b: int, parameters_str: str):
        """Out-degree strategy: constrain outgoing connections per node."""
        return BipartiteMappingStrategy._constrain_degree(p, a, b, parameters_str, in_degree=False)

def homogenize_adj_list(adj_list, max_length): # tensor may have less than max_length columns if not needed
    adj_list_tensors = [torch.tensor(sublist, dtype=torch.long) for sublist in adj_list]
    padded_tensor = pad_sequence(adj_list_tensors, batch_first=True, padding_value=-1)
    padded_tensor = padded_tensor[:, :max_length]
    valid_mask = padded_tensor != -1
    padded_tensor[~valid_mask] = 0
    no_neighbours_mask = ~valid_mask.any(dim=1) # if all neigbours are off its not the same as having no neighbours
    return padded_tensor, valid_mask, no_neighbours_mask