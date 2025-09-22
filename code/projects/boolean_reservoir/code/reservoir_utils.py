import sympy
import pandas as pd
from projects.boolean_reservoir.code.parameters import * 
import torch
from projects.boolean_reservoir.code.utils import print_pretty_binary_matrix, override_symlink
from datetime import datetime, timezone
import networkx as nx
import gzip

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def reload_history(self, history_path=None, checkpoint_path=None, include={}, exclude={}):
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

        df = combined_meta_data
        expanded_meta_data = df.loc[df.index.repeat(df['samples'])].reset_index(drop=True)
        expanded_meta_data['sample_id'] = expanded_meta_data.groupby(['phase', 's', 'f']).cumcount()
        expanded_meta_data.drop(columns=['samples'], inplace=True)

        checkpoint_path = checkpoint_path if checkpoint_path else history_path / 'checkpoint'
        load_dict = dict()
        if checkpoint_path.exists():
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
            'weights': lambda path: torch.save(config['state_dict'](), path),
        }

        for key in P.L.save_keys:
            if key not in save_map:
                raise KeyError(f"Unsupported key in save_keys: '{key}'")
            save_map[key](paths[key])

        override_symlink(checkpoint_path.name, checkpoint_path.parent / 'last_checkpoint')

        history = config.get('history')
        if history and history.record_history:
            history.save_path.mkdir(parents=True, exist_ok=True)
            override_symlink(Path('../checkpoints') / checkpoint_path.name, history.save_path / 'checkpoint')

        return paths, checkpoint_path

    @staticmethod
    def save_graph(path, graph):
        with gzip.open(path, 'wb') as f:
            nx.write_graphml(graph, f)

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
        with gzip.open(path, 'rb') as f:
            graph = nx.read_graphml(f) 
            graph = nx.relabel_nodes(graph, lambda x: int(x)) 
        return graph

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
        files.append(('graph', 'graphml.gz'))
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
