from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict, field_serializer
import yaml
from itertools import product
from typing import List, Union, Optional, Callable, Dict, Any, Type, Generic, TypeVar
T = TypeVar('T', bound=BaseModel)
from pathlib import Path, PosixPath, WindowsPath
import pandas as pd
from benchmarks.path_integration.parameters import PathIntegrationDatasetParams
from benchmarks.temporal.parameters import TemporalDatasetParams

def pydantic_init():
    def represent_pathlib_path(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

    yaml.add_representer(Path, represent_pathlib_path)
    yaml.add_representer(PosixPath, represent_pathlib_path)
    yaml.add_representer(WindowsPath, represent_pathlib_path)

pydantic_init()

def calculate_w_broadcasting(operator: Callable[[float, float], float], a, b):
    # list-list, list-value, value-list, value-value
    if isinstance(a, list) and isinstance(b, list):
        return [operator(a[i], b[i]) for i in range(len(a))]
    elif isinstance(a, list):
        return [operator(x, b) for x in a]
    elif isinstance(b, list):
        return [operator(a, y) for y in b]
    else:
        return operator(a, b)

class InputParams(BaseModel): # TODO split into Bits layer (B→I) and Input layer (I→R)
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    distribution: Union[str, List[str]] = Field(
        'identity',
        description=(
            "Select a mode for mapping input bits to input nodes. Options include:\n"
            
            "1. **Identity**: Use 'identity' for a straightforward mapping with no probabilistic adjustments.\n"
            "   - Example: 'identity'\n"
            
            "2. **Stub Matching**: Use format 'stub-a_min:a_max:b_min:b_max:p', where:\n"
            "   - stub matching in this context means constraining in-degree of b and out-degrees of a and a solution is not always guaranteed.\n"
            "   - `a_min:a_max` and `b_min:b_max` dictate the mapping from a→b, splitting elements into deterministic (`_min`) and probabilistic (`_max`) parts.\n"
            "   - Values for 'a' and 'b' can be symbolic tokens: 'a': bits, 'b': I, or numeric values directly.\n"
            "   - `p` is the connection probability, a number (0-1) or an expression like '1/b'.\n"
            "   - Example: 'stub-1:10:1:5:0.5' k_a_in is between 1 and 10, k_b_out is between 1 and 5, with a connection probability of 0.5.\n"
            
            "3. **In-Degree**: Use format 'in-b_min:b_max:p', focusing only on the in-degree of 'b', following a similar deterministic/probabilistic approach.\n"
            "   - Example: 'in-2:8:0.8' where nodes have in-degree mapped between 2 to 8 with a probability of 0.8.\n"

            "4. **Out-Degree Mapping**: Use format 'out-a_min:a_max:p', focusing only on the out-degree of 'a', following a similar deterministic/probabilistic approach.\n"
            "   - Example: 'out-2:8:0.8' where nodes have out-degree mapped between 2 to 8 with a probability of 0.8.\n"

            "Note: 'w_in' overrides this field with a manual adjacency matrix. Both aim to map bits to input nodes (bits→I)."
        )
    )
    connection: Union[str, List[str]] = Field('out-1:1:1', description="See distribution property. Produces a biparitite mapping from input nodes to reservoir nodes (I→R) so the following symbolic tokens are replaced accordingly; 'a': I, 'b': R")
    w_in: Optional[Union[Path, List[Path]]] = Field(None, description="Input distribution mapping explicitely set by an adjacency matrix B->I. Parameter is a path to a stored tensor. Overrides distribution parameter")
    pertubation: Union[str, List[str]] = Field('xor', description="Pertubation strategy given old and new states for input nodes")
    encoding: Union[str, List[str]] = Field('base2', description="Binary encoding type")
    interleaving: Union[int, List[int]] = Field(0, description="Multidimensionsional weaving of inputs, int dictates group size. n=1: abc, def -> ad, be, cf -> adb, ecf | n=2: abcd, efgh -> ab, ef, cd, gh -> abef, cdgh")
    n_nodes: Optional[Union[int, List[int]]] = Field(None, description="Number of input nodes; I")
    features: Union[int, List[int]] = Field(None, description="Dimension of input data before binary encoding")
    bits: Optional[Union[int, List[int]]] = Field(None, description="Total bits after encoding")
    resolution: Optional[Union[int, List[int]]] = Field(None, description="Bits per dimension before redundancy")
    redundancy: Union[int, List[int]] = Field(1, description="Redundancy factor of resolution")
    chunks: Optional[Union[int, List[int]]] = Field(None, description="Number of chunks to split bits into")
    chunk_size: Optional[Union[int, List[int]]] = Field(None, description="Bits per chunk. Overridden by chunks")
    ticks: Optional[Union[str, List[str]]] = Field(None, description="Number of ticks or dynamic update steps after a input step. Set as a vector corresponding to each chunk.")

    @model_validator(mode='after')
    def calculate_bits(self):
        """Calculate bits from other parameters if not set"""
        # Calculate from resolution and features
        if (self.bits is None and
            not isinstance(self.features, list) and
            not isinstance(self.resolution, list) and
            not isinstance(self.redundancy, list)):
            
            if self.resolution is not None and self.features is not None:
                self.bits = self.features * self.resolution * self.redundancy
        
        # Also handle the reverse: if bits is set, calculate resolution
        elif (self.bits is not None and 
            self.resolution is None and
            not isinstance(self.bits, list) and
            not isinstance(self.features, list) and
            not isinstance(self.redundancy, list)):
            
            if self.features is not None:
                self.resolution = self.bits // (self.features * self.redundancy)
        
        return self

    # Both set - calculate bits if not set
    @model_validator(mode='after')
    def handle_chunking(self):
        """Handle chunks and chunk_size bidirectionally"""
        if (not isinstance(self.chunks, list) and
            not isinstance(self.chunk_size, list)):
            
            if self.chunks is not None and self.chunk_size is not None:
                if self.bits is None:
                    self.bits = self.chunks * self.chunk_size
            elif self.bits is not None and not isinstance(self.bits, list):
                if self.chunks is not None:
                    self.chunk_size = self.bits // self.chunks
                elif self.chunk_size is not None:
                    self.chunks = self.bits // self.chunk_size
                elif self.features is not None and not isinstance(self.features, list):
                    self.chunks = self.features
                    if self.resolution is not None and not isinstance(self.resolution, list):
                        self.chunk_size = self.resolution * self.redundancy
        return self
    
    @model_validator(mode='after')
    def default_n_nodes(self):
        if self.n_nodes is None:
            self.n_nodes = calculate_w_broadcasting(lambda x, y: x, self.bits, None)
        return self

    @model_validator(mode='after')
    def calculate_ticks(self):
        if self.ticks is None:
            self.ticks = '1' * self.chunks
        self.ticks = self.ticks *  (self.chunks // len(self.ticks))
        return self
    

class ReservoirParams(BaseModel):
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    n_nodes: Optional[Union[int, List[int]]] = Field(None, description="Number of reservoir nodes (R) excluding input nodes (I)")
    k_min: Union[int, List[int]] = Field(0, description="Min degree of incoming nodes")
    k_avg: Union[float, List[float]] = Field(2, description="Average degree of incoming nodes")
    k_max: Union[int, List[int]] = Field(None, description="Maximum degree of incoming nodes")
    mode: Union[str, List[str]] = Field('heterogeneous', description="heterogeneous: each node can have different number of neighbours, homogeneous: each node has k neighbours set by k_avg")
    p: Union[float, List[float]] = Field(0.5, description="Probability for 1 in LUT (look up table)")
    reset: Optional[Union[bool, List[bool]]] = Field(True, description="Reset to init state after each sample")
    self_loops: Optional[Union[float, List[float]]] = Field(None, description="Probability of self-loops in graph; normalized by number of nodes")
    init: Union[str, List[str]] = Field('random', description="Initalization strategy for reservoir node states")

    @model_validator(mode='after')
    def override_for_homogeneous_mode(self):
        if self.mode == 'homogeneous':
            if isinstance(self.k_avg, list):
                self.k_min = self.k_max = 0
            else:
                self.k_min = self.k_max = int(self.k_avg)
        return self

class OutputParams(BaseModel): # TODO add w_out and distribution like in input_layer. atm we assume full readout of R
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    n_outputs: Union[int, List[int]] = Field(1, description="Dimension of output data")
    activation: Optional[Union[str, List[str]]] = Field(None, description="Activation after readout layer, fex sigmoid")
    readout_mode: Optional[Union[str, List[str]]] = Field("binary", description="Encoding of reservoir states for readout: 'binary'={0,1}, 'bipolar'={-1,1}")

class OptimizerParams(BaseModel):
    name: str = Field('adam', description="Optimizer type: 'sgd', 'adam', 'adamw', etc.")
    params: dict = Field(
        default_factory=dict,
        description="Optimizer hyperparameters (e.g., lr, momentum, weight_decay)"
    )
    
    @model_validator(mode='after')
    def normalize_name(self):
        """Convert optimizer name to lowercase for consistency."""
        self.name = self.name.lower()
        return self

class TrainingParams(BaseModel):
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    batch_size: Union[int, List[int]] = Field(32, description="Number of samples per forward pass")
    criterion: Optional[Union[str, List[str]]] = Field('MSE', description="ML criterion, fex MSE, BCE")
    epochs: Union[int, List[int]] = Field(100, description="Number of epochs")
    accuracy_threshold: Union[float, List[float]] = Field(0.5, description="Threshold for generic accuracy metric")
    evaluation: Optional[str] = Field('test', description="test, dev, train etc")
    shuffle: bool = Field(True, description="Shuffle dataset")
    drop_last: bool = Field(True, description="Drop last")
    optim: Union[OptimizerParams, List[OptimizerParams]] = Field(
        default=OptimizerParams(name='adamw', params={'lr': 0.001}),
        description="Optimizer configuration (single or list for grid search)"
    )
    
class ModelParams(BaseModel):
    input_layer: InputParams
    reservoir_layer: ReservoirParams
    output_layer: OutputParams
    training: TrainingParams

    @property
    def I(self):
        return self.input_layer 

    @property
    def R(self):
        return self.reservoir_layer 

    @property
    def O(self):
        return self.output_layer 

    @property
    def T(self):
        return self.training 

    @property
    def n_nodes(self):
        return self.R.n_nodes + self.I.n_nodes 

class GridSearchParams(BaseModel):
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    n_samples: Optional[int] = Field(1, ge=1, description="Number of samples per configuration in grid search")

class HistoryParams(BaseModel):
    record_history: Optional[bool] = Field(False, description="Reservoir dynamics state recording")
    buffer_size: Optional[int] = Field(64, description="Number of batched snapshots per output file")
    save_path: Optional[Path] = Field(Path('out/history'), description="Where model is saved when recording history")

class TrainLog(BaseModel):
    accuracy: Optional[float] = Field(None, description="accuracy")
    loss: Optional[float] = Field(None, description="loss")
    epoch: Optional[int] = Field(None, description="epoch")

class LoggingParams(BaseModel):
    timestamp_utc: Optional[str] = Field(None, description="timestamp utc")
    out_path: Path = Field(Path('out'), description="Where to save all logs for this config")
    save_path: Optional[Path] = Field(Path('out'), description="Where last run was saved")
    last_checkpoint: Optional[Path] = Field(None, description="Where last checkpoint was saved")
    grid_search: Optional[GridSearchParams] = Field(None)
    history: HistoryParams = Field(HistoryParams(), description="Parameters pertaining to recoding of reservoir dynamics")
    train_log: TrainLog = Field(TrainLog())
    save_keys: Optional[List[str]] = Field(
        default=['parameters', 'w_in', 'graph', 'init_state', 'lut', 'weights'],
        description="Only save these model objects",
        json_schema_extra={'expand': False}  # Mark as non-expandable
    )

class Params(BaseModel):
    model: ModelParams
    logging: LoggingParams = Field(LoggingParams())
    dataset: Optional[Union[PathIntegrationDatasetParams, TemporalDatasetParams]] = None

    @property
    def M(self):
        return self.model 

    @property
    def L(self):
        return self.logging 

    @property
    def D(self):
        return self.dataset 

def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    params = Params(**config)
    return params

def save_yaml_config(base_model: BaseModel, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(base_model.model_dump(), file)

def generate_param_combinations(params: BaseModel) -> List[BaseModel]:
    def expand_params(params: Any) -> List[Dict[str, Any]]:
        if isinstance(params, BaseModel):
            params_dict = params.__dict__  # Match original for consistency
            expanded_dict = {}
            for k, v in params_dict.items():
                field_info = params.model_fields.get(k)
                expand = True  # Default to expand (original behavior)
                if field_info and field_info.json_schema_extra is not None:
                    expand = field_info.json_schema_extra.get('expand', True)
                if expand:
                    expanded_dict[k] = expand_params(v)
                else:
                    expanded_dict[k] = [v]  # Treat as single value (keep list as-is)
            keys = expanded_dict.keys()
            values = [expanded_dict[k] for k in keys]
            combinations = product(*values)
            return [dict(zip(keys, combo)) for combo in combinations]
        elif isinstance(params, list):
            return params
        return [params]

    expanded_params = expand_params(params)
    result_params_list = []

    for combo_dict in expanded_params:
        new_params_dict = {k: type(params.__dict__[k])(**combo_dict[k]) if isinstance(params.__dict__[k], BaseModel) else combo_dict[k] for k in params.__dict__.keys()}
        result_params_list.append(type(params)(**new_params_dict))
    
    return result_params_list

def update_params(path):
    # update grid search hdf and yaml files to reflect changes in parameters
    path = Path(path)
    P = load_yaml_config(path)
    L = P.L
    file_path = L.out_path / 'log.h5'
    df = pd.read_hdf(file_path, key='df', mode='r')

    def update(p):
        # if p.L.last_checkpoint is None:
            # p.L.last_checkpoint = next(p.L.save_path.glob('**/*.yaml')).parent
        # if hasattr(p.model.training, 'radius_threshold'):
            # p.model.training.accuracy_threshold = p.model.training.radius_threshold
            # del p.model.training.radius_threshold
        paths_attrs = ['out_path', 'save_path', 'last_checkpoint']
        for attr in paths_attrs:
            path = getattr(p.L, attr)
            parts = path.parts
            if 'path_integration' in parts:
                continue
            elif '1D' in parts:
                idx = parts.index('1D')
            elif '2D' in parts:
                idx = parts.index('2D')
            else:
                continue
            new_parts = parts[:idx] + ('path_integration',) + parts[idx:]
            new_path = Path(*new_parts)
            setattr(P.L, attr, new_path)
        return p
    df['params'] = df['params'].apply(update)
    
    # #### UPDATE HDF ####
    # df['params'] = df['params'].apply(lambda p_dict: Params(**p_dict))
    df.to_hdf(file_path, key='df', mode='w')

    #### UPDATE YAML ####
    p = df.iloc[0]['params']
    save_yaml_config(p, P.L.out_path / 'parameters.yaml')
    for idx, row in df.iterrows():
        p = row['params']
        save_yaml_config(p, P.L.last_checkpoint / 'parameters.yaml')

    print('remmember to update config files as these were just the logged files')

if __name__ == '__main__':
    pass

    # # path integration
    # update_params('out/path_integration/grid_search/1D/initial_sweep/parameters.yaml')
    # update_params('out/path_integration/grid_search/2D/initial_sweep/parameters.yaml')
    # update_params('out/path_integration/grid_search/1D/initial_sweep2/parameters.yaml')
    # update_params('out/path_integration/grid_search/2D/initial_sweep2/parameters.yaml')

    # temporal 
    # update_params('out/temporal/density/grid_search/parameters.yaml')
    # update_params('out/grid_search/temporal/parity/parameters.yaml')
