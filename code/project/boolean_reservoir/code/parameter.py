from pydantic import BaseModel, Field, model_validator, field_validator
import yaml
from typing import List, Union, Optional
from pathlib import Path
from benchmark.path_integration.parameters import PathIntegrationDatasetParams
from benchmark.temporal.parameters import TemporalDatasetParams
from project.boolean_reservoir.code.utils.param_utils import pydantic_init, calculate_w_broadcasting, DynamicParams, CallParams, ExpressionEvaluator

pydantic_init()


class InputParams(BaseModel): # TODO split into Bits layer (B→I) and Input layer (I→R)
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    w_bi: Union[str, List[str]] = Field(
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
    w_ir: Union[str, List[str]] = Field('out-1:1:1', description="See w_bi property. Produces a biparitite mapping from input nodes to reservoir nodes (I→R) so the following symbolic tokens are replaced accordingly; 'a': I, 'b': R")
    w_in: Optional[Union[Path, List[Path]]] = Field(None, description="Input distribution mapping explicitely set by an adjacency matrix B->I. Parameter is a path to a stored tensor. Overrides distribution parameter")
    selector: Union[str, List[str]] = Field('S :I', description="Selection chain. Supports F (filter), S (slice), R (random). Variables: i: index variable for F operation, I: I.n_nodes. Ie. assuming I=2 F i<4 -> S -3: -> R I ([0, 1, 2, 3]->[1, 2, 3]->[1, 3]). R samples: R without samples => scramble")
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
        # Skip if chunks is a list (will be handled after expansion)
        if isinstance(self.chunks, list):
            return self

        # Initialize ticks as '1' repeated for each chunk count
        if self.ticks is None:
            self.ticks = calculate_w_broadcasting(lambda t, c: '1' * c, self.ticks, self.chunks)
        
        # Repeat or truncate ticks to match chunk length
        self.ticks = calculate_w_broadcasting(
            lambda t, c: t * (c // len(t)) if c >= len(t) else t[:c],
            self.ticks,
            self.chunks
        )
        return self

class ReservoirParams(BaseModel):
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    n_nodes: Optional[Union[int, List[int]]] = Field(None, description="Number of reservoir nodes (R) excluding input nodes (I)")
    k_min: Union[int, List[int]] = Field(0, description="Min degree of incoming nodes")
    k_avg: Union[float, List[float]] = Field(2, description="Average degree of incoming nodes")
    k_max: Optional[Union[str, List[str], int, List[int]]] = Field( None, description="Maximum degree of incoming nodes")
    mode: Union[str, List[str]] = Field('heterogeneous', description="heterogeneous: each node can have different number of neighbours, homogeneous: each node has k neighbours set by k_avg")
    p: Union[float, List[float]] = Field(0.5, description="Probability for 1 in LUT (look up table)")
    reset: Optional[Union[bool, List[bool]]] = Field(True, description="Reset to init state after each sample")
    self_loops: Optional[Union[float, List[float]]] = Field(None, description="Probability of self-loops in graph; normalized by number of nodes")
    init: Union[str, List[str]] = Field('random', description="Initalization strategy for reservoir node states")

    @field_validator('k_max', mode='before')
    @classmethod
    def coerce_k_max_to_str(cls, v):
        if isinstance(v, list):
            return [str(x) for x in v]
        elif isinstance(v, int):
            return str(v)
        return v

    @model_validator(mode='after')
    def resolve_and_override(self):
        if self.mode == 'homogeneous':
            if isinstance(self.k_avg, list):
                self.k_min = self.k_max = 0
            else:
                self.k_min = self.k_max = int(self.k_avg)
        elif isinstance(self.k_max, str):
            context = {'k_avg': self.k_avg, 'k_min': self.k_min}
            self.k_max = int(ExpressionEvaluator(context).eval(self.k_max))
        return self

class OutputParams(BaseModel): # TODO add w_out and distribution like in input_layer. atm we assume full readout of R
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    n_outputs: Union[int, List[int]] = Field(1, description="Dimension of output data")
    activation: Optional[Union[str, List[str]]] = Field(None, description="Activation after readout layer, fex sigmoid")
    readout_mode: Optional[Union[str, List[str]]] = Field("binary", description="Encoding of reservoir states for readout: 'binary'={0,1}, 'bipolar'={-1,1}")

class TrainingParams(BaseModel):
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    batch_size: Union[int, List[int]] = Field(128, description="Number of samples per forward pass")
    criterion: Optional[Union[str, List[str]]] = Field('MSE', description="ML criterion, fex MSE, BCE")
    epochs: Union[int, List[int]] = Field(100, description="Number of epochs")
    accuracy_threshold: Union[float, List[float]] = Field(0.5, description="Threshold for generic accuracy metric")
    evaluation: Optional[str] = Field('test', description="test, dev, train etc")
    shuffle: bool = Field(True, description="Shuffle dataset")
    drop_last: bool = Field(True, description="Drop last")
    optim: Union[DynamicParams, List[DynamicParams]] = Field(
        default=DynamicParams(name='adam', params={'lr': 1e-3, 'weight_decay': 1e-3}), # standard for RC in litterature (ridge)
        description="Optimizer configuration"
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
    record: Optional[bool] = Field(False, description="Reservoir dynamics state recording")
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

    @field_validator('save_keys', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

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

if __name__ == '__main__':
    class Params(BaseModel):
        optim: Union[DynamicParams, List[DynamicParams]] = Field(
            default=DynamicParams(name='adam', params={'lr': 1e-3, 'weight_decay': 1e-3}), # standard for RC in litterature (ridge)
            description="Optimizer configuration"
        )
        save_keys: Optional[List[str]] = Field(
            default=['parameters'],
            description="Only save these model objects",
            json_schema_extra={'expand': False}  # Mark as non-expandable
        )
    p = Params(
        optim=[
            DynamicParams(
                name="adam",
                params=CallParams(
                    lr=[1e-1, 1e-2, 1e-3],
                    a=[1, 2],
                ),
            ),
            DynamicParams(
                name="adamw",
                params=CallParams(
                    lr=[1e-1, 1e-2, 1e-3],
                ),
            ),
        ]
    )
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations 
    P = generate_param_combinations(p)
    for p in P:
        print(p)