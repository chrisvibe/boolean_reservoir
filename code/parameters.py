from pydantic import BaseModel, Field
import yaml
import itertools
from typing import List, Union, Optional
from pathlib import Path, PosixPath, WindowsPath

def represent_pathlib_path(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

yaml.add_representer(Path, represent_pathlib_path)
yaml.add_representer(PosixPath, represent_pathlib_path)
yaml.add_representer(WindowsPath, represent_pathlib_path)

class InputParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    encoding: Union[str, List[str]] = Field(..., description="Binary encoding type")
    n_inputs: Union[int, List[int]] = Field(..., description="Dimension of input data before binary encoding")
    bits_per_feature: Union[int, List[int]] = Field(..., description="Dimension per input data after binary encoding")
    redundancy: Union[int, List[int]] = Field(1, description="Encoded input can be duplicated to introduce redundancy input. 3 bits can represent 8 states, if redundancy=2 you represent 8 states with 3*2=6 bits.")
    interleaving: Union[int, List[int]] = Field(0, description="Multidimensionsional weaving of inputs, int dictates group size. n=1: abc, def -> ad, be, cf | n=2: abc, def -> ad, de, cf")

class ReservoirParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    n_nodes: Union[int, List[int]] = Field(..., description="Number of nodes in the reservoir graph")
    k_avg: Union[float, List[float]] = Field(..., description="Average degree of incoming nodes")
    k_max: Union[int, List[int]] = Field(..., description="Maximum degree of incoming nodes")
    p: Union[float, List[float]] = Field(..., description="Probability for 1 in LUT (look up table)")
    self_loops: Union[float, List[float]] = Field(..., description="Probability of self-loops in graph; normalized by number of nodes")
    init: Union[str, List[str]] = Field('random', description="Initalization strategy for reservoir node states")

class OutputParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    n_outputs: Union[int, List[int]] = Field(..., description="Dimension of output data")

class TrainingParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    batch_size: Union[int, List[int]] = Field(..., description="Number of samples per forward pass")
    epochs: Union[int, List[int]] = Field(..., description="Number of epochs")
    radius_threshold: Union[float, List[float]] = Field(..., description="Euclidian radius from target prediction. Assumes data is [0, 1]")
    learning_rate: Union[float, List[float]] = Field(..., description="Learning rate")

class ModelParams(BaseModel):
    input_layer: InputParams
    reservoir_layer: ReservoirParams
    output_layer: OutputParams
    training: TrainingParams

class GridSearchParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    n_samples: Optional[int] = Field(1, ge=1, description="Number of samples per configuration in grid search")

class HistoryParams(BaseModel):
    record_history: Optional[bool] = Field(False, description="Detailed state recoding of reservoir")
    history_buffer_size: Optional[int] = Field(10000, description="Number of log entries before saving")

class TrainLog(BaseModel):
    timestamp_utc: Optional[str] = Field(None, description="timestamp utc")
    accuracy: Optional[float] = Field(None, description="accuracy")
    loss: Optional[float] = Field(None, description="loss")
    epoch: Optional[int] = Field(None, description="epoch")

class LoggingParams(BaseModel):
    out_path: Path = Field('/out', description="Where to save logs")
    grid_search: Optional[GridSearchParams] = Field(None)
    history: HistoryParams = Field(HistoryParams())
    train_log: TrainLog = Field(TrainLog())

class Params(BaseModel):
    model: ModelParams
    logging: LoggingParams = Field(LoggingParams())

def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)

    params = Params(**config)
    return params

def save_yaml_config(base_model: BaseModel, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(base_model.model_dump(), file)

def generate_param_combinations(model_params: ModelParams):
    param_dict = {
        'input_layer': model_params.input_layer.model_dump(),
        'reservoir_layer': model_params.reservoir_layer.model_dump(),
        'output_layer': model_params.output_layer.model_dump(),
        'training': model_params.training.model_dump()
    }

    # Convert all non-list values to single-element lists
    for layer, params in param_dict.items():
        for param_name, param_value in params.items():
            if not isinstance(param_value, list):
                params[param_name] = [param_value]
    
    # Create cartesian product of all parameter values
    keys, values = zip(*{
        (layer, param_name): param_value
        for layer, params in param_dict.items()
        for param_name, param_value in params.items()
    }.items())
    
    all_combinations = list(itertools.product(*values))
    model_params_list = list() 

    # Build new ModelParams objects from combinations
    for combination in all_combinations:
        combo_dict = {key: val for key, val in zip(keys, combination)}
        new_params = {
            'input_layer': {param_name: combo_dict[('input_layer', param_name)] for param_name in param_dict['input_layer']},
            'reservoir_layer': {param_name: combo_dict[('reservoir_layer', param_name)] for param_name in param_dict['reservoir_layer']},
            'output_layer': {param_name: combo_dict[('output_layer', param_name)] for param_name in param_dict['output_layer']},
            'training': {param_name: combo_dict[('training', param_name)] for param_name in param_dict['training']}
        }
        model_params_list.append(ModelParams(**new_params))
    return model_params_list