from pydantic import BaseModel, Field, model_validator
import yaml
import itertools
from typing import List, Union, Optional, Callable
from pathlib import Path, PosixPath, WindowsPath
import pandas as pd

def represent_pathlib_path(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

yaml.add_representer(Path, represent_pathlib_path)
yaml.add_representer(PosixPath, represent_pathlib_path)
yaml.add_representer(WindowsPath, represent_pathlib_path)

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

class InputParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    dataset: Optional[Union[Path, List[Path]]] = Field(None, description="Path to dataset")
    pertubation: Union[str, List[str]] = Field('xor', description="Pertubation strategy given old and new states for input nodes")
    encoding: Union[str, List[str]] = Field(..., description="Binary encoding type")
    n_inputs: Union[int, List[int]] = Field(..., description="Dimension of input data before binary encoding")
    bits_per_feature: Union[int, List[int]] = Field(8, description="Dimension per input data after binary encoding, overriden by redundancy & resolution parameters")
    redundancy: Union[int, List[int]] = Field(1, description="Encoded input can be duplicated to introduce redundancy input. 3 bits can represent 8 states, if redundancy=2 you represent 8 states with 3*2=6 bits.")
    resolution: Optional[Union[int, List[int]]] = Field(None, description="bits_per_feature / redundancy, overrides bits_per_feature")
    interleaving: Union[int, List[int]] = Field(0, description="Multidimensionsional weaving of inputs, int dictates group size. n=1: abc, def -> ad, be, cf | n=2: abc, def -> ad, de, cf")

    @model_validator(mode='before')
    def override_bits_per_feature_by_resolution_and_redundancy(cls, values):
        if 'resolution' in values and values['resolution'] is not None:
            values['bits_per_feature'] = calculate_w_broadcasting(lambda x, y: x * y, values['resolution'], values['redundancy'])
        else:
            values['resolution'] = calculate_w_broadcasting(lambda x, y: x / y, values['bits_per_feature'], values['redundancy'])
        return values

class ReservoirParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    n_nodes: Union[int, List[int]] = Field(..., description="Number of nodes in the reservoir graph")
    k_avg: Union[float, List[float]] = Field(..., description="Average degree of incoming nodes")
    k_max: Union[int, List[int]] = Field(..., description="Maximum degree of incoming nodes")
    p: Union[float, List[float]] = Field(..., description="Probability for 1 in LUT (look up table)")
    reset: Optional[Union[bool, List[bool]]] = Field(True, description="Reset to init state after each sample")
    self_loops: Optional[Union[float, List[float]]] = Field(None, description="Probability of self-loops in graph; normalized by number of nodes")
    init: Union[str, List[str]] = Field('random', description="Initalization strategy for reservoir node states")

class OutputParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    n_outputs: Union[int, List[int]] = Field(..., description="Dimension of output data")
    activation: Optional[Union[str, List[str]]] = Field(None, description="Activation after readout layer, fex sigmoid")

class TrainingParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    batch_size: Union[int, List[int]] = Field(..., description="Number of samples per forward pass")
    criterion: Optional[Union[str, List[str]]] = Field('MSE', description="ML criterion, fex MSE")
    epochs: Union[int, List[int]] = Field(..., description="Number of epochs")
    accuracy_threshold: Union[float, List[float]] = Field(..., description="Threshold for generic accuracy metric")
    learning_rate: Union[float, List[float]] = Field(..., description="Learning rate")
    evaluation: Optional[str] = Field('test', description="test, dev, train etc")

    # @model_validator(mode='before')
    # def handle_old_name(cls, values):
    #     if 'radius_threshold' in values:
    #         values['accuracy_threshold'] = values.pop('radius_threshold')
    #     return values

class ModelParams(BaseModel):
    input_layer: InputParams
    reservoir_layer: ReservoirParams
    output_layer: OutputParams
    training: TrainingParams

class GridSearchParams(BaseModel):
    seed: int = Field(0, description="Random seed, 0 disables seed")
    n_samples: Optional[int] = Field(1, ge=1, description="Number of samples per configuration in grid search")

class HistoryParams(BaseModel):
    record_history: Optional[bool] = Field(False, description="Reservoir dynamics state recording")
    buffer_size: Optional[int] = Field(64, description="Number of batched snapshots per output file")

class TrainLog(BaseModel):
    accuracy: Optional[float] = Field(None, description="accuracy")
    loss: Optional[float] = Field(None, description="loss")
    epoch: Optional[int] = Field(None, description="epoch")

class LoggingParams(BaseModel):
    timestamp_utc: Optional[str] = Field(None, description="timestamp utc")
    out_path: Path = Field('/out', description="Where to save all logs for this config")
    save_dir: Optional[Path] = Field('/out', description="Where last run was saved")
    last_checkpoint: Optional[Path] = Field(None, description="Where last checkpoint was saved")
    grid_search: Optional[GridSearchParams] = Field(None)
    history: HistoryParams = Field(HistoryParams(), description="Parameters pertaining to recoding of reservoir dynamics")
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
    # Function to expand a single model layer into a list of parameter dictionaries
    def expand_layer(layer_params):
        # Convert model to dict and identify list parameters
        params_dict = layer_params.model_dump()
        param_lists = {k: v if isinstance(v, list) else [v] for k, v in params_dict.items()}
        
        # Get all combinations of parameters for this layer
        keys = param_lists.keys()
        values = [param_lists[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        # Convert back to dictionaries
        return [dict(zip(keys, combo)) for combo in combinations]
    
    # Expand each layer separately
    input_combos = expand_layer(model_params.input_layer)
    reservoir_combos = expand_layer(model_params.reservoir_layer)
    output_combos = expand_layer(model_params.output_layer)
    training_combos = expand_layer(model_params.training)
    
    # Create the cartesian product of all layer combinations
    model_params_list = []
    for input_combo in input_combos:
        for reservoir_combo in reservoir_combos:
            for output_combo in output_combos:
                for training_combo in training_combos:
                    # Create a new ModelParams object
                    new_params = {
                        'input_layer': InputParams(**input_combo),
                        'reservoir_layer': ReservoirParams(**reservoir_combo),
                        'output_layer': OutputParams(**output_combo),
                        'training': TrainingParams(**training_combo)
                    }
                    model_params_list.append(ModelParams(**new_params))
    
    return model_params_list

def update_params(path):
    # update grid search hdf and yaml files to reflect changes in parameters
    path = Path(path)
    P = load_yaml_config(path)
    L = P.logging
    file_path = L.out_path / 'log.h5'
    df = pd.read_hdf(file_path, key='df', mode='r')

    def update(p):
        # if p.logging.last_checkpoint is None:
            # p.logging.last_checkpoint = next(p.logging.save_dir.glob('**/*.yaml')).parent
        # if hasattr(p.model.training, 'radius_threshold'):
            # p.model.training.accuracy_threshold = p.model.training.radius_threshold
            # del p.model.training.radius_threshold
        paths_attrs = ['out_path', 'save_dir', 'last_checkpoint']
        for attr in paths_attrs:
            path = getattr(p.logging, attr)
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
            setattr(p.logging, attr, new_path)
        return p
    df['params'] = df['params'].apply(update)
    
    # #### UPDATE HDF ####
    # df['params'] = df['params'].apply(lambda p_dict: Params(**p_dict))
    df.to_hdf(file_path, key='df', mode='w')

    #### UPDATE YAML ####
    p = df.iloc[0]['params']
    save_yaml_config(p, p.logging.out_path / 'parameters.yaml')
    for idx, row in df.iterrows():
        p = row['params']
        save_yaml_config(p, p.logging.last_checkpoint / 'parameters.yaml')

    print('remmember to update config files as these were just the logged files')

if __name__ == '__main__':
    pass

    # # path integration
    # update_params('/out/grid_search/path_integration/1D/initial_sweep/parameters.yaml')
    # update_params('/out/grid_search/path_integration/2D/initial_sweep/parameters.yaml')
    # update_params('/out/grid_search/path_integration/1D/initial_sweep2/parameters.yaml')
    # update_params('/out/grid_search/path_integration/2D/initial_sweep2/parameters.yaml')

    # temporal 
    # update_params('/out/grid_search/temporal/density/parameters.yaml')
    # update_params('/out/grid_search/temporal/parity/parameters.yaml')
