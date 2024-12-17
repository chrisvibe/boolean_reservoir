from collections import namedtuple
import yaml
import itertools

Params = namedtuple('Params', ['model', 'logging'])

ModelParams = namedtuple('ModelParams', ['input_layer', 'reservoir_layer', 'output_layer', 'training'])
InputParams = namedtuple('InputParams', ['encoding', 'n_inputs', 'bits_per_feature', 'redundancy'])
ReservoirParams = namedtuple('ReservoirParams', ['n_nodes', 'k_avg', 'k_max', 'p', 'self_loops'])
OutputParams = namedtuple('OutputParams', ['n_outputs'])
TrainingParams = namedtuple('TrainingParams', ['batch_size', 'epochs', 'radius_threshold', 'learning_rate'])

LoggingParams = namedtuple('LoggingParams', ['n_samples'])


def convert_to_proper_type(value):
    """
    Convert string representations of atomic types to their proper types.
    """
    if isinstance(value, list):
        return [convert_to_proper_type(v) for v in value]
    
    if value == 'None':
        return None
    if value == 'False':
        return False
    if value == 'True':
        return True
    return value


def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)

    # Convert values to proper types and ensure they are lists
    # TODO assumes yaml is always 3 nested...
    for section in config:
        for sub_section in config[section]:
            for key, value in config[section][sub_section].items():
                config[section][sub_section][key] = convert_to_proper_type(value if isinstance(value, list) else [value])
    
    # Convert the sections into named tuples
    input_layer = InputParams(**config['model']['input_layer'])
    reservoir_layer = ReservoirParams(**config['model']['reservoir_layer'])
    output_layer = OutputParams(**config['model']['output_layer'])
    training = TrainingParams(**config['model']['training'])

    model = ModelParams(
        input_layer=input_layer,
        reservoir_layer=reservoir_layer,
        output_layer=output_layer,
        training=training
    )

    logging = LoggingParams(**config['logging']['grid_search'])

    params = Params(model=model, logging=logging)
    return params

def generate_param_combinations(parameters: ModelParams):
    input_values = list(itertools.product(*parameters.input_layer._asdict().values()))
    reservoir_values = list(itertools.product(*parameters.reservoir_layer._asdict().values()))
    output_values = list(itertools.product(*parameters.output_layer._asdict().values()))
    training_values = list(itertools.product(*parameters.training._asdict().values()))
    
    param_combinations = [
        ModelParams(
            input_layer=InputParams(*input_layer),
            reservoir_layer=ReservoirParams(*reservoir_layer),
            output_layer=OutputParams(*output_layer),
            training=TrainingParams(*training)
        )
        for input_layer in input_values
        for reservoir_layer in reservoir_values
        for output_layer in output_values
        for training in training_values
    ]
    
    return param_combinations