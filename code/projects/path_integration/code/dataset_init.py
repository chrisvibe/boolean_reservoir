from projects.boolean_reservoir.code.encoding import float_array_to_boolean, min_max_normalization
from projects.boolean_reservoir.code.parameters import InputParams
from projects.boolean_reservoir.code.utils import set_seed, balance_dataset
from projects.boolean_reservoir.code.train_model import DatasetInit
from benchmarks.path_integration.constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from pathlib import Path

class PathIntegrationDatasetInit(DatasetInit):
    def dataset_init(self, I: InputParams):
        set_seed(I.seed) # Note that model is sensitive to this init (new training needed per seed)
        parameters_from_path = self.parse_parameters_from_path(I.dataset, '-')
        parameters_from_path = self.remap_keys(parameters_from_path)
        dataset = ConstrainedForagingPathDataset(*parameters_from_path)
        bins = 100
        balance_dataset(dataset, num_bins=bins) # Note that data range affects bin assignment (outliers dangerous)
        dataset.set_normalizer_x(min_max_normalization)
        dataset.set_normalizer_y(min_max_normalization)
        dataset.normalize()
        encoder = lambda x: float_array_to_boolean(x, I)
        dataset.set_encoder_x(encoder)
        dataset.encode_x()
        dataset.split_dataset()
        return dataset

    @staticmethod
    def parse_parameters_from_path(path: Path, delimiter: str):
        parameters_from_path = {k: v for k, v in (part.split(delimiter) for part in path.parts if delimiter in part)}
        parameters_from_path['data_path'] = path 
        return parameters_from_path
    
    @staticmethod
    def remap_keys(parameter_dict):
        alias_to_parameter = {
            'd': 'n_dimensions',
            's': 'strategy',
            'b': 'boundary',
            'n': 'n_steps',
            'm': 'samples',
            'r': 'seed',
        } 
        return {k[alias_to_parameter[k]]: v for k, v in parameter_dict.items()}