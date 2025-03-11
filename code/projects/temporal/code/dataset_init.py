from benchmarks.temporal.temporal_replication_study_density_parity_datasets import TemporalDensityDataset, TemporalParityDataset
from benchmarks.temporal.parameters import TemporalDatasetParams
from projects.boolean_reservoir.code.train_model import DatasetInit
from projects.boolean_reservoir.code.parameters import InputParams
from projects.boolean_reservoir.code.utils import set_seed, balance_dataset, l2_distance
from pathlib import Path


class TemporalDatasetInit(DatasetInit):
    def dataset_init(self, I: InputParams):
        set_seed(I.seed) # Note that model is sensitive to this init (new training needed per seed)
        parameters_from_path = self.parse_parameters_from_path(I.dataset, '-')
        dataset = None
        if parameters_from_path.task == 'density':
            dataset = TemporalDensityDataset(parameters_from_path)
        elif parameters_from_path.task == 'parity':
            dataset = TemporalParityDataset(parameters_from_path)
        dataset = balance_dataset(dataset, distance_fn=l2_distance, num_bins=2, labels_are_classes=True, target_mode='minimum_bin')
        if not dataset.data_path.exists():
            dataset.save_data()
        dataset.split_dataset(split=[0.3, 0.4, 0.3])
        return dataset
    
    @staticmethod
    def parse_parameters_from_path(path: Path, delimiter: str):
        parameters_from_path = {k: v for k, v in (part.split(delimiter) for part in path.parts if delimiter in part)}
        parameters_from_path['dataset'] = path 
        if 'density' in path.parts:
            parameters_from_path['task'] = 'density'
        elif 'parity' in path.parts:
            parameters_from_path['task'] = 'parity'
        else:
            raise ValueError("dataset path should have 'density' or 'parity' in path name (choose one)")
        p = TemporalDatasetParams(**parameters_from_path)
        return p