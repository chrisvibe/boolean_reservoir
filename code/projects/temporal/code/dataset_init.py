from benchmarks.temporal.temporal_replication_study_density_parity_datasets import TemporalDensityDataset, TemporalParityDataset
from boolean_reservoir.train_model import DatasetInit
from boolean_reservoir.parameters import InputParams
from boolean_reservoir.utils import set_seed

class TemporalDatasetInit(DatasetInit):
    def dataset_init(self, I: InputParams):
        set_seed(I.seed) # Note that model is sensitive to this init (new training needed per seed)
        if 'density' in I.dataset.parts and 'parity' not in I.dataset.parts:
            dataset = TemporalDensityDataset(data_path=I.dataset)
        elif 'parity' in I.dataset.parts and 'density' not in I.dataset.parts:
            dataset = TemporalParityDataset(data_path=I.dataset)
        else:
            raise ValueError("dataset attribute should have 'density' or 'parity' in path name (choose one)")
        dataset.split_dataset(split=[0.3, 0.4, 0.3])
        return dataset