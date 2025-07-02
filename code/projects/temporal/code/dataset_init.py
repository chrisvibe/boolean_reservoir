from benchmarks.temporal.temporal_replication_study_density_parity_datasets import TemporalDensityDataset, TemporalParityDataset
from benchmarks.temporal.parameters import TemporalDatasetParams
from projects.boolean_reservoir.code.parameters import Params, InputParams
from projects.boolean_reservoir.code.train_model import DatasetInit
from projects.boolean_reservoir.code.utils import set_seed, balance_dataset, l2_distance


class TemporalDatasetInit(DatasetInit): # Note dont use I.seed here dataset init will use D.seed
    def dataset_init(self, P: Params):
        D = P.D
        I = P.M.I
        assert isinstance(D, TemporalDatasetParams) 
        if D.task == 'density':
            dataset = TemporalDensityDataset(D)
        elif D.task == 'parity':
            dataset = TemporalParityDataset(D)
        dataset = balance_dataset(dataset, distance_fn=l2_distance, num_bins=2, labels_are_classes=True, target_mode='minimum_bin')
        if not dataset.data_path.exists():
            dataset.save_data()
        if D.split:
            dataset.split_dataset()
        return dataset

