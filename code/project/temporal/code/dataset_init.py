from benchmark.temporal.temporal_density_parity_datasets import TemporalDensityDataset, TemporalParityDataset
from benchmark.temporal.parameters import TemporalDatasetParams
# from project.boolean_reservoir.code.encoding import BooleanEncoder 
from project.boolean_reservoir.code.parameter import Params, InputParams
from project.boolean_reservoir.code.train_model import DatasetInit
from project.boolean_reservoir.code.utils.utils import set_seed, balance_dataset, l2_distance


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
        # encoder = BooleanEncoder(I) # TODO not necesarry (already binary) but gives options - cant handle at the moment as the data is already in binary
        # dataset.set_encoder_x(encoder) # not necesarry (already binary) but gives options
        # dataset.encode_x() # not necesarry (already binary) but gives options
        dataset.split_dataset()
        return dataset

