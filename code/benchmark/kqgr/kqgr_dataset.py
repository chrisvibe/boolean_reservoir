from benchmark.kqgr.parameter import KQGRDatasetParams
from benchmark.temporal.temporal_density_parity_dataset import TemporalDensityDataset

class KQGRDataset(TemporalDensityDataset):
    def __init__(self, D: KQGRDatasetParams):
        # Note: TemporalDensityDataset will ignore KQGRDatasetParams fields (tau, mode) 
        # as they're used by BooleanTransformer, not the dataset itself
        super().__init__(D)