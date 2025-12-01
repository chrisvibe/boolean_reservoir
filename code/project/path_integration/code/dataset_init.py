from project.boolean_reservoir.code.encoding import BooleanEncoder, min_max_normalization
from project.boolean_reservoir.code.parameter import Params
from project.boolean_reservoir.code.utils.utils import balance_dataset
from project.boolean_reservoir.code.train_model import DatasetInit
from benchmark.path_integration.constrained_foraging_path_dataset import ConstrainedForagingPathDataset

class PathIntegrationDatasetInit(DatasetInit): # Note dont use I.seed here dataset init will use D.seed
    def dataset_init(self, P: Params):
        D = P.D
        I = P.M.I
        dataset = ConstrainedForagingPathDataset(D)
        dataset = balance_dataset(dataset, num_bins=100) # Note that data range affects bin assignment (outliers dangerous)
        dataset.set_normalizer_x(min_max_normalization)
        dataset.set_normalizer_y(min_max_normalization)
        dataset.normalize()
        encoder = BooleanEncoder(I)
        dataset.set_encoder_x(encoder)
        dataset.encode_x()
        dataset.split_dataset()
        return dataset