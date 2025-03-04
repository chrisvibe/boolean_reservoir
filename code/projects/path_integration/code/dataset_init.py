from boolean_reservoir.encoding import float_array_to_boolean, min_max_normalization
from boolean_reservoir.parameters import InputParams
from boolean_reservoir.utils import set_seed, balance_dataset
from boolean_reservoir.train_model import DatasetInit
from benchmarks.path_integration.constrained_foraging_path_dataset import ConstrainedForagingPathDataset

class PathIntegrationDatasetInit(DatasetInit):
    def dataset_init(self, I: InputParams):
        set_seed(I.seed) # Note that model is sensitive to this init (new training needed per seed)
        if I.n_inputs == 1:
            if I.dataset is None:
                I.dataset = '/data/path_integration/1D/levy_walk/n_steps/interval_boundary/dataset.pt'
            dataset = ConstrainedForagingPathDataset(data_path=I.dataset)
        elif I.n_inputs == 2:
            if I.dataset is None:
                I.dataset = '/data/path_integration/2D/levy_walk/n_steps/square_boundary/dataset.pt'
            dataset = ConstrainedForagingPathDataset(data_path=I.dataset)
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