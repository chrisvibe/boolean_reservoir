from benchmark.temporal.temporal_density_parity_dataset import TemporalDensityDataset, TemporalParityDataset
from benchmark.temporal.parameter import TemporalDatasetParams
from project.boolean_reservoir.code.encoding import BooleanTransformer
from project.boolean_reservoir.code.parameter import Params
from project.boolean_reservoir.code.train_model import DatasetInit
# from project.boolean_reservoir.code.utils.utils import balance_dataset, l2_distance


class TemporalDatasetInit(DatasetInit): # Note dont use I.seed here dataset init will use D.seed
    def dataset_init(self, P: Params):
        D = P.D
        I = P.M.I
        D.bits = I.features * I.resolution
        D.update_path()
        assert isinstance(D, TemporalDatasetParams) 
        if D.task == 'density':
            dataset = TemporalDensityDataset(D)
        elif D.task == 'parity':
            dataset = TemporalParityDataset(D)

        # reshape so BooleanTransformer gets correct shape 
        dataset.x = dataset.x.reshape(dataset.x.shape[0], 1, -1).view(
            dataset.x.shape[0], 1, I.features, -1
        )
        x = dataset.x.reshape(dataset.x.shape[0], 1, -1).view( dataset.x.shape[0], 1, I.features, -1)
        dataset.set_data({ 'x': x, 'y': dataset.y })

        # dataset = balance_dataset(dataset, distance_fn=l2_distance, num_bins=2, labels_are_classes=True, target_mode='minimum_bin')
        
        encoder = BooleanTransformer(P)
        dataset.set_encoder_x(encoder)
        dataset.encode_x()
        dataset.split_dataset()
        return dataset

if __name__ == '__main__':
    pass
    from project.boolean_reservoir.code.parameter import load_yaml_config
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
    P = load_yaml_config('config/temporal/kq_and_gr/grid_search/design_choices_prep/all2.yaml')
    P.D.samples = 10
    P.D.sampling_mode = 'random'
    P.M.I.redundancy = 2
    P.M.I.features = 2
    p = generate_param_combinations(P)[0]
    dataset = TemporalDatasetInit().dataset_init(p)

