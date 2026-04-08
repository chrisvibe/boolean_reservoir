import pytest
import torch
import numpy as np
from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.boolean_reservoir.code.train_model import train_single_model
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.path_integration.code.dataset_init import PathIntegrationDatasetInit
from project.path_integration.test.test_verification_model import PathIntegrationVerificationModel
from benchmark.path_integration.constrained_foraging_path_dataset import ConstrainedForagingPathDataset
from benchmark.path_integration.visualization import plot_random_walk
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

CONFIG_PATH = 'project/path_integration/test/config/2D/grid_search/reset_false.yaml'
VERIFICATION_CONFIG_PATH = 'project/path_integration/test/config/2D/grid_search/verification_model.yaml'
ACCURACY_THRESHOLD = 0.5
N_CONTINUITY_SAMPLES = 200  # small subset for speed


def _param_combinations():
    P = load_yaml_config(CONFIG_PATH)
    return generate_param_combinations(P)


@pytest.mark.parametrize("pi", _param_combinations())
def test_reset_false_dataset_continuity(pi):
    """With reset=False, the endpoint of sample i must be the start of sample i+1.

    In cartesian displacement mode: y[i] + sum(x[i+1]) == y[i+1]
    because start_{i+1} = y[i] and start_{i+1} + sum(displacements_{i+1}) = y[i+1].
    This is tested in raw (pre-normalisation) space to avoid coordinate transform ambiguity.
    """
    D = pi.D
    data = ConstrainedForagingPathDataset.generate_data(
        D.dimensions, N_CONTINUITY_SAMPLES, D.steps,
        D.strategy_obj, D.boundary_obj,
        coordinate_system='cartesian', reset=False, mode='displacement',
    )
    x, y = data['x'], data['y']  # x: (N, steps, dims), y: (N, dims)

    for i in range(N_CONTINUITY_SAMPLES - 1):
        torch.testing.assert_close(
            y[i] + x[i + 1].sum(dim=0), y[i + 1],
            atol=1e-5, rtol=0,
            msg=f"Continuity broken at sample {i}: endpoint of sample {i} != start of sample {i+1}",
        )


@pytest.mark.parametrize("pi", _param_combinations())
def test_reset_false_training_accuracy(pi):
    """Training with reset=False should reach accuracy > 50%."""
    model = BooleanReservoir(params=pi)
    _, trained_model, _, _ = train_single_model(model=model)
    acc = trained_model.P.L.T.accuracy
    logging.debug(f"Accuracy: {acc} (self_loops={pi.M.R.self_loops})")
    assert acc >= ACCURACY_THRESHOLD, (
        f"Expected accuracy >= {ACCURACY_THRESHOLD}, got {acc} "
        f"(self_loops={pi.M.R.self_loops})"
    )

def _data_keys(split):
    return ('x', 'y') if split == 'train' else (f'x_{split}', f'y_{split}')


def _denormalize(y_normalized, dataset):
    return dataset.inverse_normalize_y(y_normalized).numpy()


def _visualize_model_path(model, dataset, pi, file_prepend, split='train', n_steps=25, stream=0):
    """Plot a stream's real endpoint sequence (solid) vs model predictions (dashed),
    limited to the first n_steps batch segments, in raw (denormalized) coordinate space."""
    B = pi.M.T.batch_size
    x_key, y_key = _data_keys(split)
    x_s = dataset.data[x_key][stream::B][:n_steps]
    origin = np.zeros((1, pi.D.dimensions)) # assume dataset starts from 0 origin (unknown as it only contains steps)
    y_real = np.vstack([origin, _denormalize(dataset.data[y_key][stream::B][:n_steps].detach().cpu(), dataset)])
    model.eval()
    model.reset_reservoir(hard_reset=True)
    with torch.no_grad():
        y_hat = np.vstack([origin, _denormalize(model(x_s).cpu(), dataset)])
    plot_random_walk('/out/', y_real, pi.D.strategy_obj, pi.D.boundary_obj,
                     dual={'positions_opt': y_hat},
                     file_prepend=f'{file_prepend}_stream{stream}',
                     sub_dir='visualizations/test_reset',
                     label='Real path', dual_label='Predicted path')


def _run_and_visualize(model_instance, pi, file_prepend, split='train', n_steps=25):
    _, trained, dataset, _ = train_single_model(model=model_instance, save_model=False)
    for stream in range(3):
        _visualize_model_path(trained, dataset, pi, file_prepend, split, n_steps, stream=stream)


def run_single_model_plus_visualize(n_steps=25):
    pi_v = generate_param_combinations(load_yaml_config(VERIFICATION_CONFIG_PATH))[0]
    _run_and_visualize(PathIntegrationVerificationModel(pi_v), pi_v,
                       'verification_model', n_steps=n_steps)

    pi_r = _param_combinations()[0]
    _run_and_visualize(BooleanReservoir(params=pi_r), pi_r,
                       'real_model', n_steps=n_steps)


if __name__ == "__main__":
    # TODO this is postponed since the dataset y labels for PI are have a dependancy on boundary condition. one cannot just re-arrange the dataset for daisychained path when all paths are assumed to start from (0, 0).

    # combinations = _param_combinations()
    # for i, pi in enumerate(combinations):
    #     print(f"--- Testing Combination {i+1}/{len(combinations)} ---")
    #     try:
    #         print("Running: test_reset_false_dataset_continuity")
    #         test_reset_false_dataset_continuity(pi)
    #         print("Running: test_reset_false_training_accuracy")
    #         test_reset_false_training_accuracy(pi)
    #         print("PASS\n")
    #     except Exception as e:
    #         print(f"FAIL: {e}")
    #         # If using a debugger, the execution will pause here if you have 
    #         # "Uncaught Exceptions" checked.
    #         raise e 
    # print("All manual tests passed!")

    print("Running: run_single_model_plus_visualize")
    run_single_model_plus_visualize()
