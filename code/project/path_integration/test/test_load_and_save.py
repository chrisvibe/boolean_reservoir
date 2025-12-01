from shutil import rmtree
from pathlib import Path
import torch
from copy import deepcopy
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from project.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search 

def unwrap(m):
    """Return the original underlying model if compiled with torch.compile."""
    return getattr(m, "_orig_mod", m)


def _model_likeness_check(model: BooleanReservoir, model2: BooleanReservoir, dataset, accuracy=a().accuracy):
    """Test that two models have identical parameters, structure, and behavior."""
    
    # Unwrap compiled models
    m1 = unwrap(model)
    m2 = unwrap(model2)
    
    # --- Compare model parameters ---
    assert m1.P.model == m2.P.model, "model parameters do not match"
    
    # Compare tensors safely by intersecting keys
    keys = set(m1.state_dict().keys()) & set(m2.state_dict().keys())
    for k in keys:
        assert torch.all(m1.state_dict()[k] == m2.state_dict()[k]), f"{k} values do not match"
    
    # Compare remaining arrays
    assert (m1.lut == m2.lut).all(), "lookup tables do not match"
    assert (m1.initial_states == m2.initial_states).all(), "initial states do not match"
    assert (m1.w_in == m2.w_in).all(), "w_in (input mapping) do not match"
    
    # Compare graph edges
    assert list(m1.graph.edges(data=True)) == list(m2.graph.edges(data=True)), "graph structures do not match"
    
    # --- Compare model predictions ---
    x_dev, y_dev = dataset.data['x_dev'], dataset.data['y_dev']
    m1.eval()
    m2.eval()
    with torch.no_grad():
        y_hat_dev1 = m1(x_dev)
        y_hat_dev2 = m2(x_dev)
        acc1 = accuracy(y_hat_dev1, y_dev, m1.P.model.training.accuracy_threshold)
        acc2 = accuracy(y_hat_dev2, y_dev, m2.P.model.training.accuracy_threshold)
        assert acc1 == acc2, "accuracy metrics do not match"


def test_saving_and_loading_models():
    """Test that a model can be saved and loaded correctly with all its parameters."""
    path = Path('/tmp/boolean_reservoir/out') 
    if path.exists():
        rmtree(path)
    
    # Train a model and save it
    p, model, dataset, train_history = train_single_model('project/path_integration/test/config/2D/single_run/test_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    model.save()
    model.reset_reservoir(model.P.M.T.batch_size) # reset dirty state
    
    # Load the model from the saved path
    model2 = BooleanReservoir(load_path=model.P.L.last_checkpoint)
    
    # Test that the loaded model has the same parameters and behavior as the original
    _model_likeness_check(model, model2, dataset)


def test_reproducibility_of_loaded_grid_search_checkpoint():
    """Test that a model loaded from a grid search checkpoint can be retrained with the same results."""
    path = Path('/tmp/boolean_reservoir/out') 
    if path.exists():
        rmtree(path)
    
    # Run grid search and get the best parameters
    _, p = boolean_reservoir_grid_search(
        'project/path_integration/test/config/2D/grid_search/test_sweep.yaml',
        dataset_init=d().dataset_init,
        accuracy=a().accuracy,
        gpu_memory_per_job_gb = 0.5,
        cpu_memory_per_job_gb = 0.5,
        cpu_cores_per_job = 2,
    )

    # Load model from checkpoint
    model = BooleanReservoir(load_path=p.L.last_checkpoint)
    
    # Train a new model with the same parameters
    p2 = deepcopy(model.P)
    p2, model2, dataset2, train_history2 = train_single_model(parameter_override=p2, dataset_init=d().dataset_init, accuracy=a().accuracy)
    
    # Test that the models are equivalent
    _model_likeness_check(model, model2, dataset2)
    assert model.P.L.train_log.accuracy == model2.P.L.train_log.accuracy, 'log accuracies do not match'


if __name__ == '__main__':
    import logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(process)d - %(filename)s - %(message)s',
        stream=sys.stdout,
        force=True
    )
    logger = logging.getLogger(__name__)
    test_saving_and_loading_models()
    test_reproducibility_of_loaded_grid_search_checkpoint()