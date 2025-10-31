import pytest
from shutil import rmtree
from pathlib import Path
import torch
from copy import deepcopy
from projects.boolean_reservoir.code.reservoir import BooleanReservoir
from projects.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
from projects.boolean_reservoir.code.boolean_reservoir_parallel import boolean_reservoir_grid_search 

def _model_likeness_check(model: BooleanReservoir, model2: BooleanReservoir, dataset, accuracy=a().accuracy):
    """Test that two models have identical parameters and behavior."""
    # Compare model parameters
    assert model.P.model == model2.P.model, 'model parameters do not match'
    assert (model.state_dict()['readout.bias'] == model2.state_dict()['readout.bias']).all(), 'bias values do not match'
    assert (model.state_dict()['readout.weight'] == model2.state_dict()['readout.weight']).all(), 'weight values do not match'
    assert (model.lut == model2.lut).all(), 'lookup tables do not match'
    assert (model.initial_states == model2.initial_states).all(), 'initial states do not match'
    assert (model.w_in == model2.w_in).all(), 'w_in (input mapping) do not match'
    assert (list(model.graph.edges(data=True)) == list(model2.graph.edges(data=True))), 'graph structures do not match'
    
    # Compare model predictions
    x_dev, y_dev = dataset.data['x_dev'], dataset.data['y_dev']
    model.eval()
    model2.eval()
    with torch.no_grad():
        y_hat_dev = model(x_dev)
        y_hat_dev2 = model2(x_dev)
        dev_accuracy = accuracy(y_hat_dev, y_dev, model.P.model.training.accuracy_threshold)
        dev_accuracy2 = accuracy(y_hat_dev2, y_dev, model2.P.model.training.accuracy_threshold)
        assert dev_accuracy == dev_accuracy2, 'accuracy metrics do not match'

def test_saving_and_loading_models():
    """Test that a model can be saved and loaded correctly with all its parameters."""
    path = Path('/tmp/boolean_reservoir/out') 
    if path.exists():
        rmtree(path)
    
    # Train a model and save it
    p, model, dataset, train_history = train_single_model('config/path_integration/test/2D/test_model.yaml', dataset_init=d().dataset_init, accuracy=a().accuracy)
    model.save()
    
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
        'config/path_integration/test/2D/test_sweep.yaml',
        dataset_init=d().dataset_init,
        accuracy=a().accuracy,
        gpu_memory_per_job_gb = 0.5,
        cpu_memory_per_job_gb = 1,
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