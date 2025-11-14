import pytest
import torch
import torch.nn as nn
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.utils.param_utils import generate_param_combinations 
from projects.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from projects.boolean_reservoir.code.encoding import bin2dec, dec2bin
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
import logging
logging.basicConfig(level=logging.DEBUG)

ACCEPTABLE_TEST_ACCURACY_THRESHOLD = 0.9

class PathIntegrationVerificationModelBaseTwoEncoding(nn.Module):
    # Linear model for sanity check to verify:
    # a) Base 2 binary encoding is relatively lossless with a decent number of bits
    # b) Path integration task can be computed by summing steps
    # Note that x values should be in the range [0, 1] for use of bin2dec
    # Encoding assumed to be binary base 2
    def __init__(self, params: Params):
        super(PathIntegrationVerificationModelBaseTwoEncoding, self).__init__()
        self.P = params
        self.I = self.P.M.I
        # Per-dimension scaling (diagonal only - no cross effects)
        self.scale_weight = nn.Parameter(torch.ones(self.I.features))
        self.scale_bias = nn.Parameter(torch.zeros(self.I.features))
        self.test_encoding_precision(self.I.chunk_size)

    def forward(self, x):
        m, s, d, b = x.shape
        x = x.to(dtype=torch.float32)
        x = x.view(m * s * d, -1)          
        x = bin2dec(x, b) # undo binary encoding
        x = x.view(m, s, d)
        # apply per-dimension scaling
        x = x * self.scale_weight + self.scale_bias
        # sum across time steps
        x = torch.sum(x, dim=1)
        return x
        
    def save(self):
        pass

    @staticmethod
    def test_encoding_precision(bits):
        # Create a range of values
        original = torch.linspace(0, 1, 1000)
        
        # Encode and decode with b bits
        encoded = dec2bin(original, bits=bits)
        decoded = bin2dec(encoded, bits=bits)
        
        # Check error
        errors = (original - decoded).abs()
        print(f"Max single-value error: {errors.max():.6f}")
        print(f"Mean error: {errors.mean():.6f}")


class PathIntegrationVerificationModel(nn.Module):
    # Linear model for sanity check to verify:
    # a) Any reasonable binary encoding (not just base 2)
    # b) Path integration task can be computed by summing steps
    # Encoding assumed to be a generalized linear transformation
    def __init__(self, params: Params):
        super(PathIntegrationVerificationModel, self).__init__()
        self.P = params
        self.I = self.P.M.I
        # Shared decoder for all dimensions (learns the encoding)
        self.decoder = nn.Linear(self.I.chunk_size, 1)
        # Per-dimension scaling (diagonal only - no cross effects)
        self.scale_weight = nn.Parameter(torch.ones(self.I.features))
        self.scale_bias = nn.Parameter(torch.zeros(self.I.features))

    def forward(self, x):
        m, s, d, b = x.shape
        x = x.to(dtype=torch.float32)
        x = x.view(m * s * d, -1)          
        x = self.decoder(x)                # Decode bits to values
        x = x.view(m, s, d)                
        x = x * self.scale_weight + self.scale_bias  # Per-dimension scaling
        x = torch.sum(x, dim=1)            
        return x

    def save(self):
        pass

@pytest.mark.parametrize("model_class, config_path", [
    (PathIntegrationVerificationModelBaseTwoEncoding, 'config/path_integration/1D/grid_search/test/verification_model.yaml'),
    (PathIntegrationVerificationModelBaseTwoEncoding, 'config/path_integration/2D/grid_search/test/verification_model.yaml'),
    (PathIntegrationVerificationModel, 'config/path_integration/1D/grid_search/test/verification_model.yaml'),
    (PathIntegrationVerificationModel, 'config/path_integration/2D/grid_search/test/verification_model.yaml'),
])
def test_path_integration_verification_models(model_class, config_path):
    logging.debug(f"Testing model {model_class} with config {config_path}")
    P = load_yaml_config(config_path)
    for pi in generate_param_combinations(P):
        model_instance = model_class(pi)
        logging.debug(f"Model instance created: {model_instance}")
        
        p, trained_model, dataset, history = train_single_model(model=model_instance, dataset_init=d().dataset_init, accuracy=a().accuracy)
        logging.debug(f"Training completed with accuracy {trained_model.P.L.train_log.accuracy}")
        
        assert trained_model.P.L.train_log.accuracy >= ACCEPTABLE_TEST_ACCURACY_THRESHOLD, f"Accuracy {trained_model.P.L.train_log.accuracy} is below {ACCEPTABLE_TEST_ACCURACY_THRESHOLD}"




if __name__ == '__main__':
    P = load_yaml_config('projects/path_integration/test/config/1D/grid_search/verification_model.yaml')
    P = load_yaml_config('projects/path_integration/test/config/2D/grid_search/verification_model.yaml')
    for pi in generate_param_combinations(P):
        pi.M.I.seed = pi.D.seed = 0 # consistancy for debug
        model = PathIntegrationVerificationModelBaseTwoEncoding(pi)
        # model = PathIntegrationVerificationModel(pi)
        p, model, dataset, history = train_single_model(model=model, dataset_init=d().dataset_init, accuracy=a().accuracy)
        assert model.P.L.train_log.accuracy >= ACCEPTABLE_TEST_ACCURACY_THRESHOLD, f"Accuracy {model.P.L.train_log.accuracy} is below {ACCEPTABLE_TEST_ACCURACY_THRESHOLD}"