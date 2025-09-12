import pytest
import torch
import torch.nn as nn
from projects.boolean_reservoir.code.parameters import * 
from projects.boolean_reservoir.code.train_model import train_single_model, EuclideanDistanceAccuracy as a
from projects.boolean_reservoir.code.encoding import bin2dec, dec2bin
from projects.path_integration.code.dataset_init import PathIntegrationDatasetInit as d
import logging
logging.basicConfig(level=logging.DEBUG)

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
        self.scale = nn.Linear(self.I.features, self.I.features)
        self.test_encoding_precision(self.I.bits_per_feature)

    def forward(self, x):
        m, s, d, b = x.shape
        x = x.to(dtype=torch.float32)
        x = x.view(m * s * d, -1)          # role out dims
        x = bin2dec(x, b)                  # undo bit encoding 
        x = x.view(m, s, d)                # recover dimensions
        x = self.scale(x)                  # scale to y range
        x = torch.sum(x, dim=1)            # sum over s time steps
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
        self.decoder = nn.Linear(self.I.bits_per_feature, 1)
        self.scale = nn.Linear(self.I.features, self.I.features)

    def forward(self, x):
        m, s, d, b = x.shape
        x = x.to(dtype=torch.float32)
        x = x.view(m * s * d, -1)          # role out dims
        x = self.decoder(x)                # undo bit encoding 
        x = x.view(m, s, d)                # recover dimensions
        x = self.scale(x)                  # scale to y range
        x = torch.sum(x, dim=1)            # sum over s time steps
        return x

    def save(self):
        pass

@pytest.mark.parametrize("model_class, config_path", [
    (PathIntegrationVerificationModelBaseTwoEncoding, 'config/path_integration/test/1D/verification_model.yaml'),
    (PathIntegrationVerificationModelBaseTwoEncoding, 'config/path_integration/test/2D/verification_model.yaml'),
    (PathIntegrationVerificationModel, 'config/path_integration/test/1D/verification_model.yaml'),
    (PathIntegrationVerificationModel, 'config/path_integration/test/2D/verification_model.yaml'),
])
def test_path_integration_verification_models(model_class, config_path):
    # Note that the model is not saved
    logging.debug(f"Testing model {model_class} with config {config_path}")
    P = load_yaml_config(config_path)
    
    model_instance = model_class(P)
    logging.debug(f"Model instance created: {model_instance}")
    
    p, trained_model, dataset, history = train_single_model(model=model_instance, dataset_init=d().dataset_init, accuracy=a().accuracy)
    logging.debug(f"Training completedpytest /code/projects/path_integration/test/test_load_and_save.py with accuracy {trained_model.P.L.train_log.accuracy}")
    
    assert trained_model.P.L.train_log.accuracy >= 0.99, f"Accuracy {trained_model.P.L.train_log.accuracy} is below 0.99"




if __name__ == '__main__':
    P = load_yaml_config('config/path_integration/test/1D/verification_model.yaml')
    # P = load_yaml_config('config/path_integration/test/2D/verification_model.yaml')
    model = PathIntegrationVerificationModelBaseTwoEncoding(P)
    # model = PathIntegrationVerificationModel(P)
    p, model, dataset, history = train_single_model(model=model, dataset_init=d().dataset_init, accuracy=a().accuracy)
    assert model.P.L.train_log.accuracy >= 0.99, f"Accuracy {model.P.L.train_log.accuracy} is below 0.99"