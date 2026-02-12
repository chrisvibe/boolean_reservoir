import pytest
import torch
from project.boolean_reservoir.code.encoding import BooleanTransformer
from project.boolean_reservoir.code.parameter import (
    Params, ModelParams, InputParams, ReservoirParams, 
    OutputParams, TrainingParams, DatasetParams
)
from benchmark.kqgr.parameter import KQGRDatasetParams

def create_test_params(tau, eval_mode, features=2, resolution=10):
    """Create minimal params for testing"""
    return Params(
        model=ModelParams(
            input_layer=InputParams(features=features, resolution=resolution),
            reservoir_layer=ReservoirParams(n_nodes=10),
            output_layer=OutputParams(n_outputs=1),
            training=TrainingParams()
        ),
        dataset=DatasetParams(kqgr=KQGRDatasetParams(tau=tau, evaluation=eval_mode))
    )
def create_test_data(m=10, s=1, f=2, b=10):
    """Create synthetic binary data"""
    return torch.randint(0, 2, (m, s, f, b), dtype=torch.uint8)

@pytest.mark.parametrize("eval_mode", ['first', 'last', 'random'])
def test_apply_tau_basic(eval_mode):
    """Test that tau makes exactly tau bits identical per feature"""
    tau = 3
    p = create_test_params(tau, eval_mode)
    transformer = BooleanTransformer(p, apply_redundancy=False)
    
    x_orig = create_test_data()
    x = x_orig.clone()
    
    # Apply tau
    x_modified = transformer._apply_tau(x)
    
    # Create diff mask
    tau_mask = (x_modified != x_orig).any(dim=0)
    m, s, f, b = x_modified.shape
    ref_row = x_modified[0:1]
    
    # Verify exactly tau*f bits modified
    assert tau_mask.sum().item() == tau * f, \
        f"Expected {tau * f} bits modified"
    
    # Verify modified bits are identical
    assert (x_modified[:, tau_mask] == ref_row[:, tau_mask]).all(), \
        "Tau bits should be identical"
    
    # Verify unmodified bits unchanged
    non_tau_mask = ~tau_mask
    assert (x_modified[:, non_tau_mask] == x_orig[:, non_tau_mask]).all(), \
        "Non-tau bits should be unchanged"

def test_tau_per_feature():
    """Test that tau applies per feature independently"""
    tau = 4
    p = create_test_params(tau, 'last', features=3)
    transformer = BooleanTransformer(p, apply_redundancy=False)
    
    x_orig = create_test_data(f=3, b=10)  # 3 features
    x_modified = transformer._apply_tau(x_orig.clone())
    
    ref_row = x_modified[0:1]
    
    # Each feature should have exactly tau identical bits at the end
    for feat in range(3):
        tau_bits = x_modified[:, :, feat, -tau:]
        ref_tau_bits = ref_row[:, :, feat, -tau:]
        assert (tau_bits == ref_tau_bits).all(), \
            f"Feature {feat} last {tau} bits should be identical"

def test_tau_zero_is_noop():
    """Test that tau=0 doesn't modify anything"""
    p = create_test_params(tau=0, eval_mode='last')
    transformer = BooleanTransformer(p, apply_redundancy=False)
    
    x_orig = create_test_data()
    x_modified = transformer._apply_tau(x_orig.clone())
    
    assert (x_modified == x_orig).all(), \
        "tau=0 should not modify any bits"