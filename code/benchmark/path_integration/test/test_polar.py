import numpy as np
from benchmark.path_integration.constrained_foraging_path import to_polar, to_cartesian

# 2D tests
assert np.allclose(to_polar([1, 0]), [1, 0])                    # x-axis
assert np.allclose(to_polar([0, 1]), [1, np.pi/2])             # y-axis
assert np.allclose(to_polar([-1, 0]), [1, np.pi])              # -x-axis
assert np.allclose(to_polar([1, 1]), [np.sqrt(2), np.pi/4])    # 45°

# 3D tests
assert np.allclose(to_polar([1, 0, 0]), [1, 0, np.pi/2])       # x-axis
assert np.allclose(to_polar([0, 1, 0]), [1, np.pi/2, np.pi/2]) # y-axis
result = to_polar([0, 0, 1])
assert np.allclose(result[[0, 2]], [1, 0], atol=1e-4)          # z-axis (skip theta)

# Roundtrip tests
for dim in [1, 2, 3]:
    cart = np.random.randn(100, dim)
    polar = to_polar(cart)
    reconstructed = to_cartesian(polar)
    assert np.allclose(cart, reconstructed, atol=1e-6)

print("All tests passed!")