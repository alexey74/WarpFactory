import numpy as np
import pytest

from Metrics import Metric, alcubierre_comoving
from Tests.utils.arr_hash import arr_hash


alcubierre_comoving_tensor_hash: str = '2044a2a338379bfc20dde34d3d1ac4fcb71fce21c8ce6f7f45a02477abffaf079da5ced7487159c91a5e957fd0bab7263ad89c5c6e823c9551db5ad3079b505d'


def test_alcubierre_comoving():
    wrong_grid_size: np.ndarray[np.int32] = np.array([5, 20, 20, 20], dtype=np.int32)

    grid_size: np.ndarray[np.float64] = np.array([1, 20, 20, 20], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array((grid_size + 1)/2)
    v: np.float64 = 0.5
    big_r: np.float64 = 5.0
    sigma: np.float64 = 0.5
    grid_scaling: np.ndarray[np.float64] = np.array([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(AssertionError) as excinfo:
        alcubierre_comoving(wrong_grid_size, world_center, v, big_r, sigma, grid_scaling)
    assert str(excinfo.value) == 'The time grid is greater than 1, only a size of 1 can be used for the Schwarzschild solution'

    metric_val: Metric = alcubierre_comoving(grid_size, world_center, v, big_r, sigma, grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None"
    assert metric_val.type == "metric", "Type is wrong"
    assert metric_val.coords == "cartesian", "Coordinate system is wrong"
    assert metric_val.index == "covariant", "Index is wrong"
    assert metric_val.params_velocity == v, "Velocity isn't the same as the input velocity"
    assert metric_val.params_big_r == big_r, "R isn't the same as the input R"
    assert metric_val.params_sigma == sigma, "Sigma isn't the same as the input Sigma"

    assert arr_hash(metric_val.tensor) == alcubierre_comoving_tensor_hash, "Tensor hash mismatch"
