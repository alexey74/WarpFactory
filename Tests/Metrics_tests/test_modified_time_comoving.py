import numpy as np
import pytest

from Metrics import Metric, modified_time_comoving
from Tests.utils.arr_hash import arr_hash


modified_time_comoving_tensor_hash: str = '6f08402ba908cc053e85943edd94be5e0ca4c7e70cfea2f3b5318cc1ab67374545d5e3be9e1d690e11183165ae8cda80b67241e7fefef2e5e0834e0adb721acf'


def test_modified_time_comoving():
    wrong_grid_size: np.ndarray[np.int32] = np.array([5, 20, 20, 20], dtype=np.int32)

    grid_size: np.ndarray[np.int32] = np.array([1, 20, 20, 20], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array((grid_size + 1)/2)
    v: np.float64 = 0.5
    big_r: np.float64 = 5.0
    sigma: np.float64 = 0.5
    big_a: np.float64 = 0.05
    grid_scaling: np.ndarray[np.float64] = np.array([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(AssertionError) as excinfo:
        modified_time_comoving(wrong_grid_size, world_center, v, big_r, sigma, big_a, grid_scaling)
    assert str(excinfo.value) == 'The time grid is greater than 1, only a size of 1 can be used in comoving.'

    metric_val: Metric = modified_time_comoving(grid_size, world_center, v, big_r, sigma, big_a, grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None"
    assert metric_val.type == "metric", "Type is wrong"
    assert metric_val.coords == "cartesian", "Coordinate system is wrong"
    assert metric_val.index == "covariant", "Index is wrong"
    assert metric_val.params_velocity == v, "Velocity isn't the same as the input velocity"
    assert metric_val.params_big_r == big_r, "R isn't the same as the input R"
    assert metric_val.params_sigma == sigma, "Sigma isn't the same as the input Sigma"
    assert metric_val.params_big_a == big_a, "A isn't the same as the input A"

    assert arr_hash(metric_val.tensor) == modified_time_comoving_tensor_hash, "Tensor hash mismatch"
