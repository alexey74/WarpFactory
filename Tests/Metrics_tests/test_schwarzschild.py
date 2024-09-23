import numpy as np
import pytest

from Metrics import Metric, schwarzschild
from Tests.utils.arr_hash import arr_hash


schwarzschild_tensor_hash: str = '322e1aa392f845937da40b83633b7c663c5d20c6ad9e83fe3b2791231f5f4b8fb94243260e5fbbdfcd626f28d312953fe1cf0b032dba96dded53679af39e2e1c'


def test_schwarzschild():
    wrong_grid_size: np.ndarray[np.int32] = np.array([5, 20, 20, 20], dtype=np.int32)

    grid_size: np.ndarray[np.float64] = np.array([1, 20, 20, 20], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array((grid_size + 1)/2)
    rs: np.float64 = 0.01
    grid_scaling: np.ndarray[np.float64] = np.array([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(AssertionError) as excinfo:
        schwarzschild(wrong_grid_size, world_center, rs, grid_scaling)
    assert str(excinfo.value) == 'The time grid is greater than 1, only a size of 1 can be used for the Schwarzschild solution.'

    metric_val: Metric = schwarzschild(grid_size, world_center, rs, grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None"
    assert metric_val.type == "metric", "Type is wrong"
    assert metric_val.coords == "cartesian", "Coordinate system is wrong"
    assert metric_val.index == "covariant", "Index is wrong"
    assert metric_val.params_rs == rs, "Schwarzschild radius isn't the same as the input Schwarzschild radius"

    assert arr_hash(metric_val.tensor) == schwarzschild_tensor_hash, "Tensor hash mismatch"
