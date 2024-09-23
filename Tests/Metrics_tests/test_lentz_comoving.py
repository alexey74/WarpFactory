import numpy as np
import pytest

from Metrics import Metric, lentz_comoving
from Tests.utils.arr_hash import arr_hash


lentz_comoving_tensor_hash: str = 'b99a613bac1cbbdce770c0cdb9d4f506692c11166566a3c44468e05fa7b669e2923eff348afb4f8ff0fa5acb36b467347564205a287bbfc5a1ba3b6b54d7cfcf'


def test_lentz_comoving():
    wrong_grid_size: np.ndarray[np.int32] = np.array([5, 30, 30, 2], dtype=np.int32)

    grid_size: np.ndarray[np.int32] = np.array([1, 30, 30, 2], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array([1, 15.5, 15.5, 1.5])
    v: np.float64 = np.float64(0.1)
    scale: np.float64 = max(grid_size[1:3])/7
    grid_scaling: np.ndarray[np.float64] = np.array([1, 1, 1, 1])

    with pytest.raises(AssertionError) as excinfo:
        lentz_comoving(wrong_grid_size, world_center, v, scale, grid_scaling)
    assert str(excinfo.value) == 'The time grid is greater than 1, only a size of 1 can be used for the Lentz.'

    metric_val: Metric = lentz_comoving(grid_size, world_center, v, scale, grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None"
    assert metric_val.type == "metric", "Type is wrong"
    assert metric_val.coords == "cartesian", "Coordinate system is wrong"
    assert metric_val.index == "covariant", "Index is wrong"
    assert metric_val.params_velocity == v, "Velocity isn't the same as the input velocity"

    assert arr_hash(metric_val.tensor) == lentz_comoving_tensor_hash, "Tensor hash mismatch"
