import numpy as np
import pytest

from metrics import Metric, van_den_broeck_comoving
from misc import arr_hash


van_den_broeck_comoving_tensor_hash: str = '330e4d1a7a260308f08cec3327897f2db46423933541e69afb8be7e54643fc4e05dd55ed7a94f4648f46302ae938c351a99d77e743d7714d7f33c1753c52968c'


def test_van_den_broeck_comoving():
    wrong_grid_size: np.ndarray[np.int32] = np.array([5, 20, 20, 20], dtype=np.int32)

    grid_size: np.ndarray[np.float64] = np.array([1, 20, 20, 20], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array((grid_size + 1)/2)
    v: np.float64 = np.float64(0.1)
    big_r_1: np.float64 = np.float64(2)
    sigma_1: np.float64 = np.float64(1)
    big_r_2: np.float64 = np.float64(5)
    sigma_2: np.float64 = np.float64(1)
    big_a: np.float64 = np.float64(0.5)
    grid_scaling: np.ndarray[np.float64] = np.array([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(AssertionError) as excinfo:
        van_den_broeck_comoving(wrong_grid_size, world_center, v, big_r_1, sigma_1, big_r_2, sigma_2, big_a, grid_scaling)
    assert str(excinfo.value) == 'The time grid is greater than 1, only a size of 1 can be used for the comoving van den Broeck Solution.'

    metric_val: Metric = van_den_broeck_comoving(grid_size, world_center, v, big_r_1, sigma_1, big_r_2, sigma_2, big_a, grid_scaling)

    assert metric_val.params_velocity != v, "Velocity is the same as the input velocity."

    assert metric_val.tensor is not None, "Tensor is None."
    assert metric_val.type == "metric", "Type is wrong."
    assert metric_val.coords == "cartesian", "Coordinate system is wrong."
    assert metric_val.index == "covariant", "Index is wrong."
    assert metric_val.params_velocity == v * (1 + big_a) ** 2, "Velocity isn't the same as the input velocity."
    assert metric_val.params_big_r_1 == big_r_1, "Spatial expansion radius of the warp bubble isn't the same as the input."
    assert metric_val.params_sigma_1 == sigma_1, "Width factor of the spatial expansion transition isn't the same as the input."
    assert metric_val.params_big_r_2 == big_r_2, "shift vector radius of the warp bubble isn't the same as the input."
    assert metric_val.params_sigma_2 == sigma_2, "Width factor of the shift vector transition isn't the same as the input."
    assert metric_val.params_big_a == big_a, "Spatial expansion factor isn't the same as the input."

    assert arr_hash(metric_val.tensor) == van_den_broeck_comoving_tensor_hash, "Tensor hash mismatch."
