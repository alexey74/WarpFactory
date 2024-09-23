import numpy as np

from Metrics import Metric, van_den_broeck
from Tests.utils.arr_hash import arr_hash


van_den_broeck_tensor_hash: str = '6b243afc963ee583b9c105b1b2edb99b6036e411e7c5cc460dffdf9330640239d9abffe2110e797eac93b44182de4ebcea867b4757d11e7d92b641b0aaa409e0'


def test_van_den_broeck():
    grid_size: np.ndarray[np.float64] = np.array([5, 20, 20, 20], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array((grid_size + 1)/2)
    v: np.float64 = np.float64(0.1)
    big_r_1: np.float64 = np.float64(2)
    sigma_1: np.float64 = np.float64(1)
    big_r_2: np.float64 = np.float64(5)
    sigma_2: np.float64 = np.float64(1)
    big_a: np.float64 = np.float64(0.5)
    grid_scaling: np.ndarray[np.float64] = np.array([1.0, 1.0, 1.0, 1.0])

    metric_val: Metric = van_den_broeck(grid_size, world_center, v, big_r_1, sigma_1, big_r_2, sigma_2, big_a, grid_scaling)

    assert metric_val.params_velocity != v, "Velocity is the same as the input velocity."

    assert metric_val.tensor is not None, "Tensor is None."
    assert metric_val.type == "metric", "Type is wrong."
    assert metric_val.coords == "cartesian", "Coordinate system is wrong."
    assert metric_val.index == "covariant", "Index is wrong."
    assert metric_val.params_velocity == v * (1 + big_a)**2, "Velocity isn't the same as the input velocity."
    assert metric_val.params_big_r_1 == big_r_1, "Spatial expansion radius of the warp bubble isn't the same as the input."
    assert metric_val.params_sigma_1 == sigma_1, "Width factor of the spatial expansion transition isn't the same as the input."
    assert metric_val.params_big_r_2 == big_r_2, "shift vector radius of the warp bubble isn't the same as the input."
    assert metric_val.params_sigma_2 == sigma_2, "Width factor of the shift vector transition isn't the same as the input."
    assert metric_val.params_big_a == big_a, "Spatial expansion factor isn't the same as the input."

    assert arr_hash(metric_val.tensor) == van_den_broeck_tensor_hash, "Tensor hash mismatch."
