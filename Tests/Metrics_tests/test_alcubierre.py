import numpy as np

from Metrics import Metric, alcubierre
from Tests.utils.arr_hash import arr_hash


alcubierre_tensor_hash: str = '588195c813681d2d135c83cc49f3e8a45284666860af52ed3adfb5f8b6fb17ccb3d8eff5968c7bd48e3171f56b25605d1ffeb849b4fcdbc854819348c04761c8'


def test_alcubierre():
    grid_size: np.ndarray[np.int32] = np.array([5, 20, 20, 20], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array((grid_size + 1)/2)
    v: np.float64 = 0.5
    big_r: np.float64 = 5.0
    sigma: np.float64 = 0.5
    grid_scaling: np.ndarray[np.float64] = np.array([1.0, 1.0, 1.0, 1.0])

    metric_val: Metric = alcubierre(grid_size, world_center, v, big_r, sigma, grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None"
    assert metric_val.type == "metric", "Type is wrong"
    assert metric_val.coords == "cartesian", "Coordinate system is wrong"
    assert metric_val.index == "covariant", "Index is wrong"
    assert metric_val.params_velocity == v, "Velocity isn't the same as the input velocity"
    assert metric_val.params_big_r == big_r, "R isn't the same as the input R"
    assert metric_val.params_sigma == sigma, "Sigma isn't the same as the input Sigma"

    assert arr_hash(metric_val.tensor) == alcubierre_tensor_hash, "Tensor hash mismatch"
