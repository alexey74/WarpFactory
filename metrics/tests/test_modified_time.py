import numpy as np

from metrics import Metric, modified_time
from misc import arr_hash


modified_time_tensor_hash: str = 'bc5478b35a4a1abe75bc7e67c9c5851086dde288305cd5bc5bc7c991713f337bceeb64715770f40248bbe28b26bf748c99cb8909724fce110598491fa55b1865'


def test_modified_time():
    grid_size: np.ndarray[np.int32] = np.array([5, 20, 20, 20], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array((grid_size + 1)/2)
    v: np.float64 = 0.5
    big_r: np.float64 = 5.0
    sigma: np.float64 = 0.5
    big_a: np.float64 = 0.05
    grid_scaling: np.ndarray[np.float64] = np.array([1.0, 1.0, 1.0, 1.0])

    metric_val: Metric = modified_time(grid_size, world_center, v, big_r, sigma, big_a, grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None"
    assert metric_val.type == "metric", "Type is wrong"
    assert metric_val.coords == "cartesian", "Coordinate system is wrong"
    assert metric_val.index == "covariant", "Index is wrong"
    assert metric_val.params_velocity == v, "Velocity isn't the same as the input velocity"
    assert metric_val.params_big_r == big_r, "R isn't the same as the input R"
    assert metric_val.params_sigma == sigma, "Sigma isn't the same as the input Sigma"
    assert metric_val.params_big_a == big_a, "A isn't the same as the input A"

    assert arr_hash(metric_val.tensor) == modified_time_tensor_hash, "Tensor hash mismatch"
