import numpy as np

from Metrics import Metric, lentz
from Tests.utils.arr_hash import arr_hash


lentz_tensor_hash: str = 'b77cb68b8520d2ae4d3024c54cacc19af3daa3e4eceb2a108cf112e5c698412903e2ca44987747b276c7bd016402e3fd8acac299d19e0efacccc7946758d319b'


def test_lentz():
    grid_size: np.ndarray[np.int32] = np.array([5, 30, 30, 2], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array([3, 15.5, 15.5, 1.5])
    v: np.float64 = np.float64(0.1)
    scale: np.float64 = max(grid_size[1:3])/7
    grid_scaling: np.ndarray[np.float64] = np.array([1, 1, 1, 1])

    metric_val: Metric = lentz(grid_size, world_center, v, scale, grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None"
    assert metric_val.type == "metric", "Type is wrong"
    assert metric_val.coords == "cartesian", "Coordinate system is wrong"
    assert metric_val.index == "covariant", "Index is wrong"
    assert metric_val.params_velocity == v, "Velocity isn't the same as the input velocity"

    assert arr_hash(metric_val.tensor) == lentz_tensor_hash, "Tensor hash mismatch"
