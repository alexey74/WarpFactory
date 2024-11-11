import numpy as np

from metrics import Metric, minkowski
from misc import arr_hash


minkowski_tensor_hash: str = '7a78038836dd728c09f293bb4f21b13adfb0b4b36ec6cd62ca4db32a9c3567c61c667c9d7365c112a2e123669539fbd4f60e2ecd0dd65fe0e6ddd8c0197bd784'


def test_minkowski():
    grid_size: np.ndarray[np.int32] = np.array([1, 10, 10, 10], dtype=np.int32)
    grid_scaling: np.ndarray[np.float64] = np.array([1.0, 1.0, 1.0, 1.0])

    metric_val: Metric = minkowski(grid_size, grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None"
    assert metric_val.type == "metric", "Type is wrong"
    assert metric_val.coords == "cartesian", "Coordinate system is wrong"
    assert metric_val.index == "covariant", "Index is wrong"

    assert arr_hash(metric_val.tensor) == minkowski_tensor_hash, "Tensor hash mismatch"
