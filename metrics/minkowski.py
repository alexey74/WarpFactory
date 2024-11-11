"""
METRICGET_MINKOWSKI: Builds a Minkowski metric

    INPUTS:
    gridSize - 1x4 array. world size in [t, x, y, z], double type.

    gridScale - scaling of the grid in [t, x, y, z]. double type.

    OUTPUTS:
    metric - metric struct object.
"""
import numpy as np

from datetime import datetime
from metrics import Metric


def minkowski(grid_size: np.ndarray, grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):
    t_grid_size: tuple = tuple(grid_size)

    # Assign quantities to metric struct
    metric_val = Metric("Minkowski")
    metric_val.type = "metric"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().isoformat()

    # Initialization and Cross terms
    tensor: np.ndarray = np.zeros((4, 4) + t_grid_size)

    # dt^2 term
    tensor[0, 0] = np.full(t_grid_size, -1)

    # Non-time diagonal terms
    for i in range(1, 4):
        for j in range(1, 4):
            if i == j:
                tensor[i, j] = np.ones(t_grid_size)

    metric_val.tensor = tensor

    return metric_val
