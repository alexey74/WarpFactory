"""
METRICGET_SCHWARZSCHILD: Builds the Schwarzschild metric

    INPUTS:
    gridSize - 1x4 array. world size in [t, x, y, z], double type.

    worldCenter - 1x4 array. world center location in [t, x, y, z], double type.

    rs - Schwarzschild radius

    gridScale - scaling of the grid in [t, x, y, z]. double type.

    OUTPUTS:
    metric - metric struct object.
"""
from datetime import datetime

import numpy as np

from Metrics.metric import Metric
from Metrics.set_minkowski import set_minkowski


# Handle default input arguments
def metric_get_schwarzschild(grid_size: np.ndarray, world_center: np.ndarray, rs: np.float64,
                             grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):
    assert grid_size[0] == 1, 'The time grid is greater than 1, only a size of 1 can be used for the Schwarzschild solution'

    # Assign parameters to metric struct
    metric_val = Metric()
    metric_val.params_gridSize = grid_size
    metric_val.params_worldCenter = world_center
    metric_val.params_rs = rs

    # Assign quantities to metric struct
    metric_val.type = "metric"
    metric_val.frame = "comoving"
    metric_val.name = "Schwarzschild"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().strftime('%d-%m-%Y')

    # Set Minkowski terms
    metric_val.tensor = set_minkowski(grid_size)

    # Add very small offset to mitigate divide by zero errors
    epsilon = 0.0000000001
    t = 0  # Only 1 time slice
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                x = i * grid_scaling[1] - world_center[1]
                y = j * grid_scaling[2] - world_center[2]
                z = k * grid_scaling[3] - world_center[3]

                r = np.sqrt(x**2 + y**2 + z**2) + epsilon

                # Diagonal terms
                metric_val.tensor[(0, 0) + (t, i, j, k)] = -(1 - rs / r)
                metric_val.tensor[(1, 1) + (t, i, j, k)] = (x**2 / (1 - rs / r) + y**2 + z**2) / r**2
                metric_val.tensor[(2, 2) + (t, i, j, k)] = (x**2 + y**2 / (1 - rs / r) + z**2) / r**2
                metric_val.tensor[(3, 3) + (t, i, j, k)] = (x**2 + y**2 + z**2 / (1 - rs / r)) / r**2

                # dxdy cross terms
                metric_val.tensor[(1, 2) + (t, i, j, k)] = rs / (r**3 - r**2 * rs) * x * y
                metric_val.tensor[(2, 1) + (t, i, j, k)] = metric_val.tensor[(1, 2) + (t, i, j, k)]

                # dxdz cross terms
                metric_val.tensor[(1, 3) + (t, i, j, k)] = rs / (r**3 - r**2 * rs) * x * z
                metric_val.tensor[(3, 1) + (t, i, j, k)] = metric_val.tensor[(1, 3) + (t, i, j, k)]

                # dydz cross terms
                metric_val.tensor[(2, 3) + (t, i, j, k)] = rs / (r**3 - r**2 * rs) * y * z
                metric_val.tensor[(3, 2) + (t, i, j, k)] = metric_val.tensor[(2, 3) + (t, i, j, k)]

    return metric_val
