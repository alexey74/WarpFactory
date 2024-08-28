"""
METRICGET_LENTZ: Builds the Lentz metric

    INPUTS:
    gridSize - 1x4 array. world size in [t, x, y, z], double type.

    worldCenter - 1x4 array. world center location in [t, x, y, z], double type.

    v - speed of the warp drive in factors of c, along the x direction, double type.

    scale - the sizing factor of the Lentz soliton template

    gridScale - scaling of the grid in [t, x, y, z]. double type.

    OUTPUTS:
    metric - metric struct object.
"""
from datetime import datetime

import numpy as np

from Metrics import metric
from Metrics.metric import Metric
from Metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from Metrics.three_plus_one_builder import three_plus_one_builder
from Metrics.utils.get_warp_factor_by_region import get_warp_factor_by_region


def metric_get_lentz_comoving(grid_size: np.ndarray, world_center: np.ndarray, v: np.float64, scale: np.float64 = None,
                  grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):
    assert grid_size[0] == 1, 'The time grid is greater than 1, only a size of 1 can be used for the Schwarzschild solution'

    # Handle default input argument
    if scale is None:
        scale = max(grid_size[1:3]) / 7

    # Assign parameters to metric struct
    metric_val = Metric("Lentz Comoving")
    metric_val.params_grid_size = grid_size
    metric_val.params_world_center = world_center
    metric_val.params_velocity = v

    # Assign quantities to metric struct
    metric_val.type = "metric"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().isoformat()

    # Declare a Minkowski space
    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    t = 0  # only one timeslice is used

    # Lentz Soliton Terms
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):

                x = i * grid_scaling[1] - world_center[1]
                y = j * grid_scaling[2] - world_center[2]

                # Get Lentz template values
                wfx, wfy = get_warp_factor_by_region(x, y, scale)

                # Assign dxdt term
                beta[(0,) + (t, i, j, k)] = v * (1 - wfx)

                # Assign dydt term
                beta[(1,) + (t, i, j, k)] = wfy * v

    metric_val.tensor = three_plus_one_builder(alpha, beta, gamma)

    return metric
