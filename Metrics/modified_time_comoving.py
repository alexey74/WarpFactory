"""
METRICGET_MODIFIEDTIMECOMOVING: Builds the Modified Time metric in Galilean comoving frame

    INPUTS:
    gridSize - 1x4 array. world size in [t, x, y, z], double type.

    worldCenter - 1x4 array. world center location in [t, x, y, z], double type.

    v - speed of the warp drive in factors of c, along the x direction, double type.

    R - radius of the warp bubble, double type.

    sigma - thickness parameter of the bubble, double type.

    A - lapse rate modification, double type.

    gridScale - scaling of the grid in [t, x, y, z]. double type.

    OUTPUTS:
    metric - metric struct object.
"""
from datetime import datetime

import numpy as np

from Metrics import Metric, set_minkowski, shape_func_alcubierre


# Handle default input arguments
def modified_time_comoving(grid_size: np.ndarray, world_center: np.ndarray, v: np.double, big_r: np.double, sigma: np.double,
                             big_a: np.double, grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):
    assert grid_size[0] == 1, 'The time grid is greater than 1, only a size of 1 can be used in comoving'

    # Assign parameters to metric struct
    metric_val = Metric("Modified Time Comoving")
    metric_val.params_gridSize = grid_size
    metric_val.params_worldCenter = world_center
    metric_val.params_velocity = v
    metric_val.params_big_r = big_r
    metric_val.params_sigma = sigma
    metric_val.params_big_a = big_a

    # Assign quantities to metric struct
    metric_val.type = "metric"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().isoformat()

    # Set Minkowski terms
    metric_val.tensor = set_minkowski(grid_size)

    # Add the Modified Time changes
    t = 0 # only one timeslice is used
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                x = i * grid_scaling[0] - world_center[0]
                y = j * grid_scaling[1] - world_center[1]
                z = k * grid_scaling[2] - world_center[2]

                # Find the radius from the center of the bubble
                r = np.sqrt((x**2 + y**2 + z**2))

                # Find shape function at this point in r
                fs: np.double = shape_func_alcubierre(r, big_r, sigma)

                # Add alcubierre term to dxdt
                metric_val.tensor[(0, 1) + (t, i, j, k)] = v * (1 - fs)
                metric_val.tensor[(1, 0) + (t, i, j, k)] = metric_val.tensor[(0, 1) + (t, i, j, k)]

                # Add dt term modification
                metric_val.tensor[(0, 0) + (t, i, j, k)] = -((1 - fs) + fs/big_a)**2 + (fs * v)**2

    return metric_val
