"""
METRICGET_MODIFIEDTIME: Builds the Modified Time metric

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

from metrics import Metric, set_minkowski, shape_func_alcubierre
from constants import c


# Handle default input arguments
def modified_time(grid_size: np.ndarray, world_center: np.ndarray, v: np.double, big_r: np.double, sigma: np.double,
                             big_a: np.double, grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):
    # Assign parameters to metric struct
    metric_val = Metric("Modified Time")
    metric_val.params_gridSize = grid_size
    metric_val.params_worldCenter = world_center
    metric_val.params_velocity = v
    metric_val.params_big_r = big_r
    metric_val.params_sigma = sigma
    metric_val.params_big_a = big_a

    # Assign quantities to metric struct
    metric_val.type = "metric"

    # TODO: Also figure out why this is here
    metric_val.frame = "comoving"

    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().isoformat()

    # Set Minkowski terms
    metric_val.tensor = set_minkowski(grid_size)

    # Add the Modified Time changes
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                x: np.float64 = (1 + i) * grid_scaling[0] - world_center[0]
                y: np.float64 = (1 + j) * grid_scaling[1] - world_center[1]
                z: np.float64 = (1 + k) * grid_scaling[2] - world_center[2]

                for t in range(grid_size[0]):
                    # Determine the x offset of the center of the bubble, centered in time
                    # TODO: figure out why it was GridScale originally
                    xs: np.float64 = ((1 + t) * grid_scaling[0] - world_center[0]) * v * c

                    # Find the radius from the center of the bubble
                    r: np.float64 = np.sqrt((x - xs)**2 + y**2 + z**2)

                    # Find shape function at this point in r
                    fs: np.float64 = shape_func_alcubierre(r, big_r, sigma)

                    # Add alcubierre term to dxdt
                    cross_term: np.float64 = -v * fs
                    metric_val.tensor[(0, 1) + (t, i, j, k)] = cross_term

                    # TODO: figure out why it was originally supposed to be ... = metric_val.tensor[(0, 1) + (t, i, k, k)]
                    metric_val.tensor[(1, 0) + (t, i, j, k)] = cross_term

                    # Add dt term modification
                    metric_val.tensor[(0, 0) + (t, i, j, k)] = -((1 - fs) + fs/big_a)**2 + (fs * v)**2

    return metric_val
