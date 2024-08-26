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
import scipy as sp

from Metrics.metric import Metric
from Metrics.set_minkowski import set_minkowski
from Metrics.utils.shape_func_alcubierre import shape_func_alcubierre


# Handle default input arguments
def metric_get_modified_time(grid_size: np.ndarray, world_center: np.ndarray, v: np.double, big_r: np.double, sigma: np.double,
                             big_a: np.double, grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):
    # Assign parameters to metric struct
    metric_val = Metric()
    metric_val.params_gridSize = grid_size
    metric_val.params_worldCenter = world_center
    metric_val.params_velocity = v
    metric_val.params_big_r = big_r
    metric_val.params_sigma = sigma
    metric_val.params_big_a = big_a

    # Assign quantities to metric struct
    metric_val.type = "metric"
    metric_val.frame = "comoving"
    metric_val.name = "Modified Time"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().strftime('%d-%m-%Y')

    # Set Minkowski terms
    metric_val.tensor = set_minkowski(grid_size)

    # Add the Modified Time changes
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                x = i * grid_scaling[0] - world_center[0]
                y = j * grid_scaling[1] - world_center[1]
                z = k * grid_scaling[2] - world_center[2]

                for t in range(grid_size[0]):
                    # Determine the x offset of the center of the bubble, centered in time
                    # Originally it said grid_scale, and I'm really unsure if I missed a reference, so yeah...
                    xs: np.ndarray = (t * grid_scaling[0] - world_center[0]) * v * sp.constants.c

                    # Find the radius from the center of the bubble
                    r = np.sqrt(((x - xs)**2 + y**2 + z**2))

                    # Find shape function at this point in r
                    fs: np.double = shape_func_alcubierre(r, big_r, sigma)

                    # Add alcubierre term to dxdt
                    metric_val.tensor[(0, 1) + (t, i, j, k)] = -v * fs
                    metric_val.tensor[(1, 0) + (t, i, j, k)] = metric_val.tensor[(0, 1) + (t, i, j, k)]

                    # Add dt term modification
                    metric_val.tensor[(0, 0) + (t, i, j, k)] = -((1 - fs) + fs/big_a)**2 + (fs * v)**2

    return metric_val
