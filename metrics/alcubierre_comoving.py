"""
METRICGET_ALCUBIERRECOMOVING: Builds the Alcubierre metric in a Galilean comoving frame

    INPUTS:
    gridSize - 1x4 array. world size in [t, x, y, z], double type.

    worldCenter - 1x4 array. world center location in [t, x, y, z], double type.

    v - speed of the warp drive in factors of c, along the x direction, double type.

    R - radius of the warp bubble, double type.

    sigma - thickness parameter of the bubble, double type.

    gridScale - scaling of the grid in [t, x, y, z]. double type.

    OUTPUTS:
    metric - metric struct object.
"""
import numpy as np

from datetime import datetime
from metrics import Metric, set_minkowski_three_plus_one, shape_func_alcubierre, three_plus_one_builder


def alcubierre_comoving(grid_size: np.ndarray, world_center: np.ndarray, v: np.float64, big_r: np.float64,
                     sigma: np.float64, grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):
    assert grid_size[0] == 1, 'The time grid is greater than 1, only a size of 1 can be used for the Schwarzschild solution'

    # Assign parameters to metric struct
    metric_val = Metric("Alcubierre Comoving")
    metric_val.params_grid_size = grid_size
    metric_val.params_world_center = world_center
    metric_val.params_velocity = v
    metric_val.params_big_r = big_r
    metric_val.params_sigma = sigma

    # Assign quantities to metric struct
    metric_val.type = "metric"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().isoformat()

    # Declare a Minkowski space
    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    t = 0  # only one timeslice is used

    # Add the Alcubierre modification
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):

                # Find grid center x, y, z
                x: np.float64 = (1 + i) * grid_scaling[1] - world_center[1]
                y: np.float64 = (1 + j) * grid_scaling[2] - world_center[2]
                z: np.float64 = (1 + k) * grid_scaling[3] - world_center[3]
                # Find the radius from the center of the bubble
                r: np.float64 = np.sqrt(x**2 + y**2 + z**2)

                # Find shape function at this point in r
                fs: np.float64 = shape_func_alcubierre(r, big_r, sigma)

                # Add alcubierre modification to shift vector along x
                beta[(0,) + (t, i, j, k)] = v * (1 - fs)

    # Create tensor from the 3+1 functions
    metric_val.tensor = three_plus_one_builder(alpha, beta, gamma)

    return metric_val
