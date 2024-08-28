"""
METRICGET_VANDENBROECKCOMOVING: Builds the Van Den Broeck metric in Galilean comoving frame

    INPUTS:
    gridSize - 1x4 array. world size in [t, x, y, z], double type.

    worldCenter - 1x4 array. world center location in [t, x, y, z], double type.

    v - speed of the warp drive in factors of c, along the x direction, double type.

    R1 - spatial expansion radius of the warp bubble, double type.

    sigma1 - width factor of the spatial expansion transition

    R2 - shift vector radius of the warp bubble, double type.

    sigma2 - width factor of the shift vector transition

    A - spatial expansion factor, double type.

    gridScale - scaling of the grid in [t, x, y, z]. double type.

    OUTPUTS:
    metric - metric struct object.
"""
from datetime import datetime

import numpy as np

from Metrics.metric import Metric
from Metrics.set_minkowski import set_minkowski
from Metrics.utils.shape_func_alcubierre import shape_func_alcubierre


def metric_get_van_den_broeck_comoving(grid_size: np.ndarray, world_center: np.ndarray, v: np.float64, big_r_1: np.float64,
                              sigma_1: np.float64, big_r_2: np.float64, sigma_2: np.float64, big_a: np.float64,
                              grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):
    assert grid_size[0] == 1, 'The time grid is greater than 1, only a size of 1 can be used for the Schwarzschild solution'

    # Assign parameters to metric struct
    metric_val: Metric = Metric("Van Den Broeck Comoving")
    metric_val.params_grid_size = grid_size
    metric_val.params_world_center = world_center
    metric_val.params_velocity = v * (1 + big_a)**2
    metric_val.params_big_r_1 = big_r_1
    metric_val.params_sigma_1 = sigma_1
    metric_val.params_big_r_2 = big_r_2
    metric_val.params_sigma_2 = sigma_2
    metric_val.params_big_a = big_a

    # Assign quantities to metric struct
    metric_val.type = "metric"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().isoformat()

    # Declare a Minkowski space
    metric_val.tensor = set_minkowski(grid_size)

    t = 0  # only one timeslice is used

    # Add the Van Den Brock modification
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):

                # Find grid center x, y, z
                x = i * grid_scaling[1] - world_center[1]
                y = j * grid_scaling[2] - world_center[2]
                z = k * grid_scaling[3] - world_center[3]

                # Find the radius from the center of the bubble
                r = np.sqrt(x**2 + y**2 + z**2)

                # Define the B function value in Van Den Broeck
                B = 1 + shape_func_alcubierre(r, big_r_1, sigma_1) * big_a

                # Define the f function value in Van Den Broeck
                fs = shape_func_alcubierre(r, big_r_2, sigma_2) * v

                # Assign fs and B to the proper terms
                metric_val.tensor[(1, 1) + (t, i, j, k)] = B ** 2
                metric_val.tensor[(2, 2) + (t, i, j, k)] = B ** 2
                metric_val.tensor[(3, 3) + (t, i, j, k)] = B ** 2

                cross_term = -B**2 * (v - fs)
                metric_val.tensor[(0, 1) + (t, i, j, k)] = cross_term
                metric_val.tensor[(1, 0) + (t, i, j, k)] = cross_term

                metric_val.tensor[(0, 0) + (t, i, j, k)] = -(1 - B**2 * fs**2)

    return metric_val
