"""
METRICGET_VANDENBROECK: Builds the Van Den Broeck metric

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
import scipy as sp

from Metrics import Metric, set_minkowski, shape_func_alcubierre


def van_den_broeck(grid_size: np.ndarray, world_center: np.ndarray, v: np.float64, big_r_1: np.float64,
                                   sigma_1: np.float64, big_r_2: np.float64, sigma_2: np.float64, big_a: np.float64,
                                   grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):

    # Assign parameters to metric struct
    metric_val = Metric("Van Den Broeck")
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

    # Add the Van Den Brock modification
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):

                # Find grid center x, y, z
                x: np.float64 = (1 + i) * grid_scaling[1] - world_center[1]
                y: np.float64 = (1 + j) * grid_scaling[2] - world_center[2]
                z: np.float64 = (1 + k) * grid_scaling[3] - world_center[3]

                for t in range(grid_size[0]):
                    # Determine the x offset of the center of the bubble, centered in time
                    xs: np.float64 = ((1 + t) * grid_scaling[0] - world_center[0]) * v * (1 + big_a)**2 * sp.constants.c

                    # Find the radius from the center of the bubble
                    r: np.float64 = np.sqrt((x - xs)**2 + y**2 + z**2)

                    # Define the B function value in Van Den Broeck
                    big_b: np.float64 = 1 + shape_func_alcubierre(r, big_r_1, sigma_1) * big_a

                    # Define the f function value in Van Den Broeck
                    fs: np.float64 = shape_func_alcubierre(r, big_r_2, sigma_2) * v

                    # Assign fs and B to the proper terms
                    eq_terms: np.float64 = big_b ** 2
                    metric_val.tensor[(1, 1) + (t, i, j, k)] = eq_terms
                    metric_val.tensor[(2, 2) + (t, i, j, k)] = eq_terms
                    metric_val.tensor[(3, 3) + (t, i, j, k)] = eq_terms

                    cross_term: np.float64 = -big_b**2 * fs
                    metric_val.tensor[(0, 1) + (t, i, j, k)] = cross_term
                    metric_val.tensor[(1, 0) + (t, i, j, k)] = cross_term

                    metric_val.tensor[(0, 0) + (t, i, j, k)] = -(1 - big_b**2 * fs**2)

    return metric_val
