from datetime import datetime
import numpy as np
import scipy as sp
import scipy.constants

from Metrics import metric
from Metrics.utils.alphanumeric_solver import alphanumeric_solver
from Metrics.utils.compact_sigmoid import compact_sigmoid
from Metrics.utils.sph2cart_diag import sph2cart_diag
from Metrics.utils.tov_const_density import tov_const_density
from Solver.utils.legendre_radial_interp import legendre_radial_interp


def metric_val_get_warpshell_comoving(grid_size: np.ndarray, world_center: np.array(np.double), big_m: np.double, R1: float,
                                      R2: float, r_buff: float = 0.0, sigma: float = 0.0, smooth_factor: float = 1.0,
                                      v_warp: float = 0.0, do_warp: bool = False, grid_scaling: np.array(np.double) =
                                      np.array([1, 1, 1, 1])):
    metric_val: metric = metric.Metric()
    metric_val.type = "metric_val"
    metric_val.name = "Comoving Warp Shell"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().strftime('%d-%m-%Y')

    # declare radius array
    world_size = np.sqrt((grid_size[2] * grid_scaling[2] - world_center[2]) ** 2 +
                         (grid_size[3] * grid_scaling[3] - world_center[3]) ** 2 +
                         (grid_size[4] * grid_scaling[4] - world_center[4]) ** 2)
    r_sample_res = 10 ** 5
    r_sample = np.linspace(0, world_size * 1.2, r_sample_res)

    # construct rho profile
    rho = np.zeros(1, r_sample) + big_m / (4 / 3 * np.pi * (R2 ** 3 - R1 ** 3)) * (r_sample > R1 & r_sample < R2)
    metric_val.params_rho = rho

    max_big_r = np.min(np.diff(rho > 0))[1]
    max_big_r = r_sample[max_big_r]

    # construct mass profile
    big_m = sp.cumtrapz(r_sample, 4 * np.pi * rho * r_sample ** 2)

    # construct pressure profile
    big_p: np.ndarray = tov_const_density(R2, big_m, rho, r_sample)
    metric_val.params_big_p = big_p

    # smooth functions
    rho: np.ndarray = sp.smooth(sp.smooth(sp.smooth(sp.smooth(rho, 1.79 * smooth_factor), 1.79 * smooth_factor),
                                          1.79 * smooth_factor), 1.79 * smooth_factor).conjugate()
    metric_val.params_rho_smooth = rho

    big_p: np.ndarray = sp.smooth(sp.smooth(sp.smooth(sp.smooth(big_p, smooth_factor), smooth_factor),
                                            smooth_factor), smooth_factor).conjugate()
    metric_val.params_p_smooth = big_p

    # reconstruct mass profile
    big_m = sp.cumtrapz(r_sample, 4 * np.pi * rho * r_sample ** 2)
    big_m[big_m < 0] = max(big_m)

    # save variables
    metric_val.params_big_m = big_m
    metric_val.params_r_vec = r_sample

    # set shift line vector
    shift_radial_vector = compact_sigmoid(r_sample, R1, R2, sigma, r_buff)
    shift_radial_vector = sp.smooth(sp.smooth(shift_radial_vector, smooth_factor), smooth_factor)

    # construct metric_val using spherical symmetric_val solution:
    # solve for B
    big_b = (1 - 2 * sp.constants.G * big_m / r_sample / scipy.constants.c ** 2) ** (-1)
    big_b[1] = 1

    # solve for a
    a = alphanumeric_solver(big_m, big_p, max_big_r, r_sample)

    # solve for A from a
    big_a = -np.exp(2 * a)

    # save variables to the metric_val.params:
    metric_val.params_big_a = big_a
    metric_val.params_big_b = big_b

    # return metric_val boosted and in cartesian space
    metric_val.tensor = np.zeros((4, 4) + tuple(grid_size))

    shift_matrix = np.zeros(tuple(grid_size))

    # set offset value to handle r = 0
    epsilon = 0

    for i in range(1, grid_size[4]):
        for j in range(1, grid_size[1]):
            for k in range(1, grid_size[2]):
                x = (i * grid_scaling[2] - world_center[2])
                y = (j * grid_scaling[3] - world_center[3])
                z = (k * grid_scaling[4] - world_center[4])

                # ref Catalog of Spacetimes, Eq.(1.6.2) for coords def.
                r = np.sqrt(x ^ 2 + y ^ 2 + z ^ 2) + epsilon
                theta = np.arctan2(np.sqrt(x ^ 2 + y ^ 2), z)
                phi = np.arctan2(y, x)

                min_idx = min(abs(r_sample - r))[1]
                if r_sample[min_idx] > r:
                    min_idx = min_idx - 1

                min_idx = min_idx + (r - r_sample[min_idx]) / (r_sample[min_idx + 1] - r_sample[min_idx])

                g11_sph = legendre_radial_interp(big_a, min_idx)
                g22_sph = legendre_radial_interp(big_b, min_idx)

                g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart = sph2cart_diag(theta, phi, g11_sph, g22_sph)

                metric_val.tensor[(1, 1) + (1, i, j, k)] = g11_cart

                metric_val.tensor[(1, 1) + (1, i, j, k)] = g22_cart

                metric_val.tensor[(1, 1) + (1, i, j, k)] = g23_cart
                metric_val.tensor[(1, 1) + (1, i, j, k)] = metric_val.tensor[(2, 3) + (1, i, j, k)]

                metric_val.tensor[(1, 1) + (1, i, j, k)] = g24_cart
                metric_val.tensor[(1, 1) + (1, i, j, k)] = metric_val.tensor[(2, 4) + (1, i, j, k)]

                metric_val.tensor[(1, 1) + (1, i, j, k)] = g33_cart

                metric_val.tensor[(1, 1) + (1, i, j, k)] = g34_cart
                metric_val.tensor[(1, 1) + (1, i, j, k)] = metric_val.tensor[(3, 4) + (1, i, j, k)]

                metric_val.tensor[(1, 1) + (1, i, j, k)] = g44_cart

                shift_matrix[1, i, j, k] = legendre_radial_interp(shift_radial_vector, min_idx)

    # Add warp effect
    if do_warp:
        metric_val.tensor[(1, 2)] = metric_val.tensor[(1, 2)] - metric_val.tensor[(1, 2)] * shift_matrix - shift_matrix * v_warp
        metric_val.tensor[(2, 1)] = metric_val.tensor[(1, 2)]
