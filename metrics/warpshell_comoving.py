"""
METRICGET_WARPSHELLCOMOVING: Builds the Warp Shell metric in a comoving frame
https://iopscience.iop.org/article/10.1088/1361-6382/ad26aa

    INPUTS:
    gridSize - 1x4 array. world size in [t, x, y, z], double type.

    worldCenter - 1x4 array. world center location in [t, x, y, z], double type.

    m - total mass of the warp shell

    R1 - inner radius of the shell

    R2 - outer radius of the shell

    Rbuff - buffer distance between the shell wall and when the shift
    starts to change

    sigma - sharpness parameter of the shift sigmoid

    smoothfactor - factor by which to smooth the walls of the shell

    vWarp - speed of the warp drive in factors of c, along the x direction, double type.

    doWarp - 0 or 1, whether or not to create the warp effect inside the
    shell

    gridScale - scaling of the grid in [t, x, y, z]. double type.

    OUTPUTS:
    metric - metric struct object.
"""
# TODO: check possible use of numba
from datetime import datetime

import numpy as np

from metrics import Metric, tov_const_density, compact_sigmoid, sph2cart_diag, find_min_idx, alphanumeric_solver, cumulative_trapz
from solver import legendre_radial_interp
from constants import c, G
from misc.smooth import smooth


def warpshell_comoving(grid_size: np.ndarray, world_center: np.ndarray, m: np.float64, big_r_1: np.float64,
                                      big_r_2: np.float64, r_buff: np.float64 = 0.0, sigma: np.float64 = 0.0,
                                      smooth_factor: np.float64 = 1.0, v_warp: np.float64 = 0.0, do_warp: bool = False,
                                      grid_scaling: np.ndarray = np.array([1, 1, 1, 1])):

    metric_val: Metric = Metric("Comoving Warp Shell")
    metric_val.type = "metric"
    metric_val.scaling = grid_scaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().strftime('%d-%m-%Y')

    # declare radius array
    world_size: np.float64 = np.sqrt((grid_size[1] * grid_scaling[1] - world_center[1]) ** 2 +
                         (grid_size[2] * grid_scaling[2] - world_center[2]) ** 2 +
                         (grid_size[3] * grid_scaling[3] - world_center[3]) ** 2)
    r_sample: np.ndarray[np.float64, ...] = np.linspace(0.0, world_size * 1.2, 10 ** 5, dtype=np.float64)

    # construct rho profile
    rho: np.ndarray[np.float64] = (np.array([1 if (big_r_1 < i < big_r_2) else 0 for i in r_sample]) *
                                   m / (4 / 3 * np.pi * (big_r_2 ** 3 - big_r_1 ** 3)))
    metric_val.params_rho = rho

    # construct mass profile
    big_m: np.ndarray[np.float64] = cumulative_trapz(4 * np.pi * rho * r_sample ** 2, r_sample, initial=0)

    # construct pressure profile
    big_p: np.ndarray[np.float64] = tov_const_density(big_r_2, big_m, rho, r_sample)
    metric_val.params_big_p = big_p

    # smooth functions
    wsz: int = np.int32(np.floor(1.79 * smooth_factor) - 1 if np.floor(1.79 * smooth_factor) % 2 == 0
                        else np.floor(1.79 * smooth_factor))
    rho: np.ndarray[np.float64] = smooth(smooth(smooth(smooth(rho, wsz), wsz), wsz), wsz).conjugate()
    metric_val.params_rho_smooth = rho

    wsz: int = int(np.floor(smooth_factor) - 1 if np.floor(smooth_factor) % 2 == 0 else np.floor(smooth_factor))
    big_p: np.ndarray[np.float64] = smooth(smooth(smooth(smooth(big_p, wsz), wsz), wsz), wsz).conjugate()
    metric_val.params_p_smooth = big_p

    # reconstruct mass profile
    big_m: np.ndarray[np.float64] = np.array(cumulative_trapz(4 * np.pi * rho * r_sample ** 2, r_sample, initial=0))
    big_m[big_m < 0] = np.max(big_m)

    # save variables
    metric_val.params_big_m = big_m
    metric_val.params_r_vec = r_sample

    # set shift line vector
    shift_radial_vector = np.array(compact_sigmoid(r_sample.astype(np.longdouble), big_r_1, big_r_2, sigma, r_buff),
                                   dtype=np.float64)
    shift_radial_vector = smooth(smooth(shift_radial_vector, wsz), wsz)
    shift_radial_vector = np.array([np.abs(i) for i in shift_radial_vector])

    # construct metric_val using spherical symmetric_val solution:
    # solve for B
    big_b: np.ndarray[np.float64] = (1 - 2 * G * big_m[1:] / r_sample[1:] / c**2)**(-1)
    big_b = np.insert(big_b, 0, np.float64(0))

    # solve for a
    a: np.ndarray[np.float64] = alphanumeric_solver(big_m, big_p, r_sample)

    # solve for A from a
    big_a: np.float64 = -np.exp(2 * a)

    # save variables to the metric_val.params:
    metric_val.params_big_a = big_a
    metric_val.params_big_b = big_b

    # return metric_val boosted and in cartesian space
    metric_val.tensor = np.zeros((4, 4) + tuple(grid_size))

    shift_matrix: np.ndarray[np.float64] = np.zeros(tuple(grid_size))

    # set offset value to handle r = 0
    epsilon: np.float64 = np.float64(0.0)

    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                x: np.float64 = (1 + i) * grid_scaling[1] - world_center[1]
                y: np.float64 = (1 + j) * grid_scaling[2] - world_center[2]
                z: np.float64 = (1 + k) * grid_scaling[3] - world_center[3]

                # ref Catalog of Spacetimes, Eq.(1.6.2) for coords def.
                r: np.float64 = np.sqrt(x**2 + y**2 + z**2) + epsilon
                theta: np.float64 = np.arctan2(np.sqrt(x**2 + y**2), z)
                phi: np.float64 = np.arctan2(y, x)

                min_idx: int = find_min_idx(r_sample, r)
                if r_sample[min_idx] > r:
                    min_idx = min_idx - 1

                min_idx = min_idx + (r - r_sample[min_idx]) / (r_sample[min_idx + 1] - r_sample[min_idx])

                g11_sph: np.float64 = legendre_radial_interp(big_a, min_idx)
                g22_sph: np.float64 = legendre_radial_interp(big_b, min_idx)

                g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart = sph2cart_diag(theta, phi, g22_sph)

                metric_val.tensor[(0, 0, 0, i, j, k)] = g11_sph

                metric_val.tensor[(1, 1, 0, i, j, k)] = g22_cart

                metric_val.tensor[(1, 2, 0, i, j, k)] = g23_cart
                metric_val.tensor[(2, 1, 0, i, j, k)] = g23_cart

                metric_val.tensor[(1, 3, 0, i, j, k)] = g24_cart
                metric_val.tensor[(3, 1, 0, i, j, k)] = g24_cart

                metric_val.tensor[(2, 2, 0, i, j, k)] = g33_cart

                metric_val.tensor[(2, 3, 0, i, j, k)] = g34_cart
                metric_val.tensor[(3, 2, 0, i, j, k)] = g34_cart

                metric_val.tensor[(3, 3, 0, i, j, k)] = g44_cart

                shift_matrix[0, i, j, k] = legendre_radial_interp(shift_radial_vector, min_idx)

    # Add warp effect
    if do_warp:
        metric_val.tensor[(0, 1)] = metric_val.tensor[(0, 1)] - metric_val.tensor[(0, 1)] * shift_matrix - shift_matrix * v_warp
        metric_val.tensor[(1, 0)] = metric_val.tensor[(0, 1)]

    return metric_val