import scipy as sp
import numpy as np


def tov_const_density(big_r: np.float64, big_m: np.ndarray, rho: np.ndarray, r: np.ndarray):
    return np.real((sp.constants.c ** 2 * rho * (
                (big_r * np.sqrt(big_r - 2 * sp.constants.G * big_m[-1] / sp.constants.c ** 2)
                 - np.emath.sqrt(big_r ** 3 - 2 * sp.constants.G * big_m[-1] * r ** 2 / sp.constants.c ** 2)) /
                (np.emath.sqrt(big_r ** 3 - 2 * sp.constants.G * big_m[-1] * r ** 2 / sp.constants.c ** 2) - 3 * big_r *
                 np.sqrt(big_r - 2 * sp.constants.G * big_m[-1] / sp.constants.c ** 2))) * (r < big_r).astype(np.float64)))
