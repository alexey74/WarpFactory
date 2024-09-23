import scipy as sp
import numpy as np


def tov_const_density(big_r: np.float64, big_m: np.ndarray, rho: np.ndarray, r: np.ndarray):
    c: np.float64 = sp.constants.c
    big_g: np.float64 = sp.constants.G

    return np.real((c ** 2 * rho * ((big_r * np.sqrt(big_r - 2 * big_g * big_m[-1] / c ** 2)
                 - np.emath.sqrt(big_r ** 3 - 2 * big_g * big_m[-1] * r ** 2 / c ** 2)) /
                (np.emath.sqrt(big_r ** 3 - 2 * big_g * big_m[-1] * r ** 2 / c ** 2) - 3 * big_r *
                 np.sqrt(big_r - 2 * big_g * big_m[-1] / c ** 2))) * (r < big_r)))
