import numpy as np
import scipy as sp


def alphanumeric_solver(big_m, big_p, r):
    # Trapezoidal Method:
    dalpha = ((sp.constants.G * big_m / sp.constants.c ** 2 + 4 * sp.constants.pi * sp.constants.G * r ** 3 * big_p / sp.constants.c ** 4)
              / (r ** 2 - 2 * sp.constants.G * big_m * r / sp.constants.c ** 2))
    dalpha[0] = 0
    alpha_temp = sp.integrate.cumulative_trapezoid(dalpha, r)
    offset = (1 / 2 * np.log(1 - 2 * sp.constants.G * big_m[-1] / r[-1] / sp.constants.c ** 2)) - alpha_temp[-1]
    return alpha_temp + offset
