import numpy as np
import scipy as sp


def alphanumeric_solver(M, P, R, r):
    # Trapezoidal Method:
    dalpha = ((sp.constants.G * M / sp.constants.c ** 2 + 4 * sp.constants.pi * sp.constants.G * r ** 3 * P / sp.constants.c ** 4)
              / (r ** 2 - 2 * sp.constants.G * M * r / sp.constants.c ** 2))
    dalpha[1] = 0
    alpha_temp = sp.cumtrapz(r, dalpha)
    offset = (1 / 2 * np.log(1 - 2 * sp.constants.G * M[-1] / r[-1] / sp.constants.c ** 2)) - alpha_temp[-1]
    return alpha_temp + offset
