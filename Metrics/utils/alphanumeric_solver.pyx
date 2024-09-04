import numpy as np
import scipy as sp

cimport cython
cimport numpy as np

np.import_array()


cdef extern from "math.h":
    double log(double x)

@cython.boundscheck(False)
@cython.wraparound(True)
cpdef np.ndarray[double] alphanumeric_solver(big_m, big_p, r):
    cdef np.ndarray[double] dalpha = ((sp.constants.G * big_m / sp.constants.c ** 2 + 4 * sp.constants.pi * sp.constants.G * r**3
                                       * big_p / sp.constants.c**4) / (r**2 - 2 * sp.constants.G * big_m * r / sp.constants.c**2))
    dalpha[0] = 0
    cdef np.ndarray[double] alpha_temp = sp.integrate.cumulative_trapezoid(dalpha, r)
    cdef double offset = (1 / 2 * np.log(1 - 2 * sp.constants.G * big_m[-1] / r[-1] / sp.constants.c ** 2)) - alpha_temp[-1]
    return alpha_temp + offset
