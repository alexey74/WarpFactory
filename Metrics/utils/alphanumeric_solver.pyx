# TODO: Actually cythonize this
import numpy as np
import scipy as sp

from cpython.tuple cimport PyTuple_SetItem, PyTuple_Size
cimport cython
cimport numpy as np


cdef extern from "math.h":
    double log(double x)


@cython.boundscheck(False)
@cython.wraparound(True)
cpdef np.ndarray[double] cumulative_trapz(np.ndarray[double] y, np.ndarray x=None, double dx=1.0, int axis=-1, np.int initial=None):
    cdef np.ndarray[double] d
    cdef tuple shape
    cdef int x_nd = x.ndim
    cdef int y_nd = y.ndim
    cdef np.ndarray[double] ret
    cdef list[slice] slice1
    cdef list[slice] slice2

    if x_nd == 1:
        d = np.diff(x)
        # reshape to correct shape
        shape = tuple((1 * y_nd,))
        PyTuple_SetItem(shape, PyTuple_Size(shape) - 1, -1)
        d = d.reshape(shape)
    else:
        d = np.diff(x, axis=axis)

    slice1 = [slice(None)] * y_nd
    slice2 = [slice(None)] * y_nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    ret = np.cumulative_sum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)

    shape = np.shape(ret)
    PyTuple_SetItem(shape, PyTuple_Size(shape) - 1, 1)

    if initial is None:
        return ret

    return np.concatenate([np.full(shape, initial, dtype=ret.dtype), ret], axis=axis)


@cython.boundscheck(False)
@cython.wraparound(True)
cpdef np.ndarray[double] alphanumeric_solver(np.ndarray[double] big_m, np.ndarray[double] big_p, np.ndarray[double] r):
    cdef np.ndarray[double] dalpha = r
    dalpha[0] = 1

    dalpha = ((sp.constants.G * big_m / sp.constants.c ** 2 + 4 * sp.constants.pi * sp.constants.G * dalpha**3
                  * big_p / sp.constants.c**4) / (dalpha**2 - 2 * sp.constants.G * big_m * dalpha / sp.constants.c**2))
    dalpha[0] = 0
    cdef np.ndarray[double] alpha_temp = cumulative_trapz(dalpha, r)
    cdef double offset = (1 / 2 * np.log(1 - 2 * sp.constants.G * big_m[-1] / r[-1] / sp.constants.c ** 2)) - alpha_temp[-1]
    return alpha_temp + offset
