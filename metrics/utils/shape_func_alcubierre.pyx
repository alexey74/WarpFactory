cimport cython

cdef extern from "math.h" nogil:
    double fabs(double x) nogil
    double tanh(double x) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double shape_func_alcubierre(double r, double big_r, double sigma) nogil:
    return (tanh(sigma * (big_r + r)) + tanh(sigma * (big_r - r))) / (2 * tanh(big_r * sigma))
