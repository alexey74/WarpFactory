cimport cython
from libc.math cimport isinf, isnan

cdef extern from "math.h":
    long double fabsl(long double x) nogil
    long double expl(long double x) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long double[::1] compact_sigmoid(long double[::1] r, double R1, double R2, double sigma, double r_buff) nogil:
    cdef long double[::1] f = r
    cdef Py_ssize_t n = r.shape[0]

    cdef long double cr

    for i in range(n):
        cr = r[i]
        f[i] = fabsl((1/(expl(((R2 - R1 - 2 * r_buff) * (sigma + 2)) / 2 * (1 / (cr - R2 + r_buff) + 1 / (cr - R1 - r_buff)))
                             + 1) * (1 if cr > R1 + r_buff else 0) * (1 if cr < R2 - r_buff else 0) +
                          ((1 if cr >= R2 - r_buff else 0) - 1)))

        if isinf(f[i]) or isnan(f[i]):
            raise Exception('compact sigmoid returns non-numeric values!')

    return f
