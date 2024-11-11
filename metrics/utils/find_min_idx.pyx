cimport cython


cdef extern from "math.h" nogil:
    double fabs(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int find_min_idx(double[::1] r_sample, double r) nogil:
    cdef int min_idx = 0
    cdef double min_val = 2147483647
    cdef double cur_val
    for i in range(1, r_sample.shape[0]):
        cur_val = fabs(r_sample[i] - r)
        if cur_val < min_val:
            min_idx = i
            min_val = cur_val
    return min_idx
