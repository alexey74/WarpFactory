cimport cython
cimport numpy as np

np.import_array()


cdef extern from "math.h":
    int floor(double x)
    int ceil(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef legendre_radial_interp(input_array, double r, int r_scale = 1):
    # 3rd Order Legendre Polynomial Interpolation
    cdef int x0 = max([floor(r / r_scale - 1), 0])
    cdef int x1 = max([floor(r / r_scale), 0])
    cdef int x2 = max([ceil(r / r_scale), 0])
    cdef int x3 = max([ceil(r / r_scale + 0), 0])

    cdef double y0 = input_array[x0]
    cdef double y1 = input_array[x1]
    cdef double y2 = input_array[x2]
    cdef double y3 = input_array[x3]

    cdef double x_0 = x0 * r_scale
    cdef double x_1 = x1 * r_scale
    cdef double x_2 = x2 * r_scale
    cdef double x_3 = x3 * r_scale

    return (((y0 * (r - x_1) * (r - x_2) * (r - x_3) / ((x_0 - x_1) * (x_0 - x_2) * (x_0 - x_3))
              + y1 * (r - x_0) * (r - x_2) * (r - x_3) / ((x_1 - x_0) * (x_1 - x_2) * (x_1 - x_3)))
             + y2 * (r - x_0) * (r - x_1) * (r - x_3) / ((x_2 - x_0) * (x_2 - x_1) * (x_2 - x_3)))
            + y3 * (r - x_0) * (r - x_1) * (r - x_2) / ((x_3 - x_0) * (x_3 - x_1) * (x_3 - x_2)))
