cimport cython


cdef extern from "math.h":
    int floor(double x)
    int ceil(double x)

cdef int imax(int x, int y):
    return x if x > y else y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double legendre_radial_interp(double[::1] input_array, double r, int r_scale = 1):
    # 3rd Order Legendre Polynomial Interpolation
    cdef int x0 = floor((r / r_scale) - 1)
    cdef int x1 = floor(r / r_scale)
    cdef int x2 = ceil(r / r_scale)
    cdef int x3 = ceil((r / r_scale) + 1)

    cdef double y0 = input_array[imax(x0, 1)]
    cdef double y1 = input_array[imax(x1, 1)]
    cdef double y2 = input_array[imax(x2, 1)]
    cdef double y3 = input_array[imax(x3, 1)]

    cdef double x_0 = x0 * r_scale
    cdef double x_1 = x1 * r_scale
    cdef double x_2 = x2 * r_scale
    cdef double x_3 = x3 * r_scale

    cdef double r_x_0 = r - x_0
    cdef double r_x_1 = r - x_1
    cdef double r_x_2 = r - x_2
    cdef double r_x_3 = r - x_3

    cdef double x_0_1 = x_0 - x_1
    cdef double x_0_2 = x_0 - x_2
    cdef double x_0_3 = x_0 - x_3
    cdef double x_1_2 = x_1 - x_2
    cdef double x_1_3 = x_1 - x_3
    cdef double x_2_3 = x_2 - x_3

    return (((y0 * r_x_1 * r_x_2 * r_x_3 / (x_0_1 * x_0_2 * x_0_3) + y1 * r_x_0 * r_x_2 * r_x_3 / (-x_0_1 * x_1_2 * x_1_3))
             + y2 * r_x_0 * r_x_1 * r_x_3 / (x_0_2 * x_1_2 * x_2_3)) + y3 * r_x_0 * r_x_1 * r_x_2 / (x_0_3 * x_1_3 * -x_2_3))
