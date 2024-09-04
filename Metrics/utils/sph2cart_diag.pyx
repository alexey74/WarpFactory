cimport cython


cdef extern from "math.h":
    double sqrt(double x)
    double atan2(double x, double y)
    double sin(double x)
    double cos(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef (double, double, double, double, double, double) sph2cart_diag(double theta, double phi, double g22_sph):
    cdef double cos_phi = cos(phi)
    cdef double cos_theta = cos(theta)
    cdef double sin_theta = sin(theta)
    cdef double sin_phi = sin(phi)

    cdef double g22_cart = g22_sph * cos_phi**2 * sin_theta**2 + cos_phi**2 * cos_theta**2 + sin_phi**2
    cdef double g33_cart = g22_sph * sin_phi**2 * sin_theta**2 + cos_theta**2 * sin_phi**2 + cos_phi**2
    cdef double g44_cart = g22_sph * cos_theta**2 + sin_theta**2

    cdef double g23_cart = g22_sph * cos_phi * sin_phi * sin_theta**2 + cos_phi * cos_theta**2 * sin_phi - cos_phi * sin_phi
    cdef double g24_cart = g22_sph * cos_phi * cos_theta * sin_theta - cos_phi * cos_theta * sin_theta
    cdef double g34_cart = g22_sph * cos_theta * sin_phi * sin_theta - cos_theta * sin_phi * sin_theta

    return g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart
