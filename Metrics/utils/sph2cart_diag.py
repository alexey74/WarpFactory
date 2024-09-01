import numpy as np


def sph2cart_diag(theta, phi, g22_sph):
    if np.abs(phi) == np.pi / 2:
        cos_phi = 0
    else:
        cos_phi = np.cos(phi)

    if np.abs(theta) == np.pi / 2:
        cos_theta = 0
    else:
        cos_theta = np.cos(theta)

    g22_cart = (g22_sph * cos_phi**2 * np.sin(theta)**2 + (cos_phi**2 * cos_theta**2)) + np.sin(phi)**2
    g33_cart = (g22_sph * np.sin(phi)**2 * np.sin(theta)**2 + (cos_theta**2 * np.sin(phi)**2)) + cos_phi**2
    g44_cart = (g22_sph * cos_theta**2 + np.sin(theta)**2)

    g23_cart = (g22_sph * cos_phi * np.sin(phi) * np.sin(theta)**2 + (cos_phi * cos_theta**2 * np.sin(phi))
                - cos_phi * np.sin(phi))
    g24_cart = (g22_sph * cos_phi * cos_theta * np.sin(theta) - (cos_phi * cos_theta * np.sin(theta)))
    g34_cart = (g22_sph * cos_theta * np.sin(phi) * np.sin(theta) - (cos_theta * np.sin(phi) * np.sin(theta)))

    return g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart
