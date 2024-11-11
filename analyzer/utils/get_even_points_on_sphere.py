import numpy as np

from constants.constants import golden_ratio


def get_even_points_on_sphere(radius: np.float64, number_of_points: int, use_gpu: bool = False) -> np.ndarray[np.float64]:
    vec = np.zeros((3, number_of_points))

    for i in range(number_of_points - 1):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.acos(1 - 2 * (i + 0.5)/number_of_points)

        vec[0, i + 1] = radius * np.cos(theta) * np.sin(phi)
        vec[1, i + 1] = radius * np.sin(theta) * np.sin(phi)
        vec[2, i + 1] = radius * np.cos(phi)
    return np.real(vec)
