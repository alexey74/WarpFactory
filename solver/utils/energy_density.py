# EIND Calculates the Energy Tensor from the Einstein Tensor
# Returns value in Joules/m^3
import numpy as np
from constants import c, G


def energy_density(e_tensor: np.ndarray[np.float64], inverse_tensor: np.ndarray[np.float64],
                   use_gpu: bool) -> np.ndarray[np.float64]:
    density_: np.ndarray[np.float64] = np.zeros(e_tensor.shape)
    density: np.ndarray[np.float64] = np.zeros(e_tensor.shape)

    for i in range(4):
        for j in range(4):
            density_[i, j] = c**4 / (8 * np.pi * G) * e_tensor[i, j]

    # Turn into contravariant form
    for i in range(4):
        for j in range(4):
            for a in range(4):
                for b in range(4):
                    density[i, j] += density_[a, b] * inverse_tensor[a, i] * inverse_tensor[b, j]
    return density
