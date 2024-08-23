import numpy as np


class Metric:
    type: str
    name: str
    scaling: np.array(np.double)
    coords: str
    index: str
    date: str
    params_p: float
    params_rho_smooth: np.ndarray
