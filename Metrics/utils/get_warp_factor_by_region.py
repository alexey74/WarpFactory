import numpy as np


def get_warp_factor_by_region(x: np.float64, y_in: np.float64, size_scale) -> tuple[float, float]:
    y = np.abs(y_in)
    wfx: float = 0
    wfy: float = 0

    if (size_scale <= x <= 2 * size_scale) and (x - size_scale >= y):
        wfx = -2.0
        wfy = 0.0
    elif (size_scale < x <= 2 * size_scale) and (x - size_scale <= y) and (-y + 3 * size_scale >= x):
        wfx = -1.0
        wfy = 1.0
    elif (0 < x <= size_scale) and (y < x + size_scale) and (-y + size_scale < x):
        wfx = 0.0
        wfy = 1.0
    elif (0 < x <= size_scale) and (x + size_scale <= y) and (x <= -y + 3 * size_scale):
        wfx = -0.5
        wfy = 0.5
    elif (-size_scale < x <= 0) and (-x + size_scale < y) and (-x <= -y + 3 * size_scale):
        wfx = 1.0
        wfy = 0.0
    elif (-size_scale <= x <= size_scale) and (y < x + size_scale):
        wfx = 1.0
        wfy = 0.0

    wfy *= np.sign(y_in)

    return wfx, wfy