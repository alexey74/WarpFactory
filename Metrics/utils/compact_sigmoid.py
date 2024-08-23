import numpy as np
import scipy as sp


def compact_sigmoid(r, R1, R2, sigma, r_buff):
    f = abs(1 / (np.exp(((R2 - R1 - 2 * r_buff) * (sigma + 2)) / 2 * (1 / (r - R2 + r_buff) + 1 / (r - R1 - r_buff))) + 1) *
            (r > R1 + r_buff) * (r < R2 - r_buff) + (r >= R2 - r_buff) - 1)
    if np.any(np.isinf(f)) or np.any(~np.isreal(f)):
        raise Exception('compact sigmoid returns non-numeric values!')
    return f
