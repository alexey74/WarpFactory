# TODO: Implement
import numpy as np

from Metrics import Metric


def trace(tensor: Metric, metric: Metric) -> np.float64:
    trace: np.ndarray[np.float64] = np.zeros(tensor.tensor.shape[2:])
    for a in range(4):
        for b in range(4):
            trace += metric.tensor[a, b] * tensor.tensor[a, b]

    print(trace)
    return np.float64(1.0)