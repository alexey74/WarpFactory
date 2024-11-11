# TODO: test
import numpy as np

from metrics import Metric
from solver import verify_tensor, tensor_inverse, Energy


def trace(tensor_val: Metric or Energy, metric_val: Metric) -> np.ndarray[np.float64]:
    assert verify_tensor(metric_val, True), "Metric is not verified. Please verify metric using verifyTensor(metric)."

    tr: np.ndarray[np.float64] = np.zeros(tensor_val.tensor.shape[2:])

    metric_tensor: np.ndarray[np.float64] = metric_val.tensor

    if tensor_val.index == metric_val.index:
        metric_tensor = tensor_inverse(metric_val.tensor)

    for a in range(4):
        for b in range(4):
            tr += metric_tensor[a, b] * tensor_val.tensor[a, b]
    return tr
