import numpy as np

from metrics import Metric
from solver import verify_tensor, tensor_inverse


# TODO: Test


def inner_product(vec_a: np.ndarray, vec_b: np.ndarray,
                  metric_val: Metric = None, same_index: bool = False) -> np.ndarray[np.float64]:
    assert verify_tensor(metric_val, True), "Metric is not verified. Please verify metric using verifyTensor(metric)."

    prod: np.ndarray[np.float64] = np.zeros(metric_val.tensor.shape[2:])

    if metric_val is not None:
        if same_index:
            inv_t = tensor_inverse(metric_val.tensor)

            for i in range(4):
                for j in range(4):
                    prod += vec_a[i] * vec_b[j] * inv_t[i, j]
        else:
            for i in range(4):
                for j in range(4):
                    prod += vec_a[i] * vec_b[j] * metric_val.tensor[i, j]
    else:
        for i in range (4):
            for j in range (4):
                prod += vec_a[i] * vec_b[j]
    return prod
