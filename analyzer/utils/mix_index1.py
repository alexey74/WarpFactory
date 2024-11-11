import numpy as np

from metrics import Metric


# TODO: Test


def mix_index1(input_tensor: Metric, metric_tensor: Metric):
    temp_output_tensor: np.ndarray[np.float64] = np.zeros(input_tensor.tensor.shape)

    for i in range(4):
        for j in range(4):
            for a in range(4):
                temp_output_tensor[(i, j)] = (temp_output_tensor[(i, j)] + input_tensor.tensor[(a, j)] *
                                              metric_tensor.tensor[(a, i)])
    return temp_output_tensor
