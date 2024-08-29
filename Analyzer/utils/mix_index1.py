import numpy as np

from Metrics.metric import Metric


def mix_index1(input_tensor: Metric, metric_tensor: Metric):
    s: tuple[int, ...] = input_tensor.tensor[0, 0].shape

    temp_output_tensor = np.zeros((4, 4) + s)
    for i in range(4):
        for j in range(4):
            for a in range(4):
                temp_output_tensor[(i, j)] = (temp_output_tensor[(i, j)] + input_tensor.tensor[(a, j)] *
                                              metric_tensor.tensor[(a, i)])
    return temp_output_tensor
