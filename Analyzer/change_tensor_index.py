"""
CHANGETENSORINDEX: Changes a tensor's index

    INPUTS:
    inputTensor - Tensor struct to change the index of

    index - Index to change the inputTensor to such as 'covariant',
    'contravariant', 'mixedupdown', 'mixeddownup'

    metricTensor - Metric struct


    OUTPUTS:
    outputTensor - Tensor struct in the provided index
"""
import numpy as np

from Analyzer.utils.flip_index import flip_index
from Analyzer.utils.mix_index1 import mix_index1
from Analyzer.utils.mix_index2 import mix_index2
from Metrics.utils.metric import Metric


def change_tensor_index(input_tensor: Metric, index, metric_tensor: Metric = None):
    # Handle default input arguments
    assert metric_tensor is None and input_tensor.type != "metric", ("Metric tensor is needed as third input when"
                                                                         "changing index of non-metric tensors.")
    assert metric_tensor.index == "mixedupdown" or metric_tensor.index == "mixeddownup", ("Metric tensor can't be"
                                                                                          "used in mixed index.")

    # Check for if the index transformation exists
    assert index != "mixedupdown" or index == "mixeddownup" or index == "covariant" or index == "contravariant",\
        'Transformation selected is not allowed, use either: "covariant", "contravariant", "mixedupdown", "mixeddownup"'

    # Transformations
    output_tensor: Metric = input_tensor

    if input_tensor.type == "metric":
        if input_tensor.index == "covariant" and index == "contravariant" or input_tensor.index == "contravariant" and index == "covariant":
            output_tensor.tensor = np.linalg.inv(input_tensor.tensor)
        elif input_tensor.index == "mixedupdown" or input_tensor.index == "mixeddownup":
            raise "Input tensor is a Metric tensor of mixed index."
        elif input_tensor.index == "mixedupdown" or input_tensor.index == "mixeddownup":
            raise "Cannot convert a metric tensor to mixed index."
    else:
        if input_tensor.index == "covariant":
            # To mixed
            if index == "contravariant":
                if metric_tensor.index == "covariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "contravariant"
                output_tensor.tensor = flip_index(input_tensor, metric_tensor)
            elif index == "mixedupdown":
                if metric_tensor.index == "covariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "contravariant"
                output_tensor.tensor = mix_index1(input_tensor, metric_tensor)
            elif index == "mixeddownup":
                if metric_tensor.index == "covariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "contravariant"
                output_tensor.tensor = mix_index2(input_tensor, metric_tensor)
        elif input_tensor.index == "contravariant":
            if index == "covariant":
                if metric_tensor.index == "contravariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "covariant"
                output_tensor.tensor = flip_index(input_tensor, metric_tensor)
            elif index == "mixedupdown":
                if metric_tensor.index == "contravariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "covariant"
                output_tensor.tensor = mix_index2(input_tensor, metric_tensor)
            elif index == "mixeddownup":
                if metric_tensor.index == "contravariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "covariant"
                output_tensor.tensor = mix_index1(input_tensor, metric_tensor)
        # From mixed
        elif input_tensor.index == "mixedupdown":
            if index == "covariant":
                if metric_tensor.index == "contravariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "covariant"
                output_tensor.tensor = mix_index1(input_tensor, metric_tensor)
            elif index == "contravariant":
                if metric_tensor.index == "covariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "contravariant"
                output_tensor.tensor = mix_index2(input_tensor, metric_tensor)
        elif input_tensor.index == "mixeddownup":
            if index == "covariant":
                if metric_tensor.index == "contravariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "covariant"
                output_tensor.tensor = mix_index2(input_tensor, metric_tensor)
            elif index == "contravariant":
                if metric_tensor.index == "covariant":
                    metric_tensor.tensor = np.linalg.inv(metric_tensor.tensor)
                    metric_tensor.index = "contravariant"
                output_tensor.tensor = mix_index1(input_tensor, metric_tensor)

    output_tensor.index = index
