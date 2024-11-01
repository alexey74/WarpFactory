"""
GETENERGYTENSOR: Converts the metric into the stress energy tensor

INPUTS:
metric - A metric struct

tryGPU - A flag on whether or not to use GPU computation (0=no, 1=yes)

diffOrder - Order of finite difference, either 'second' or 'fourth'

OUTPUTS:
energy - energy tensor struct
"""
from datetime import datetime
from logging import info, error

import numpy as np
#import cupy as cx

from Analyzer import change_tensor_index
from Solver import verify_tensor
from Metrics import Metric
from Solver.utils.energy import Energy
from Solver.utils.met2den import met2den


# TODO: Add Cupy whenever available for 3.13
# TODO: Test
# TODO: GPU Implementation


def get_energy_tensor(metric_val: Metric, use_gpu: bool = False, diff_order: int = 4) -> Energy:
    energy_val: Energy = Energy(metric_val.name)
    energy_tensor: np.ndarray

    if not verify_tensor(metric_val):
        raise ValueError('Metric is not verified. Please verify metric using verifyTensor(metric).')

    if metric_val.index != 'covariant':
        metric_val = change_tensor_index(metric_val, 'covariant')
        info('Changed metric from %s index to covariant index.', metric_val.index)

    if use_gpu:
        np.array(metric_val.tensor)
        energy_tensor = met2den(metric_val.tensor, metric_val.scaling, diff_order)
    else:
        energy_tensor = met2den(metric_val.tensor, metric_val.scaling, diff_order)

    energy_val.type = "Stress-Energy"
    energy_val.tensor = energy_tensor
    energy_val.coords = metric_val.coords
    energy_val.index = "contravariant"
    energy_val.order = diff_order
    energy_val.date = datetime.today().strftime('%d-%m-%Y')

    return energy_val
