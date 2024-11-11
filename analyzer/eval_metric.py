import numpy as np

from analyzer.frame_transfer import frame_transfer
from analyzer.get_energy_conditions import get_energy_conditions
from analyzer.scalar import scalar
from metrics import Metric
from solver import get_energy_tensor, Energy


# TODO: Test


def eval_metric(metric_val: Metric, keep_positive: bool = False, num_angular_vec: int = 100,
                num_time_vec:int = 10, use_gpu: bool = False):
    # Energy tensor outputs
    energy_tensor: Energy = get_energy_tensor(metric_val, use_gpu)
    energy_tensor_eulerian: Energy = frame_transfer(metric_val, energy_tensor, "eulerian", use_gpu)

    # Energy condition outputs
    null_condition = get_energy_conditions(energy_tensor, metric_val, "null", num_angular_vec, num_time_vec, False, use_gpu)
    weak_condition = get_energy_conditions(energy_tensor, metric_val, "weak", num_angular_vec, num_time_vec, False, use_gpu)
    strong_condition = get_energy_conditions(energy_tensor, metric_val, "strong", num_angular_vec, num_time_vec, False, use_gpu)
    dominant_condition = get_energy_conditions(energy_tensor, metric_val, "dominant", num_angular_vec, num_time_vec, False, use_gpu)

    if not keep_positive:
        null_condition = np.where(null_condition > 0, null_condition, 0)
        weak_condition = np.where(weak_condition > 0, weak_condition, 0)
        strong_condition = np.where(strong_condition > 0, strong_condition, 0)
        dominant_condition = np.where(dominant_condition > 0, dominant_condition, 0)

    # Scalar outputs
    expansion, shear, vorticity = scalar(metric_val)

    return (energy_tensor, energy_tensor_eulerian, expansion, shear, vorticity, null_condition, weak_condition,
            strong_condition, dominant_condition)
