import numpy as np

from analyzer import change_tensor_index, trace
from analyzer.frame_transfer import frame_transfer
from analyzer.utils.generate_uniform_field import generate_uniform_field
from analyzer.utils.get_inner_product import inner_product
from metrics import Metric, minkowski
from solver import Energy, verify_tensor


# TODO: Test


def get_energy_conditions(energy_tensor: Energy, metric_val: Metric, condition: str, num_angular_vec: int = 100,
                          num_time_vec: int = 10, return_all: bool = False, use_gpu: bool = False):
    # Check if correct conditions input
    if condition.casefold() in ["null", "weak", "dominant", "strong"]:
        raise Exception("Incorrect energy condition input, use either: null, weak, dominant, strong")

    # Return warning for any coordinate system not cartesian
    if metric_val.coords.casefold() == "cartesian":
        raise AssertionError("Evaluation not verified for coordinate systems other than Cartesian!")

    # Check tensor formats are correct
    if not verify_tensor(metric_val, True):
        raise Exception("Metric is not verified. Please verify metric using verifyTensor(metric).")

    if not verify_tensor(energy_tensor, True):
        raise Exception("Stress-energy is not verified. Please verify stress-energy using verifyTensor(EnergyTensor).")

    # Get size of the spacetime
    size: tuple = metric_val.tensor.shape[2:]

    # optional return vector
    vec: np.ndarray[np.float64] = np.zeros(size + (num_angular_vec, num_time_vec))

    # Declare variables to be determined in eval of energy conditions
    map_tensor: np.ndarray = np.full(size, np.nan)

    # Convert energy tensor into the local inertial frame if not eulerian
    transformed_energy_tensor: Energy = frame_transfer(metric_val, energy_tensor, "eulerian", use_gpu=use_gpu)

    # Build vector fields
    field_type: str = ""
    if condition.casefold() in ["null", "dominant"]:
        field_type = "nulllike"
    elif condition.casefold() in ["weak", "strong"]:
        field_type = "timelike"
    vec_field: np.ndarray[np.float64] = generate_uniform_field(field_type, num_angular_vec, num_time_vec, use_gpu)

    # Find energy conditions
    # Null energy condition
    if condition.casefold() == "null":
        transformed_energy_tensor = change_tensor_index(transformed_energy_tensor, "covariant",
                                                        metric_val) # double check that it is covariant

        for i in range(num_angular_vec):
            temp: np.ndarray[np.float64] = np.zeros(size)
            for j in range(4):
                for k in range(4):
                    temp += transformed_energy_tensor.tensor[j, k] * vec_field[j, i] * vec_field[k, i]
            map_tensor = np.fmin(map_tensor, temp)
            if return_all:
                vec[:, :, :, :, i] = temp
    # Weak energy condition
    elif condition.casefold() == "weak":
        transformed_energy_tensor = change_tensor_index(transformed_energy_tensor, "covariant",
                                                        metric_val) # double check that it is covariant

        for i in range(num_time_vec):
            for j in range(num_angular_vec):
                temp: np.ndarray[np.float64] = np.zeros(size)
                for k in range(4):
                    for m in range(4):
                        temp += transformed_energy_tensor.tensor[k, m] * vec_field[k, j, i] * vec_field[m, j, i]
                map_tensor = np.fmin(map_tensor, temp)
                if return_all:
                    vec[:, :, :, :, j, i] = temp
    # Dominant energy condition
    elif condition.casefold() == "dominant":
        # Build minkowski reference metric
        metric_minkowski: Metric = change_tensor_index(minkowski(np.ndarray(size)), "covariant") # make sure it is covariant

        transformed_energy_tensor = change_tensor_index(transformed_energy_tensor, "mixedupdown",
                                                        metric_minkowski)  # convert to mixed up down with minkowski

        for i in range(num_angular_vec):
            temp: np.ndarray[np.float64] = np.zeros(size + (4,))
            for j in range(4):
                for k in range(4):
                    temp[:, :, :, :, j] -= transformed_energy_tensor.tensor[j, k] * vec_field[k, i]

            # Wrap into vector struct.
            vector: np.ndarray = np.array([temp[:, :, :, :, 0], temp[:, :, :, :, 1], temp[:, :, :, :, 2], temp[:, :, :, :, 3]])
            vector.index = "contravariant"
            vector.type = "4-vector"

            # Find inner product to determine if timelike or null
            diff = inner_product(vector, vector, metric_minkowski)
            diff = np.sign(diff) * np.sqrt(np.abs(diff))

            map_tensor = np.fmax(map_tensor, diff)
            if return_all:
                vec[:, :, :, :, i] = diff

            # Flip sign of the dominant energy condition to better align with evaluations of
            # other conditions (i.e. negative is violating)
            map_tensor = -map_tensor
            if return_all:
                vec = -vec

    # Strong energy condition
    elif condition.casefold() == "strong":
        # Build minkowski reference metric
        metric_minkowski: Metric = change_tensor_index(minkowski(np.ndarray(size)), "covariant")  # make sure it is covariant

        transformed_energy_tensor = change_tensor_index(transformed_energy_tensor, "covariant",
                                                        metric_minkowski)  # Make sure the energy tensor is covariant

        # Find the trace
        e_trace: np.ndarray[np.float64] = trace(transformed_energy_tensor, metric_minkowski)

        for i in range(num_time_vec):
            for j in range(num_angular_vec):
                temp: np.ndarray[np.float64] = np.zeros(size)
                for k in range(4):
                    for m in range(4):
                        temp += (transformed_energy_tensor.tensor[k, m] - 1/2 * e_trace * metric_minkowski.tensor[k, m]) * vec_field[k, j, i] * vec_field[m, j, i]
                map_tensor = np.fmin(map_tensor, temp)
                if return_all:
                    vec[:, :, :, :, j, i] = temp
    else:
        raise Exception("Unrecognized input energy condition, use either: null, weak, dominant, strong")

    # If returnVec is set, return both the vec (other places in code) and the vectorField
    if return_all:
        return map_tensor, vec, vec_field
    return map_tensor


