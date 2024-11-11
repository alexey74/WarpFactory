import numpy as np

from analyzer import change_tensor_index
from analyzer.utils.get_eulerian_transformation_matrix import get_eulerian_transformation_matrix
from metrics import Metric
from solver import Energy, verify_tensor


def frame_transfer(metric_val: Metric, energy_tensor: Energy, frame: str, use_gpu: bool = False) -> Energy:
    transformed_energy_tensor: Energy = energy_tensor
    transformed_energy_tensor = np.zeros(energy_tensor.tensor.shape)

    assert not verify_tensor(metric_val, True), ("Metric is not verified."
                                                 "Please verify metric using verifyTensor(metric).")
    assert not verify_tensor(energy_tensor, True), ("Stress-energy is not verified."
                                                    "Please verify Stress-energy tensor using verifyTensor(energyTensor).")

    if frame == "eulerian" and energy_tensor.frame == "eulerian":
        # Convert to covariant (lower) index
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric_val)

        # Do transformations at each point in space
        big_m = metric_val
        big_m.tensor = get_eulerian_transformation_matrix(metric_val.tensor)
        big_m = np.transpose(big_m, (4, 5, 0, 1, 2, 3))
        energy_tensor = np.transpose(energy_tensor, (4, 5, 0, 1, 2, 3))

        transformed_temp_tensor: Energy = Energy("Temp Energy Tensor")
        transformed_temp_tensor.tensor = np.einsum('abcdmp,abcdpn->abcdmn', np.einsum('abcdmn,abcdnp->abcdmp',
                                                    np.transpose(big_m, (0, 1, 2, 3, 5, 4)), energy_tensor, optimize=True),
                                                   big_m, optimize=True)

        # Transform to contravariant T^{0, i} = -T_{0, i}
        for i in range(1, 4):
            transformed_energy_tensor.tensor[0, i] = -transformed_energy_tensor.tensor[0, i]
            transformed_energy_tensor.tensor[i, 0] = -transformed_energy_tensor.tensor[i, 0]

        # Update the tensor metadata
        transformed_energy_tensor.frame = "eulerian"
        transformed_energy_tensor.index = "contravariant"
    else:
        raise AssertionError("Frame not found")
    return transformed_energy_tensor
