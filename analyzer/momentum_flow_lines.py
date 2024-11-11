"""
GETMOMENTUMFLOWLINES: Gets the momentum flow lines for an energy tensor

INPUTS:
    energyTensor - Energy struct

    startPoints - 1x3 cell array of the start points of flowlines
    startPoints{1} = X;
    startPoints{2} = Y;
    startPoints{3} = Z;

    stepSize - Step size of the flowline propagation

    maxSteps - The max number of propagation steps to run

    scaleFactor - The scaling factor that multiplies the momentum density


OUTPUTS:
    paths - 1xN cell array containing N paths. The path in each cell is an Mx3 array.
"""
import numpy as np

from solver import Energy, trilinear_interpolation


def momentum_flow_lines(input_energy: Energy, start_points, step_size: np.float64, max_steps: int,
                        scale_factor: np.float64):
    # Check that the energyTensor is contravariant
    if input_energy.index != "contravariant":
        raise ValueError('Energy tensor for momentum flowlines should be contravariant.')

    # Load in the momentum data
    x_mom = np.squeeze(input_energy.tensor[0, 1]) * scale_factor
    y_mom = np.squeeze(input_energy.tensor[0, 2]) * scale_factor
    z_mom = np.squeeze(input_energy.tensor[0, 3]) * scale_factor

    # Reshape the starting points X, Y, and Z
    str_pts_x = start_points[0].reshape(-1)
    str_pts_y = start_points[1].reshape(-1)
    str_pts_z = start_points[2].reshape(-1)

    # Make the paths
    paths = []
    for j in range(1, len(str_pts_x) + 1):
        pos = np.zeros((max_steps, 3))
        pos[0, :] = np.array([str_pts_x[j - 1], str_pts_y[j - 1], str_pts_z[j - 1]])

        for i in range(max_steps - 1):
            # Check if the particle is outside the world
            if (np.sum(np.isnan(pos[i, :])) > 0 or (np.floor(pos[i, 0]) <= 1) or (np.ceil(pos[i, 0]) >= x_mom.shape[0]) or
                    (np.floor(pos[i, 1]) <= 1) or (np.ceil(pos[i, 1]) >= x_mom.shape[1]) or (np.floor(pos[i, 2]) <= 1) or
                    (np.ceil(pos[i, 2]) >= x_mom.shape[2])):
                break

            x_momentum = trilinear_interpolation(x_mom, pos[i, :])
            y_momentum = trilinear_interpolation(y_mom, pos[i, :])
            z_momentum = trilinear_interpolation(z_mom, pos[i, :])

            # Propagate position
            pos[i + 1, 0] = pos[i, 0] + x_momentum * step_size
            pos[i + 1, 1] = pos[i, 1] + y_momentum * step_size
            pos[i + 1, 2] = pos[i, 2] + z_momentum * step_size

        paths.append(pos[:j - 1, :])
    return np.array(paths, dtype=np.ndarray)
