import numpy as np

from Solver import c_det


def c4_inv(tensor: np.ndarray) -> np.ndarray:
    h, w = tensor.shape[:2]
    assert h == 4 and w == 4, 'Tensor is not 4x4'

    inv_det = 1/c_det(tensor)
    return np.array([[inv_det * (tensor[1,1] * tensor[2,2] * tensor[3,3] - tensor[1,1] * tensor[2,3] * tensor[3,2] -
                        tensor[1,2] * tensor[2,1] * tensor[3,3] + tensor[1,2] * tensor[2,3] * tensor[3,1] + tensor[1,3] *
                        tensor[2,1] * tensor[3,2] - tensor[1,3] * tensor[2,2] * tensor[3,1]), inv_det *
             (tensor[0,1] *tensor[2,3] * tensor[3,2] - tensor[0,1] * tensor[2,2] * tensor[3,3] + tensor[0,2] * tensor[2,1] *
              tensor[3,3] - tensor[0,2] * tensor[2,3] * tensor[3,1] - tensor[0,3] * tensor[2,1] * tensor[3,2] + tensor[0,3] *
              tensor[2,2] * tensor[3,1]), inv_det *
             (tensor[0,1] * tensor[1,2] * tensor[3,3] - tensor[0,1] * tensor[1,3] *tensor[3,2] - tensor[0,2] * tensor[1,1] *
              tensor[3,3] + tensor[0,2] * tensor[1,3] * tensor[3,1] + tensor[0,3] * tensor[1,1] * tensor[3,2] - tensor[0,3] *
              tensor[1,2] * tensor[3,1]), inv_det *
             (tensor[0,1] * tensor[1,3] * tensor[2,2] - tensor[0,1] * tensor[1,2] *tensor[2,3] + tensor[0,2] * tensor[1,1] *
              tensor[2,3] - tensor[0,2] * tensor[1,3] * tensor[2,1] - tensor[0,3] * tensor[1,1] * tensor[2,2] + tensor[0,3] *
              tensor[1,2] * tensor[2,1])],
            [inv_det * (tensor[1,0] * tensor[2,3] * tensor[3,2] - tensor[1,0] * tensor[2,2] * tensor[3,3] +
                        tensor[1,2] * tensor[2,0] * tensor[3,3] - tensor[1,2] * tensor[2,3] * tensor[3,0] - tensor[1,3] *
                        tensor[2,0] * tensor[3,2] + tensor[1,3] * tensor[2,2] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[2,2] * tensor[3,3] - tensor[0,0] * tensor[2,3] * tensor[3,2] - tensor[0,2] * tensor[2,0] *
              tensor[3,3] + tensor[0,2] * tensor[2,3] * tensor[3,0] + tensor[0,3] * tensor[2,0] * tensor[3,2] - tensor[0,3] *
              tensor[2,2] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[1,3] * tensor[3,2] - tensor[0,0] * tensor[1,2] * tensor[3,3] + tensor[0,2] * tensor[1,0] *
              tensor[3,3] - tensor[0,2] * tensor[1,3] * tensor[3,0] - tensor[0,3] * tensor[1,0] * tensor[3,2] + tensor[0,3] *
              tensor[1,2] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[1,2] * tensor[2,3] - tensor[0,0] * tensor[1,3] * tensor[2,2] - tensor[0,2] * tensor[1,0] *
              tensor[2,3] + tensor[0,2] * tensor[1,3] * tensor[2,0] + tensor[0,3] * tensor[1,0] * tensor[2,2] - tensor[0,3] *
              tensor[1,2] * tensor[2,0])],
            [inv_det * (tensor[1,0] * tensor[2,1] * tensor[3,3] - tensor[1,0] * tensor[2,3] * tensor[3,1] -
                        tensor[1,1] * tensor[2,0] * tensor[3,3] + tensor[1,1] * tensor[2,3] * tensor[3,0] + tensor[1,3] *
                        tensor[2,0] * tensor[3,1] - tensor[1,3] * tensor[2,1] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[2,3] * tensor[3,1] - tensor[0,0] * tensor[2,1] * tensor[3,3] + tensor[0,1] * tensor[2,0] *
              tensor[3,3] - tensor[0,1] * tensor[2,3] * tensor[3,0] - tensor[0,3] * tensor[2,0] * tensor[3,1] + tensor[0,3] *
              tensor[2,1] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[1,1] * tensor[3,3] - tensor[0,0] * tensor[1,3] * tensor[3,1] - tensor[0,1] * tensor[1,0] *
              tensor[3,3] + tensor[0,1] * tensor[1,3] * tensor[3,0] + tensor[0,3] * tensor[1,0] * tensor[3,1] - tensor[0,3] *
              tensor[1,1] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[1,3] * tensor[2,1] - tensor[0,0] * tensor[1,1] * tensor[2,3] + tensor[0,1] * tensor[1,0] *
              tensor[2,3] - tensor[0,1] * tensor[1,3] * tensor[2,0] - tensor[0,3] * tensor[1,0] * tensor[2,1] + tensor[0,3] *
              tensor[1,1] * tensor[2,0])],
            [inv_det * (tensor[1,0] * tensor[2,2] * tensor[3,1] - tensor[1,0] * tensor[2,1] * tensor[3,2] + tensor[1,1] *
                        tensor[2,0] * tensor[3,2] - tensor[1,1] * tensor[2,2] * tensor[3,0] - tensor[1,2] * tensor[2,0] *
                        tensor[3,1] + tensor[1,2] * tensor[2,1] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[2,1] * tensor[3,2] - tensor[0,0] * tensor[2,2] * tensor[3,1] - tensor[0,1] * tensor[2,0] *
              tensor[3,2] + tensor[0,1] * tensor[2,2] * tensor[3,0] + tensor[0,2] * tensor[2,0] * tensor[3,1] - tensor[0,2] *
              tensor[2,1] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[1,2] * tensor[3,1] - tensor[0,0] * tensor[1,1] * tensor[3,2] + tensor[0,1] * tensor[1,0] *
              tensor[3,2] - tensor[0,1] * tensor[1,2] * tensor[3,0] - tensor[0,2] * tensor[1,0] * tensor[3,1] + tensor[0,2] *
              tensor[1,1] * tensor[3,0]), inv_det *
             (tensor[0,0] * tensor[1,1] * tensor[2,2] - tensor[0,0] * tensor[1,2] * tensor[2,1] - tensor[0,1] * tensor[1,0] *
              tensor[2,2] + tensor[0,1] * tensor[1,2] * tensor[2,0] + tensor[0,2] * tensor[1,0] * tensor[2,1] - tensor[0,2] *
              tensor[1,1] * tensor[2,0])]])
