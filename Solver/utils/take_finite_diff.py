import numpy as np


def take_finite_diff_1dir(tensor: np.ndarray[np.float64], k: int, delta: np.ndarray[np.float64], phi_flag: bool):
    size: tuple[int, ...] = tensor.shape

    big_b: np.ndarray[np.float64] = np.zeros(size)

    if size[k] >= 5:
        if k == 0:
            big_b[2:-2, :, :, :] = (-(tensor[4:, :, :, :] - tensor[:-4, :, :, :]) +
                                    8 * (tensor[3:-1, :, :, :] - tensor[1:-3, :, :, :])) / (12 * delta[k])
            big_b[0, :, :, :] = big_b[2, :, :, :]
            big_b[1, :, :, :] = big_b[2, :, :, :]
            big_b[-2, :, :, :] = big_b[-3, :, :, :]
            big_b[-1, :, :, :] = big_b[-3, :, :, :]
        elif k == 1:
            big_b[:, 2:-2, :, :] = (-(tensor[:, 4:, :, :] - tensor[:, :-4, :, :]) +
                                    8 * (tensor[:, 3:-1, :, :] - tensor[:, 1:-3, :, :])) / (12 * delta[k])
            big_b[:, 0, :, :] = big_b[:, 2, :, :]
            big_b[:, 1, :, :] = big_b[:, 2, :, :]
            big_b[:, -2, :, :] = big_b[:, -3, :, :]
            big_b[:, -1,  :, :] = big_b[:, -3, :, :]
        elif k == 2:
            big_b[:, :, 2:-2, :] = (-(tensor[:, :, 4:, :] - tensor[:, :, :-4, :]) +
                                    8 * (tensor[:, :, 3:-1, :] - tensor[:, :, 1:-3, :])) / (12 * delta[k])
            if phi_flag:
                big_b[:, :, -2, :] = 2 * (size[2] - 6)
                big_b[:, :, -1, :] = 2 * (size[2] - 5)
                big_b[:, :, 0, :] = 2 * 4
                big_b[:, :, 1, :] = 2 * 3
            else:
                big_b[:, :, -2, :] = big_b[:, :, -3, :]
                big_b[:, :, -1, :] = big_b[:, :, -3, :]
                big_b[:, :, 0, :] = big_b[:, :, 2, :]
                big_b[:, :, 1, :] = big_b[:, :, 2, :]
        elif k == 3:
            big_b[:, :, :, 2:-2] = (-(tensor[:, :, :, 4:] - tensor[:, :, :, :-4]) +
                                    8 * (tensor[:, :, :, 3:-1] - tensor[:, :, :, 1:-3])) / (12 * delta[k])
            big_b[:, :, :, 0] = big_b[:, :, :, 2]
            big_b[:, :, :, 1] = big_b[:, :, :, 2]
            big_b[:, :, :, -2] = big_b[:, :, :, -3]
            big_b[:, :, :, -1] = big_b[:, :, :, -3]
    return big_b


def take_finite_diff_2dirs(tensor: np.ndarray[np.float64], k1: int, k2: int,
                           delta: int, phi_flag: bool) -> np.ndarray[np.float64]:
    size: tuple[int, ...] = tensor.shape

    big_b: np.ndarray[np.float64] = np.zeros(size)

    if size[k1] >= 5 and size[k2] >= 5:
        if k1 == k2:
            if k1 == 0:
                big_b[2:-2, :, :, :] = ((-(tensor[4:, :, :, :] + tensor[:-4, :, :, :]) +
                                        16 * (tensor[3:-1, :, :, :] + tensor[1:-3, :, :, :]) - 30 * tensor[2:-2, :, :, :]) /
                                        (12 * delta[k1]**2))
                big_b[0, :, :, :] = big_b[2, :, :, :]
                big_b[1, :, :, :] = big_b[2, :, :, :]
                big_b[-2, :, :, :] = big_b[-3, :, :, :]
                big_b[-1, :, :, :] = big_b[-3, :, :, :]
            elif k1 == 1:
                big_b[:, 2:-2, :, :] = ((-(tensor[:, 4:, :, :] + tensor[:, :-4, :, :]) +
                                         16 * (tensor[:, 3:-1, :, :] + tensor[:, 1:-3, :, :]) - 30 * tensor[:, 2:-2, :, :]) /
                                        (12 * delta[k1]**2))
                big_b[:, 0, :, :] = big_b[:, 2, :, :]
                big_b[:, 1, :, :] = big_b[:, 2, :, :]
                big_b[:, -2, :, :] = big_b[:, -3, :, :]
                big_b[:, -1, :, :] = big_b[:, -3, :, :]
            elif k1 == 2:
                big_b[:, :, 2:-2, :] = ((-(tensor[:, :, 4:, :] + tensor[:, :, :-4, :]) +
                                         16 * (tensor[:, :, 3:-1, :] + tensor[:, :, 1:-3, :]) - 30 * tensor[:, :, 2:-2, :]) /
                                        (12 * delta[k1]**2))
                if phi_flag:
                    big_b[:, :, 0, :] = -2
                    big_b[:, :, 1, :] = -2
                    big_b[:, :, -2, :] = 2
                    big_b[:, :, -1, :] = 2
                else:
                    big_b[:, :, -2, :] = big_b[:, :, -3, :]
                    big_b[:, :, -1, :] = big_b[:, :, -3, :]
                    big_b[:, :, 0, :] = big_b[:, :, 2, :]
                    big_b[:, :, 1, :] = big_b[:, :, 2, :]
            elif k1 == 3:
                big_b[:, :, :, 2:-2] = ((-(tensor[:, :, :, 4:] + tensor[:, :, :, :-4]) +
                                         16 * (tensor[:, :, :, 3:-1] + tensor[:, :, :, 1:-3]) - 30 * tensor[:, :, :, 2:-2]) /
                                        (12 * delta[k1]**2))
                big_b[:, :, :, 0] = big_b[:, :, :, 2]
                big_b[:, :, :, 1] = big_b[:, :, :, 2]
                big_b[:, :, :, -2] = big_b[:, :, :, -3]
                big_b[:, :, :, -1] = big_b[:, :, :, -3]
    else:
        # Todo: Test this part too
        k_l: np.int64 = np.max([k1, k2])
        k_s: np.int64 = np.min([k1, k2])

        x0: range = range(3, size[k_s] - 1)
        x1: range = range(4, size[k_s])
        x2: range = range(5, size[k_s] + 1)
        x_1: range = range(2, size[k_s] - 2)
        x_2: range = range(1, size[k_s] - 3)

        y0: range = range(3, size[k_l] - 1)
        y1: range = range(4, size[k_l])
        y2: range = range(5, size[k_l] + 1)
        y_1: range = range(2, size[k_l] - 2)
        y_2: range = range(1, size[k_l] - 3)

        if k_s == 0:
            if k_l == 1:  # partial 0 / partial 1
                big_b[x0, y0, :, :] = 1 / (12**2 * delta[k_l] * delta[k_s]) * (
                        -(-(tensor[x2, y2, :, :] - tensor[x_2, y2, :, :]) + 8 * (tensor[x1, y2, :, :] - tensor[x_1, y2, :, :]))
                        + ( - (tensor[x2, y_2, :, :] - tensor[x_2, y_2, :, :]) + 8 * (tensor[x1, y_2, :, :] - tensor[x_1, y_2, :, :]))
                        + 8 * ( - (tensor[x2, y1, :, :] - tensor[x_2, y1, :, :]) + 8 * (tensor[x1, y1, :, :] - tensor[x_1, y1, :, :]))
                        - 8 * ( - (tensor[x2, y_1, :, :] - tensor[x_2, y_1, :, :]) + 8 * (tensor[x1, y_1, :, :] - tensor[x_1, y_1, :, :])))
            elif k_l == 2:  # partial 0 / partial 2
                big_b[x0, :, y0, :] = 1 / (12**2 * delta[k_l] * delta[k_s]) * (
                        -(-(tensor[x2, :, y2, :] - tensor[x_2, :, y2, :]) + 8 * (tensor[x1, :, y2, :] - tensor[x_1, :, y2, :]))
                        + ( - (tensor[x2, :, y_2, :] - tensor[x_2, :, y_2, :]) + 8 * (tensor[x1, :, y_2, :] - tensor[x_1, :, y_2, :]))
                        + 8 * ( - (tensor[x2, :, y1, :] - tensor[x_2, :, y1, :]) + 8 * (tensor[x1, :, y1, :] - tensor[x_1, :, y1, :]))
                        - 8 * ( - (tensor[x2, :, y_1, :] - tensor[x_2, :, y_1, :]) + 8 * (tensor[x1, :, y_1, :] - tensor[x_1, :, y_1, :])))
            elif k_l == 3:  # partial 0 / partial 3
                big_b[x0, :, :, y0] = 1 / (12**2 * delta[k_l] * delta[k_s]) * (
                        -(-(tensor[x2, :, :, y2] - tensor[x_2, :, :, y2]) + 8 * (tensor[x1, :, :, y2] - tensor[x_1, :, :, y2]))
                        + ( - (tensor[x2, :, :, y_2] - tensor[x_2, :, :, y_2]) + 8 * (tensor[x1, :, :, y_2] - tensor[x_1, :, :, y_2]))
                        + 8 * ( - (tensor[x2, :, :, y1] - tensor[x_2, :, :, y1]) + 8 * (tensor[x1, :, :, y1] - tensor[x_1, :, :, y1]))
                        - 8 * ( - (tensor[x2, :, :, y_1] - tensor[x_2, :, :, y_1]) + 8 * (tensor[x1, :, :, y_1] - tensor[x_1, :, :, y_1])))
        elif k_s == 1:
            if k_l == 2:  # partial 1 / partial 2
                big_b[:, x0, y0, :] = 1 / (12**2 * delta[k_l] * delta[k_s]) * (
                        -(-(tensor[:, x2, y2, :] - tensor[:, x_2, y2, :]) + 8 * (tensor[:, x1, y2, :] - tensor[:, x_1, y2, :]))
                        + ( - (tensor[:, x2, y_2, :] - tensor[:, x_2, y_2, :]) + 8 * (tensor[:, x1, y_2, :] - tensor[:, x_1, y_2, :]))
                        + 8 * ( - (tensor[:, x2, y1, :] - tensor[:, x_2, y1, :]) + 8 * (tensor[:, x1, y1, :] - tensor[:, x_1, y1, :]))
                        - 8 * ( - (tensor[:, x2, y_1, :] - tensor[:, x_2, y_1, :]) + 8 * (tensor[:, x1, y_1, :] - tensor[:, x_1, y_1, :])))
            elif k_l == 3:  # partial 1 / partial 3
                big_b[:, x0, :, y0] = 1 / (12**2 * delta[k_l] * delta[k_s]) * (
                        -(-(tensor[:, x2, :, y2] - tensor[:, x_2, :, y2]) + 8 * (tensor[:, x1, :, y2] - tensor[:, x_1, :, y2]))
                        + ( - (tensor[:, x2, :, y_2] - tensor[:, x_2, :, y_2]) + 8 * (tensor[:, x1, :, y_2] - tensor[:, x_1, :, y_2]))
                        + 8 * ( - (tensor[:, x2, :, y1] - tensor[:, x_2, :, y1]) + 8 * (tensor[:, x1, :, y1] - tensor[:, x_1, :, y1]))
                        - 8 * ( - (tensor[:, x2, :, y_1] - tensor[:, x_2, :, y_1]) + 8 * (tensor[:, x1, :, y_1] - tensor[:, x_1, :, y_1])))
        elif k_s == 2:
            if k_l == 3:  # partial 2 / partial 3
                big_b[:, :, x0, y0] = 1 / (12**2 * delta[k_l] * delta[k_s]) * (
                        -(-(tensor[:, :, x2, y2] - tensor[:, :, x_2, y2]) + 8 * (tensor[:, :, x1, y2] - tensor[:, :, x_1, y2]))
                        + ( - (tensor[:, :, x2, y_2] - tensor[:, :, x_2, y_2]) + 8 * (tensor[:, :, x1, y_2] - tensor[:, :, x_1, y_2]))
                        + 8 * ( - (tensor[:, :, x2, y1] - tensor[:, :, x_2, y1]) + 8 * (tensor[:, :, x1, y1] - tensor[:, :, x_1, y1]))
                        - 8 * ( - (tensor[:, :, x2, y_1] - tensor[:, :, x_2, y_1]) + 8 * (tensor[:, :, x1, y_1] - tensor[:, :, x_1, y_1])))
    return big_b
