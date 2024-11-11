import numpy as np

from analyzer import get_even_points_on_sphere


# TODO: Test
# TODO: Implement GPU Support


def generate_uniform_field(field_type: str, num_angular_vec: int, num_time_vec: int, use_gpu: bool = False) -> np.ndarray[np.float64]:
    if not field_type == 'nulllike' or not field_type == 'timelike':
        assert 'Vector field type not generated, use either: "nulllike", "timelike"'

    if field_type == 'timelike':
        # generate timelike vectors c^2t^2 > r^2
        bb: np.ndarray[np.float64] = np.linspace(0, 1, num_time_vec)
        vec_field: np.ndarray[np.float64] = np.ones((4, num_angular_vec, num_time_vec))

        for i in range(num_time_vec):
            # build vector field in cartesian coords
            vec_field[:, :, i] = np.stack((np.ones(num_angular_vec),
                                           get_even_points_on_sphere(1 - bb[i], num_angular_vec, 1)), axis=0)
            vec_field[:, :, i] = vec_field[:, :, i] / np.emath.sqrt(vec_field[0, :, i]**2 + vec_field[1, :, i]**2 + vec_field[2, :, i]**2 + vec_field[3, :, i]**2)
    elif field_type == 'nulllike':
        # build vector field in cartesian coords
        vec_field: np.ndarray[np.float64] = np.ones((4, num_angular_vec))
        vec_field[1:] = get_even_points_on_sphere(1.0, num_angular_vec, use_gpu)
        vec_field = vec_field / np.emath.sqrt(vec_field[0]**2 + vec_field[1]**2 + vec_field[2]**2 + vec_field[3]**2)
    return vec_field
