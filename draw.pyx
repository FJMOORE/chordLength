import cython
cimport numpy as c_np
import numpy as np
import math
from libc.math cimport sqrt


cdef binary_3d(c_np.ndarray[c_np.float_t, ndim=3] radius_map, float radius):
    cdef int x_max = radius_map.shape[0]
    cdef int y_max = radius_map.shape[1]
    cdef int z_max = radius_map.shape[2]
    cdef float alpha = 0.2765
    cdef c_np.ndarray[c_np.float_t, ndim=3] binary = np.zeros((x_max, y_max, z_max), dtype=np.float_)

    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                if radius_map[x, y, z] <= radius:
                    binary[x, y, z] = 1
    return binary


def simulate_binary_3d(c_np.ndarray[c_np.uint8_t, ndim=3] image,
                         c_np.ndarray[c_np.float64_t, ndim=2] positions,
                         float radius):
    cdef int x_max = image.shape[0]
    cdef int y_max = image.shape[1]
    cdef int z_max = image.shape[2]
    cdef Py_ssize_t dimension = 3
    cdef Py_ssize_t particle_num = positions.shape[1]
    cdef int image_shape[3]
    image_shape[:] = [x_max, y_max, z_max]
    cdef Py_ssize_t lower_bound = 0
    cdef Py_ssize_t upper_bound = 0

    cdef c_np.ndarray[c_np.float_t, ndim=3] new_image = np.zeros(image_shape, dtype=np.float_)
    cdef c_np.ndarray[c_np.float_t, ndim=2] sub_image_indices = np.empty((3, np.max(image_shape)), dtype=np.float_)

    cdef double p[3]
    cdef double box_size[3]
    cdef int sub_image_box[3][2]
    box_size[:] = [radius + 1, radius + 1, radius + 1]
    cdef int sub_image_lengths[3]

    for num in range(particle_num):  # for each particle
        sub_image_indices[:, :] = 0
        p[:] = positions[:, num]
        for dim in range(dimension):
            lower_bound = max([0, math.floor(p[dim] - box_size[dim])])
            upper_bound = min([image_shape[dim] - 1, math.ceil(p[dim] + box_size[dim])])
            sub_image_box[dim][0] = lower_bound
            sub_image_box[dim][1] = upper_bound + 1
            sub_image_lengths[dim] = upper_bound - lower_bound + 1

            try:
                sub_image_indices[dim, :sub_image_lengths[dim]] = np.linspace(
                    lower_bound - p[dim],
                    upper_bound - p[dim],
                    upper_bound - lower_bound + 1,
                    endpoint=True
                )
            except ValueError:
                print(f'lower boundary: {lower_bound}, upper boundary: {upper_bound}')
                print(f'image shape: {image_shape}, dimension: {dim}, particle: {p}, box_size: {box_size}')
                print(f'radius: {radius}')
                raise ValueError()

        mesh = np.meshgrid(
            sub_image_indices[0, :sub_image_lengths[0]],
            sub_image_indices[1, :sub_image_lengths[1]],
            sub_image_indices[2, :sub_image_lengths[2]],
            indexing='ij', sparse=True)

        radius_map = np.sqrt(np.sum(np.array(mesh) ** 2, axis=0))

        new_image[
        sub_image_box[0][0]: sub_image_box[0][1],
        sub_image_box[1][0]: sub_image_box[1][1],
        sub_image_box[2][0]: sub_image_box[2][1]
        ] += binary_3d(radius_map, radius)

    return new_image
