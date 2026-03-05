import numpy as np
import tensorflow as tf

from pinn_source.data_handler import FEMDataHandler


class PINNDataSet:

    def __init__(self, data_handler: FEMDataHandler, tf_random_generator, np_random_generator, dtype, pressure):

        self._num_points = {
            'train': {'bc': {}, 'pde': {}, 'data': {}, 'reg': {}},
            'test': {'bc': {}, 'pde': {}, 'data': {}, 'reg': {}}
        }
        self._points = {
            'train': {'bc': {}, 'pde': {}, 'data': {}, 'reg': {}},
            'test': {'bc': {}, 'pde': {}, 'data': {}, 'reg': {}}
        }
        self._batch_size_fraction = 1.0

        self.dtype = dtype
        self.data_handler = data_handler
        self.tf_random_generator = tf_random_generator
        self.np_random_generator = np_random_generator

        self.bc_plane_corners = {
            'min': {k: [] for k in ['nxminus', 'nxplus', 'nyminus', 'nyplus', 'nzminus', 'nzplus']},
            'max': {k: [] for k in ['nxminus', 'nxplus', 'nyminus', 'nyplus', 'nzminus', 'nzplus']}
        }

        self.boundary_data = {
            'nxminus': tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nxplus':  tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nyminus': tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nyplus':  tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nzminus': tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nzplus':  -pressure * tf.constant([0.0, 0.0, 1.0], dtype=dtype),
        }

#############################################################################
# Data access
#############################################################################

    def get_labels(self, category, learning_type):
        return self._num_points[learning_type][category].keys()

    def set_batch_size_fraction(self, batch_size):
        self._batch_size_fraction = batch_size

    def get_data(self, name, category, learning_type):
        data = self._points[learning_type][category][name]
        num = self._num_points[learning_type][category][name]
        num_batched = int(num * self._batch_size_fraction)
        return data, num, num_batched

#############################################################################
# Point setup
#############################################################################

    def set_bc_plane_corners(self):
        min_spatial = self.data_handler.min_dim
        max_spatial = self.data_handler.max_dim

        self.bc_plane_corners = {
            'min': {
                'nxminus': [min_spatial[0], min_spatial[1], min_spatial[2]],
                'nxplus':  [max_spatial[0], min_spatial[1], min_spatial[2]],
                'nyminus': [min_spatial[0], min_spatial[1], min_spatial[2]],
                'nyplus':  [min_spatial[0], max_spatial[1], min_spatial[2]],
                'nzminus': [min_spatial[0], min_spatial[1], min_spatial[2]],
                'nzplus':  [min_spatial[0], min_spatial[1], max_spatial[2]]
            },
            'max': {
                'nxminus': [min_spatial[0], max_spatial[1], max_spatial[2]],
                'nxplus':  [max_spatial[0], max_spatial[1], max_spatial[2]],
                'nyminus': [max_spatial[0], min_spatial[1], max_spatial[2]],
                'nyplus':  [max_spatial[0], max_spatial[1], max_spatial[2]],
                'nzminus': [max_spatial[0], max_spatial[1], min_spatial[2]],
                'nzplus':  [max_spatial[0], max_spatial[1], max_spatial[2]]
            }
        }

    def set_bc_points(self, nxminus, nxplus, nyminus, nyplus, nzminus, nzplus, learning_type):
        self._num_points[learning_type]['bc'] = {
            'nxminus': nxminus,
            'nxplus': nxplus,
            'nyminus': nyminus,
            'nyplus': nyplus,
            'nzminus': nzminus,
            'nzplus': nzplus
        }

    def set_pde_points(self, pde, learning_type):
        self._num_points[learning_type]['pde'] = {
            0: pde
        }

    def set_data_points(self, displacement, learning_type):
        self._num_points[learning_type]['data'] = {
            'displacement': displacement,
            'x_displacement': displacement,
        }

#############################################################################
# Sampling (Sobol sequences)
#############################################################################

    @staticmethod
    def _sobol_cube(num_points, minval, maxval, dim, dtype):
        scalings = [u - v for u, v in zip(maxval, minval)]
        points = tf.math.sobol_sample(dim, num_points, dtype=dtype)
        scaling = tf.linalg.diag(tf.constant(scalings, dtype=dtype))
        return tf.math.add(tf.linalg.matmul(points, scaling), tf.constant(minval, dtype=dtype))

    @staticmethod
    def _sobol_plane(num_points, minval, maxval, dtype):
        scalings = [u - v for u, v in zip(maxval, minval)]
        dim = len(scalings)

        if scalings[0] == 0:
            axis1, axis2 = 1, 2
        elif scalings[1] == 0:
            axis1, axis2 = 0, 2
        elif scalings[2] == 0:
            axis1, axis2 = 0, 1
        else:
            raise ValueError("No constant axis found — not a plane")

        points = tf.math.sobol_sample(2, num_points, dtype=dtype)
        result = np.zeros((num_points, dim), dtype=np.float64)
        for i in range(num_points):
            result[i, axis1] = scalings[axis1] * points[i, 0]
            result[i, axis2] = scalings[axis2] * points[i, 1]
            result[i] += minval
        return tf.constant(result, dtype=dtype)

    def sample_bc_points(self):
        dim = self.data_handler.mesh_dimension

        for type in self._num_points['train']['bc']:
            num_train = self._num_points['train']['bc'][type]
            num_test = self._num_points['test']['bc'][type]

            min_dim = self.bc_plane_corners['min'][type]
            max_dim = self.bc_plane_corners['max'][type]

            num = num_train + num_test
            sample = self._sobol_plane(num, min_dim, max_dim, self.dtype)
            sample_train, sample_test = sample[:num_train], sample[-num_test:]

            self._points['train']['bc'][type] = sample_train[:, :dim]
            self._points['test']['bc'][type] = sample_test[:, :dim]

    def sample_pde_points(self):
        dim = self.data_handler.mesh_dimension
        min_spatial = self.data_handler.min_dim
        max_spatial = self.data_handler.max_dim

        num_train = self._num_points['train']['pde'][0]
        num_test = self._num_points['test']['pde'][0]

        num = num_train + num_test
        sample = self._sobol_cube(num, min_spatial, max_spatial, dim, self.dtype)
        sample_train, sample_test = sample[:num_train], sample[-num_test:]

        self._points['train']['pde'][0] = sample_train
        self._points['test']['pde'][0] = sample_test

    def sample_displacement_points(self, time_scale = 2.0):
        num_train = self._num_points['train']['data']['displacement']
        num_test = self._num_points['test']['data']['displacement']

        space_dim = self.data_handler.mesh_dimension

        x_test, x_displaced_test, u_test, x_data, x_displaced_data, u_data = self.data_handler.get_random(num_train, num_test, time_scale, self.np_random_generator)

        self._points['train']['data']['displacement'] = u_data
        self._points['test']['data']['displacement'] = u_test

        self._points['train']['data']['x_displacement'] = x_data[:, :space_dim]
        self._points['test']['data']['x_displacement'] = x_test[:, :space_dim]

    def combine_collocation_points(self, categories: list[str], learning_type: str):
        x_data = []
        for category in categories:
            for name in self._points[learning_type][category]:
                d = self._points[learning_type][category][name]
                if isinstance(d, tuple):
                    d = d[0]
                x_data += [d]

        x_prior = np.concatenate(x_data, axis=0)

        self._points[learning_type]['reg']['x_prior'] = x_prior
        self._num_points[learning_type]['reg']['x_prior'] = x_prior.shape[0]