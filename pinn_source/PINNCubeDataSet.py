import numpy as np
import tensorflow as tf

from Codebase.PINNDataSet import PINNDataSet
from Codebase.Sampler import Sampler
from Codebase.FEMDataHandler import FEMDataHandler

from Codebase.constitutive import LinearElasticity_Neumann_approx

class PINNCubeDataSet(PINNDataSet):

    def __init__(self, sampler: Sampler, data_handler: FEMDataHandler, tf_random_generator, np_random_generator, dtype):
        super().__init__()

        self.sampler = sampler
        self.data_handler = data_handler

        self.tf_random_generator = tf_random_generator
        self.np_random_generator = np_random_generator

        self.bc_plane_corners = {
            'min': {
                'nxminus': [],
                'nxplus': [],
                'nyminus': [],
                'nyplus': [],
                'nzminus': [],
                'nzplus': []
            },
            'max': {
                'nxminus': [],
                'nxplus': [],
                'nyminus': [],
                'nyplus': [],
                'nzminus': [],
                'nzplus': []
            }
        }

        self.boundary_data = {
            'nxminus': tf.zeros(self.data_handler.mesh_dimension, dtype=dtype),
            'nxplus': tf.zeros(self.data_handler.mesh_dimension, dtype=dtype),
            'nyminus': tf.zeros(self.data_handler.mesh_dimension, dtype=dtype),
            'nyplus': tf.zeros(self.data_handler.mesh_dimension, dtype=dtype),
            'nzminus': tf.zeros(self.data_handler.mesh_dimension, dtype=dtype),
            'nzplus': tf.zeros(self.data_handler.mesh_dimension, dtype=dtype)
        }

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

    def set_data_points(self, displacement, stress, strain, learning_type):
        self._num_points[learning_type]['data'] = {
            'displacement': displacement,
            'x_displacement': displacement,
            'stress': stress,
            'x_stress': stress,
            'strain': strain,
            'x_strain': strain,
        }

    def sample_bc_points(self, sample_type, time_scale = 2.0):

        dim = self.data_handler.mesh_dimension

        for type in self._num_points['train']['bc']:

            num_train = self._num_points['train']['bc'][type]
            num_test = self._num_points['test']['bc'][type]

            min_dim = self.bc_plane_corners['min'][type]
            max_dim = self.bc_plane_corners['max'][type]

            sample_train, sample_test = self.sampler.get_boundary_sample(num_train, num_test, dim,
                                        min_dim, max_dim, sample_type,
                                        self.tf_random_generator, False)
            
            #sample_test, strain_test, sample_train, strain_data, mu_test, mu_train = self.data_handler.get_random_strain(num_train, num_test, time_scale, self.np_random_generator, reference_submesh_names = [type])

            #mult = 1.0
            #if type[1] == 'x':
            #    axis = 0
            #elif type[1] == 'y':
            #    axis = 1
            #    mult = 0.0
            #elif type[1] == 'z':
            #    axis = 2
            #    mult = 0.0

            #if type[2:] == 'minus':
            #    direction = 1.0
            #elif type[2:] == 'plus':
            #    direction = -1.0
            
            self._points['train']['bc'][type] = sample_train[:, :dim]#, mult * LinearElasticity_Neumann_approx(strain_data, axis, direction, 650., mu_train)
            self._points['test']['bc'][type] = sample_test[:, :dim]#, mult * LinearElasticity_Neumann_approx(strain_test, axis, direction, 650., mu_test)

    def sample_pde_points(self, sample_type):
        dim = self.data_handler.mesh_dimension
        min_spatial = self.data_handler.min_dim
        max_spatial = self.data_handler.max_dim

        num_train = self._num_points['train']['pde'][0]
        num_test = self._num_points['test']['pde'][0]

        sample_train, sample_test = self.sampler.get_sample(num_train, num_test, dim, min_spatial, max_spatial, sample_type,
                           self.tf_random_generator)

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

    def sample_strain_points(self, time_scale = 2.0):
        num_train = self._num_points['train']['data']['strain']
        num_test = self._num_points['test']['data']['strain']
        
        space_dim = self.data_handler.mesh_dimension
        x_test, u_test, x_data, u_data = self.data_handler.get_random_strain(num_train, num_test, time_scale, self.np_random_generator)
        
        self._points['train']['data']['strain'] = u_data
        self._points['test']['data']['strain'] = u_test

        self._points['train']['data']['x_strain'] = x_data[:, :space_dim]
        self._points['test']['data']['x_strain'] = x_test[:, :space_dim]

    def sample_stress_points(self, time_scale = 2.0):
        num_train = self._num_points['train']['data']['stress']
        num_test = self._num_points['test']['data']['stress']

        space_dim = self.data_handler.mesh_dimension
        x_test, u_test, x_data, u_data = self.data_handler.get_random_stress(num_train, num_test, time_scale, self.np_random_generator)

        self._points['train']['data']['stress'] = u_data
        self._points['test']['data']['stress'] = u_test

        self._points['train']['data']['x_stress'] = x_data[:, :space_dim]
        self._points['test']['data']['x_stress'] = x_test[:, :space_dim]

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
