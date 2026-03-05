import tensorflow as tf

from Codebase.PINNCubeDataSet import PINNCubeDataSet
from Codebase.Sampler import Sampler
from Codebase.FEMDataHandler import FEMDataHandler
class PINNLinearElasticityDataSet(PINNCubeDataSet):

    def __init__(self, sampler: Sampler, data_handler: FEMDataHandler, tf_random_generator, np_random_generator, dtype, pressure):
        super().__init__(sampler, data_handler, tf_random_generator, np_random_generator, dtype)

        self.boundary_data = {
            'nxminus': tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nxplus':  tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nyminus': tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nyplus':  tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nzminus': tf.constant([0.0, 0.0, 0.0], dtype=dtype),
            'nzplus':  -pressure*tf.constant([0.0, 0.0, 1.0], dtype=dtype),
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