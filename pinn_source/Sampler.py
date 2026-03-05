import tensorflow as tf
import numpy as np
from scipy.stats import qmc

class Sampler:

    def __init__(self, dtype):

        self.dtype = dtype

#############################################################################
# Sobol sampling
#############################################################################
    def sobol_sequence_cube(self, num_points, minval, maxval, dim):

        scalings = [u - v for u, v in zip(maxval, minval)]
        points = tf.math.sobol_sample(dim, num_points, dtype=self.dtype)
        scaling = tf.linalg.diag(tf.constant(scalings, dtype=self.dtype))
        return tf.math.add(tf.linalg.matmul(points, scaling), tf.constant(minval, dtype=self.dtype))

    def sobol_sequence_plane(self, num_points, axis1, axis2, scaling1, scaling2,
                             shift=tf.zeros(4, dtype=tf.double), dim=4, time_axis=True, time_scaling=None):
        points = tf.math.sobol_sample(dim - 1, num_points, dtype=self.dtype)
        points_3d = np.zeros((num_points, dim), dtype=np.float64)
        for i, p in enumerate(points):
            points_3d[i, axis1] = scaling1 * points[i, 0]
            points_3d[i, axis2] = scaling2 * points[i, 1]
            if time_axis:
                points_3d[i, dim - 1] = time_scaling * points[i, 2]
            points_3d[i] += shift
        return tf.constant(points_3d, dtype=self.dtype)

#############################################################################
# Latin Hypercube sampling
#############################################################################
    def latin_hypercube_sequence_cube(self, num_points, minval, maxval, dim):
        sampler = qmc.LatinHypercube(d=dim)  # , optimization="random-cd") # optimization step slow
        points = sampler.random(n=num_points)
        return tf.constant(qmc.scale(points, minval, maxval), dtype=self.dtype)

    def latin_hypercube_sequence_plane(self, num_points, axis1, axis2, scaling1, scaling2,
                                       shift=tf.zeros(4, dtype=tf.double), dim=4, time_axis=True,
                                       time_scaling=None):
        sampler = qmc.LatinHypercube(d=dim - 1)  # , optimization="random-cd")
        points = sampler.random(n=num_points)
        points_3d = np.zeros((num_points, dim), dtype=np.float64)
        for i, p in enumerate(points):
            points_3d[i, axis1] = scaling1 * points[i, 0]
            points_3d[i, axis2] = scaling2 * points[i, 1]
            if time_axis:
                points_3d[i, dim - 1] = time_scaling * points[i, 2]
            points_3d[i] += shift
        return tf.constant(points_3d, dtype=self.dtype)

#############################################################################
# Utility
#############################################################################
    def get_boundary_sample(self, num_train, num_test, dim, minval, maxval, type, tf_random_generator, time_axis=True):
        scalings = [u - v for u, v in zip(maxval, minval)]
        if scalings[0] == 0:
            axis1, axis2 = 1, 2
            scaling1, scaling2 = scalings[1], scalings[2]
        elif scalings[1] == 0:
            axis1, axis2 = 0, 2
            scaling1, scaling2 = scalings[0], scalings[2]
        elif scalings[2] == 0:
            axis1, axis2 = 0, 1
            scaling1, scaling2 = scalings[0], scalings[1]
        else:
            raise ValueError

        num = num_train + num_test

        if time_axis:
            time_scaling = scalings[3]
        else:
            time_scaling = None

        if type == "uniform":
            sample = tf_random_generator.uniform(shape=[num, dim], minval=minval, maxval=maxval,
                                                 dtype=self.dtype)
        elif type == "sobol":
            sample = self.sobol_sequence_plane(num, axis1, axis2, scaling1, scaling2,
                                          tf.constant(minval, dtype=self.dtype), dim, time_axis,
                                          time_scaling)
        elif type == "lathyp":
            sample = self.latin_hypercube_sequence_plane(num, axis1, axis2, scaling1, scaling2,
                                                    tf.constant(minval, dtype=self.dtype), dim, time_axis,
                                                    time_scaling)
        else:
            raise Exception(type + " not implemented")

        return sample[:num_train], sample[-num_test:]

    def get_sample(self, num_train, num_test, dim, minval, maxval, type, tf_random_generator):
        scalings = [u - v for u, v in zip(maxval, minval)]

        num = num_train + num_test

        if type == "uniform":
            sample = tf_random_generator.uniform(shape=[num, dim], minval=minval, maxval=maxval,
                                                 dtype=self.dtype)
        elif type == "sobol":
            sample = self.sobol_sequence_cube(num, minval, maxval, dim)
        elif type == "lathyp":
            sample = self.latin_hypercube_sequence_cube(num, minval, maxval, dim)
        else:
            raise Exception(type + " not implemented")

        return sample[:num_train], sample[-num_test:]