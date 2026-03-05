import tensorflow as tf
import nisaba as ns

class HLoss(ns.loss.LossMeanSquares):
    def __init__(self, name, eval_roots, weight = 1.0, normalization = 1.0, expected_shape = None, full_size = None, dataset_key = None):

        if full_size is not None:
            self.lagrange_mul = tf.Variable(tf.zeros(full_size, dtype=ns.config.get_dtype()))
        else:
            self.lagrange_mul = tf.Variable(tf.zeros(expected_shape, dtype=ns.config.get_dtype()))

        self.full_size = full_size
        self.dataset_key = dataset_key
        
        super().__init__(name, eval_roots, weight, normalization, expected_shape)

    def loss_base_call(self, data):
        self.loss_values = self._eval_roots(data)
        if self.dataset_key is not None:
            data = data[self.dataset_key]

        if self.full_size is not None:
            lagrange_mul = tf.gather(self.lagrange_mul, tf.cast(data[:,-1], dtype=tf.int32))
        else:
            lagrange_mul = self.lagrange_mul
        
        val = tf.math.multiply(lagrange_mul, self.loss_values)
        loss = tf.reduce_mean(tf.square(val))
        return loss

    def normalized_values(self, data, normalization = None):
        abs_values = tf.math.abs(self._eval_roots(data))
        if normalization is not None:
            max_value = normalization
        else:
            max_value = tf.reduce_max(abs_values)
        return abs_values/max_value

    def roots(self, data):
        roots = self._eval_roots(data)
        n_squares = tf.cast(tf.reduce_prod(tf.shape(roots)), self.dtype)
        return tf.reshape(roots, (-1,)) / tf.sqrt( n_squares * self.normalization )