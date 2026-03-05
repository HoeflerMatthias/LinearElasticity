import tensorflow as tf
class DistanceLayer(tf.keras.layers.Layer):

    @staticmethod
    def signed_distance(x, x0):
        return x - x0

    @staticmethod
    def distance(x, x0):
        m = 2
        distance = (x - x0) ** m
        output = tf.math.divide(1., tf.math.divide(1., distance) ** (1 / m))
        return output

    @staticmethod
    def make_add_func(func, axis = None, output_dims = 3):
        if callable(func):
            funclam = func
        else:
            funclam = lambda x: func*tf.ones_like(x[:,0])

        if axis == None:
            def f(inputs):
                return funclam(inputs)
        else:
            def f(inputs):
                ones = tf.zeros_like(inputs[:,0])
                stacked = [ones for _ in range(output_dims)]
                stacked[axis] = funclam(inputs)
                return tf.stack(stacked, axis=-1)

        return f

    def __init__(self, axis=0, value=0, output_dims=3, distance_axes=None, signed_distance=True, **kwargs):
        """Initializes the class and sets up the internal variables"""
        self.axis = axis
        self.value = value
        self.output_dims = output_dims
        self.signed_distance = signed_distance
        self.distance_axes = distance_axes
        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        if self.signed_distance:
            dfunc = DistanceLayer.signed_distance
        else:
            dfunc = DistanceLayer.distance

        output = dfunc(inputs[:, self.axis], self.value)

        if self.distance_axes is not None:
            ones = tf.ones_like(output)
            stacked = [ones for _ in range(self.output_dims)]
            for ax in self.distance_axes:
                stacked[ax] = output

            result = tf.stack(stacked, axis=-1)

        else:
            result = tf.tile(tf.expand_dims(output, -1), [1, self.output_dims])

        return result