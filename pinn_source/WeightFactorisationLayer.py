import tensorflow as tf
import nisaba as ns
class WeightFactorisationLayer(tf.keras.layers.Layer):
    def __init__(self, input, output, mu, sigma, np_random_generator, seed, name="WeightFactorLayer", CAF=False):
        '''Initializes the class and sets up the internal variables'''
        super(WeightFactorisationLayer, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        V = initializer(shape=(output, input))

        s = tf.math.exp(np_random_generator.normal(loc=mu, scale=sigma, size=(output)))
        V = tf.matmul(tf.linalg.diag(1 / s), V)

        self.V = ns.Variable(V, dtype=ns.config.get_dtype())
        self.s = ns.Variable(s, dtype=ns.config.get_dtype())
        self.b = ns.Variable(tf.zeros([output], dtype=ns.config.get_dtype()))

        if CAF:
            beta = tf.math.exp(np_random_generator.normal(loc=mu, scale=sigma, size=(output)))
            self.beta = ns.Variable(beta, dtype=ns.config.get_dtype())

            self.activation = lambda x: tf.keras.activations.tanh(x) * (
                        tf.ones_like(x) + tf.matmul(tf.linalg.diag(self.beta), x))
        else:
            self.activation = tf.keras.activations.tanh

    def call(self, inputs):
        W = tf.linalg.diag(self.s)
        M = tf.matmul(W, self.V)

        return self.activation(tf.matmul(inputs, tf.linalg.matrix_transpose(M)) + self.b)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'V': self.V.numpy(),
            's': self.s.numpy(),
            'b': self.b.numpy()
        })
        return config