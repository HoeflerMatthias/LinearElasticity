import tensorflow as tf
class DenseActivationLayer(tf.keras.layers.Layer):
    def __init__(self, output, shared_weights=True, name="layerDenseActivation"):
        '''Initializes the class and sets up the internal variables'''
        super(DenseActivationLayer, self).__init__()

        self.gate = lambda x: tf.keras.activations.softmax(x)
        self.functions = [
            tf.keras.activations.tanh,
            tf.math.sin,
            # tf.keras.activations.gelu,
            # tf.keras.activations.swish,
        ]

        self.act_weights = tf.Variable([[0.] for i in range(len(self.functions))], dtype=ns.config.get_dtype(),
                                       name='weights')
        self.shared_weights = shared_weights

        if self.shared_weights:
            self.dense = tf.keras.layers.Dense(output, activation=None, name=name)
        else:
            self.dense = [tf.keras.layers.Dense(output, activation=None, name=f'{name}_{i}') for i, _ in
                          enumerate(self.functions)]

    def act(self, inputs):
        w = self.gate(self.act_weights)
        sum = 0
        for i, f in enumerate(self.functions):
            sum += w[i] * f(inputs)
        return sum

    def call(self, inputs):
        w = self.gate(self.act_weights)
        output = 0

        if self.shared_weights:
            for i, f in enumerate(self.functions):
                output += w[i] * f(self.dense(inputs))
        else:
            for i, f in enumerate(self.functions):
                output += w[i] * f(self.dense[i](inputs))

        return output