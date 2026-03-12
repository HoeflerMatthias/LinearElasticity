import tensorflow as tf
import numpy as np


class FourierLayer(tf.keras.layers.Layer):

    @staticmethod
    def random_weight_matrix(sigma, num_frequencies, input_dim, np_random_generator):
        return np_random_generator.normal(scale=sigma, size=(input_dim, num_frequencies))

    @staticmethod
    def log_spaced_weight_matrix(num_frequencies, input_dim):
        if num_frequencies % input_dim != 0:
            raise ValueError(
                f"For log_spaced Fourier features, layers[0] ({num_frequencies}) "
                f"must be divisible by input_dim ({input_dim})")
        L = num_frequencies // input_dim
        freqs = 2.0 ** np.arange(L)
        B = np.zeros((input_dim, num_frequencies))
        for d in range(input_dim):
            B[d, d * L:(d + 1) * L] = freqs
        return B

    def __init__(self, num_frequencies, B, name="layerFourier", **kwargs):
        super(FourierLayer, self).__init__()
        self.B = B
        self.num_frequencies = num_frequencies
        init = tf.constant_initializer(self.B)
        self.fixed_B_Layer = tf.keras.layers.Dense(
            self.num_frequencies, kernel_initializer=init, trainable=False, use_bias=False, name=name)

    def call(self, inputs):
        freq_inputs = self.fixed_B_Layer(inputs)
        cosx = tf.cos(2 * np.pi * freq_inputs)
        sinx = tf.sin(2 * np.pi * freq_inputs)
        return tf.keras.layers.Concatenate(axis=1)([cosx, sinx])

    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_frequencies': self.num_frequencies, 'B': self.B})
        return config

    @classmethod
    def from_config(cls, config):
        config["B"] = np.array(config['B']['config']['value'], dtype=config['B']['config']['dtype'])
        return cls(**config)


class DistanceLayer(tf.keras.layers.Layer):

    def __init__(self, axis=0, value=0, output_dims=3, distance_axes=None, **kwargs):
        self.axis = axis
        self.value = value
        self.output_dims = output_dims
        self.distance_axes = distance_axes
        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        output = inputs[:, self.axis] - self.value

        if self.distance_axes is not None:
            ones = tf.ones_like(output)
            stacked = [ones for _ in range(self.output_dims)]
            for ax in self.distance_axes:
                stacked[ax] = output

            result = tf.stack(stacked, axis=-1)

        else:
            result = tf.tile(tf.expand_dims(output, -1), [1, self.output_dims])

        return result


def load_network(path):
    return tf.keras.models.load_model(path, custom_objects={
            'DistanceLayer': DistanceLayer,
            'FourierLayer': FourierLayer,
        })

def get_network(input_dim, input_mean, input_variance, output_dim, output_mean, output_variance, params, np_random_generator, name = ""):

    activation = tf.nn.tanh

    inputs_orig = tf.keras.Input(shape=(input_dim,), name="layerInput")

    inputs = tf.keras.layers.Normalization(mean = input_mean, variance = input_variance, name = "InputNormalisation")(inputs_orig)

    if params.get('fourier', False):
        fourier_params = params['fourier_params']
        num_frequencies = params['layers'][0]
        mode = fourier_params.get('mode', 'random')
        if mode == 'random':
            B = FourierLayer.random_weight_matrix(fourier_params['sig'], num_frequencies, input_dim, np_random_generator)
        elif mode == 'log_spaced':
            B = FourierLayer.log_spaced_weight_matrix(num_frequencies, input_dim)
        else:
            raise ValueError(f"Unknown Fourier mode: {mode}")
        l1 = FourierLayer(num_frequencies, B)(inputs)
    else:
        l1 = tf.keras.layers.Dense(params['layers'][0], activation=activation, name="layer0")(inputs)

    layers = [l1]
    for i, layer in enumerate(params['layers'][1:]):
        l = tf.keras.layers.Dense(layer, activation=activation, name=f'layer{i + 1}')(layers[-1])
        layers.append(l)

    outputs_orig = tf.keras.layers.Dense(output_dim, name="layerOutput")(layers[-1])

    if params.get('residual_connection', False):
        outputs_orig += inputs

    outputs = tf.keras.layers.Normalization(mean = output_mean, variance = output_variance, invert=True, name = "OutputNormalisation")(outputs_orig)

    if 'BL' in params and params['BL']:
        for i, (ax, val, dax) in enumerate(zip(params['BL_params']['axis'], params['BL_params']['value'], params['BL_params']['distance_axes'])):
            distance_layer = DistanceLayer(axis = ax, value = val, distance_axes=dax, output_dims=output_dim, name="DistanceLayer_"+ str(i))(inputs_orig)
            outputs = tf.keras.layers.Multiply()([outputs, distance_layer])

    model = tf.keras.Model(inputs=inputs_orig, outputs=outputs, name=name)

    return model