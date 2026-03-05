import tensorflow as tf
import numpy as np

class FourierLayer(tf.keras.layers.Layer):

    @staticmethod
    def initialise_weight_matrix(sigma, input_dim, output_dim, np_random_generator):
        B = np_random_generator.normal(scale=sigma, size=(output_dim, input_dim))
        return B
    
    def __init__(self, output_fourier, B, name = "layerFourier", **kwargs):
        '''Initializes the class and sets up the internal variables'''
        super(FourierLayer,self).__init__()

        self.B = B
        self.output_fourier = output_fourier
        init = tf.constant_initializer(self.B)
        self.fixed_B_Layer = tf.keras.layers.Dense(self.output_fourier, kernel_initializer=init, trainable = False, use_bias = False, name = name)

    def call(self, inputs):
        freq_inputs = self.fixed_B_Layer(inputs)
        cosx = tf.cos(2*np.pi*freq_inputs)
        sinx = tf.sin(2*np.pi*freq_inputs)
        concat = tf.keras.layers.Concatenate(axis=1)([cosx, sinx])

        return concat

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_fourier': self.output_fourier,
            'B': self.B
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Note that you can also use `keras.saving.deserialize_keras_object` here
        #import pdb; pdb.set_trace()
        config["B"] = np.array(config['B']['config']['value'], dtype=config['B']['config']['dtype'])
        return cls(**config)