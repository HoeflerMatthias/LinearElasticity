import tensorflow as tf

# Set Keras default float type to float64 for scientific computing
tf.keras.backend.set_floatx('float64')


def get_dtype():
    return tf.float64
