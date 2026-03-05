import nisaba as ns
import nisaba.experimental as nse
import tensorflow as tf
import numpy as np

class IndexedDataSet(ns.DataSet):

    def __init__(self, data, name = 'data', batch_size = None, epochs_per_batch = 1, shuffle = False):
        
        indices = tf.expand_dims(tf.range(data.shape[0], dtype=data.dtype), axis=-1)
        data = tf.concat([data, indices], axis=-1)

        super().__init__(data, name, batch_size, epochs_per_batch, shuffle)