import numpy as np
import tensorflow as tf


def get_filter_count(layer_idx, filters_root):
    """
    At each downsampling step we double the number
    of feature channels.
    """
    return 2 ** layer_idx * filters_root

def get_kernel_initalizer(filters, kernel_size):
    """
    Drawing the initial weights from a Gaussian distribution with a
    standard deviation of sqrt(2/N), where N denotes the number of
    incoming nodes of one neuron.
    """
    stddev = np.sqrt(2 / (kernel_size **2 * filters))
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)
