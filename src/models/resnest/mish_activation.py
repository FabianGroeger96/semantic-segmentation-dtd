import tensorflow as tf
from tensorflow.keras.layers import Activation


class Mish(Activation):
    """
    based on https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
    Mish Activation Function.
    """

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


def mish(inputs):
    # with tf.device("CPU:0"):
    result = inputs * tf.math.tanh(tf.math.softplus(inputs))
    return result
