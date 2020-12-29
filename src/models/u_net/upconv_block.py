import tensorflow as tf

from src.models.u_net.utils import get_filter_count, get_kernel_initalizer


class UpConvBlock(tf.keras.layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, **kwargs):
        super(UpConvBlock, self).__init__(**kwargs)

        # configs
        self.layer_idx = layer_idx
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        # get filter count
        filters = get_filter_count(layer_idx + 1, filters_root)
        # get kernel initializer
        kernel_initializer = get_kernel_initalizer(filters, kernel_size)

        # initialize layers in the block
        self.upconv = tf.keras.layers.Conv2DTranspose(filters // 2,
                                                      kernel_size=(pool_size, pool_size),
                                                      kernel_initializer=kernel_initializer,
                                                      strides=pool_size,
                                                      padding='valid')
        # TODO: compare when inserting here a BN
        self.activation = tf.keras.layers.Activation('relu')

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        x = self.upconv(inputs)
        x = self.activation(x)

        return x
