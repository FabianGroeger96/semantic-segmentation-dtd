import tensorflow as tf

from src.models.u_net.utils import get_filter_count, get_kernel_initalizer


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        # configs
        self.layer_idx = layer_idx
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # get filter count
        filters = get_filter_count(layer_idx, filters_root)
        # get kernel initializer
        kernel_initializer = get_kernel_initalizer(filters, kernel_size)

        # initialize layers in the block
        self.conv_1 = tf.keras.layers.Conv2D(filters=filters,
                                             kernel_size=(kernel_size, kernel_size),
                                             kernel_initializer=kernel_initializer,
                                             strides=1,
                                             padding='valid')
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        # TODO: compare when inserting here a BN
        self.activation_1 = tf.keras.layers.Activation('relu')

        self.conv_2 = tf.keras.layers.Conv2D(filters=filters,
                                             kernel_size=(kernel_size, kernel_size),
                                             kernel_initializer=kernel_initializer,
                                             strides=1,
                                             padding='valid')
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)
        # TODO: compare when inserting here a BN
        self.activation_2 = tf.keras.layers.Activation('relu')

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        # 1st block conv
        x = self.conv_1(inputs)
        x = self.dropout_1(x, training=training)
        x = self.activation_1(x)

        # 2nd block conv
        x = self.conv_2(x)
        x = self.dropout_2(x, training=training)
        x = self.activation_2(x)

        return x
