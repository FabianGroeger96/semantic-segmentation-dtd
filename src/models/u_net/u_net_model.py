import tensorflow as tf

from src.models.u_net.conv_block import ConvBlock
from src.models.u_net.crop_concat_block import CropConcatBlock
from src.models.u_net.upconv_block import UpConvBlock
from src.models.u_net.utils import get_kernel_initalizer


class UNet(tf.keras.Model):

    def __init__(self,
                 num_classes: int,
                 img_size: int,
                 img_border: int,
                 nr_channels: int = 1,
                 layer_depth: int = 3,
                 filters_root: int = 64,
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 dropout_rate: float = 0.5,
                 name='UNet',
                 **kwargs):
        super(UNet, self).__init__(**kwargs)

        # configs
        self.model_name = name
        self.img_size_out = img_size-2*img_border
        self.out_shape = (self.img_size_out**2, num_classes)
        self.layer_depth = layer_depth

        # keep dict of outputs from contracting layers
        self.contracting_layers_out = {}

        # parameters for the layers
        conv_params = dict(filters_root=filters_root,
                           kernel_size=kernel_size,
                           dropout_rate=dropout_rate)
        upconv_params = dict(filters_root=filters_root,
                             kernel_size=kernel_size,
                             pool_size=pool_size)

        # create the contracting layers of the model
        self.contracting_layers = []
        for layer_idx in range(0, layer_depth - 1):
            conv_block = ConvBlock(layer_idx, **conv_params)
            pool_block = tf.keras.layers.MaxPooling2D((pool_size, pool_size))
            # append to the layers
            self.contracting_layers.append({'conv': conv_block,
                                            'pool': pool_block})

        # create the middle layer of the model (bottom of the U)
        self.bottom_layer = ConvBlock(layer_idx+1, **conv_params)

        # create the expansive layers of the model
        self.expansive_layers = []
        for layer_idx in range(layer_idx, -1, -1):
            upconv_block = UpConvBlock(layer_idx, **upconv_params)
            crop_concat_block = CropConcatBlock()
            conv_block = ConvBlock(layer_idx, **conv_params)
            # append to the layers
            self.expansive_layers.append({'upconv': upconv_block,
                                          'crop_concat': crop_concat_block,
                                          'conv': conv_block})

        # last layer of the model (output convolution)
        kernel_initializer = get_kernel_initalizer(filters_root, kernel_size)
        self.conv_out = tf.keras.layers.Conv2D(
            filters=num_classes, kernel_size=1,
            kernel_initializer=kernel_initializer,
            strides=1, padding='valid')
        # TODO: check if BN here
        self.conv_act_out = tf.keras.layers.Activation('relu')
        self.act_out = tf.keras.layers.Activation('softmax', name='outputs')

        # reshape before output (converts 2D to 1D), for pixel-wise loss
        self.reshape_out = tf.keras.layers.Reshape(target_shape=self.out_shape)

    @tf.function
    def call(self, inputs, training=None):
        x = inputs

        # contracting path
        for layer_idx, layer in enumerate(self.contracting_layers):
            x = layer['conv'](x, training=training)
            # save the output for the expansive path
            self.contracting_layers_out[layer_idx] = x
            x = layer['pool'](x, training=training)

        # middle layer (bottom of the U)
        x = self.bottom_layer(x, trianing=training)

        # expansive path
        for layer_idx, layer in enumerate(self.expansive_layers):
            x = layer['upconv'](x, training=training)
            contracting_idx = (len(self.contracting_layers_out) - 1) - layer_idx
            x = layer['crop_concat'](x, self.contracting_layers_out[contracting_idx])
            x = layer['conv'](x, training=training)

        # output layer
        x = self.conv_out(x)
        x = self.conv_act_out(x)
        prediction = self.act_out(x)
        prediction = self.reshape_out(prediction)

        return prediction
