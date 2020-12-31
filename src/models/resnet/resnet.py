from enum import Enum

import tensorflow as tf

from src.models.resnet.residual_block import BasicBlock, BottleNeck


class ResNetType(Enum):
    """
    Type of ResNet. Determines the type of layers to use for the ResNet implementation.
    TypeI: uses basic blocks as layers.
    TypeII: uses bottle neck as layers.
    """
    TypeI = 0
    TypeII = 1


class ResNet(tf.keras.Model):
    """ Abstract implementation for creating different ResNet models_embedding. """

    def __init__(self, layer_params, resnet_type, num_classes=47, model_name="ResNet"):
        """
        Initialises a ResNet model.

        :param layer_params: the number of filters in each layer of the model.
        :param resnet_type: the type of ResNet, determines which layers to use.
            TypeI: uses basic blocks as layers.
            TypeII: uses bottle neck as layers.
        :param num_classes: number of classes in the dataset, used for output.
        :param model_name: the name of the model.
        """
        super(ResNet, self).__init__()

        self.model_name = model_name

        self.l2_amount = 1e-5
        self.l2_regularization = tf.keras.regularizers.l2(1e-5)

        self.create_encoder_layers(layer_params, resnet_type)
        self.create_decoder_layers(num_classes)

        # output reshape layer
        self.out_reshape = tf.keras.layers.Reshape(target_shape=(128*128, 47))

    def create_encoder_layers(self, layer_params, resnet_type):
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            bias_regularizer=self.l2_regularization,
                                            kernel_regularizer=self.l2_regularization)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                padding="same")

        if resnet_type == ResNetType.TypeI:
            self.layer1 = self.make_basic_block_layer(filter_num=64,
                                                      blocks=layer_params[0],
                                                      l2_amount=self.l2_amount)
            self.layer2 = self.make_basic_block_layer(filter_num=128,
                                                      blocks=layer_params[1],
                                                      stride=2,
                                                      l2_amount=self.l2_amount)
            self.layer3 = self.make_basic_block_layer(filter_num=256,
                                                      blocks=layer_params[2],
                                                      stride=2,
                                                      l2_amount=self.l2_amount)
            self.layer4 = self.make_basic_block_layer(filter_num=512,
                                                      blocks=layer_params[3],
                                                      stride=2,
                                                      l2_amount=self.l2_amount)
        elif resnet_type == ResNetType.TypeII:
            self.layer1 = self.make_bottleneck_layer(filter_num=64,
                                                     blocks=layer_params[0],
                                                     l2_amount=self.l2_amount)
            self.layer2 = self.make_bottleneck_layer(filter_num=128,
                                                     blocks=layer_params[1],
                                                     stride=2,
                                                     l2_amount=self.l2_amount)
            self.layer3 = self.make_bottleneck_layer(filter_num=256,
                                                     blocks=layer_params[2],
                                                     stride=2,
                                                     l2_amount=self.l2_amount)
            self.layer4 = self.make_bottleneck_layer(filter_num=512,
                                                     blocks=layer_params[3],
                                                     stride=2,
                                                     l2_amount=self.l2_amount)



    def create_decoder_layers(self, num_classes):
        self.deconv1 = tf.keras.layers.Conv2DTranspose(512, (4, 4),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            name='deconv1')
        self.debn1 = tf.keras.layers.BatchNormalization()
        self.deact1 = tf.keras.layers.Activation('relu')

        self.deconv2 = tf.keras.layers.Conv2DTranspose(256, (4, 4),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            name='deconv2')
        self.debn2 = tf.keras.layers.BatchNormalization()
        self.deact2 = tf.keras.layers.Activation('relu')

        self.deconv3 = tf.keras.layers.Conv2DTranspose(128, (4, 4),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            name='deconv3')
        self.debn3 = tf.keras.layers.BatchNormalization()
        self.deact3 = tf.keras.layers.Activation('relu')

        self.deconv4 = tf.keras.layers.Conv2DTranspose(64, (4, 4),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            name='deconv4')
        self.debn4 = tf.keras.layers.BatchNormalization()
        self.deact4 = tf.keras.layers.Activation('relu')

        self.deconv5 = tf.keras.layers.Conv2DTranspose(64, (4, 4),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            name='deconv5')
        self.debn5 = tf.keras.layers.BatchNormalization()
        self.deact5 = tf.keras.layers.Activation('relu')

        self.pred_conv_layer = tf.keras.layers.Conv2D(num_classes, (1, 1),
                                                 activation='softmax',
                                                 padding='same',
                                                 kernel_initializer='he_normal',
                                                 name='pred_conv_layer')

    @tf.function
    def call(self, inputs, training=None):
        """
        Executes a forward pass through the model.
        Based on the input, it will execute the forward pass for a single channel or for multiple channels.
        Will be executed as a graph (@tf.function).

        :param inputs: the input that will be passed through the model.
        :param training: if the model is training.
        :return: returns the output of the model.
        """
        features = self.forward_pass(inputs, training=training)

        return features

    @tf.function
    def forward_pass(self, inputs, training=None):
        """
        The forward pass through the network.

        :param inputs: the input that will be passed through the model.
        :param training: if the model is training, for disabling dropout, batch norm. etc.
        :return: the output of the forward pass.
        """
        # encoder
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        if training:
            x = tf.keras.layers.Dropout(0.5)(x)

        # decoder
        x = self.deconv1(x)
        x = self.debn1(x, training=training)
        x = self.deact1(x)

        x = self.deconv2(x)
        x = self.debn2(x, training=training)
        x = self.deact2(x)

        x = self.deconv3(x)
        x = self.debn3(x, training=training)
        x = self.deact3(x)

        x = self.deconv4(x)
        x = self.debn4(x, training=training)
        x = self.deact4(x)

        x = self.deconv5(x)
        x = self.debn5(x, training=training)
        x = self.deact5(x)

        seg_map = self.pred_conv_layer(x)
        seg_map = self.out_reshape(seg_map)

        return seg_map

    def make_basic_block_layer(
            self, filter_num, blocks, stride=1, l2_amount=1.0):
        res_block = tf.keras.Sequential()
        res_block.add(
            BasicBlock(
                filter_num,
                stride=stride,
                l2_amount=l2_amount))

        for _ in range(1, blocks):
            res_block.add(
                BasicBlock(
                    filter_num,
                    stride=1,
                    l2_amount=l2_amount))

        return res_block

    def make_bottleneck_layer(self, filter_num, blocks,
                              stride=1, l2_amount=1.0):
        res_block = tf.keras.Sequential()
        res_block.add(
            BottleNeck(
                filter_num,
                stride=stride,
                l2_amount=l2_amount))

        for _ in range(1, blocks):
            res_block.add(
                BottleNeck(
                    filter_num,
                    stride=1,
                    l2_amount=l2_amount))

        return res_block

class ResNet18(ResNet):
    """ Concrete implementation of the standard ResNet18 architecture. """

    def __init__(self, model_name="ResNet18"):
        super(ResNet18, self).__init__(layer_params=[2, 2, 2, 2], resnet_type=ResNetType.TypeI,
                                       model_name=model_name)


class ResNet34(ResNet):
    """ Concrete implementation of the standard ResNet34 architecture. """

    def __init__(self, model_name="ResNet34"):
        super(ResNet34, self).__init__(layer_params=[3, 4, 6, 3], resnet_type=ResNetType.TypeI,
                                       model_name=model_name)


class ResNet50(ResNet):
    """ Concrete implementation of the standard ResNet50 architecture. """

    def __init__(self, model_name="ResNet50"):
        super(ResNet50, self).__init__(layer_params=[3, 4, 6, 3], resnet_type=ResNetType.TypeII,
                                       model_name=model_name)


class ResNet101(ResNet):
    """ Concrete implementation of the standard ResNet101 architecture. """

    def __init__(self, model_name="ResNet101"):
        super(ResNet101, self).__init__(layer_params=[3, 4, 23, 3], resnet_type=ResNetType.TypeII,
                                        model_name=model_name)


class ResNet152(ResNet):
    """ Concrete implementation of the standard ResNet152 architecture. """

    def __init__(self, model_name="ResNet152"):
        super(ResNet152, self).__init__(layer_params=[3, 8, 36, 3], resnet_type=ResNetType.TypeII,
                                        model_name=model_name)
