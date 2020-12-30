# based on: https://github.com/QiaoranC/tf_ResNeSt_RegNet_model

import tensorflow as tf

from src.models.resnest.mish_activation import Mish, mish
from src.models.resnest.grouped_convolution import GroupedConv2D

from tensorflow.keras import models
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
    Reshape
)

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


class ResNest:
    def __init__(
            self, verbose=False, input_shape=(128, 128, 1),
            active="relu", n_classes=47, dropout_rate=0.2, fc_activation=None,
            blocks_set=[3, 4, 6, 3],
            radix=2, groups=1, bottleneck_width=64, deep_stem=True,
            stem_width=32, block_expansion=4, avg_down=True, avd=True,
            avd_first=False, preact=False, using_basic_block=False,
            using_cb=False):

        self.channel_axis = -1  # not for change
        self.verbose = verbose
        self.active = active  # default relu
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.fc_activation = fc_activation

        self.blocks_set = blocks_set
        self.radix = radix
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.block_expansion = block_expansion
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first

        # self.cardinality = 1
        self.dilation = 1
        self.preact = preact
        self.using_basic_block = using_basic_block
        self.using_cb = using_cb

    def _make_stem(self, input_tensor, stem_width=64, deep_stem=False):
        x = input_tensor
        if deep_stem:
            x = Conv2D(
                stem_width,
                kernel_size=3,
                strides=2,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
                data_format="channels_last")(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(
                stem_width,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
                data_format="channels_last")(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(
                stem_width * 2,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
                data_format="channels_last")(x)

            # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = Activation(self.active)(x)
        else:
            x = Conv2D(
                stem_width,
                kernel_size=7,
                strides=2,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
                data_format="channels_last")(x)
            # x = BatchNormalization(axis=self.channel_axis,epsilon=1.001e-5)(x)
            # x = Activation(self.active)(x)
        return x

    def _rsoftmax(self, input_tensor, filters, radix, groups):
        x = input_tensor
        if radix > 1:
            x = tf.reshape(x, [-1, groups, radix, filters // groups])
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.keras.activations.softmax(x, axis=1)
            x = tf.reshape(x, [-1, 1, 1, radix * filters])
        else:
            x = Activation("sigmoid")(x)

        return x

    def _SplAtConv2d(
            self, input_tensor, filters=64, kernel_size=3, stride=1,
            dilation=1, groups=1, radix=0):
        x = input_tensor
        in_channels = input_tensor.shape[-1]

        x = GroupedConv2D(
            filters=filters * radix,
            kernel_size=[
                kernel_size for i in range(
                    groups * radix)],
            use_keras=True,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
            data_format="channels_last",
            dilation_rate=dilation)(x)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        if radix > 1:
            splited = tf.split(x, radix, axis=-1)
            gap = sum(splited)
        else:
            gap = x

        # print('sum',gap.shape)
        gap = GlobalAveragePooling2D(data_format="channels_last")(gap)
        gap = tf.reshape(gap, [-1, 1, 1, filters])
        # print('adaptive_avg_pool2d',gap.shape)

        reduction_factor = 4
        inter_channels = max(in_channels * radix // reduction_factor, 32)

        x = Conv2D(inter_channels, kernel_size=1)(gap)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters * radix, kernel_size=1)(x)

        atten = self._rsoftmax(x, filters, radix, groups)

        if radix > 1:
            logits = tf.split(atten, radix, axis=-1)
            out = sum([a * b for a, b in zip(splited, logits)])
        else:
            out = atten * x
        return out

    def _make_block(
            self, input_tensor, first_block=True, filters=64, stride=2,
            radix=1, avd=False, avd_first=False, is_first=False):
        x = input_tensor
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * self.block_expansion:
            short_cut = input_tensor
            if self.avg_down:
                if self.dilation == 1:
                    short_cut = AveragePooling2D(
                        pool_size=stride, strides=stride, padding="same",
                        data_format="channels_last")(short_cut)
                else:
                    short_cut = AveragePooling2D(
                        pool_size=1, strides=1, padding="same",
                        data_format="channels_last")(short_cut)
                short_cut = Conv2D(
                    filters * self.block_expansion,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    kernel_initializer="he_normal",
                    use_bias=False,
                    data_format="channels_last")(short_cut)
            else:
                short_cut = Conv2D(
                    filters * self.block_expansion,
                    kernel_size=1,
                    strides=stride,
                    padding="same",
                    kernel_initializer="he_normal",
                    use_bias=False,
                    data_format="channels_last")(short_cut)

            short_cut = BatchNormalization(
                axis=self.channel_axis, epsilon=1.001e-5)(short_cut)
        else:
            short_cut = input_tensor

        group_width = int(
            filters * (self.bottleneck_width / 64.0)) * self.cardinality
        x = Conv2D(
            group_width,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
            data_format="channels_last")(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = AveragePooling2D(
                pool_size=3, strides=stride, padding="same",
                data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = self._SplAtConv2d(
                x,
                filters=group_width,
                kernel_size=3,
                stride=stride,
                dilation=self.dilation,
                groups=self.cardinality,
                radix=radix)
        else:
            x = Conv2D(
                group_width,
                kernel_size=3,
                strides=stride,
                padding="same",
                kernel_initializer="he_normal",
                dilation_rate=self.dilation,
                use_bias=False,
                data_format="channels_last")(x)
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

        if avd and not avd_first:
            x = avd_layer(x)
            # print('can')
        x = Conv2D(
            filters * self.block_expansion,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            dilation_rate=self.dilation,
            use_bias=False,
            data_format="channels_last")(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)

        m2 = Add()([x, short_cut])
        m2 = Activation(self.active)(m2)
        return m2

    def _make_block_basic(
            self, input_tensor, first_block=True, filters=64, stride=2,
            radix=1, avd=False, avd_first=False, is_first=False):
        """Conv2d_BN_Relu->Bn_Relu_Conv2d
        """
        x = input_tensor
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        short_cut = x
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * self.block_expansion:
            if self.avg_down:
                if self.dilation == 1:
                    short_cut = AveragePooling2D(
                        pool_size=stride, strides=stride, padding="same",
                        data_format="channels_last")(short_cut)
                else:
                    short_cut = AveragePooling2D(
                        pool_size=1, strides=1, padding="same",
                        data_format="channels_last")(short_cut)
                short_cut = Conv2D(
                    filters,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    kernel_initializer="he_normal",
                    use_bias=False,
                    data_format="channels_last")(short_cut)
            else:
                short_cut = Conv2D(
                    filters,
                    kernel_size=1,
                    strides=stride,
                    padding="same",
                    kernel_initializer="he_normal",
                    use_bias=False,
                    data_format="channels_last")(short_cut)

        group_width = int(
            filters * (self.bottleneck_width / 64.0)) * self.cardinality
        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = AveragePooling2D(
                pool_size=3, strides=stride, padding="same",
                data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = self._SplAtConv2d(
                x,
                filters=group_width,
                kernel_size=3,
                stride=stride,
                dilation=self.dilation,
                groups=self.cardinality,
                radix=radix)
        else:
            x = Conv2D(
                filters,
                kernel_size=3,
                strides=stride,
                padding="same",
                kernel_initializer="he_normal",
                dilation_rate=self.dilation,
                use_bias=False,
                data_format="channels_last")(x)

        if avd and not avd_first:
            x = avd_layer(x)
            # print('can')

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(
            filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            dilation_rate=self.dilation,
            use_bias=False,
            data_format="channels_last")(x)
        m2 = Add()([x, short_cut])
        return m2

    def _make_layer(self, input_tensor, blocks=4,
                    filters=64, stride=2, is_first=True):
        x = input_tensor
        if self.using_basic_block is True:
            x = self._make_block_basic(
                x,
                first_block=True,
                filters=filters,
                stride=stride,
                radix=self.radix,
                avd=self.avd,
                avd_first=self.avd_first,
                is_first=is_first)
            # print('0',x.shape)

            for i in range(1, blocks):
                x = self._make_block_basic(
                    x,
                    first_block=False,
                    filters=filters,
                    stride=1,
                    radix=self.radix,
                    avd=self.avd,
                    avd_first=self.avd_first)
                # print(i,x.shape)

        elif self.using_basic_block is False:
            x = self._make_block(
                x,
                first_block=True,
                filters=filters,
                stride=stride,
                radix=self.radix,
                avd=self.avd,
                avd_first=self.avd_first,
                is_first=is_first)
            # print('0',x.shape)

            for i in range(1, blocks):
                x = self._make_block(
                    x,
                    first_block=False,
                    filters=filters,
                    stride=1,
                    radix=self.radix,
                    avd=self.avd,
                    avd_first=self.avd_first)
                # print(i,x.shape)
        return x

    def _make_Composite_layer(
            self, input_tensor, filters=256, kernel_size=1, stride=1,
            upsample=True):
        x = input_tensor
        x = Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        if upsample:
            x = UpSampling2D(size=2)(x)
        return x

    def build(self):
        get_custom_objects().update({'mish': Mish(mish)})

        print(self.input_shape)
        input_sig = Input(shape=self.input_shape)
        x = self._make_stem(
            input_sig,
            stem_width=self.stem_width,
            deep_stem=self.deep_stem)
        f1 = x
        print(x.shape)

        if self.preact is False:
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)
        if self.verbose:
            print("stem_out", x.shape)

        x = MaxPool2D(
            pool_size=3,
            strides=2,
            padding="same",
            data_format="channels_last")(x)
        if self.verbose:
            print("MaxPool2D out", x.shape)

        if self.preact is True:
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

        if self.using_cb:
            second_x = x
            second_x = self._make_layer(
                x, blocks=self.blocks_set[0],
                filters=64, stride=1, is_first=False)
            second_x_tmp = self._make_Composite_layer(
                second_x, filters=x.shape[-1], upsample=False)
            if self.verbose:
                print('layer 0 db_com', second_x_tmp.shape)
            x = Add()([second_x_tmp, x])
        x = self._make_layer(
            x, blocks=self.blocks_set[0],
            filters=64, stride=1, is_first=False)
        if self.verbose:
            print("-" * 5, "layer 0 out", x.shape, "-" * 5)
        f2 = x
        print(x.shape)

        fs = []
        b1_b3_filters = [64, 128, 256, 512]
        for i in range(3):
            idx = i+1
            if self.using_cb:
                second_x = self._make_layer(
                    x, blocks=self.blocks_set[idx],
                    filters=b1_b3_filters[idx],
                    stride=2)
                second_x_tmp = self._make_Composite_layer(
                    second_x, filters=x.shape[-1])
                if self.verbose:
                    print(
                        'layer {} db_com out {}'.format(
                            idx, second_x_tmp.shape))
                x = Add()([second_x_tmp, x])
                print(x.shape)
            x = self._make_layer(
                x, blocks=self.blocks_set[idx],
                filters=b1_b3_filters[idx],
                stride=2)
            print(x.shape)
            fs.append(x)
            if self.verbose:
                print('----- layer {} out {} -----'.format(idx, x.shape))

        o = fs[-1]
        print(o.shape)
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = Conv2D(
                512, (3, 3),
                padding='valid', activation='relu',
                data_format=IMAGE_ORDERING)(o)
        o = (BatchNormalization())(o)

        f4 = fs[-2]
        print('f4', f4.shape)
        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        print('o', o.shape)
        o = (concatenate([o, f4], axis=MERGE_AXIS))
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = Conv2D(
                256, (3, 3),
                padding='valid', activation='relu',
                data_format=IMAGE_ORDERING)(o)
        o = (BatchNormalization())(o)
        print('out', o.shape)

        f3 = fs[-3]
        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (concatenate([o, f3], axis=MERGE_AXIS))
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (
            Conv2D(
                128, (3, 3),
                padding='valid', activation='relu',
                data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        print('out', o.shape)

        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (concatenate([o, f2], axis=MERGE_AXIS))
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (
            Conv2D(
                64, (3, 3),
                padding='valid', activation='relu',
                data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        print('out', o.shape)

        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (concatenate([o, f1], axis=MERGE_AXIS))
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (
            Conv2D(
                32, (3, 3),
                padding='valid', activation='relu',
                data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        print('out', o.shape)

        o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
        o = Conv2D(self.n_classes, (3, 3), padding='same',
                   data_format=IMAGE_ORDERING)(o)

        o = Activation('softmax')(o)
        o = Reshape(target_shape=(128*128, 47))(o)
        print('out', o.shape)

        model = models.Model(inputs=input_sig, outputs=o)

        if self.verbose:
            print(
                "Resnest builded with input {}, output{}".format(
                    input_sig.shape,
                    o.shape))
            print("-------------------------------------------")
            print("")

        return model
