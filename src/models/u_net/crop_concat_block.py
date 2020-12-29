import tensorflow as tf


class CropConcatBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CropConcatBlock, self).__init__(**kwargs)

    @tf.function
    def call(self, x, contracting_input, **kwargs):
        """
        x: input from the expansive path
        contracting_input: input from the contracting path
        """

        x1_shape = tf.shape(contracting_input)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        # crop down the input from the contracting path, to match the expansive
        # path
        contracting_input_cropped = contracting_input[:,
                                        height_diff: (x1_shape[1] - height_diff),
                                        width_diff: (x1_shape[2] - width_diff),
                                        :]

        # combine the features / inputs
        x = tf.concat([contracting_input_cropped, x], axis=-1)

        return x
