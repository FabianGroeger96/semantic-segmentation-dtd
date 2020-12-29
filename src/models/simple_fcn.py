import tensorflow as tf


class SimpleFCN(tf.keras.Model):

    def __init__(self, nr_classes, nr_channels, dropout_rate, name='SimpleFCN', **kwargs):
        super(SimpleFCN, self).__init__(**kwargs)

        # configs
        self.model_name = name

        # conv block 1
        self.conv_1 = tf.keras.layers.Conv2D(input_shape=(None, None, nr_channels), filters=64, kernel_size=3, padding='same')
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.ac_1 = tf.keras.layers.Activation('relu')

        # conv block 2
        self.conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.ac_2 = tf.keras.layers.Activation('relu')

        # conv block 3
        self.conv_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.ac_3 = tf.keras.layers.Activation('relu')

        # fully connected conv block (output)
        self.conv_out = tf.keras.layers.Conv2D(filters=nr_classes, kernel_size=1,
                                               kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=None))
        self.dropout_out = tf.keras.layers.Dropout(dropout_rate)
        self.bn_out = tf.keras.layers.BatchNormalization()
        self.ac_out = tf.keras.layers.Activation('softmax')
        self.reshape_out = tf.keras.layers.Reshape(target_shape=(128*128, nr_classes))

    @tf.function
    def call(self, inputs, training=None):
        # conv block 1
        x = self.conv_1(inputs)
        x = self.dropout_1(x, training=training)
        x = self.bn_1(x, training=training)
        x = self.ac_1(x)

        # conv block 2
        x = self.conv_2(x)
        x = self.dropout_2(x, training=training)
        x = self.bn_2(x, training=training)
        x = self.ac_2(x)

        # conv block 3
        x = self.conv_3(x)
        x = self.dropout_3(x, training=training)
        x = self.bn_3(x, training=training)
        x = self.ac_3(x)

        # fully connected conv (output layer)
        x = self.conv_out(x)
        x = self.dropout_out(x, training=training)
        x = self.bn_out(x, training=training)

        prediction = self.ac_out(x)
        prediction = self.reshape_out(prediction)

        return prediction
