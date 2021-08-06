import tensorflow as tf


class ResidualBlock(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=48, kernel_size=3, strides=1, activation="relu", padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=48, kernel_size=3, strides=1, activation=None, padding="same"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = tf.keras.layers.Add()([inputs, x])
        x = tf.keras.layers.ReLU()(x)
        return x
