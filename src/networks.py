import tensorflow as tf
import tensorflow_addons as tfa

from .blocks import ResidualBlock


def build_style_network(input_shape):
    return tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", input_shape=input_shape),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same"),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(filters=48, kernel_size=3, strides=2, padding="same"),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation("relu"),
        *[ResidualBlock() for _ in range(5)],
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same"),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same"),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same"),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation("tanh"),
    ],
        name="StyleNet"
)


class MultiOutputVGG(tf.keras.models.Model):

    def __init__(self, input_shape, output_layers):
        super().__init__()
        self.output_layers = output_layers
        self.vgg = tf.keras.applications.vgg16.VGG16(input_shape=input_shape, include_top=False)
        self.vgg.trainable = False

    def call(self, inputs):
        output = []
        x = inputs
        for i, layer in enumerate(self.vgg.layers):
            x = layer(x)
            if i in self.output_layers:
                output.append(x)
        return output
