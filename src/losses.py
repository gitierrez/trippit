import tensorflow as tf


# TODO: should this return reduced or return Tensor of same length as inputs?
def content_loss(input_feature_maps, stylized_feature_maps):
    dim = tf.reduce_prod(input_feature_maps[0].shape)
    dim = tf.cast(dim, tf.float32)
    return tf.reduce_sum(tf.square(input_feature_maps - stylized_feature_maps)) / dim


def style_loss(style_feature_maps, stylized_feature_maps):
    channels = stylized_feature_maps.shape[-1]
    channels = tf.cast(channels, tf.float32)
    style_gram = gram_matrix(style_feature_maps)
    stylized_gram = gram_matrix(stylized_feature_maps)
    return tf.reduce_sum(tf.square(style_gram - stylized_gram)) / tf.square(channels)


def gram_matrix(x):
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    input_shape = tf.shape(x)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def variation_loss(x, eta=1):
    a = tf.reduce_sum(tf.square(x[:, :, 1:, :] - x[:, :, :-1, :]))
    b = tf.reduce_sum(tf.square(x[:, 1:, :, :] - x[:, :-1, :, :]))
    return tf.pow(a + b, eta/2)
