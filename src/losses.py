import tensorflow as tf


# TODO: add unit tests
# TODO: should this return reduced or return Tensor of same length as inputs?
def content_loss(input_feature_maps, stylized_feature_maps):
    batch_size, h, w, channels = input_feature_maps.shape
    diff = tf.reshape(input_feature_maps - stylized_feature_maps, (batch_size, -1))
    content_loss_per_batch = tf.reduce_sum(tf.square(diff), axis=1) / tf.cast(channels * h * w, tf.float32)
    return tf.reduce_mean(content_loss_per_batch)


def style_loss(style_feature_maps, stylized_feature_maps):
    style_gram = gram_matrix(style_feature_maps)
    stylized_gram = gram_matrix(stylized_feature_maps)
    batch_size, channels, _ = style_gram.shape
    diff = tf.reshape(style_gram - stylized_gram, (batch_size, -1))
    style_loss_per_batch = tf.reduce_sum(tf.square(diff), axis=1) / tf.cast(channels, tf.float32) ** 2
    return tf.reduce_mean(style_loss_per_batch)


def gram_matrix(x):
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    input_shape = tf.shape(x)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def variation_loss(x, eta=1):
    total_variation_per_batch = tf.pow(tf.image.total_variation(x), eta/2)
    return tf.reduce_mean(total_variation_per_batch)
