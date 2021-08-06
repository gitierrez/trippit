import tensorflow as tf

from src.data import resize_image, normalize_img, denormalize_img, decode_jpeg_file, show_image

model = tf.saved_model.load("model")

img = decode_jpeg_file("resources/images/1.jpg")
img = tf.cast(img, tf.float32)
img = resize_image(img, 360, 640)
img = normalize_img(img)
img = tf.expand_dims(img, axis=0)

styled_img = model(img)
show_image(denormalize_img(styled_img[0]))