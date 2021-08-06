import datetime
import tensorflow as tf

from src.data import build_dataset
from src.networks import build_style_network, MultiOutputVGG
from src.losses import content_loss, style_loss, variation_loss

train_dataset = build_dataset(
    images_path="resources/images/*.jpg",
    style_path="resources/styles/starry-night.jpg",
    batch_size=2
)

style_network = build_style_network(input_shape=(360, 640, 3))
loss_network = MultiOutputVGG(input_shape=(360, 640, 3), output_layers=[2, 5, 8, 12])

frame_input = tf.keras.layers.Input(shape=(360, 640, 3))
style_input = tf.keras.layers.Input(shape=(360, 640, 3))

stylized_frame = style_network(frame_input)
frame_vgg_outputs = loss_network(frame_input)
style_vgg_outputs = loss_network(style_input)
stylized_frame_vgg_outputs = loss_network(stylized_frame)

nst_model = tf.keras.models.Model(
    inputs=[frame_input, style_input],
    outputs=[stylized_frame, frame_vgg_outputs, style_vgg_outputs, stylized_frame_vgg_outputs]
)

#tf.keras.utils.plot_model(nst_model, to_file="model.png")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

content_strength = 1
style_strength = 10
variation_strength = 0.001

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/nst/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Might fail: loss is reduced by batch-sum, causing large gradients
cum_step = 0
for epoch in range(1):
    print("\nStart of epoch %d" % (epoch,))
    for step, (img, style) in enumerate(train_dataset):
        cum_step += 1
        with tf.GradientTape() as tape:
            styled_img, img_vgg, style_vgg, stylized_vgg = nst_model([img, style])
            loss_c = content_loss(img_vgg[-1], stylized_vgg[-1])
            loss_s = 0
            for i in range(len(style_vgg)):
                loss_s += style_loss(style_vgg[i], stylized_vgg[i])
            loss_v = variation_loss(styled_img)
            loss = content_strength * loss_c + style_strength * loss_s + variation_strength * loss_v

        grads = tape.gradient(loss, nst_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, nst_model.trainable_weights))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=cum_step)
            tf.summary.scalar('content_loss', loss_c, step=cum_step)
            tf.summary.scalar('style_loss', loss_s, step=cum_step)
            tf.summary.scalar('variation_loss', loss_v, step=cum_step)

style_network.save("model")