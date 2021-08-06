import tensorflow as tf
import matplotlib.pyplot as plt


def show_image(img: tf.Tensor):
    plt.figure()
    plt.imshow(img)
    plt.show()


def decode_jpeg_file(file_path: str):
    img = tf.io.read_file(file_path)
    return tf.image.decode_jpeg(img)


def resize_image(img: tf.Tensor, height: int, width: int):
    return tf.image.resize(
        img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )


def normalize_img(img: tf.Tensor):
    """Normalizes image to [-1, 1] range"""
    return (img / 127.5) - 1


def denormalize_img(img: tf.Tensor):
    return img * 0.5 + 0.5


class ImageStyleLoader:

    def __init__(self, style_path: str, height: int, width: int):
        self.style_path = style_path
        self.height = height
        self.width = width

    def _load(self, img_path: str):
        image = decode_jpeg_file(img_path)
        style = decode_jpeg_file(self.style_path)

        return tf.cast(image, tf.float32), tf.cast(style, tf.float32)

    def _preprocess(self, img: str):
        img = resize_image(img, self.height, self.width)
        img = normalize_img(img)
        return img

    def load_and_preprocess(self, img_path: str):
        img, style = self._load(img_path)
        img = self._preprocess(img)
        style = self._preprocess(style)
        return img, style


def build_dataset(images_path, style_path, batch_size, height=360, width=640, buffer_size=500):
    loader = ImageStyleLoader(style_path, height, width)

    dataset = tf.data.Dataset.list_files(images_path)
    dataset = dataset.map(
        loader.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset

#for img, style in train_dataset.take(1):
#    show_image(denormalize_img(img[0]))
#    show_image(denormalize_img(style[0]))
