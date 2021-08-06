import cv2
import numpy as np


def process_image(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    img = np.reshape(img, [1, height, width, 3]).astype(np.float)
    return img
