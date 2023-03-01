import tensorflow as tf
import os
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)


def preprocess_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [224, 224])
    return img / 255.


dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname,  'data/handwritten/train')

imgs = os.listdir(dirname)

for i in range(len(imgs)):
    imgs[i] = os.path.join(dirname, imgs[i])

img_raw = tf.gfile.FastGFile(imgs[0], 'rb').read()
img = preprocess_img(img_raw)

plt.figure(1)
plt.imshow(img)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
