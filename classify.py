# Import libraries
import tensorflow as tf
import shutil
import numpy as np
import os

class_names = ['handwritten', 'non-handwritten']

dirname = os.path.dirname(__file__)
model_dir = os.path.join(dirname,  'model')
model = tf.saved_model.load(model_dir)

img_height = 180
img_width = 180

data2_dir = os.path.join(dirname, "data2")
data2_dirs = os.listdir(data2_dir)

for classname in class_names:
    os.makedirs(os.path.join(data2_dir,  classname), exist_ok=True)

for file_path in data2_dirs:
    full_file_path = os.path.join(data2_dir,  file_path)

    if os.path.isdir(full_file_path):
        continue

    if file_path == '.DS_Store':
        continue

    img = tf.keras.utils.load_img(
        full_file_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model(img_array)
    classname = class_names[np.argmax(predictions)]
    shutil.move(full_file_path, os.path.join(data2_dir, classname, file_path))
