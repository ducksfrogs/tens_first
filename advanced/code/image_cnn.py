import tensorflow as tf

from tensorflow.keras.models import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras,preprocessing.image import ImageDataGenerator


import os
import numpy as np
import matplotlib.pyplot as plt

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
