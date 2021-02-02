import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser','Pullover', 'Dress','Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Amkle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0
