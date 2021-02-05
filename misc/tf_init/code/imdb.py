import tensorflow as tf

from tensorflow import keras
import numpy as np
print(tf.__version__)

imob = keras.datasets.imdb

(train_data, train_label), (test_data,test_label) = imob.load_data()

print("Training entries : {}. Labels: {}".format(len(train_data), len(test_label)))
