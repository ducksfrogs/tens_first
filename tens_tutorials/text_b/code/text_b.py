import tensorflow as tf
from tensorflow import keras

import numpy as np
print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

word_index = imdb.get_word_index()
