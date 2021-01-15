import functools

import numpy as np
import tensorflow as tf

TRAIN_DATA_URL = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
TEST_DATA_URL = 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'

train_file_path = tf.keras.utils.get_file('train.csv', TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file('eval.csv', TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)

!head {train_file_path}
