import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train /255.0, X_test /255.0

X_train = X_train[..., tf.newaxis]
X_test  = X_test[..., tf.newaxis]
