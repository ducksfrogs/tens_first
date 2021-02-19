from keras import layers

layer = layers.Dense(32, input_shape=(784,))

from keras import models

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,)))
model.add(layers.Dense(32))
