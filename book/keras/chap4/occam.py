from keras import regularizers
from keras import models


model = models.Sequential()
model.add(laysers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                        activation='relu', input_shape=(10000, )))
model.add(laysers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(laysers.Dense(1, activation='sigmoid'))
