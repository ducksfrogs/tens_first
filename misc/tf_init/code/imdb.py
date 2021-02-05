import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

print("Training entries: {}, labels {}".format(len(train_data), len(train_labels)))

print(train_data[0])

len(train_data[0]), len(train_data[1])

word_index = imdb.get_word_index()

word_index = {kk(v+3) for k, v in word_index.items()}

word_index['<PAD>'] = 0
word_index['<STRAT>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                    value=word_index['<PAD>'],
                                    padding='post',
                                    maxlen=256)


test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                    value=word_index['<PAD>'],
                                    padding='post',
                                    maxlen=256)


len(train_data[0]), len(train_data[1])

print(train_data[0])

vocab_size = 100000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    validation_data=(x_val, y_val),
                    batch_size=512,verbose=1)


results = model.evaluate(test_data, test_labels, verbose=2)

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss history_dict['val_loss']

epochs = range(1, len(acc) +1)

plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and Validation loss")
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()
plt.show()
