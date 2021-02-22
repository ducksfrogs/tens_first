import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

len(train_data[0]), len(train_data[1]))


word_index = imdb.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverve_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverve_word_index.get(i, '?') for i in text])


train_data = keras.preprocessing.sequence.parse_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)


len(train_data[0])
len(train_data[1])

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimiser='adam',loss='binary_crossentropy',metrics=['accuracy'])

x_val = train_data[:10000]
parital_x_train = train_data[10000:]

y_val = train_labels[:10000]
parital_y_train = train_labels[10000:]

history = model.fit(parital_x_train,
                    parital_y_train, epochs=40, batch_size=512,
                    validation_data=(x_val, y_val),verbose=1)

results = model.evaluate(test_data, test_labels, verbose=2)

history_dict = history.history

dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()

plt.clf()

plt.plot(epochs, acc, 'bo', label="Training acc")
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
