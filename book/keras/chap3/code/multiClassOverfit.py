from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = \
            reuters.load_data(num_words=10000)

words_index = reuters.get_word_index()

reverse_word_index = \
    dict([(val, key) for (key, val) in words_index.items()])

decoded_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
)

decoded_newswire

import numpy as np

def vactorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results

X_train = vactorize_sequences(train_data)
X_test = vactorize_sequences(test_data)


from keras import models
from keras import layers


model = models.Sequential()
model.add(layers.Dense(64,activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

x_val = X_train[:1000]
partial_x_train = X_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                partial_y_train,batch_size=512,
                epochs=20,
                validation_data=(x_val, y_val))
