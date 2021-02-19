from keras.datasets import imdb

(train_data, train_labels),  (test_data,test_labels) = imdb.load_data(num_words=10000)

word_idx = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_idx.items()])

decode_review = ' '.join(
 [reverse_word_index.get(i-3, "?") for i in train_data[0]]
)

#vectorize

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)
