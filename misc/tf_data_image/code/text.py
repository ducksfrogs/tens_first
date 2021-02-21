import tensorflow as tf
import tensorflow_datasets as tfds

import os

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
  text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)

parent_dir = os.path.dirname(text_dir)

parent_dir

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, filename in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_datasets = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_datasets)

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]

for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatnate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
