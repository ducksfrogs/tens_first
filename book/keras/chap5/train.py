import os, shutil

original_dataset_dir = '.~/Documents'

base_dir = '.~/Documents/cats_and_dogs_small'

os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
valid_dir = os.path.join(base_dir, 'validation')

os.mkdir(valid_dir)
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

valid_cats_dir = os.path.join(valid_dir, 'cats')
os.mkdir(valid_cats_dir)
