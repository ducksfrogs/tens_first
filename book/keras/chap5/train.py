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

valid_dogs_dir = os.path.join(valid_dir, 'dogs')
os.mkdir(valid_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir. 'dogs')
os.mkdir(test_dogs_dir)


fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(valid_cats_dir, fname)
    shutil.copyfile(src, dst)
