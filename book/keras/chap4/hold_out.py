import numpy as np

num_validation_samples = 10000

np.random.shuffle(data)

val_data = data[:num_validation_samples]
data = data[num_validation_samples:]

train_data = data[:]

model = get_model()
model.train(train_data)
val_score = model.evaluate(val_data)

model = get_model()
model.train(np.concatenate([train_data, val_data]))
test_score = model.evaluate(test_data)
