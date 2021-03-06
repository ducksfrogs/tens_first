k = 4
num_validation_samples = len(data) // 4
np.random.shuffle(data)
validation_scores = []

for fold in range(k):
    validation_data = data[num_validation_samples * fold:
                        num_validation_samples*(fold+1)]
    training_data = data[:num_validation_samples*fold] +
                    data[num_validation_samples*(fold+1)]

    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)
