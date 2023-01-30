from car_insurance.insurance_genome import GeneticHelper, RfGenome
import numpy as np
import pickle


BASELINE = int(GeneticHelper.get_baseline())

data_path = './data/iaat_insurance_training_data.pickle'
truth_path = './data/iaat_insurance_training_data_test_truth.pickle'
data = np.array(pickle.load(open(data_path, 'rb')))

x = data[:, 0:-1]
y = data[:, -1:]
y_truth = np.array(pickle.load(open(truth_path, 'rb'))).reshape(-1, 1)

assert len(x) == len(y) == len(y_truth)

print(y_truth.shape)

indices = list(range(len(x)))
np.random.shuffle(indices)

cutoff_idx = int(len(indices) * (2 / 3))
indices_train, indices_test = indices[:cutoff_idx], indices[cutoff_idx:]

x_train, x_test, y_train, y_test, y_truth = x[indices_train, :], x[indices_test, :], y[indices_train, :], \
                                            y[indices_test, :], y_truth[indices_test, :]

assert x_train.shape[0] == y_train.shape[0]
assert x_test.shape[0] == y_test.shape[0] == y_truth.shape[0]

genome = RfGenome(BASELINE)
genome.load_data()

matches, n_correct_when_match, bb_n_correct, iaat_n_correct, n_total = 0, 0, 0, 0, len(x_test)

for i in range(n_total):
    arry = x_test[i, :].reshape(1, -1)
    pred = genome.predict(arry)
    bb_pred = y_test[i, :][0]
    truth = y_truth[i, :][0]

    bb_n_correct += 1 if bb_pred == truth else 0
    iaat_n_correct += 1 if pred == truth else 0

    if pred == bb_pred:
        matches += 1

        if pred == truth:
            n_correct_when_match += 1

print(f'Matches = {matches}, percentage = {matches / n_total}')
print(f'Num correct when matching = {n_correct_when_match}, percentage = {n_correct_when_match / n_total}')
print(f'BB num correct = {bb_n_correct}, percentage = {bb_n_correct / n_total}')
print(f'iAAT num correct = {iaat_n_correct}, percentage = {iaat_n_correct / n_total}')
print(genome.feature_names())
