from bank_credit.credit_assumptions import CreditAssumptions, CreditFeatures
from bank_credit.data_cleaner import DataCleaner
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Get the data
x, y = DataCleaner.get_credit_data()

# Balance the data
labels = np.argmax(np.array(y), axis=1)
_, counts = np.unique(labels, return_counts=True)
n = min(counts)
mask = np.hstack([np.random.choice(np.where(labels == label)[0], n, replace=False) for label in np.unique(labels)])
x, y = np.array(x)[mask, :], np.array(y)[mask, :]

assert x.shape[0] == y.shape[0]

# Create 1 column for y
y = np.argmax(y, axis=1)

# Create train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=2 / 3)
input_data_shape = x_train.shape[1:]

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train SVM on best params
param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'degree': [3, 5, 7, 9, 11],
              'gamma': ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
              'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]}

random_search = RandomizedSearchCV(SVC(), param_grid, cv=2, n_iter=10)
random_search.fit(x_train_scaled, y_train)
model = random_search.best_estimator_

print(f'SVM accuracy = {random_search.best_score_}')

# Get predictions from trained SVM
y_pred = model.predict(x_test_scaled)

# Create training set for inverse AAT
train_iaat = np.array([CreditAssumptions(CreditFeatures(*list(x_test[i, :])), y_pred[i, ]).create_aat_tuple()
                       for i in range(len(x_test))])

assert len(train_iaat) == len(y_pred)

print(np.unique(train_iaat[:, -1], return_counts=True))

# Create training set for regular surrogate models
train_surrogate = np.append(x_test, y_pred.reshape(-1, 1), 1)

assert x_test.shape[0] == train_surrogate.shape[0] and x_test.shape[-1] + 1 == train_surrogate.shape[-1]

# Save the inverse AAT training data and AAT feature names
with open('./data/iaat_credit_training_data_svm.pickle', 'wb') as f:
    pickle.dump(train_iaat, f)

aat_feature_names = CreditAssumptions(CreditFeatures(*list(x_test[0, :])), y_pred[0, ]).assumption_names()

with open('./data/iaat_credit_training_features_svm.pickle', 'wb') as f:
    pickle.dump(aat_feature_names, f)

# Save the ground truth
with open('./data/iaat_credit_training_data_test_truth_svm.pickle', 'wb') as f:
    pickle.dump(y_test, f)

# Save the surrogate training data
with open('./data/surrogate_credit_training_data_svm.pickle', 'wb') as f:
    pickle.dump(train_surrogate, f)

# Save the SVM and its scaler
scaler_file = './data/svm_scaler.pickle'
svm_file = './data/svm.pickle'

with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)

with open(svm_file, 'wb') as f:
    pickle.dump(model, f)
