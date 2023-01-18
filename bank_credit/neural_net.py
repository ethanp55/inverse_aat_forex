from bank_credit.credit_assumptions import CreditAssumptions, CreditFeatures
from bank_credit.data_cleaner import DataCleaner
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential


# Get the data
x, y = DataCleaner.get_credit_data()

# Balance the data
labels = np.argmax(np.array(y), axis=1)
_, counts = np.unique(labels, return_counts=True)
n = min(counts)
mask = np.hstack([np.random.choice(np.where(labels == label)[0], n, replace=False) for label in np.unique(labels)])
x, y = np.array(x)[mask, :], np.array(y)[mask, :]

assert x.shape[0] == y.shape[0]

# Create train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=2 / 3)
input_data_shape = x_train.shape[1:]

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create the neural network
model = Sequential()
model.add(Dense(32, input_shape=input_data_shape, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

# Training parameters
n_epochs = 1000
batch_size = 32
patience_percentage = 0.2
path = f'./data/trained_nn'
early_stop = EarlyStopping(monitor='val_accuracy', verbose=1, patience=int(patience_percentage * n_epochs))
model_checkpoint = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True, verbose=1)
optimizer = Adam()

# Compile and train
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(
    x_train_scaled, y_train,
    batch_size=batch_size,
    epochs=n_epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, model_checkpoint]
)

# Get predictions from trained neural network
y_pred = model.predict(x_test_scaled)

# Create training set for inverse AAT
train_iaat = np.array([CreditAssumptions(CreditFeatures(*list(x_test[i, :])),
                                         np.argmax(y_pred[i, :])).create_aat_tuple() for i in range(len(x_test))])

assert len(train_iaat) == len(y_pred)

print(np.unique(train_iaat[:, -1], return_counts=True))

# Create training set for regular surrogate models
train_surrogate = np.append(x_test, y_pred, 1)

assert x_test.shape[0] == train_surrogate.shape[0] and x_test.shape[-1] + 2 == train_surrogate.shape[-1]

# Save the inverse AAT training data and AAT feature names
with open('./data/iaat_credit_training_data.pickle', 'wb') as f:
    pickle.dump(train_iaat, f)

aat_feature_names = CreditAssumptions(CreditFeatures(*list(x_test[0, :])), np.argmax(y_pred[0, :])).assumption_names()

with open('./data/iaat_credit_training_features.pickle', 'wb') as f:
    pickle.dump(aat_feature_names, f)

# Save the ground truth
with open('./data/iaat_credit_training_data_test_truth.pickle', 'wb') as f:
    pickle.dump(np.argmax(y_test, axis=1), f)

# Save the surrogate training data
with open('./data/surrogate_credit_training_data.pickle', 'wb') as f:
    pickle.dump(train_surrogate, f)
