from baseball.baseball_assumptions import BaseballAssumptions, BaseballFeatures
from baseball.data_cleaner import DataCleaner
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential


# Get the data
x, y = DataCleaner.get_baseball_x_and_y()
x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), train_size=2 / 3)
input_data_shape = x_train.shape[1:]

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
model.add(Dense(1, activation='relu'))

# Training parameters
n_epochs = 1000
batch_size = 32
patience_percentage = 0.1
path = f'./data/trained_nn'
early_stop = EarlyStopping(monitor='mean_squared_error', verbose=1, patience=int(patience_percentage * n_epochs))
model_checkpoint = ModelCheckpoint(path, monitor='mean_squared_error', save_best_only=True, verbose=1)
optimizer = Adam()

# Compile and train
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=n_epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, model_checkpoint]
)

# Get predictions from trained neural network
y_pred = model.predict(x_test)

# Create training set for inverse AAT
train_iaat = np.array([BaseballAssumptions(BaseballFeatures(*list(x_test[i, :])), y_pred[i, ][0]).create_aat_tuple()
                       for i in range(len(x_test))])

assert len(train_iaat) == len(y_pred)


# Save the inverse AAT training data and AAT feature names
with open('./data/iaat_baseball_training_data.pickle', 'wb') as f:
    pickle.dump(train_iaat, f)

aat_feature_names = BaseballAssumptions(BaseballFeatures(*list(x_test[0, :])), y_pred[0, ][0]).assumption_names()

with open('./data/iaat_baseball_training_features.pickle', 'wb') as f:
    pickle.dump(aat_feature_names, f)



