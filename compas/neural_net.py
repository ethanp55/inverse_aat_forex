from compas.compas_dist_assumptions import CompasDistributionAssumptions, CompasFeatureDistributions
from compas.data_cleaner import DataCleaner
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential


# Get the data
x, y = DataCleaner.get_score_data()
x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), train_size=2 / 3)
input_data_shape = x_train.shape[1:]

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Save the scaler in case we need it later
with open('./trained_nn_scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)

# Estimate the distributions of each scaled feature (for the assumptions)
feature_distributions = CompasFeatureDistributions(x_train_scaled, './nn_feature_dists')

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
n_epochs = 2500
batch_size = 32
patience_percentage = 0.1
path = f'./data/trained_nn'
early_stop = EarlyStopping(monitor='mean_squared_error', verbose=1, patience=int(patience_percentage * n_epochs))
model_checkpoint = ModelCheckpoint(path, monitor='mean_squared_error', save_best_only=True, verbose=1)
optimizer = Adam()

# Compile and train
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

model.fit(
    x_train_scaled, y_train,
    batch_size=batch_size,
    epochs=n_epochs,
    validation_split=0.33,
    callbacks=[early_stop, model_checkpoint]
)

# Get predictions from trained neural network
y_pred = model.predict(x_test_scaled)

# Create training set for inverse AAT
aat_scaler = MinMaxScaler()
train_iaat = aat_scaler.fit_transform(x_test)
train_iaat = np.append(train_iaat, y_pred, 1)
# train_iaat = np.array([CompasDistributionAssumptions(*feature_distributions.calculate_probs(list(x_test_scaled[i, :])),
#                                                      y_pred[i, ][0]).create_tuple() for i in range(len(x_test_scaled))])

assert len(train_iaat) == len(y_pred)

# Create training set for regular surrogate models
train_surrogate = np.append(x_test_scaled, y_pred, 1)

assert x_test_scaled.shape[0] == train_surrogate.shape[0] and x_test_scaled.shape[-1] + 1 == train_surrogate.shape[-1]

# Save the inverse AAT training data and AAT feature names
with open('./data/iaat_compas_training_data.pickle', 'wb') as f:
    pickle.dump(train_iaat, f)

aat_feature_names = CompasDistributionAssumptions(*feature_distributions.calculate_probs(list(x_test_scaled[0, :])),
                                                  y_pred[0, ][0]).assumption_names()

with open('./data/iaat_compas_training_features.pickle', 'wb') as f:
    pickle.dump(aat_feature_names, f)

# Save the surrogate training data
with open('./data/surrogate_compas_training_data.pickle', 'wb') as f:
    pickle.dump(train_surrogate, f)
