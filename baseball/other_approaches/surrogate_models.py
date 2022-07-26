import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


N_ITERS = 50
CV = 5

# Read in and process the training data
data_path = '../data/surrogate_baseball_training_data.pickle'
training_data = np.array(pickle.load(open(data_path, 'rb')))

x, y = training_data[:, 0:-1], training_data[:, -1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=2 / 3)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Set up model types and parameters
model_types = {'RF': RandomForestRegressor, 'LR': LinearRegression, 'KNN': KNeighborsRegressor,
               'DT': DecisionTreeRegressor}

model_params = {
    'RF': {'n_estimators': [5, 10, 15, 20, 25, 50],
           'min_samples_leaf': [5, 10, 15, 20, 25, 50],
           'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
           'min_samples_split': [2, 3, 4, 5, 10, 15]},

    'LR': {'fit_intercept': [True, False]},

    'KNN': {'n_neighbors': [3, 5, 10, 15, 30],
            'weights': ['uniform', 'distance']},

    'DT': {'min_samples_leaf': [5, 10, 15, 20, 25, 50],
           'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
           'min_samples_split': [2, 3, 4, 5, 10, 15]}
}


# Train and test each surrogate model
for model_name, model_type in model_types.items():
    # Train
    param_grid = model_params[model_name]
    random_search = RandomizedSearchCV(model_type(), param_grid, cv=CV, n_iter=N_ITERS)
    random_search.fit(x_train_scaled, y_train.ravel())
    model = random_search.best_estimator_

    # Calculate test results
    y_test_pred = model.predict(x_test_scaled).reshape(-1, 1)
    errors = np.square(y_test_pred - y_test)
    mse = mean_squared_error(y_test_pred, y_test)
    assert round(mse, 5) == round(errors.mean(), 5)

    print(f'MSE for {model_name} = {round(mse, 5)}')

    # Save test results
    test_data_path = f'../data/{model_name}_errors.csv'
    np.savetxt(test_data_path, errors, delimiter=',')
