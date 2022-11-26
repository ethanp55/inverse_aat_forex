from aat.assumptions import Assumptions, TechnicalIndicators
from market_proxy.currency_pairs import CurrencyPairs
from market_proxy.trades import TradeType
import numpy as np
from pandas import DataFrame
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import List, Optional


class AatMarketTrainer:
    def __init__(self, currency_pair: CurrencyPairs, file_specifier: Optional[str] = None) -> None:
        self.currency_pair = currency_pair
        self.file_specifier = file_specifier
        self.training_data = []
        self.feature_names = None

    def record_tuple(self, curr_idx: int, market_data: DataFrame, trade_type: TradeType) -> None:
        ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi, slowd_rsi, \
        vo, willy, willy_ema, key_level, is_support = \
            market_data.loc[market_data.index[curr_idx - 1], ['ema200', 'ema100', 'atr', 'atr_sma', 'rsi', 'rsi_sma',
                                                              'adx', 'macd', 'macdsignal', 'slowk_rsi', 'slowd_rsi',
                                                              'vo', 'willy', 'willy_ema', 'key_level', 'is_support']]
        bid_open, ask_open = market_data.loc[market_data.index[curr_idx], ['Bid_Open', 'Ask_Open']]

        ti_vals = TechnicalIndicators(ema200, ema100, atr, atr_sma, rsi, rsi_sma, adx, macd, macdsignal, slowk_rsi,
                                      slowd_rsi, vo, willy, willy_ema)

        new_assumptions = Assumptions(ti_vals, bid_open, ask_open, key_level, trade_type)
        new_tup = new_assumptions.create_aat_tuple()

        if self.feature_names is None:
            self.feature_names = new_assumptions.assumption_names()

        self.training_data.append(new_tup)

    def save_data(self) -> None:
        self._data_dir = '../aat/training_data'
        specifier = f'{self.file_specifier}_' if self.file_specifier is not None else ''

        file_path = f'{self._data_dir}/{self.currency_pair.value}_{specifier}training_data.pickle'
        feature_names_path = f'{self._data_dir}/{self.currency_pair.value}_{specifier}training_features.pickle'

        training_array = np.array(self.training_data)
        y = training_array[:, -1]
        unique_labels, counts = np.unique(y, return_counts=True)
        n = min(counts)
        mask = np.hstack([np.random.choice(np.where(y == label)[0], n, replace=False) for label in unique_labels])

        training_array = training_array[mask, :]

        with open(file_path, 'wb') as f:
            pickle.dump(training_array, f)

        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_names, f)


class AatKnnMarketTrainerForCnnModel(AatMarketTrainer):
    def __init__(self, currency_pair: CurrencyPairs) -> None:
        AatMarketTrainer.__init__(self, currency_pair)

    def save_data(self) -> None:
        AatMarketTrainer.save_data(self)

        x = np.array(self.training_data)[:, 0:-1]
        y = np.array(self.training_data)[:, -1]

        print('X train shape: ' + str(x.shape))
        print('Y train shape: ' + str(y.shape))

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        model = NearestNeighbors(n_neighbors=15)
        model.fit(x_scaled)

        trained_knn_file = f'{self.currency_pair.value}_trained_knn_aat.pickle'
        trained_knn_scaler_file = f'{self.currency_pair.value}_trained_knn_scaler_aat.pickle'

        with open(f'{self._data_dir}/{trained_knn_file}', 'wb') as f:
            pickle.dump(model, f)

        with open(f'{self._data_dir}/{trained_knn_scaler_file}', 'wb') as f:
            pickle.dump(scaler, f)


class AatRfMarketTrainerForCnnModel(AatMarketTrainer):
    def __init__(self, currency_pair: CurrencyPairs) -> None:
        AatMarketTrainer.__init__(self, currency_pair)

    def save_data(self) -> None:
        AatMarketTrainer.save_data(self)

        x = np.array(self.training_data)[:, 0:-1]
        y = np.array(self.training_data)[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=2 / 3)

        print('X train shape: ' + str(x_train.shape))
        print('Y train shape: ' + str(y_train.shape))

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)

        param_grid = {'n_estimators': [1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200],
                      'min_samples_leaf': [1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200],
                      'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      'min_samples_split': [2, 3, 4, 5, 10, 15, 20]}

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
        grid_search.fit(x_train_scaled, y_train)

        print(f'Best random forest parameters:\n{grid_search.best_params_}')

        model = grid_search.best_estimator_
        print(model.feature_importances_)
        print(len(model.feature_importances_))
        print(len(self.feature_names))

        trained_rf_file = f'{self.currency_pair.value}_trained_rf_aat.pickle'
        trained_rf_scaler_file = f'{self.currency_pair.value}_trained_rf_scaler_aat.pickle'
        feature_importances_file = f'{self.currency_pair.value}_rf_feature_importances.pickle'

        with open(f'{self._data_dir}/{trained_rf_file}', 'wb') as f:
            pickle.dump(model, f)

        with open(f'{self._data_dir}/{trained_rf_scaler_file}', 'wb') as f:
            pickle.dump(scaler, f)

        with open(f'{self._data_dir}/{feature_importances_file}', 'wb') as f:
            pickle.dump(list(model.feature_importances_), f)


class AatReducedRfMarketTrainerForCnnModel(AatMarketTrainer):
    def __init__(self, currency_pair: CurrencyPairs, possible_top_n_features: List[int]) -> None:
        AatMarketTrainer.__init__(self, currency_pair)
        self.possible_top_n_features = possible_top_n_features

    def save_data(self) -> None:
        AatMarketTrainer.save_data(self)

        file_path = f'../aat/training_data/{self.currency_pair.value}_rf_feature_importances.pickle'
        feature_importances = pickle.load(open(file_path, 'rb'))
        best_test_accuracy = -np.inf

        for top_n_features in self.possible_top_n_features:
            top_n_feature_indices = [feature_importances.index(x) for x in
                                     sorted(feature_importances, reverse=True)[:top_n_features]]
            features_used = [self.feature_names[i] for i in top_n_feature_indices]

            x = np.array(self.training_data)[:, top_n_feature_indices]
            y = np.array(self.training_data)[:, -1]

            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=2 / 3)

            print(f'---------- Training with the top {top_n_features} features ----------')
            print('X train shape: ' + str(x_train.shape))
            print('Y train shape: ' + str(y_train.shape))

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.fit_transform(x_test)

            param_grid = {'n_estimators': [1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200],
                          'min_samples_leaf': [1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200],
                          'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                          'min_samples_split': [2, 3, 4, 5, 10, 15, 20]}

            grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
            grid_search.fit(x_train_scaled, y_train)

            print(f'Best random forest parameters:\n{grid_search.best_params_}')

            model = grid_search.best_estimator_
            y_pred = model.predict(x_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)

            if test_accuracy > best_test_accuracy:
                print(f'Saving model with test accuracy {test_accuracy}')
                best_test_accuracy = test_accuracy

                trained_rf_file = f'{self.currency_pair.value}_trained_reduced_rf_aat.pickle'
                trained_rf_scaler_file = f'{self.currency_pair.value}_trained_reduced_rf_scaler_aat.pickle'
                features_used_file = f'{self.currency_pair.value}_reduced_rf_features_used.pickle'

                with open(f'{self._data_dir}/{trained_rf_file}', 'wb') as f:
                    pickle.dump(model, f)

                with open(f'{self._data_dir}/{trained_rf_scaler_file}', 'wb') as f:
                    pickle.dump(scaler, f)

                with open(f'{self._data_dir}/{features_used_file}', 'wb') as f:
                    pickle.dump(features_used, f)

            print('---------------------------------------------------------------------\n')
