from genetics.genome import Genome, GeneticHelper
from market_proxy.currency_pairs import CurrencyPairs
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from typing import List, Optional


class RfGenome(Genome):
    def __init__(self, currency_pair: CurrencyPairs, baseline: int, genome_length: int = 1, n_estimators: int = 10,
                 min_samples_leaf: int = 5, max_depth: int = 10, min_samples_split: int = 2,
                 optimize_params: bool = True, features: Optional[List[int]] = None,
                 file_specifier: Optional[str] = None) -> None:
        Genome.__init__(self, currency_pair, baseline, genome_length, features, file_specifier)

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(self.x_train)
        y_train = self.y_train[:, -1]

        if optimize_params:
            param_grid = {'n_estimators': [5, 10, 15, 20, 25, 50],
                          'min_samples_leaf': [5, 10, 15, 20, 25, 50],
                          'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                          'min_samples_split': [2, 3, 4, 5, 10, 15]}

            random_search = RandomizedSearchCV(RandomForestRegressor(), param_grid, cv=2, n_iter=20)
            random_search.fit(x_train_scaled, y_train)
            self.rf = random_search.best_estimator_

        else:

            self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                            max_depth=max_depth, min_samples_split=min_samples_split)
            self.rf.fit(x_train_scaled, y_train)

    def predict(self, x: np.array) -> int:
        column_mask = np.array(self.features).astype(np.bool)
        x_scaled = self.scaler.transform(x[:, column_mask])
        correction_pred = self.rf.predict(x_scaled)[0]
        pred = self.baseline * correction_pred

        return round(pred) - 1

    def performance(self) -> float:
        x_test_scaled = self.scaler.transform(self.x_test)
        corrections = self.rf.predict(x_test_scaled)
        predictions = np.round(self.baseline * corrections)
        # predictions = corrections
        n_total = len(predictions)
        matches = [GeneticHelper.prediction_comparison(predictions[i], self.y_test[i, -2])
                   for i in range(n_total)]
        n_matches = sum(matches)

        return n_matches / n_total

    def save_data(self) -> None:
        specifier = f'{self.file_specifier}_' if self.file_specifier is not None else ''

        scaler_file = f'../aat/training_data/{self.currency_pair.value}_{specifier}genetic_rf_scaler.pickle'
        rf_file = f'../aat/training_data/{self.currency_pair.value}_{specifier}genetic_rf.pickle'
        features_file = f'../aat/training_data/{self.currency_pair.value}_{specifier}genetic_rf_features.pickle'

        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(rf_file, 'wb') as f:
            pickle.dump(self.rf, f)

        with open(features_file, 'wb') as f:
            pickle.dump(self.features, f)

    def load_data(self) -> None:
        specifier = f'{self.file_specifier}_' if self.file_specifier is not None else ''

        scaler_file = f'../aat/training_data/{self.currency_pair.value}_{specifier}genetic_rf_scaler.pickle'
        rf_file = f'../aat/training_data/{self.currency_pair.value}_{specifier}genetic_rf.pickle'
        features_file = f'../aat/training_data/{self.currency_pair.value}_{specifier}genetic_rf_features.pickle'

        self.scaler = pickle.load(open(scaler_file, 'rb'))
        self.rf = pickle.load(open(rf_file, 'rb'))
        self.features = pickle.load(open(features_file, 'rb'))
