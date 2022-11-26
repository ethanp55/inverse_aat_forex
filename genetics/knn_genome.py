from genetics.genome import Genome, GeneticHelper
from market_proxy.currency_pairs import CurrencyPairs
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import List, Optional


class KnnGenome(Genome):
    def __init__(self, currency_pair: CurrencyPairs, baseline: int, genome_length: int = 1, n_neighbors: int = 15,
                 features: Optional[List[int]] = None) -> None:
        Genome.__init__(self, currency_pair, baseline, genome_length, features)
        self.n_neighbors = n_neighbors

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(self.x_train)

        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn.fit(x_train_scaled)

    def predict(self, x: np.array) -> int:
        if x.shape[-1] == sum(self.features):
            x_filtered = x

        else:
            column_mask = np.array(self.features).astype(np.bool)
            x_filtered = x[:, column_mask]

        x_scaled = self.scaler.transform(x_filtered)
        neighbor_distances, neighbor_indices = self.knn.kneighbors(x_scaled, self.n_neighbors)
        distances = []

        for i in range(len(neighbor_distances[0])):
            neighbor_dist = neighbor_distances[0][i]
            distances.append(neighbor_dist)

        inverse_distance_sum = 0

        for dist in distances:
            inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

        pred = 0

        for i in range(len(distances)):
            distance_i = distances[i]
            inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
            distance_weight = inverse_distance_i / inverse_distance_sum

            neighbor_idx = neighbor_indices[0][i]
            neighbor_correction = self.y_train[neighbor_idx, -1]
            pred += distance_weight * neighbor_correction * self.baseline

        return round(pred)

    def performance(self) -> float:
        n_matches, n_total = 0, len(self.x_test)

        for i in range(n_total):
            pred = self.predict(self.x_test[i, :].reshape(1, -1))
            true = self.y_test[i, -2]

            n_matches += 1 if GeneticHelper.prediction_comparison(pred, true) else 0

        return n_matches / n_total

    def save_data(self) -> None:
        scaler_file = f'../aat/training_data/{self.currency_pair.value}_genetic_scaler.pickle'
        knn_file = f'../aat/training_data/{self.currency_pair.value}_genetic_knn.pickle'
        features_file = f'../aat/training_data/{self.currency_pair.value}_genetic_knn_features.pickle'

        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(knn_file, 'wb') as f:
            pickle.dump(self.knn, f)

        with open(features_file, 'wb') as f:
            pickle.dump(self.features, f)

    def load_data(self) -> None:
        scaler_file = f'../aat/training_data/{self.currency_pair.value}_genetic_scaler.pickle'
        knn_file = f'../aat/training_data/{self.currency_pair.value}_genetic_knn.pickle'
        features_file = f'../aat/training_data/{self.currency_pair.value}_genetic_knn_features.pickle'

        self.scaler = pickle.load(open(scaler_file, 'rb'))
        self.knn = pickle.load(open(knn_file, 'rb'))
        self.features = pickle.load(open(features_file, 'rb'))
