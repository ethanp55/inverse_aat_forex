from copy import deepcopy
import numpy as np
import pickle
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple


class Genome:
    def __init__(self, baseline: int, genome_length: int, features: Optional[List[int]] = None) -> None:
        self.baseline = baseline
        self.genome_length = genome_length

        data_path = './data/iaat_baseball_training_data.pickle'
        data = np.array(pickle.load(open(data_path, 'rb')))
        data = GeneticHelper.add_correction_term(data, self.baseline)

        k = data.shape[-1] - 2

        if features is None:
            self.features = random.choices([0, 1], k=k)

        else:
            self.features = features

        self.features = self._adjust_features(self.features)
        column_mask = np.array(self.features + [1, 1]).astype(np.bool)
        training_data = data[:, column_mask]

        x = training_data[:, 0:-2]
        y = training_data[:, -2:]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, train_size=2 / 3)

    def _adjust_features(self, features) -> List[int]:
        n_used, adjusted_features = sum(features), [val for val in features]
        indices_where_true = [i for i in range(len(features)) if features[i] == 1]

        while n_used > self.genome_length:
            idx = random.choice(indices_where_true)
            adjusted_features[idx] = 0
            n_used -= 1
            indices_where_true.remove(idx)

        if sum(adjusted_features) == 0:
            idx = random.randrange(len(adjusted_features))
            adjusted_features[idx] = 1

        return adjusted_features

    def predict(self, x: np.array) -> int:
        pass

    def performance(self) -> float:
        pass

    def save_data(self) -> None:
        pass

    def load_data(self) -> None:
        pass

    def feature_names(self) -> List[str]:
        pass


class RfGenome(Genome):
    def __init__(self, baseline: int, genome_length: int = 1, n_estimators: int = 10,
                 min_samples_leaf: int = 5, max_depth: int = 10, min_samples_split: int = 2,
                 optimize_params: bool = True, features: Optional[List[int]] = None) -> None:
        Genome.__init__(self, baseline, genome_length, features)

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

    def predict(self, x: np.array) -> float:
        column_mask = np.array(self.features).astype(np.bool)
        x_scaled = self.scaler.transform(x[:, column_mask])
        correction_pred = self.rf.predict(x_scaled)[0]
        pred = self.baseline * correction_pred

        return pred

    def performance(self) -> float:
        x_test_scaled = self.scaler.transform(self.x_test)
        corrections = self.rf.predict(x_test_scaled)
        predictions = self.baseline * corrections

        return mean_squared_error(self.y_test[:, -2], predictions)

    def save_data(self) -> None:
        x_test_scaled = self.scaler.transform(self.x_test)
        corrections = self.rf.predict(x_test_scaled)
        y_test_pred = self.baseline * corrections
        errors = np.square(y_test_pred - self.y_test[:, -2])

        print(f'Final MSE = {round(errors.mean(), 5)}')

        scaler_file = './data/baseball_genetic_rf_scaler.pickle'
        rf_file = './data/baseball_genetic_rf.pickle'
        features_file = './data/baseball_genetic_rf_features.pickle'
        errors_file = './data/baseball_genetic_rf_errors.csv'

        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(rf_file, 'wb') as f:
            pickle.dump(self.rf, f)

        with open(features_file, 'wb') as f:
            pickle.dump(self.features, f)

        np.savetxt(errors_file, errors, delimiter=',')

    def load_data(self) -> None:
        scaler_file = './data/baseball_genetic_rf_scaler.pickle'
        rf_file = './data/baseball_genetic_rf.pickle'
        features_file = './data/baseball_genetic_rf_features.pickle'

        self.scaler = pickle.load(open(scaler_file, 'rb'))
        self.rf = pickle.load(open(rf_file, 'rb'))
        self.features = pickle.load(open(features_file, 'rb'))

    def feature_names(self) -> List[str]:
        feature_names = pickle.load(open('./data/iaat_baseball_training_features.pickle', 'rb'))

        if self.features is None:
            self.load_data()

        assert len(self.features) == len(feature_names)

        return [feature_names[i] for i in range(len(feature_names)) if self.features[i] == 1]


class Population:
    def __init__(self, genomes: List[Genome]) -> None:
        self.genomes = genomes
        self.performances = [genome.performance() for genome in self.genomes]


class GeneticHelper(object):
    @staticmethod
    def generate_population(population_size: int, baseline: float, genome_length: int, genome_type: type) -> Population:
        assert issubclass(genome_type, Genome)
        genomes = [genome_type(baseline, genome_length) for _ in range(population_size)]

        return Population(genomes)

    @staticmethod
    def selection(population: Population, k: int = 2) -> List[Genome]:
        random_selection = random.choices(population.genomes, weights=population.performances, k=k)
        return [deepcopy(genome) for genome in random_selection]

    @staticmethod
    def single_point_crossover(a: Genome, b: Genome, genome_length: int) -> Tuple[Genome, Genome]:
        genome_type = type(a)

        feature_length = len(a.features)
        idx = random.randint(1, feature_length - 1)

        new_a_features = a.features[0:idx] + b.features[idx:]
        new_b_features = b.features[0:idx] + a.features[idx:]

        new_a = genome_type(a.baseline, genome_length, features=new_a_features)
        new_b = genome_type(b.baseline, genome_length, features=new_b_features)

        return new_a, new_b

    @staticmethod
    def mutation(genome: Genome, genome_length: int, prob: float = 0.5, n_mutations: int = 1) -> Genome:
        genome_type = type(genome)

        mutated_features = [val for val in genome.features]

        for _ in range(n_mutations):
            idx = random.randrange(len(mutated_features))
            mutated_features[idx] = mutated_features[idx] if random.random() > prob else 1 - mutated_features[idx]

        return genome_type(genome.baseline, genome_length,  features=mutated_features)

    @staticmethod
    def n_features() -> int:
        training_data = np.array(pickle.load(open('./data/iaat_baseball_training_data.pickle', 'rb')))

        return training_data.shape[-1] - 1

    @staticmethod
    def get_baseline() -> float:
        training_data = np.array(pickle.load(open('./data/iaat_baseball_training_data.pickle', 'rb')))
        avg = training_data[:, -1].mean()

        return avg

    @staticmethod
    def add_correction_term(training_data: np.array, baseline: int) -> np.array:
        correction_column = (training_data[:, -1] / baseline).reshape(-1, 1)

        return np.append(training_data, correction_column, 1)
