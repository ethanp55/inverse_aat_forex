from copy import deepcopy
from market_proxy.currency_pairs import CurrencyPairs
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple


class Genome:
    def __init__(self, currency_pair: CurrencyPairs, baseline: int, genome_length: int,
                 features: Optional[List[int]] = None, file_specifier: Optional[str] = None) -> None:
        self.currency_pair = currency_pair
        self.baseline = baseline
        self.genome_length = genome_length
        self.file_specifier = file_specifier

        specifier = f'{self.file_specifier}_' if self.file_specifier is not None else ''
        data_path = f'../aat/training_data/{currency_pair.value}_{specifier}training_data.pickle'
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


class Population:
    def __init__(self, genomes: List[Genome]) -> None:
        self.genomes = genomes
        self.performances = [genome.performance() for genome in self.genomes]


class GeneticHelper(object):
    @staticmethod
    def generate_population(population_size: int, currency_pair: CurrencyPairs, baseline: int,
                            genome_length: int, genome_type: type, file_specifier: Optional[str] = None) -> Population:
        assert issubclass(genome_type, Genome)
        genomes = [genome_type(currency_pair, baseline, genome_length, file_specifier=file_specifier)
                   for _ in range(population_size)]

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

        new_a = genome_type(a.currency_pair, a.baseline, genome_length, features=new_a_features,
                            file_specifier=a.file_specifier)
        new_b = genome_type(b.currency_pair, b.baseline, genome_length, features=new_b_features,
                            file_specifier=b.file_specifier)

        return new_a, new_b

    @staticmethod
    def mutation(genome: Genome, genome_length: int, prob: float = 0.5, n_mutations: int = 1) -> Genome:
        genome_type = type(genome)

        mutated_features = [val for val in genome.features]

        for _ in range(n_mutations):
            idx = random.randrange(len(mutated_features))
            mutated_features[idx] = mutated_features[idx] if random.random() > prob else 1 - mutated_features[idx]

        return genome_type(genome.currency_pair, genome.baseline, genome_length,  features=mutated_features,
                           file_specifier=genome.file_specifier)

    @staticmethod
    def n_features(currency_pair: CurrencyPairs, file_specifier: Optional[str] = None) -> int:
        specifier = f'{file_specifier}_' if file_specifier is not None else ''
        data_path = f'../aat/training_data/{currency_pair.value}_{specifier}training_data.pickle'
        training_data = np.array(pickle.load(open(data_path, 'rb')))

        return training_data.shape[-1] - 1

    @staticmethod
    def get_baseline(currency_pair: CurrencyPairs, file_specifier: Optional[str] = None) -> int:
        specifier = f'{file_specifier}_' if file_specifier is not None else ''
        data_path = f'../aat/training_data/{currency_pair.value}_{specifier}training_data.pickle'
        training_data = np.array(pickle.load(open(data_path, 'rb')))
        arry, counts = np.unique(training_data[:, -1], return_counts=True)
        y = int(arry[counts == counts.max()][0])

        return y + 1

    @staticmethod
    def prediction_comparison(pred: int, true: int) -> bool:
        # print(pred - 1, true)
        return (pred - 1) == true

    @staticmethod
    def add_correction_term(training_data: np.array, baseline: int) -> np.array:
        correction_column = ((training_data[:, -1] + 1) / baseline).reshape(-1, 1)

        return np.append(training_data, correction_column, 1)
