from genetics.genome import GeneticHelper, Population
from genetics.rf_genome import RfGenome
from market_proxy.currency_pairs import CurrencyPairs
import numpy as np
import random


FITNESS_LIMIT = 0.99
MAX_ITERATIONS = 500
POPULATION_SIZE = 50
CURRENCY_PAIR = CurrencyPairs.EUR_USD
FILE_SPECIFIER = 'bars'
BASELINE = GeneticHelper.get_baseline(CURRENCY_PAIR, file_specifier=FILE_SPECIFIER)
N_FEATURES = GeneticHelper.n_features(CURRENCY_PAIR, file_specifier=FILE_SPECIFIER)
GENOME_PERCENTAGE = 1 / 3
GENOME_LENGTH = int(np.ceil(GENOME_PERCENTAGE * N_FEATURES))
POSSIBLE_N_MUTATIONS = list(range(int(N_FEATURES / 4)))
GENOME_TYPE = RfGenome

population = GeneticHelper.generate_population(POPULATION_SIZE, CURRENCY_PAIR, BASELINE, GENOME_LENGTH, GENOME_TYPE,
                                               FILE_SPECIFIER)

for i in range(MAX_ITERATIONS):
    print(f'Generation {i + 1} / {MAX_ITERATIONS}')

    assert len(population.genomes) == POPULATION_SIZE

    sorted_performances_with_indices = sorted(zip(enumerate(population.performances)), key=lambda x: x[0][-1],
                                              reverse=True)
    best_idx, best_performance = sorted_performances_with_indices[0][0]
    second_best_idx, second_best_performance = sorted_performances_with_indices[1][0]
    best_genome, second_best_genome = population.genomes[best_idx], population.genomes[second_best_idx]

    print(f'Best performance = {best_performance}, features = {sum(best_genome.features)} / {N_FEATURES}')

    if best_performance >= FITNESS_LIMIT or i >= MAX_ITERATIONS - 1:
        best_genome.save_data()
        break

    new_genomes = [best_genome, second_best_genome]

    for j in range(int(POPULATION_SIZE / 2) - 1):
        parents = GeneticHelper.selection(population)
        offspring_a, offspring_b = GeneticHelper.single_point_crossover(parents[0], parents[1], GENOME_LENGTH)
        n_mutations = random.choice(POSSIBLE_N_MUTATIONS)
        offspring_a = GeneticHelper.mutation(offspring_a, GENOME_LENGTH, n_mutations=n_mutations)
        offspring_b = GeneticHelper.mutation(offspring_b, GENOME_LENGTH, n_mutations=n_mutations)
        new_genomes += [offspring_a, offspring_a]

    population = Population(new_genomes)



