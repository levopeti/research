import random
import numpy as np

from operator import attrgetter
from chromosome import Chromosome


def random_selection(population):
    """Select and return a random member of the population."""
    return random.choice(population.get_all())


def tournament_selection(population):
    """Select a random number of individuals from the population and
    return the fittest member of them all.
    """
    tournament_size = population.pop_size // 2
    members = random.sample(population, tournament_size)
    members.sort(key=attrgetter('fitness'), reverse=False)
    return members[0]


def betterhalf_selection(population):
    """Select the better half of individuals from the population and
    return a random member.
    """
    member = random.sample(population[:(population.pop_size // 2)], 1)
    return member[0]


def crossover(population, selection):
    child_1 = Chromosome(population.chrom_size, population.fitness_function)
    child_2 = Chromosome(population.chrom_size, population.fitness_function)

    parent_1 = selection(population)
    parent_2 = selection(population)

    crossover_index = random.randrange(1, population.chrom_size - 1)
    # child_1.genes = parent_1.genes[:crossover_index] + parent_2.genes[crossover_index:]
    # child_2.genes = parent_2.genes[:crossover_index] + parent_1.genes[crossover_index:]
    child_1.genes = np.concatenate((parent_1.genes[:crossover_index], parent_2.genes[crossover_index:]), axis=None)
    child_2.genes = np.concatenate((parent_2.genes[:crossover_index], parent_1.genes[crossover_index:]), axis=None)

    child_1.fitness = child_1.calculate_fitness()
    child_2.fitness = child_2.calculate_fitness()

    population.add_individual_to_pop(child_1)
    population.add_individual_to_pop(child_2)


