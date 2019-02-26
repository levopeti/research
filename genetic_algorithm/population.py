from chromosome import Chromosome, Particle
from operator import attrgetter
import copy
import random
import numpy as np

from abc import ABC, abstractmethod


class PopulationBase(ABC):
    """Base class of population for metaheuristic algorithms."""

    __slots__ = "__current_generation", "pop_size", "chromosome_size", "counter",\
                "fitness_function", "global_best_individual"

    def __init__(self, pop_size, chromosome_size, fitness_function):
        """Initialise the Population."""
        self.__current_population = []
        self.pop_size = pop_size
        self.chromosome_size = chromosome_size
        self.counter = -1
        self.fitness_function = fitness_function
        self.global_best_individual = None

        self.create_initial_population()

    def __repr__(self):
        """Return initialised Population representation in human readable form."""
        return repr((self.pop_size, self.__current_population))

    def __next__(self):
        self.counter += 1
        if self.counter < len(self.__current_population):
            return self.__current_population[self.counter]
        else:
            self.counter = -1
            raise StopIteration()

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return self.__current_population[index]

    def __len__(self):
        return len(self.__current_population)

    @property
    def current_population(self):
        return self.__current_population

    @current_population.setter
    def current_population(self, chromosomes):
        self.__current_population = chromosomes

    # def get_all(self):
    #     return self.__current_population
    #
    # def set_fitness(self, fitness, i):
    #     self.__current_population[i].fitness = fitness
    #
    # def set_genes(self, genes, i):
    #     self.__current_population[i].genes = genes

    @abstractmethod
    def create_initial_population(self):
        """Create members of the first population randomly."""
        pass

    def rank_population(self):
        """Sort the population by fitness ascending order."""
        self.__current_population.sort(key=attrgetter('fitness'), reverse=False)

    def add_individual_to_pop(self, individual):
        """Add an individual to the current population."""
        self.__current_population.append(individual)

    @abstractmethod
    def add_new_individual(self):
        """Add new individual to the current population."""
        pass

    def cut_pop_size(self):
        """Resize the current population to pop size."""
        self.__current_population = self.__current_population[:self.pop_size]

    def get_best_fitness(self):
        self.rank_population()
        return self.__current_population[0].fitness

    def get_best_genes(self):
        self.rank_population()
        return self.__current_population[0].genes

    def init_global_best(self):
        pass

    def set_global_best(self):
        pass

    def set_personal_bests(self):
        pass


class Population(PopulationBase):
    """ Population class that encapsulates all of the chromosomes."""

    def __init__(self, pop_size, chromosome_size, fitness_function):
        super().__init__(pop_size, chromosome_size, fitness_function)

    def create_initial_population(self):
        """Create members of the first population randomly."""

        for _ in range(self.pop_size):
            individual = Chromosome(self.chromosome_size, self.fitness_function)
            individual.calculate_fitness()
            self.add_individual_to_pop(individual)

    def add_new_individual(self):
        """Add new individual with fitness value to the current population."""
        individual = Chromosome(self.chromosome_size, self.fitness_function)
        individual.calculate_fitness()
        self.add_individual_to_pop(individual)

    def crossover(self, selection_function):
        """Add new individuals to the population with crossover."""

        child_1 = Chromosome(self.chromosome_size, self.fitness_function)
        child_2 = Chromosome(self.chromosome_size, self.fitness_function)

        parent_1 = selection_function(self)
        parent_2 = selection_function(self)

        crossover_index = random.randrange(1, self.chromosome_size - 1)

        child_1.genes = np.concatenate((parent_1.genes[:crossover_index], parent_2.genes[crossover_index:]), axis=None)
        child_2.genes = np.concatenate((parent_2.genes[:crossover_index], parent_1.genes[crossover_index:]), axis=None)

        child_1.calculate_fitness()
        child_2.calculate_fitness()

        self.add_individual_to_pop(child_1)
        self.add_individual_to_pop(child_2)


class Swarm(PopulationBase):
    """ Swarm class that encapsulates all of the particles."""

    def __init__(self, pop_size, chromosome_size, fitness_function):
        super().__init__(pop_size, chromosome_size, fitness_function)

    def create_initial_population(self):
        """Create members of the first population randomly."""

        for _ in range(self.pop_size):
            individual = Particle(self.chromosome_size, self.fitness_function)
            self.add_individual_to_pop(individual)

    def add_new_individual(self):
        """Add new individual to the current population."""
        individual = Particle(self.chromosome_size, self.fitness_function)
        self.add_individual_to_pop(individual)

    def init_global_best(self):
        self.global_best_individual = copy.deepcopy(self.__current_population[0])

    def set_global_best(self):
        if self.__current_population[0].fitness < self.global_best_individual.fitness:
            self.global_best_individual = copy.deepcopy(self.__current_population[0])

        for particle in self.__current_population:
            particle.set_global_best(self.global_best_individual)

    def set_personal_bests(self):
        for particle in self.__current_population:
            particle.set_personal_best()


