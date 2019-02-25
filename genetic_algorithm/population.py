from chromosome import Chromosome, Particle
from operator import attrgetter
import copy

from abc import ABC, abstractmethod


class PopulationBase(ABC):
    """Base class of population for metaheuristic algorithms."""

    __slots__ = "current_generation", "pop_size", "chromosome_size", "counter",\
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
        if self.counter < self.pop_size:
            self.counter += 1
            return self.__current_population[self.counter]
        else:
            self.counter = -1
            raise StopIteration()

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return self.__current_population[index]

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
        """Add new individual to the current population."""
        self.__current_population.append(individual)

    def cut_pop_size(self):
        """Resize the current population to pop size."""
        self.__current_population = self.__current_population[:self.pop_size]

    def get_best_fitness(self):
        self.rank_population()
        return self.__current_population[0].get_fitness()

    def get_best_genes(self):
        self.rank_population()
        return self.__current_population[0].get_genes()

    def init_global_best(self):
        pass

    def set_global_best(self):
        pass

    def set_personal_bests(self):
        pass


class Population(PopulationBase):
    """ Population class that encapsulates all of the chromosomes."""

    __slots__ = "current_generation", "pop_size", "chrom_size", "counter", "fitness_function"

    def __init__(self, pop_size, chrom_size, fitness_function):
        """Initialise the Population."""
        self.current_generation = []
        self.pop_size = pop_size
        self.chrom_size = chrom_size
        self.counter = self.pop_size
        self.fitness_function = fitness_function

        self.create_initial_population()

    def create_initial_population(self):
        """Create members of the first population randomly."""
        initial_population = []

        for _ in range(self.pop_size):
            individual = Chromosome(self.chrom_size, self.fitness_function)
            initial_population.append(individual)

        self.current_generation = initial_population


class Swarm(PopulationBase):
    """ Swarm class that encapsulates all of the particles."""

    def __init__(self, pop_size, chromosome_size, fitness_function):
        super().__init__(pop_size, chromosome_size, fitness_function)

    def create_initial_population(self):
        """Create members of the first population randomly."""
        initial_population = []

        for _ in range(self.pop_size):
            individual = Particle(self.chromosome_size, self.fitness_function)
            initial_population.append(individual)

        self.__current_population = initial_population

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


