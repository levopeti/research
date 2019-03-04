import numpy as np
import time

from abc import ABC, abstractmethod

np.random.seed(int(time.time()))


class ChromosomeBase(ABC):
    """Base class of chromosome for metaheuristic algorithms."""

    __slots__ = "chromosome_size", "__fitness", "__genes",\
                "__fitness_test", "__genes_test", "fitness_function"

    def __init__(self, chromosome_size, fitness_function):
        self.chromosome_size = chromosome_size
        self.__fitness = None
        self.__genes = []

        self.__fitness_test = None
        self.__genes_test = None

        self.create_individual()
        self.fitness_function = fitness_function

        self.num_fitness_eval = 0
        self.counter = -1

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form."""
        return repr((self.__fitness, self.__genes))

    def __next__(self):
        self.counter += 1
        if self.counter < self.chromosome_size:
            return self.__genes[self.counter]
        else:
            self.counter = -1
            raise StopIteration()

    def __iter__(self):
        return self

    def __getitem__(self, index):
        return self.__genes[index]

    def __len__(self):
        return self.chromosome_size

    @abstractmethod
    def create_individual(self):
        """Create a candidate solution representation."""
        pass

    def calculate_fitness(self):
        """Calculate the fitness value of the chromosome."""
        self.__fitness = self.fitness_function.calculate(self.__genes)
        self.num_fitness_eval += 1

    @property
    def fitness(self):
        return self.__fitness

    @property
    def genes(self):
        return self.__genes

    @fitness.setter
    def fitness(self, value):
        if value >= 0:
            self.__fitness = value
        else:
            raise ValueError("Fitness value must be greater or equal then 0!")

    @genes.setter
    def genes(self, genes):
        if len(genes) != self.chromosome_size:
            raise ValueError("Length of genes is not valid!")
        self.__genes = genes
        self.resize_invalid_genes()

    @property
    def genes_test(self):
        return self.__genes_test

    @genes_test.setter
    def genes_test(self, genes):
        if len(genes) != self.chromosome_size:
            raise ValueError("Length of genes test is not valid!")
        self.__genes_test = genes
        self.resize_invalid_genes_test()

    def calculate_fitness_test(self):
        """Calculate the test fitness value of the test genes."""
        if self.genes_test is None:
            raise ValueError("Genes test is not set!")

        self.__fitness_test = self.fitness_function.calculate(self.__genes_test)
        self.num_fitness_eval += 1

    def resize_invalid_genes(self):
        """Resize invalid genes to valid."""
        pass

    def resize_invalid_genes_test(self):
        """Resize invalid genes test to valid."""
        pass

    def set_test(self):
        """Set test values to current values."""
        self.genes_test = self.__genes.copy()
        self.__fitness_test = self.__fitness

    def apply_test(self):
        """Set current values to test values and set test values to None."""

        if self.__genes_test is None or self.__fitness_test is None:
            raise ValueError("Test values should not be None.")

        self.genes = self.__genes_test
        self.__fitness = self.__fitness_test

        self.__genes_test = None
        self.__fitness_test = None

    def reject_test(self):
        """Set test values to None."""
        self.__genes_test = None
        self.__fitness_test = None

    def apply_test_if_better(self):
        """
        Apply test values and return True if they are better
        and set to None, else return False.
        """

        if self.__genes_test is None or self.__fitness_test is None:
            raise ValueError("Test values should not be None.")

        # if test is better
        if self.__fitness_test < self.__fitness:
            self.genes = self.__genes_test
            self.__fitness = self.__fitness_test

            self.__genes_test = None
            self.__fitness_test = None

            return True

        # if original is better
        else:
            self.__genes_test = None
            self.__fitness_test = None

            return False


class Chromosome(ChromosomeBase):
    """ Chromosome class that encapsulates an individual's fitness and solution representation."""

    def __init__(self, chromosome_size, fitness_function):
        super().__init__(chromosome_size, fitness_function)

    def create_individual(self):
        """Create a candidate solution representation."""
        self.genes = np.random.rand(self.chromosome_size)

    def resize_invalid_genes(self):
        """Resize invalid genes to valid."""

        for i in range(self.chromosome_size):
            if self.genes[i] > 1:
                self.genes[i] = 1
            elif self.genes[i] < 0:
                self.genes[i] = 0

    def resize_invalid_genes_test(self):
        """Resize invalid genes test to valid."""

        for i in range(self.chromosome_size):
            if self.genes_test[i] > 1:
                self.genes_test[i] = 1
            elif self.genes_test[i] < 0:
                self.genes_test[i] = 0

    def apply_on_chromosome(self, func):
        """Apply function on the chromosome."""
        func(self)


class Particle(ChromosomeBase):
    """
    Particle class that encapsulates an individual's fitness,
    solution and velocity representation.
    """
    __slots__ = "velocity", "personal_best", "personal_best_fitness",\
                "inertia", "phi_p", "phi_g", "global_best", "norm"

    def __init__(self, chromosome_size, fitness_function):
        super().__init__(chromosome_size, fitness_function)

        self.velocity = None
        self.personal_best = None
        self.global_best = None
        self.personal_best_fitness = 0

        self.inertia = 3
        self.phi_p = 2
        self.phi_g = 5

        self.norm = 3

        self.create_individual()

    def create_individual(self):
        """Create a candidate solution representation."""
        self.genes = (np.random.rand(self.chromosome_size) * 10) - 5
        self.personal_best = copy.deepcopy(self.genes)
        self.velocity = (np.random.rand(self.chromosome_size) * 1) - 0.5

    def calculate_fitness(self):
        if not self.personal_best_fitness:
            fitness_value = self.fitness_function(self)
            self.personal_best_fitness = copy.deepcopy(fitness_value)
            return fitness_value

        return self.fitness_function(self)

    def set_personal_best(self):
        if self.fitness < self.personal_best_fitness:
            self.personal_best_fitness = copy.deepcopy(self.fitness)
            self.personal_best = copy.deepcopy(self.genes)

    def set_global_best(self, global_best):
        self.global_best = global_best.genes

    def update_velocity(self):
        r_p = np.random.rand()
        r_g = np.random.rand()

        self.velocity = self.inertia * self.velocity + self.phi_p * r_p * (self.personal_best - self.genes) + self.phi_g * r_g * (self.global_best - self.genes)
        self.velocity = (self.velocity * self.norm) / np.linalg.norm(self.velocity)

    def iterate(self):
        self.genes += self.velocity

    def sso_iterate(self):

        c_w = 0.2
        c_p = 0.4
        c_g = 0.9

        for i in range(self.chrom_size):
            p = np.random.rand()

            if p < c_w:
                continue
            elif p < c_p:
                self.genes[i] = self.personal_best[i]
            elif p < c_g:
                self.genes[i] = self.global_best[i]
            else:
                self.genes[i] = np.random.rand()





