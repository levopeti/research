import random
import fitness_functions
import numpy as np
import time

np.random.seed(int(time.time()))


class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    __slots__ = "chrom_size", "fitness", "step_size", "genes", "fitness_function"

    def __init__(self, chrom_size, fitness_function):
        """Initialise the Chromosome."""
        self.chrom_size = chrom_size
        self.fitness = 0
        self.step_size = 0.05
        self.genes = []

        self.create_individual()
        self.fitness_function = fitness_function
        # print("max of g: {0:.2f}".format(np.amax(self.genes)))
        # print("min of g: {0:.2f}\n".format(np.amin(self.genes)))
        # print(self.calculate_fitness())

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))

    def create_individual(self):
        """Create a candidate solution representation.
        """
        # self.genes = [random.random() for _ in range(self.chrom_size)]
        self.genes = (np.random.rand(self.chrom_size) * 1) - 0.5

    def calculate_fitness(self):
        return self.fitness_function(self)

