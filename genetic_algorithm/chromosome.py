import random
import fitness_functions


class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, chrom_size, fitness_function):
        """Initialise the Chromosome."""
        self.chrom_size = chrom_size
        self.fitness = 0
        self.step_size = 0.05
        self.genes = []

        self.create_individual()
        self.fitness_function = fitness_function

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))

    def create_individual(self):
        """Create a candidate solution representation.
        """
        self.genes = [random.random() for _ in range(self.chrom_size)]

    def calculate_fitness(self):
        return self.fitness_function(self)

