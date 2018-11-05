import numpy as np
import time
import copy

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
        self.genes = None

        self.create_individual()
        self.fitness_function = fitness_function

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))

    def create_individual(self):
        """Create a candidate solution representation.
        """
        self.genes = (np.random.rand(self.chrom_size) * 1) - 0.5

    def calculate_fitness(self):
        return self.fitness_function(self)


class Particle(Chromosome):
    """ Particle class that encapsulates an individual's fitness, solution and velocity
    representation.
    """
    __slots__ = "velocity", "personal_best", "personal_best_fitness", "inertia", "phi_p", "phi_g", "global_best", "norm"

    def __init__(self, chrom_size, fitness_function):
        super().__init__(chrom_size, fitness_function)

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
        """Create a candidate solution representation.
        """
        self.genes = (np.random.rand(self.chrom_size) * 10) - 5
        self.personal_best = copy.deepcopy(self.genes)
        self.velocity = (np.random.rand(self.chrom_size) * 1) - 0.5

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





