from chromosome import Chromosome, Particle
from operator import attrgetter
import copy


class Population(object):
    """ Population class that encapsulates all of the chromosomes.
    """
    __slots__ = "current_generation", "pop_size", "chrom_size", "counter", "fitness_function"

    def __init__(self, pop_size, chrom_size, fitness_function):
        """Initialise the Population."""
        self.current_generation = []
        self.pop_size = pop_size
        self.chrom_size = chrom_size
        self.counter = self.pop_size
        self.fitness_function = fitness_function

        self.create_initial_population()

    def __repr__(self):
        """Return initialised Population representation in human readable form.
        """
        return repr((self.pop_size, self.current_generation))

    def __next__(self):
        if self.counter < 0:
            self.counter -= 1
            return self.current_generation[self.counter]
        else:
            self.counter = self.pop_size
            raise StopIteration()

    def __iter__(self):
        return self

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []

        for _ in range(self.pop_size):
            individual = Chromosome(self.chrom_size, self.fitness_function)
            initial_population.append(individual)

        self.current_generation = initial_population

    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(key=attrgetter('fitness'), reverse=False)

    def add_individual_to_pop(self, individual: Chromosome):
        self.current_generation.append(individual)

    def insp_pop_size(self):
        self.current_generation = self.current_generation[:self.pop_size]

    def get_the_best(self) -> Chromosome:
        return self.current_generation[0]

    def get_all(self):
        return self.current_generation

    def set_fitness(self, fitness, i):
        self.current_generation[i].fitness = fitness

    def set_genes(self, genes, i):
        self.current_generation[i].genes = genes


class Swarm(Population):
    """ Swarm class that encapsulates all of the particles.
    """
    __slots__ = "global_best"

    def __init__(self, pop_size, chrom_size, fitness_function):
        super().__init__(pop_size, chrom_size, fitness_function)

        self.global_best = None

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []

        for _ in range(self.pop_size):
            individual = Particle(self.chrom_size, self.fitness_function)
            initial_population.append(individual)

        self.current_generation = initial_population

    def init_global_best(self):
        self.global_best = copy.deepcopy(self.current_generation[0])

    def set_global_best(self):
        if self.current_generation[0].fitness < self.global_best.fitness:
            self.global_best = copy.deepcopy(self.current_generation[0])

        for particle in self.current_generation:
            particle.set_global_best(self.global_best)

    def set_personal_bests(self):
        for particle in self.current_generation:
            particle.set_personal_best()


