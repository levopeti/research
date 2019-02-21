from abc import ABC, abstractmethod
import time


class BaseAlgorithmClass(ABC):
    """
    Base class of metaheuristic algoritms.

    TODO: How to use
    """
    def __init__(self,
                 population_size=50,
                 chromosome_size=10,
                 max_iteration=None,
                 max_fitness_eval=None,
                 patience=None,
                 lamarck=False,
                 pool_size=None,
                 pool=False):

        self.patience = float("inf") if patience is None else patience
        self.max_iteration = float("inf") if max_iteration is None else max_iteration
        self.max_fitness_eval= float("inf") if max_fitness_eval is None else max_fitness_eval
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.lamarck = lamarck
        self.pool_size = pool_size
        self.pool = pool

        self.population = None

        self.fitness_function = None
        self.selection_function = None
        self.mutation_function = None
        self.memetic_function = None
        self.crossover_function = None

        self.iteration_steps = []
        self.iteration = 0
        self.no_improvement = 0
        self.num_of_fitness_eval = 0
        self.best_fitness = None

    def compile(self,
                fitness_function,
                selection_function,
                mutation_function,
                memetic_function,
                crossover_function):
        """Compile the functions of the algorithm."""

        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.mutation_function = mutation_function
        self.memetic_function = memetic_function
        self.crossover_function = crossover_function

    def init_population(self):
        """
        Create and initialize the population.
        Calculate the population's fitness and rank the population
        by fitness ascending order.
        """
        self.create_population()
        self.calculate_fitness()
        self.population.rank_population()
        self.population.init_global_best()
        self.population.set_global_best()
        self.population.set_personal_bests()
        self.init_steps()

    @abstractmethod
    def calculate_fitness(self):
        """Calculate and set the fitness values of the individuals of the population."""
        pass

    @abstractmethod
    def create_population(self):
        """Set the appropriate population type."""
        pass

    @abstractmethod
    def init_steps(self):
        """Initialize the iteration steps."""
        pass

    def next_iteration(self):
        """Create the next iteration or generation with the corresponding steps."""
        for step in self.iteration_steps:
            step()

    def run(self):
        """Run (solve) the algorithm."""
        self.init_population()

        self.iteration = 1
        self.best_fitness = self.population.get_best_fitness()

        while self.no_improvement < self.patience and self.max_iteration >= self.iteration:
            start = time.time()
            print('{}. iteration'.format(self.iteration))
            self.next_iteration()
            print('best fitness values:')
            for j in range(4 if self.population_size >= 4 else self.population_size):
                print('{0:.3f}'.format(self.population.get_fitness(j)))

            end = time.time()
            print('Process time: {0:.2f}s\n'.format(end - start))

            if self.best_fitness > self.population.get_best_fitness():
                self.no_improvement = 0
                self.best_fitness = self.population.get_best_fitness()
            else:
                self.no_improvement += 1

            self.iteration += 1

    def last_iteration(self):
        """Return members of the last iteration as a generator function."""
        return ((member.fitness, member.genes) for member in self.population.get_all())

    def best_individual(self):
        """Return the individual with the best fitness in the current generation."""
        best_genes = self.population.get_best_genes()
        best_fitness = self.population.get_best_fitness()
        return best_genes, best_fitness
