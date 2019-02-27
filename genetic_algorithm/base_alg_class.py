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
                 min_fitness=None,
                 patience=None,
                 pool_size=None,
                 pool=False):

        self.patience = float("inf") if patience is None else patience
        self.max_iteration = float("inf") if max_iteration is None else max_iteration
        self.max_fitness_eval = float("inf") if max_fitness_eval is None else max_fitness_eval
        self.min_fitness = 0 if min_fitness is None else min_fitness
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.pool_size = pool_size
        self.pool = pool

        self.population = None

        self.fitness_function = None
        self.selection_function = None
        self.mutation_function = None
        self.memetic_function = None
        self.crossover_function = None

        self.callbacks = None
        self.logs = None

        self.iteration_steps = []
        self.iteration = 0
        self.no_improvement = 0
        self.num_of_fitness_eval = 0  # TODO
        self.best_fitness = None

        # TODO: Time dict

    def compile(self,
                fitness_function,
                selection_function,
                mutation_function=None,
                memetic_function=None,
                crossover_function=None,
                callbacks=None):
        """Compile the functions of the algorithm."""

        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.mutation_function = mutation_function
        self.memetic_function = memetic_function
        self.crossover_function = crossover_function
        self.callbacks = callbacks if callbacks else []

        self.init_population()

    def init_population(self):
        """
        Create and initialize the population.
        Calculate the population's fitness and rank the population
        by fitness ascending order.
        """
        self.create_population()
        # self.calculate_fitness()
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

        self.callbacks_on_iteration_begin()

        for step in self.iteration_steps:
            self.callbacks_on_step_begin()
            step()

            self.rank_population()
            self.get_all_fitness_eval()
            self.print_best_values(top_n=4)
            self.callbacks_on_step_end()

        self.cut_pop_size()
        self.callbacks_on_iteration_end()

    def run(self):
        """Run (solve) the algorithm."""

        self.iteration = 1
        self.best_fitness = self.population.get_best_fitness()
        self.callbacks_on_search_begin()

        while self.no_improvement < self.patience and self.max_iteration >= self.iteration \
                and self.min_fitness < self.best_fitness and self.max_fitness_eval > self.num_of_fitness_eval:
            start = time.time()
            print('*' * 36, '{}. iteration'.format(self.iteration), '*' * 36)
            self.next_iteration()

            end = time.time()
            print('Number of fitness evaluation so far: ', self.num_of_fitness_eval)
            print('Iteration process time: {0:.2f}s\n\n'.format(end - start))

            if self.best_fitness > self.population.get_best_fitness():
                self.no_improvement = 0
                self.best_fitness = self.population.get_best_fitness()
            else:
                self.no_improvement += 1

            self.iteration += 1

        self.callbacks_on_search_end()

    def callbacks_on_search_begin(self):
        """Call the on_search_begin function of the callbacks."""
        for callback in self.callbacks:
            callback.on_iteration_end(self.logs)
            callback.set_model(self)

    def callbacks_on_search_end(self):
        """Call the on_search_end function of the callbacks."""
        for callback in self.callbacks:
            callback.on_iteration_end(self.logs)

    def callbacks_on_iteration_begin(self):
        """Call the on_iteration_begin function of the callbacks."""
        for callback in self.callbacks:
            callback.on_iteration_end(self.logs)

    def callbacks_on_iteration_end(self):
        """Call the on_iteration_end function of the callbacks."""
        for callback in self.callbacks:
            callback.on_iteration_end(self.logs)

    def callbacks_on_step_begin(self):
        """Call the on_step_begin function of the callbacks."""
        for callback in self.callbacks:
            callback.on_iteration_end(self.logs)

    def callbacks_on_step_end(self):
        """Call the on_step_end function of the callbacks."""
        for callback in self.callbacks:
            callback.on_iteration_end(self.logs)

    def last_iteration(self):
        """Return members of the last iteration as a generator function."""
        return ((member.fitness, member.genes) for member in self.population.get_all())

    def best_individual(self):
        """Return the individual with the best fitness in the current generation."""
        best_genes = self.population.get_best_genes()
        best_fitness = self.population.get_best_fitness()
        return best_genes, best_fitness

    def rank_population(self):
        """Sort the population by fitness ascending order."""
        self.population.rank_population()

    def cut_pop_size(self):
        """Resize the current population to pop size."""
        self.population.cut_pop_size()

    def add_individual_to_pop(self, individual):
        """Add an individual to the current population."""
        self.population.add_individual_to_pop(individual)

    def add_new_individual(self):
        """Add new individual to the current population."""
        self.population.add_new_individual()

    def get_all_fitness_eval(self):
        """Count all evaluation and set they to 0."""
        for i, member in enumerate(self.population):
            self.num_of_fitness_eval += member.num_fitness_eval
            member.num_fitness_eval = 0

    def print_best_values(self, top_n=4):
        """Print the top n pieces fitness values"""
        print('Best fitness values:')
        for j in range(top_n if self.population_size >= top_n else self.population_size):
            print('{0:.3f}'.format(self.population[j].fitness))
        print('\n')
