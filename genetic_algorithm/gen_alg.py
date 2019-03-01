import time
from functools import partial

from population import Population
from base_alg_class import BaseAlgorithmClass

from pathos.multiprocessing import Pool
from multiprocessing import cpu_count, Manager
from progressbar import ProgressBar, Bar, Percentage, ETA


class GeneticAlgorithm(BaseAlgorithmClass):
    """
    Genetic Algorithm class.
    This is the main class that controls the functionality of the Genetic Algorithm.
    """

    def __init__(self,
                 population_size=50,
                 chromosome_size=10,
                 max_iteration=None,
                 max_fitness_eval=None,
                 min_fitness=None,
                 patience=None,
                 pool_size=cpu_count(),
                 pool=False,
                 num_of_new_individual=None,
                 num_of_crossover=None,
                 elitism=False,
                 *args,
                 **kwargs):
        super().__init__(population_size=population_size,
                         chromosome_size=chromosome_size,
                         max_iteration=max_iteration,
                         max_fitness_eval=max_fitness_eval,
                         min_fitness=min_fitness,
                         patience=patience,
                         pool_size=pool_size,
                         pool=pool)

        self.num_of_new_individual = self.population_size // 2 if num_of_new_individual is None else num_of_new_individual
        self.num_of_crossover = self.population_size // 4 if num_of_crossover is None else num_of_crossover
        self.elitism = elitism

    def create_population(self):
        """
        Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.population = Population(self.population_size, self.chromosome_size, self.fitness_function)

    def init_steps(self):
        """Initialize the iteration steps."""
        self.iteration_steps = []

        if self.config["crossover"]:
            self.iteration_steps.append(partial(self.add_new_individuals_function, "Crossover"))

        if self.config["differential_evolution"]:
            self.iteration_steps.append(partial(self.add_new_individuals_function, "Differential evolution"))

        if self.config["invasive_weed"]:
            self.iteration_steps.append(partial(self.add_new_individuals_function, "Invasive weed"))

        if self.config["add_pure_new"]:
            self.iteration_steps.append(partial(self.add_new_individuals_function, "Add pure new"))

        if self.config["mutation"]:
            self.iteration_steps.append(partial(self.modify_one_by_one_function, "Mutation"))

        if self.config["memetic"]:
            self.iteration_steps.append(partial(self.modify_one_by_one_function, "Local search"))

    def add_new_individuals_function(self, name):
        """
        Add new individuals to the population with a given method
        (crossover, differential evolution, invasive weed, add pure individual).
        """
        start = time.time()

        if name == "Crossover":
            for _ in range(self.num_of_crossover):
                self.population.crossover(self.selection_function)

        elif name == "Differential evolution":
            self.population.differential_evolution(**self.config)

        elif name == "Invasive weed":
            self.population.invasive_weed(**self.config)

        elif name == "Add pure new":
            for _ in range(self.num_of_new_individual):
                self.population.add_new_individual()
        else:
            raise NameError("Bad type of function.")

        end = time.time()
        print('{0} time: {1:.2f}s\n'.format(name, end - start))

    def modify_one_by_one_function(self, name):
        """Apply a function (local search, mutation) to all chromosomes."""
        start = time.time()
        print("{}:".format(name))

        if name == "Local search":
            current_function = self.memetic_function
        elif name == "Mutation":
            current_function = self.mutation_function
        else:
            raise NameError("Bad type of function.")

        if self.pool:
            p = Pool(self.pool_size)
            manager = Manager()
            lock = manager.Lock()
            counter = manager.Value('i', 0)
            if self.progress_bar:
                pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=60), ETA()], maxval=len(self.population)).start()

            def pool_function(inside_lock, inside_counter, inside_member):
                inside_member.apply_on_chromosome(current_function)

                inside_lock.acquire()
                inside_counter.value += 1
                if self.progress_bar:
                    pbar.update(inside_counter.value)
                inside_lock.release()

                return inside_member

            func = partial(pool_function, lock, counter)
            first = 1 if self.elitism and name == "Mutation" else 0

            members = p.map(func, self.population[first:])

            if self.elitism and name == "Mutation":
                members.append(self.population[0])

            self.population.current_population = members
            p.terminate()
        else:
            if self.progress_bar:
                pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=60), ETA()], maxval=len(self.population)).start()
            ignor_first = self.elitism and name == "Mutation"

            for i, member in enumerate(self.population):
                if self.progress_bar:
                    pbar.update(i + 1)
                if not ignor_first:
                    member.apply_on_chromosome(current_function)
                ignor_first = False

        if self.progress_bar:
            pbar.finish()

        end = time.time()
        print('{0} time: {1:.2f}s\n'.format(name, end - start))

