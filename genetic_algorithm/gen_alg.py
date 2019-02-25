import time
import matplotlib.pyplot as plt
import numpy as np

from population import Population
from chromosome import Chromosome

from base_alg_class import BaseAlgorithmClass

from pathos.multiprocessing import Pool
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
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
                 patience=None,
                 lamarck=False,
                 pool_size=cpu_count(),
                 pool=False,
                 num_of_new_individual=None,
                 num_of_crossover=None,
                 elitism=True,
                 *args,
                 **kwargs):
        super().__init__(population_size=population_size,
                         chromosome_size=chromosome_size,
                         max_iteration=max_iteration,
                         max_fitness_eval=max_fitness_eval,
                         patience=patience,
                         lamarck=lamarck,
                         pool_size=pool_size,
                         pool=pool)

        self.num_of_new_individual = self.population_size // 2 if num_of_new_individual is None else num_of_new_individual
        self.num_of_crossover = self.population_size if num_of_crossover is None else num_of_crossover
        self.elitism = elitism

    def create_population(self):
        """
        Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.population = Population(self.population_size, self.chromosome_size, self.fitness_function)

    def init_steps(self):
        """Initialize the iteration steps."""

        self.iteration_steps.append(self.selection)
        if self.lamarck:
            self.iteration_steps.append(self.local_search)
        # if self.mutation_function:
        #     self.iteration_steps.append(self.mutation)
        self.iteration_steps.append(self.calculate_fitness)
        self.iteration_steps.append(self.rank_population)
        self.iteration_steps.append(self.cut_pop_size)

    def selection(self):
        """Create an individuals using the genetic operators (selection, crossover) supplied."""

        start = time.time()

        for _ in range(self.num_of_crossover):
            self.population.crossover(self.selection_function)

        for _ in range(self.num_of_new_individual):
            self.population.add_new_individual()

        end = time.time()
        print('Selection time: {0:.2f}s'.format(end - start))

    def mutation(self):
        """
        Mutation on all members of the population.
        If elitism is True the best is exception.
        """
        start = time.time()
        print("Mutation:")

        pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=100), ETA()], maxval=self.population_size).start()

        if self.pool:
            pass
            # print('Use process pool for mutation with pool size {}.'.format(self.pool_size))
            # p = Pool(self.pool_size)
            # members = p.map(self.mutation_function, self.population.get_all()[self.first:], chunksize=1)
            #
            # for i, member in enumerate(members):
            #     self.population.set_fitness(member.fitness, i + self.first)
            #     self.population.set_genes(member.genes, i + self.first)
            #
            # p.terminate()
        else:
            ignor_first = self.elitism
            for i, member in enumerate(self.population):
                pbar.update(i + 1)
                if not ignor_first:
                    member.mutation(self.mutation_function)
                ignor_first = False

        pbar.finish()
        end = time.time()
        print('Time of mutation: {0:.2f}s\n'.format(end - start))

    def local_search(self):
        """Gradient search based on memetic evolution."""

        start = time.time()

        if self.pool:  # self.pool:
            print('Use process pool for local search with pool size {}.'.format(self.pool_size))
            p = Pool(self.pool_size)
            members = p.map(self.memetic_function, self.population.get_all())

            for i, member in enumerate(members):
                self.population.set_genes(member, i)

            p.terminate()
        elif self.thread:
            print('Use thread pool for local search with pool size {}.'.format(self.pool_size))
            with ThreadPoolExecutor(max_workers=self.pool_size) as p:
                members = p.map(self.memetic_function, self.population.get_all())

                for i, member in enumerate(members):
                    self.population.set_genes(member, i)
        else:
            for i, member in enumerate(self.population.get_all()):
                member = self.memetic_function(member)
                self.population.set_genes(member, i)

        end = time.time()
        print('Memetic for weights time: {0:.2f}s'.format(end - start))

    def calculate_fitness(self):
        """
        Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        start = time.time()
        print("Calculate fitness:")

        pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=100), ETA()], maxval=self.population_size).start()

        if self.pool:
            pass
            # print('Use process pool for calculate fitness with pool size {}.'.format(self.pool_size))
            # p = Pool(self.pool_size)
            # fitness_values = p.map(self.fitness_function, self.population.get_all())
            #
            # for i, value in enumerate(fitness_values):
            #     self.population.set_fitness(value, i)
            #
            # p.terminate()
        else:
            for i, member in enumerate(self.population):
                pbar.update(i + 1)
                member.calculate_fitness()

        pbar.finish()
        end = time.time()
        print('Time of calculation fitness: {0:.2f}s\n'.format(end - start))
