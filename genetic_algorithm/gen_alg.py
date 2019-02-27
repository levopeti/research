import time
import matplotlib.pyplot as plt
import numpy as np
import functools

from population import Population
from chromosome import Chromosome

from base_alg_class import BaseAlgorithmClass

from pathos.multiprocessing import Pool
from multiprocessing import cpu_count, Lock, Value, Manager
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

        if self.selection_function:
            self.iteration_steps.append(self.selection)

        if self.memetic_function:
            self.iteration_steps.append(self.local_search)

        if self.mutation_function:
            self.iteration_steps.append(self.mutation)

    def selection(self):
        """Create an individuals using the genetic operators (selection, crossover) supplied."""

        start = time.time()

        for _ in range(self.num_of_crossover):
            self.population.crossover(self.selection_function)

        for _ in range(self.num_of_new_individual):
            self.population.add_new_individual()

        end = time.time()
        print('Selection time: {0:.2f}s\n'.format(end - start))

    def mutation(self):
        """
        Mutation on all members of the population.
        If elitism is True the best is exception.
        """
        start = time.time()
        print("Mutation:")

        if self.pool:
            print('Use process pool for mutation with pool size {}.'.format(self.pool_size))
            p = Pool(self.pool_size)
            manager = Manager()
            lock = manager.Lock()
            counter = manager.Value('i', 0)
            pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=100), ETA()], maxval=len(self.population)).start()

            def pool_mutation(inside_lock, inside_counter, inside_member):
                inside_member.apply_on_chromosome(self.mutation_function)

                inside_lock.acquire()
                inside_counter.value += 1
                pbar.update(inside_counter.value)
                inside_lock.release()

                return inside_member

            func = functools.partial(pool_mutation, lock, counter)
            first = 1 if self.elitism else 0

            members = p.map(func, self.population[first:], chunksize=1)

            if self.elitism:
                members.append(self.population[0])

            self.population.current_population = members
            p.terminate()
        else:
            pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=100), ETA()], maxval=len(self.population)).start()
            ignor_first = self.elitism

            for i, member in enumerate(self.population):
                pbar.update(i + 1)
                if not ignor_first:
                    member.apply_on_chromosome(self.mutation_function)
                ignor_first = False

        pbar.finish()
        end = time.time()
        print('Mutation time: {0:.2f}s\n'.format(end - start))

    def local_search(self):
        """Gradient search based on memetic evolution."""
        start = time.time()
        print("Local search:")

        if self.pool:
            print('Use process pool for local search with pool size {}.'.format(self.pool_size))
            p = Pool(self.pool_size)
            manager = Manager()
            lock = manager.Lock()
            counter = manager.Value('i', 0)
            pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=100), ETA()], maxval=len(self.population)).start()

            def pool_memetic(inside_lock, inside_counter, inside_member):
                inside_member.apply_on_chromosome(self.memetic_function)

                inside_lock.acquire()
                inside_counter.value += 1
                pbar.update(inside_counter.value)
                inside_lock.release()

                return inside_member

            func = functools.partial(pool_memetic, lock, counter)
            members = p.map(func, self.population[:], chunksize=1)

            self.population.current_population = members
            p.terminate()
        else:
            pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=100), ETA()], maxval=len(self.population)).start()

            for i, member in enumerate(self.population):
                pbar.update(i + 1)
                member.apply_on_chromosome(self.memetic_function)

        pbar.finish()
        end = time.time()
        print('Local search time: {0:.2f}s\n'.format(end - start))

    def calculate_fitness(self):
        """
        Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        start = time.time()
        print("Calculate fitness:")

        if self.pool:
            print('Use process pool for calculate fitness with pool size {}.'.format(self.pool_size))
            p = Pool(self.pool_size)
            manager = Manager()
            lock = manager.Lock()
            counter = manager.Value('i', 0)
            pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=100), ETA()], maxval=len(self.population)).start()

            def pool_fitness(inside_lock, inside_counter, inside_member):
                inside_member.calculate_fitness()

                inside_lock.acquire()
                inside_counter.value += 1
                pbar.update(inside_counter.value)
                inside_lock.release()

                return inside_member

            func = functools.partial(pool_fitness, lock, counter)
            members = p.map(func, self.population[:], chunksize=1)

            self.population.current_population = members
            p.terminate()
        else:
            pbar = ProgressBar(widgets=[Percentage(), Bar(dec_width=100), ETA()], maxval=len(self.population)).start()

            for i, member in enumerate(self.population):
                pbar.update(i + 1)
                member.calculate_fitness()

        pbar.finish()
        end = time.time()
        print('Time of calculation fitness: {0:.2f}s\n'.format(end - start))
