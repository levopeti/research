import time
import matplotlib.pyplot as plt
import numpy as np

from population import Population
from chromosome import Chromosome
from selections import crossover
from base_alg_class import BaseAlgorithmClass

from pathos.multiprocessing import Pool
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor


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
                 pool=False):
        super().__init__(population_size=population_size,
                         chromosome_size=chromosome_size,
                         max_iteration=max_iteration,
                         max_fitness_eval=max_fitness_eval,
                         patience=patience,
                         lamarck=lamarck,
                         pool_size=pool_size,
                         pool=pool)

    def create_new_population(self):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        selection = self.selection_function
        start = time.time()

        for _ in range(self.population_size):
            self.crossover_function(self.population, selection)

        for _ in range(self.population_size // 2):
            individual = Chromosome(self.chromosome_size, self.fitness_function)
            individual.fitness = individual.calculate_fitness()
            self.population.add_individual_to_pop(individual)

        end = time.time()
        print('Create new pop: {0:.2f}s'.format(end - start))

    def create_first_generation(self):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.population = Population(self.population_size, self.chromosome_size, self.fitness_function)
        self.calculate_population_fitness()
        self.population.rank_population()

    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """

        self.create_new_population()
        if self.lamarck:
            self.local_search()
        self.mutation()
        self.calculate_population_fitness()
        self.population.rank_population()
        self.population.insp_pop_size()

    def run(self):
        """Run (solve) the Genetic Algorithm."""
        self.create_first_generation()

        if self.patience:
            no_improvement = 0
            generation = 1
            best_fitness = self.population.get_the_best().fitness

            while no_improvement < self.patience:
                start = time.time()
                print('{}. generation'.format(generation))
                self.create_next_generation()
                print('best fitness values: ', [self.population.get_all()[j].fitness for j in range(8 if self.population_size >= 8 else self.population_size)])
                # print('best member: ', max(self.population.get_the_best().genes), min(self.population.get_the_best().genes))

                loss, acc = self.fitness_function(self.population.get_the_best(), acc=True)
                print("Accurate: {0:.2f}%\t".format(acc * 100), "Loss: {0:.2f}\t".format(loss))

                end = time.time()
                print('Process time: {0:.2f}s\n'.format(end - start))

                if best_fitness > self.population.get_the_best().fitness:
                    no_improvement = 0
                    best_fitness = self.population.get_the_best().fitness

                no_improvement += 1
                generation += 1

        else:
            for i in range(0, self.generations):
                start = time.time()
                print('{}. generation'.format(i + 1))
                self.create_next_generation()
                print('best fitness values: ', [self.population.get_all()[j].fitness for j in range(10 if self.population_size >= 10 else self.population_size)])
                end = time.time()
                print('Process time: {0:.2f}s\n'.format(end - start))

    def best_individual(self):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.population.get_the_best()
        return best.fitness, best.genes

    def last_generation(self):
        """Return members of the last generation as a generator function."""
        return ((member.fitness, member.genes) for member in self.population.get_all())

    def mutation(self):
        """Mutation on the all members of the population except the best.
        """
        start = time.time()

        if 1:
            print('Use process pool for mutation with pool size {}.'.format(self.pool_size))
            p = Pool(self.pool_size)
            members = p.map(self.mutation_function, self.population.get_all()[self.first:], chunksize=1)
            for i, member in enumerate(members):
                self.population.set_fitness(member.fitness, i + self.first)
                self.population.set_genes(member.genes, i + self.first)

            p.terminate()
        elif 0:
            print('Use thread pool for mutation with pool size {}.'.format(self.pool_size))

            # threads = []
            # for i, member in enumerate(self.population.get_all()):
            #     #start = time.time()
            #     t = threading.Thread(target=self.mutation_function, args=(member,))
            #     threads.append(t)
            #     t.start()
            #
            #     if i % self.pool_size == 0 or i == len(self.population.get_all()) - 1:
            #         for t in threads:
            #             t.join()
            #
            #         #end = time.time()
            #         threads = []
            #         #print('Thread process: {0:.2f}s'.format(end - start))

            with ThreadPoolExecutor(max_workers=self.pool_size) as p:
                p.map(self.mutation_function, self.population.get_all(), chunksize=1)

        else:
            for member in self.population.get_all()[self.first:]:
                self.mutation_function(member)

        end = time.time()
        print('Mutation time: {0:.2f}s'.format(end - start))

    def local_search(self):
        """Gradient search based on memetic evolution.
        """
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

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        start = time.time()

        if 0:  # self.pool:
            print('Use process pool for calculate fitness with pool size {}.'.format(self.pool_size))
            p = Pool(self.pool_size)
            fitness_values = p.map(self.fitness_function, self.population.get_all())

            for i, value in enumerate(fitness_values):
                self.population.set_fitness(value, i)

            p.terminate()
        elif 0:  # self.thread:
            print('Use thread pool for calculate fitness with pool size {}.'.format(self.pool_size))
            with ThreadPoolExecutor(max_workers=self.pool_size) as p:
                fitness_values = p.map(self.fitness_function, self.population.get_all())

                for i, value in enumerate(fitness_values):
                    self.population.set_fitness(value, i)
        else:
            for i, member in enumerate(self.population.get_all()):
                fitness_values = self.fitness_function(member)
                self.population.set_fitness(fitness_values, i)

        end = time.time()
        print('Calculate pop fitness time: {0:.2f}s'.format(end - start))
