import time
import matplotlib.pyplot as plt
import numpy as np

from elements.population import Swarm

from pathos.multiprocessing import Pool
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import threading


class ParticleSwarm(object):
    """
    Particle Swarm class.
    This is the main class that controls the functionality of the Particle Swarm.
    """

    def __init__(self,
                 population_size=50,
                 chromosome_size=10,
                 iterations=100,
                 patience=None,
                 lamarck=False,
                 pool_size=cpu_count(),
                 pool=False,
                 thread=False):

        self.patience = patience
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.iterations = iterations
        self.lamarck = lamarck
        self.pool_size = pool_size
        self.pool = pool
        self.thread = thread

        self.population = None

        self.fitness_function = None
        self.memetic_function = None

        if self.pool:
            self.thread = False

    def create_first_generation(self):
        """
        Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.population = Swarm(self.population_size, self.chromosome_size, self.fitness_function)
        self.calculate_population_fitness()
        self.population.rank_population()
        self.population.init_global_best()
        self.population.set_global_best()
        self.population.set_personal_bests()

    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """

        self.iterate()
        if self.lamarck:
            self.local_search()
        self.calculate_population_fitness()
        self.population.rank_population()
        self.population.set_global_best()
        self.population.set_personal_bests()

    def run(self):
        """Run (solve) the Genetic Algorithm."""
        self.create_first_generation()

        if self.patience:
            no_improvement = 0
            iterations = 1
            best_fitness = self.population.get_the_best().fitness

            while no_improvement < self.patience:
                start = time.time()
                print('{}. iteration'.format(iterations))
                self.create_next_generation()
                print('best fitness values: ', [self.population.get_all()[j].fitness for j in range(4 if self.population_size >= 4 else self.population_size)])
                # print('best member: ', max(self.population.get_the_best().genes), min(self.population.get_the_best().genes))

                loss, acc = self.fitness_function(self.population.get_the_best(), acc=True)
                print("Accurate: {0:.2f}%\t".format(acc * 100), "Loss: {0:.2f}\t".format(loss))
                loss, acc = self.fitness_function(self.population.global_best, acc=True)
                print("Accurate: {0:.2f}%\t".format(acc * 100), "Loss: {0:.2f}\t".format(loss))

                end = time.time()
                print('Process time: {0:.2f}s\n'.format(end - start))

                # print("max of V: {0:.3f}".format(np.amax(self.population.current_generation[0].velocity)))
                # print("min of V: {0:.3f}".format(np.amin(self.population.current_generation[0].velocity)))
                # print("min of V: {0:.3f}\n".format(np.average(self.population.current_generation[0].velocity)))
                #
                # print("max of G: {0:.3f}".format(np.amax(self.population.current_generation[0].genes)))
                # print("min of G: {0:.3f}\n".format(np.amin(self.population.current_generation[0].genes)))
                # print("avg of G: {0:.3f}\n".format(np.average(self.population.current_generation[0].genes)))
                #
                # print("avg of dG: {0:.3f}".format(np.average(self.population.current_generation[0].genes - self.population.current_generation[100].genes)))
                # print("min of dG {0:.3f}".format(np.amin(self.population.current_generation[0].genes - self.population.current_generation[100].genes)))
                # print("max of dG: {0:.3f}\n".format(np.amax(self.population.current_generation[0].genes - self.population.current_generation[100].genes)))
                #
                # print(self.population.current_generation[100].genes[:10])
                # print(self.population.current_generation[0].genes[:10])
                # input()

                # if iterations % 300 == 0:
                #     # result = np.array(self.population.get_the_best().genes)
                #     # result = result.reshape((50, 50, 3))
                #     # plt.imshow(result)
                #     # plt.show()
                #
                #     result = np.array(self.population.global_best.genes)
                #     result = result.reshape((50, 50, 3))
                #     plt.imshow(result)
                #     plt.show()

                    # print("max of W: {0:.2f}".format(np.amax(self.population.get_the_best().velocity)))
                    # print("min of W: {0:.2f}\n".format(np.amin(self.population.get_the_best().velocity)))
                    # input()

                if best_fitness > self.population.get_the_best().fitness:
                    no_improvement = 0
                    best_fitness = self.population.get_the_best().fitness

                no_improvement += 1
                iterations += 1

        else:
            for i in range(0, self.iterations):
                start = time.time()
                print('{}. iterations'.format(i + 1))
                self.create_next_generation()
                print('best fitness values: ', [self.population.get_all()[j].fitness for j in range(10 if self.population_size >= 10 else self.population_size)])
                end = time.time()
                print('Process time: {0:.2f}s\n'.format(end - start))

    def best_individual(self):
        """Return the individual with the best fitness in the current generation."""
        best = self.population.get_the_best()
        return best.fitness, best.genes

    def last_generation(self):
        """Return members of the last generation as a generator function."""
        return ((member.fitness, member.genes) for member in self.population.get_all())

    def local_search(self):
        """Gradient search based on memetic evolution."""
        start = time.time()

        if self.pool: #self.pool:
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

        if 0: #self.pool:
            print('Use process pool for calculate fitness with pool size {}.'.format(self.pool_size))
            p = Pool(self.pool_size)
            fitness_values = p.map(self.fitness_function, self.population.get_all())

            for i, value in enumerate(fitness_values):
                self.population.set_fitness(value, i)

            p.terminate()
        elif 0: #self.thread:
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

    def iterate(self):
        """One iteration on the swarm."""
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
            for member in self.population.get_all():
                member.update_velocity()
                member.iterate()
                # member.sso_iterate()

        end = time.time()
        print('Calculate pop iteration time: {0:.2f}s'.format(end - start))

