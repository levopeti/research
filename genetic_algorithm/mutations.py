import random
import numpy as np
import time

np.random.seed(int(time.time()))


def mutation(mut_id):

    def basic_mutation(member):
        """Mutation on the all members of the population. the best is exception.
        """
        mutation_probability = 0.1

        for i in range(member.chrom_size):

            if np.random.rand() < mutation_probability:
                member.genes[i] = (np.random.rand() * 4) - 2

        return member

    def bac_mutation(member):
        """Bacterial mutation.
        """
        random_index = list(range(member.chrom_size))
        random.shuffle(random_index)

        for i in random_index[:300]:  # *1.4s
            number = member.genes[i]
            member.genes[i] = (np.random.rand() * 4) - 2
            new_fitness = member.calculate_fitness()
            if new_fitness < member.fitness:
                member.fitness = new_fitness
            else:
                member.genes[i] = number

        return member

    if mut_id == 1:
        return basic_mutation

    elif mut_id == 2:
        return bac_mutation
