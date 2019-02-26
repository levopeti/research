import random
import numpy as np
import time

np.random.seed(int(time.time()))


def mutation(name, mutation_probability, mutation_random):
    """If mutation random is True, the sequence in the genes is random."""

    def basic_mutation(member):
        """Mutation on all genes with mutation probability."""

        random_index = list(range(member.chromosome_size))
        if mutation_random:
            random.shuffle(random_index)

        new_genes = []
        for i in random_index:
            gene = member[i]
            if np.random.rand() < mutation_probability:
                gene = np.random.rand()
            new_genes.append(gene)

        member.genes = new_genes
        member.calculate_fitness()

    def bac_mutation(member):
        """Bacterial on all genes."""
        # TODO: szakdoga
        random_index = list(range(member.chromosome_size))
        if mutation_random:
            random.shuffle(random_index)

        for i in random_index:
            member.set_test()
            member.genes_test[i] = np.random.rand()
            member.calculate_fitness_test()
            member.apply_test_if_better()

    if name == "basic":
        return basic_mutation

    elif name == "bacterial":
        return bac_mutation
