import copy
import random


def local_search(number_of_steps, step_size, lamarck_random, member):
    """
    Local search for all chromosomes in a given steps.
    :param number_of_steps: Number of steps to the direction of the negative gradient.
    :param step_size: Size of step.
    :param lamarck_random: If True, the sequence in the genes is random.
    :param member: Current chromosome.
    """

    for _ in range(number_of_steps):
        random_index = list(range(member.chromosome_size))
        if lamarck_random:
            random.shuffle(random_index)

        for i in random_index:
            # take one step positive direction
            member.set_test()
            member.genes_test[i] += step_size
            member.resize_invalid_genes_test()
            member.calculate_fitness_test()
            test_is_better = member.apply_test_if_better()

            if not test_is_better:
                # take one step negative direction, if positive is not better
                member.set_test()
                member.genes_test[i] -= step_size
                member.resize_invalid_genes_test()
                member.calculate_fitness_test()
                member.apply_test_if_better()


def lamarck_one(member):
    random_index = list(range(member.chrom_size))
    random.shuffle(random_index)

    for i in random_index:
        old_fitness = float("inf")

        while member.fitness < old_fitness:
            number = member.genes[i]
            member.genes[i] = random.random()
            old_fitness = member.fitness

            #new_fitness = member.calculate_fitness()
            #tmp_member = copy.deepcopy(member)

            if member.genes[i] + member.step_size <= 1:
                member.genes[i] += member.step_size
                tmp_fitness = member.calculate_fitness()
                if tmp_fitness < member.fitness:
                    member.fitness = tmp_fitness
                else:
                    member.genes[i] = number

            if member.genes[i] - member.step_size >= 0 and old_fitness == member.fitness:
                member.genes[i] -= member.step_size
                tmp_fitness = member.calculate_fitness()
                if tmp_fitness < member.fitness:
                    member.fitness = tmp_fitness
                else:
                    member.genes[i] = number

    return member


def lamarck_two(member):
    for i in range(member.chrom_size - 1):
        for j in range(i + 1, member.chromosome_size):
            old_fitness = 0

            while member.fitness > old_fitness:
                tmp_member = copy.deepcopy(member)
                old_fitness = member.fitness

                if tmp_member.genes[i] + member.step_size <= 1 and tmp_member.genes[j] + member.step_size <= 1:
                    tmp_member.genes[i] += member.step_size
                    tmp_member.genes[j] += member.step_size
                    tmp_fitness = tmp_member.calculate_fitness()
                    if tmp_fitness < member.fitness:
                        member = tmp_member
                        member.fitness = tmp_fitness
                        continue

                if tmp_member.genes[i] - member.step_size >= 0 and tmp_member.genes[j] - member.step_size >= 0:
                    tmp_member.genes[i] -= member.step_size
                    tmp_member.genes[j] -= member.step_size
                    tmp_fitness = tmp_member.calculate_fitness()
                    if tmp_fitness < member.fitness:
                        member = tmp_member
                        member.fitness = tmp_fitness

    return member


