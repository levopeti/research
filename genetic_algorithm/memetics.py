import copy
import random


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


