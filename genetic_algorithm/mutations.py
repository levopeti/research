import random


def mutation(mut_id):

    def basic_mutation(member):
        """Mutation on the all members of the population. the best is exception.
        """
        mutation_probability = 0.2
        #print(666)
        for i in range(member.chrom_size):

            if random.random() < mutation_probability:
                member.genes[i] = random.random()
                #member.fitness = member.calculate_fitness()

        return member

    def bac_mutation(member):
        """Bacterial mutation.
        """
        #print(111)
        random_index = list(range(member.chrom_size))
        random.shuffle(random_index)

        for i in random_index:
            number = member.genes[i]
            member.genes[i] = random.random()
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

