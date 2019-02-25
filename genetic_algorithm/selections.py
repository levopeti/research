import random
from operator import attrgetter


def random_selection(population):
    """Select and return a random member of the population."""
    return random.choice(population)


def tournament_selection(population):
    """
    Select a random number of individuals from the population and
    return the fittest member of them all.
    """
    tournament_size = population.pop_size // 2
    members = random.sample(population[:], tournament_size)
    members.sort(key=attrgetter('fitness'), reverse=False)
    return members[0]


def better_half_selection(population):
    """
    Select the better half of individuals from the population and
    return a random member.
    """

    member = random.sample(population[:(population.pop_size // 2)], 1)
    return member[0]



