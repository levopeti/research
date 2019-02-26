import yaml
import time
import functools
import fitness_functions
from memetics import local_search
from selections import random_selection, tournament_selection, better_half_selection
import mutations
import os
import numpy as np
import matplotlib.pyplot as plt

from gen_alg import GeneticAlgorithm

# TODO: Callbacks, RemoteControl

config = yaml.load(open("config.yml", 'r'))
mutation_probability = config["mutation_probability"]
mutation_random = config["mutation_random"]

ga = GeneticAlgorithm(**config)

ff = fitness_functions.FitnessFunction(1)
mut = mutations.mutation("bacterial", mutation_probability, mutation_random)
mem_f = functools.partial(local_search, 1, 0.1, True)

# TODO: config
ga.compile(fitness_function=ff,
           selection_function=tournament_selection,
           mutation_function=mut,
           memetic_function=mem_f)

print('Run genetic algorithm\n')

for key, item in config.items():
    if item is not None:
        print("{0}: {1}".format(key, item))
print('\n')

ga.run()
#ff.model.base_line(500)

#ff.model.sess.close()
# best = ga.best_individual()
# result = np.array(best[1])
# result = result.reshape((ff.size, ff.size, 3))
# plt.imshow(result)
# plt.show()
