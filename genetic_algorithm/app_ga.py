import yaml
import time
import fitness_functions
from callbacks import LogToFile
import os
import numpy as np
import matplotlib.pyplot as plt

from gen_alg import GeneticAlgorithm

# TODO: LogsCallback, Fitness function class

with open("config.yml", 'r') as config_file:
    config = yaml.load(config_file)

ga = GeneticAlgorithm(**config)

ff = fitness_functions.FitnessFunction(1)

ltf = LogToFile(file_path="/home/biot/projects/research/logs")

ga.compile(config=config,
           fitness_function=ff,
           remote_config="config.yml",
           callbacks=None)

print("Run genetic algorithm\n")

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
