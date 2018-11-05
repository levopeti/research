import yaml
import fitness_functions
import selections
import mutations
import os
import numpy as np
import matplotlib.pyplot as plt

from pso_alg import ParticleSwarm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

config = yaml.load(open("config.yml", 'r'))
mutation_probability = config["mutation_probability"]

pso = ParticleSwarm(population_size=config["population_size"],
                    iterations=config["generations"],
                    chromosome_size=config["chromosome_size"],
                    patience=config["patience"],
                    lamarck=config["lamarck"],
                    pool_size=config["pool_size"],
                    pool=config["pool"],
                    thread=config["thread"])

ff = fitness_functions.FitnessFunction(5)

pso.fitness_function = ff.calculate
pso.memetic_function = ff.train_steps(number_of_steps=1)

print('Run genetic algorithm\n')

print('Population size: {}\nChromosome size: {}\nLamarck: {}\nPool: {}\nThread: {}\n'.
      format(config["population_size"], config["chromosome_size"], config["lamarck"], pso.pool, pso.thread))

pso.run()
#ff.model.base_line(500)

# best = pso.best_individual()
# result = np.array(best[1])
# result = result.reshape((ff.size, ff.size, 3))
# plt.imshow(result)
# plt.show()
