import yaml
import time
import fitness_functions
import memetics
import selections
import mutations
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from gen_alg import GeneticAlgorithm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(1))
# sess1 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
#
# gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(3))
# sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
#
# time.sleep(10)
# sess1.close()
# sess2.close()
# exit()

config = yaml.load(open("config.yml", 'r'))
mutation_probability = config["mutation_probability"]

ga = GeneticAlgorithm(population_size=config["population_size"],
                      generations=config["generations"],
                      chromosome_size=config["chromosome_size"],
                      patience=config["patience"],
                      lamarck=config["lamarck"],
                      pool_size=config["pool_size"],
                      pool=config["pool"],
                      thread=config["thread"])

ff = fitness_functions.FitnessFunction(5)

ga.fitness_function = ff.calculate
ga.selection_function = selections.random_selection
ga.mutation_function = mutations.mutation(1)        # basic mutation - 1    bacterial mutation - 2
ga.first = 0
ga.memetic_function = memetics.lamarck_one

print('Run genetic algorithm\n')

print('Population size: {}\nChromosome size: {}\nLamarck: {}\nPool: {}\nThread: {}\n'.
      format(config["population_size"], config["chromosome_size"], config["lamarck"], ga.pool, ga.thread))

ga.run()
#ff.model.base_line(500)

#ff.model.sess.close()
best = ga.best_individual()
result = np.array(best[1])
result = result.reshape((ff.size, ff.size, 3))
plt.imshow(result)
plt.show()
